"""Triton kernels for nanoops (Tier 3 — opt-in CUDA kernel rewrites).

Each kernel here mirrors the math of the corresponding Python op in
`functional.py` but fuses multiple passes into a single GPU kernel.
Wins come from:
  - fewer kernel-launch overheads (~30 us each)
  - fewer round-trips to HBM (a chain like `norm → linear → relu²` is
    3× HBM traffic; fused is 1× input read + 1× output write)
  - smaller working set (no intermediate buffers between ops)

Activated via env var per op (e.g. `NANOOPS_TRITON_NORM_MLP=1`).
If triton isn't installed or the env var isn't set, callers fall back
to the eager Python implementation in `functional.py`.
"""

from __future__ import annotations

import os
import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# Cached at module load — same pattern as NANOOPS_LOOKUP_SORTED so the
# hot path doesn't pay an os.environ.get dict lookup on every call.
NORM_MLP_ENABLED = _HAS_TRITON and bool(os.environ.get("NANOOPS_TRITON_NORM_MLP"))


# ─────────────────────────────────────────────────────────────────────
# Fused RMSNorm + Linear + ReluSquare
#
# Forward math (per row m of input x ∈ ℝ^(M, K)):
#     y_norm[m, k] = x[m, k] * rsqrt(mean_k(x[m, k]²) + eps)
#     x_hat[m, k] = y_norm[m, k] * norm_weight[k]                     (RMSNorm)
#     z[m, n] = sum_k x_hat[m, k] * lin_weight[n, k]                  (Linear)
#     y[m, n] = max(z[m, n], 0)² = relu(z)²                           (ReluSquare)
#
# Forward fuses all three into one tiled (BLOCK_M × BLOCK_N) Triton
# kernel: each tile streams K in BLOCK_K chunks, computing rms per row
# in pass 1 and the matmul+relu² in pass 2. RMSNorm's per-row rsqrt
# lives in registers across both passes.
#
# Backward saves (x, rms_inv, x_hat, z) in ctx and uses:
#   - a small Triton kernel for the relu²-backward ⊙ elementwise step
#     (dz = 2 * relu(z) * dy)
#   - torch.matmul (cuBLAS) for the linear backward (dW = dz^T @ x_hat,
#     dx_hat = dz @ W) — cuBLAS already wins at big matmuls
#   - a Triton kernel for the RMSNorm backward + dnw reduction
#     (computes dx, dnw together with one K-pass per row)
# ─────────────────────────────────────────────────────────────────────


if _HAS_TRITON:

    @triton.jit
    def _norm_linear_relu_square_fwd_kernel(
        x_ptr, norm_w_ptr, lin_w_ptr,
        y_ptr, z_ptr, rms_inv_ptr,
        M, N, K,
        eps,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        stride_zm, stride_zn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Forward kernel. Computes y = relu(norm(x) @ W^T)² for one (BLOCK_M, BLOCK_N)
        tile, and saves z (= norm(x) @ W^T, pre-relu²) + rms_inv for backward.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < M
        col_mask = cols < N

        # Pass 1: per-row mean(x²). Computed locally per m-tile; later
        # cached in `rms_inv_ptr` so the backward doesn't re-derive it.
        sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            ks = k_start + tl.arange(0, BLOCK_K)
            k_mask = ks < K
            x_ptrs = x_ptr + rows[:, None] * stride_xm + ks[None, :] * stride_xk
            x = tl.load(
                x_ptrs,
                mask=row_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            sum_sq += tl.sum(x * x, axis=1)
        rms_inv = 1.0 / tl.sqrt(sum_sq / K + eps)

        # Only the n=0 tile writes rms_inv (it doesn't depend on n) —
        # avoid redundant writes by checking pid_n.
        if pid_n == 0:
            tl.store(rms_inv_ptr + rows, rms_inv, mask=row_mask)

        # Pass 2: matmul x_hat @ W^T into fp32 accumulator, then relu².
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            ks = k_start + tl.arange(0, BLOCK_K)
            k_mask = ks < K
            x_ptrs = x_ptr + rows[:, None] * stride_xm + ks[None, :] * stride_xk
            x = tl.load(
                x_ptrs,
                mask=row_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            nw = tl.load(norm_w_ptr + ks, mask=k_mask, other=0.0).to(tl.float32)
            x_hat = x * rms_inv[:, None] * nw[None, :]
            w_ptrs = (
                lin_w_ptr
                + cols[:, None] * stride_wn
                + ks[None, :] * stride_wk
            )
            w = tl.load(
                w_ptrs,
                mask=col_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            # input_precision="ieee" disables Triton's default TF32 downcast
            # on Ampere. TF32 has only 10-bit mantissa, while PyTorch's `@`
            # keeps full fp32 (under default torch.set_float32_matmul_precision
            # = "highest") — without this flag, fp32 parity drifts by ~1%.
            acc += tl.dot(x_hat, tl.trans(w), input_precision="ieee")

        # Save z (pre-relu²) for backward — backward reads it to compute
        # 2 * relu(z) * dy. Storing in fp32 keeps the sign info exact.
        z_ptrs = z_ptr + rows[:, None] * stride_zm + cols[None, :] * stride_zn
        tl.store(
            z_ptrs,
            acc.to(z_ptr.dtype.element_ty),
            mask=row_mask[:, None] & col_mask[None, :],
        )

        # Epilogue: y = relu(z)²
        relu_z = tl.where(acc > 0.0, acc, 0.0)
        y = relu_z * relu_z
        y_ptrs = y_ptr + rows[:, None] * stride_ym + cols[None, :] * stride_yn
        tl.store(
            y_ptrs,
            y.to(y_ptr.dtype.element_ty),
            mask=row_mask[:, None] & col_mask[None, :],
        )

    @triton.jit
    def _relu_square_bwd_kernel(
        z_ptr, dy_ptr, dz_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """dz = 2 * relu(z) * dy. Bandwidth-bound elementwise."""
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        z = tl.load(z_ptr + offs, mask=mask, other=0.0)
        dy = tl.load(dy_ptr + offs, mask=mask, other=0.0)
        relu_z = tl.where(z > 0.0, z, 0.0)
        dz = 2.0 * relu_z * dy
        tl.store(dz_ptr + offs, dz, mask=mask)

    @triton.jit
    def _rms_norm_bwd_kernel(
        x_ptr, rms_inv_ptr, nw_ptr,
        dxhat_ptr,
        dx_ptr, dnw_partial_ptr,
        M, K,
        stride_xm, stride_xk,
        stride_dxhat_m, stride_dxhat_k,
        stride_dx_m, stride_dx_k,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """RMSNorm backward + partial dnw (reduced over M into row tiles).

        Per row m:
          g_eff[m, k] = dx_hat[m, k] * nw[k]
          y_norm[m, k] = x[m, k] * rms_inv[m]
          inner[m] = mean_k(g_eff[m, k] * y_norm[m, k])
          dx[m, k] = rms_inv[m] * (g_eff[m, k] - y_norm[m, k] * inner[m])
          dnw_partial[m_tile, k] = sum_{m in tile} (dx_hat[m, k] * y_norm[m, k])

        dnw_partial is (num_m_tiles, K); caller does a final sum over the
        m_tile axis to get dnw. (Avoids atomicAdd on a (K,) buffer at the
        cost of one extra small reduction.)
        """
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = rows < M
        k_mask = ks < K

        # Load full K for this (m_tile) — need to compute inner over full K.
        # Strategy: pass 1 computes inner per row with a separate K-loop;
        # pass 2 computes dx for THIS k-tile.
        rms_inv = tl.load(rms_inv_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        nw = tl.load(nw_ptr + ks, mask=k_mask, other=0.0).to(tl.float32)

        # Pass 1: compute inner[m] = mean over k of g_eff * y_norm
        inner = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k2_start in range(0, K, BLOCK_K):
            ks2 = k2_start + tl.arange(0, BLOCK_K)
            k2_mask = ks2 < K
            x_ptrs = x_ptr + rows[:, None] * stride_xm + ks2[None, :] * stride_xk
            x = tl.load(
                x_ptrs, mask=row_mask[:, None] & k2_mask[None, :], other=0.0,
            ).to(tl.float32)
            dxh_ptrs = (
                dxhat_ptr
                + rows[:, None] * stride_dxhat_m
                + ks2[None, :] * stride_dxhat_k
            )
            dxh = tl.load(
                dxh_ptrs, mask=row_mask[:, None] & k2_mask[None, :], other=0.0,
            ).to(tl.float32)
            nw2 = tl.load(nw_ptr + ks2, mask=k2_mask, other=0.0).to(tl.float32)
            y_norm2 = x * rms_inv[:, None]
            g_eff2 = dxh * nw2[None, :]
            inner += tl.sum(g_eff2 * y_norm2, axis=1)
        inner = inner / K  # mean

        # Pass 2: compute dx for THIS k-tile
        x_ptrs = x_ptr + rows[:, None] * stride_xm + ks[None, :] * stride_xk
        x = tl.load(
            x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float32)
        dxh_ptrs = (
            dxhat_ptr
            + rows[:, None] * stride_dxhat_m
            + ks[None, :] * stride_dxhat_k
        )
        dxh = tl.load(
            dxh_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float32)
        y_norm = x * rms_inv[:, None]
        g_eff = dxh * nw[None, :]
        dx_tile = rms_inv[:, None] * (g_eff - y_norm * inner[:, None])
        dx_ptrs = dx_ptr + rows[:, None] * stride_dx_m + ks[None, :] * stride_dx_k
        tl.store(
            dx_ptrs,
            dx_tile.to(dx_ptr.dtype.element_ty),
            mask=row_mask[:, None] & k_mask[None, :],
        )

        # Per-m-tile partial dnw[k]: sum over this tile's rows of (dxh * y_norm)
        dnw_partial = tl.sum(dxh * y_norm, axis=0)  # (BLOCK_K,)
        dnw_p_ptrs = dnw_partial_ptr + pid_m * K + ks
        tl.store(dnw_p_ptrs, dnw_partial.to(dnw_partial_ptr.dtype.element_ty), mask=k_mask)


class NormMLPReluSquare(torch.autograd.Function):
    """Fused full MLP block: y = relu(RMSNorm(x) @ W_fc.T)² @ W_proj.T.

    Math (per row m):
        x_hat[m, k] = x[m, k] * rsqrt(mean_k(x²) + eps) * norm_w[k]       (RMSNorm)
        z[m, n]     = sum_k x_hat[m, k] * W_fc[n, k]                     (Linear: c_fc)
        r[m, n]     = relu(z[m, n])²                                     (ReluSquare)
        y[m, p]     = sum_n r[m, n] * W_proj[p, n]                       (Linear: c_proj)

    Mixed-implementation strategy:
      Forward:  Triton fuses (norm + c_fc + relu²) into one kernel, saves
                z and rms_inv. The second linear (c_proj) uses
                torch.matmul (cuBLAS) — fusing it into the same kernel
                would require holding the (M, 4K) intermediate r in
                shared memory across the second matmul, which doesn't
                fit on Ampere SMs at nanchat scale.
      Backward: relu²-backward is Triton (elementwise);  both linear
                backwards use cuBLAS; RMSNorm backward is Triton
                (per-row reduction + elementwise).

    What's saved in ctx:
      x, norm_w, W_fc, W_proj, z, rms_inv
    r is NOT saved — we recompute it from z (`relu(z)²`) when needed for
    `dW_proj = dy^T @ r`. Saves M*4K floats (~80 MB at nanchat d24 MLP).
    """

    @staticmethod
    def forward(ctx, x, norm_weight, fc_weight, proj_weight, eps=1e-6):
        assert x.is_cuda and x.is_contiguous()
        assert norm_weight.is_cuda and fc_weight.is_cuda and proj_weight.is_cuda
        M, K = x.shape
        N_fc, K_w = fc_weight.shape
        K_proj_out, N_proj_in = proj_weight.shape
        assert K == K_w, f"x last dim {K} != fc_weight in dim {K_w}"
        assert N_fc == N_proj_in, (
            f"fc out dim {N_fc} != proj in dim {N_proj_in}"
        )

        # Block sizes chosen to fit 3090's 100 KB shared memory per SM.
        # Larger tiles = better arithmetic intensity but need room for
        # x_hat (BLOCK_M×BLOCK_K) + w slab (BLOCK_N×BLOCK_K) + acc
        # (BLOCK_M×BLOCK_N), all in fp32. 32×64×32 → ~16 KB total.
        BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
        # r will hold relu(z)² (the c_fc output post-activation). We
        # materialize it because c_proj is done by torch.matmul.
        r = torch.empty((M, N_fc), dtype=x.dtype, device=x.device)
        z = torch.empty((M, N_fc), dtype=x.dtype, device=x.device)
        rms_inv = torch.empty((M,), dtype=torch.float32, device=x.device)

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N_fc, BLOCK_N))
        _norm_linear_relu_square_fwd_kernel[grid](
            x, norm_weight, fc_weight,
            r, z, rms_inv,
            M, N_fc, K, eps,
            x.stride(0), x.stride(1),
            fc_weight.stride(0), fc_weight.stride(1),
            r.stride(0), r.stride(1),
            z.stride(0), z.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # Second linear via cuBLAS: y = r @ W_proj.T
        y = r @ proj_weight.t()

        ctx.save_for_backward(x, norm_weight, fc_weight, proj_weight, z, rms_inv)
        ctx.M, ctx.N_fc, ctx.K = M, N_fc, K
        ctx.K_out = K_proj_out
        return y

    @staticmethod
    def backward(ctx, dy):
        x, norm_w, W_fc, W_proj, z, rms_inv = ctx.saved_tensors
        M, N_fc, K = ctx.M, ctx.N_fc, ctx.K
        dy = dy.contiguous()

        # Step A: c_proj backward (cuBLAS).
        # y = r @ W_proj.T → dr = dy @ W_proj ; dW_proj = dy.T @ r.
        # Recompute r = relu(z)² to avoid saving M*N_fc floats in ctx.
        relu_z = z.clamp(min=0)
        r = relu_z * relu_z
        dr = dy @ W_proj         # (M, N_fc)
        dW_proj = dy.t() @ r     # (K_out, N_fc)

        # Step B: relu²-backward (Triton). dz = 2 * relu(z) * dr.
        dz = torch.empty_like(z)
        BLOCK_SIZE = 1024
        grid_rs = (triton.cdiv(M * N_fc, BLOCK_SIZE),)
        _relu_square_bwd_kernel[grid_rs](
            z, dr, dz, M * N_fc, BLOCK_SIZE=BLOCK_SIZE,
        )

        # Step C: c_fc backward (cuBLAS).
        # z = x_hat @ W_fc.T → dx_hat = dz @ W_fc ; dW_fc = dz.T @ x_hat.
        # Recompute x_hat = x * rms_inv * norm_w (cheap elementwise).
        x_hat = x * rms_inv.unsqueeze(1) * norm_w
        dx_hat = dz @ W_fc       # (M, K)
        dW_fc = dz.t() @ x_hat   # (N_fc, K)

        # Step D: RMSNorm backward (Triton) — dx + per-m-tile partial dnw.
        # Block sizes chosen so 2×(BLOCK_M, BLOCK_K) fp32 tiles fit in 3090
        # shared memory (~100 KB). 32×64 fp32 = 8 KB per tile, plenty of room.
        BLOCK_M_BWD, BLOCK_K_BWD = 32, 64
        num_m_tiles = triton.cdiv(M, BLOCK_M_BWD)
        dx = torch.empty_like(x)
        dnw_partials = torch.empty(
            (num_m_tiles, K), dtype=norm_w.dtype, device=x.device,
        )
        grid_bwd = (num_m_tiles, triton.cdiv(K, BLOCK_K_BWD))
        _rms_norm_bwd_kernel[grid_bwd](
            x, rms_inv, norm_w,
            dx_hat,
            dx, dnw_partials,
            M, K,
            x.stride(0), x.stride(1),
            dx_hat.stride(0), dx_hat.stride(1),
            dx.stride(0), dx.stride(1),
            BLOCK_M=BLOCK_M_BWD, BLOCK_K=BLOCK_K_BWD,
        )
        dnw = dnw_partials.sum(dim=0)

        # eps non-differentiable
        return dx, dnw, dW_fc, dW_proj, None


def norm_mlp_relu_square(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    fc_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused nanchat-style MLP block: y = relu(RMSNorm(x) @ W_fc.T)² @ W_proj.T

    The first three ops (RMSNorm + c_fc + ReluSquare) are fused into a
    single Triton kernel; the final c_proj uses cuBLAS. Caller is
    responsible for falling back to the eager Python chain when Triton
    isn't available — check `NORM_MLP_ENABLED` or `_HAS_TRITON`.
    """
    return NormMLPReluSquare.apply(x, norm_weight, fc_weight, proj_weight, eps)


# ─────────────────────────────────────────────────────────────────────
# Fused RMSNorm + QKV projection (the first half of CausalSelfAttention).
#
# nanchat's attention forward looks like:
#     x_norm = norm(x)                       (RMSNorm at Block level)
#     q = c_q(x_norm); k = c_k(x_norm); v = c_v(x_norm)        ← FUSED HERE
#     q, k = apply_rotary(q, k, cos_sin)
#     q, k = norm(q), norm(k)
#     q, k = q * 1.2, k * 1.2
#     y = sliding_window_sdpa(q, k, v, window_size)
#     y = c_proj(y)
#
# This kernel folds the OUTER RMSNorm + Q/K/V linear projections into
# one tiled Triton pass — three matmuls share one read of x_norm and
# one rms-inv computation per row, plus we co-locate the per-row writes
# of q, k, v into a single (M, 3*N) output buffer. Backward chains the
# three matmul gradients + a single RMSNorm-backward Triton kernel,
# reusing the same _rms_norm_bwd_kernel as the MLP fusion.
# ─────────────────────────────────────────────────────────────────────


if _HAS_TRITON:

    @triton.jit
    def _norm_qkv_fwd_kernel(
        x_ptr, norm_w_ptr, qkv_w_ptr,
        out_ptr, rms_inv_ptr,
        M, N_qkv, K,
        eps,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_om, stride_on,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Computes one (BLOCK_M, BLOCK_N) tile of out = RMSNorm(x) @ W_qkv.T,
        where W_qkv = concat([c_q.weight, c_k.weight, c_v.weight], dim=0)
        and N_qkv = (H_q + 2*H_kv) * D_head. Caller splits the output
        into q/k/v slices along dim=-1."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < M
        col_mask = cols < N_qkv

        # Pass 1: per-row mean(x²)
        sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            ks = k_start + tl.arange(0, BLOCK_K)
            k_mask = ks < K
            x_ptrs = x_ptr + rows[:, None] * stride_xm + ks[None, :] * stride_xk
            x = tl.load(
                x_ptrs,
                mask=row_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            sum_sq += tl.sum(x * x, axis=1)
        rms_inv = 1.0 / tl.sqrt(sum_sq / K + eps)

        if pid_n == 0:
            tl.store(rms_inv_ptr + rows, rms_inv, mask=row_mask)

        # Pass 2: matmul x_hat @ W_qkv.T into fp32 accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            ks = k_start + tl.arange(0, BLOCK_K)
            k_mask = ks < K
            x_ptrs = x_ptr + rows[:, None] * stride_xm + ks[None, :] * stride_xk
            x = tl.load(
                x_ptrs,
                mask=row_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            nw = tl.load(norm_w_ptr + ks, mask=k_mask, other=0.0).to(tl.float32)
            x_hat = x * rms_inv[:, None] * nw[None, :]
            w_ptrs = (
                qkv_w_ptr
                + cols[:, None] * stride_wn
                + ks[None, :] * stride_wk
            )
            w = tl.load(
                w_ptrs,
                mask=col_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            acc += tl.dot(x_hat, tl.trans(w), input_precision="ieee")

        out_ptrs = out_ptr + rows[:, None] * stride_om + cols[None, :] * stride_on
        tl.store(
            out_ptrs,
            acc.to(out_ptr.dtype.element_ty),
            mask=row_mask[:, None] & col_mask[None, :],
        )


class NormQKVProjection(torch.autograd.Function):
    """Fused RMSNorm + concatenated Q/K/V linear projection.

    Forward: one Triton kernel folds the outer RMSNorm and all three
    QKV matmuls together, writing the concatenated output (M, N_q+N_k+N_v).
    Caller slices it into q, k, v.

    Backward: three linear backwards via cuBLAS + one RMSNorm-backward
    Triton kernel (reused from the MLP fusion's _rms_norm_bwd_kernel).
    """

    @staticmethod
    def forward(ctx, x, norm_weight, qkv_weight, eps=1e-6):
        assert x.is_cuda and x.is_contiguous()
        assert norm_weight.is_cuda and qkv_weight.is_cuda
        M, K = x.shape
        N_qkv, K_w = qkv_weight.shape
        assert K == K_w, f"x last dim {K} != qkv_weight in dim {K_w}"

        BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
        out = torch.empty((M, N_qkv), dtype=x.dtype, device=x.device)
        rms_inv = torch.empty((M,), dtype=torch.float32, device=x.device)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N_qkv, BLOCK_N))
        _norm_qkv_fwd_kernel[grid](
            x, norm_weight, qkv_weight,
            out, rms_inv,
            M, N_qkv, K, eps,
            x.stride(0), x.stride(1),
            qkv_weight.stride(0), qkv_weight.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        ctx.save_for_backward(x, norm_weight, qkv_weight, rms_inv)
        ctx.M, ctx.N_qkv, ctx.K = M, N_qkv, K
        return out

    @staticmethod
    def backward(ctx, d_out):
        x, norm_w, qkv_w, rms_inv = ctx.saved_tensors
        M, N_qkv, K = ctx.M, ctx.N_qkv, ctx.K
        d_out = d_out.contiguous()

        # Linear backward (cuBLAS): out = x_hat @ qkv_w.T
        #   dx_hat = d_out @ qkv_w
        #   d_qkv_w = d_out.T @ x_hat
        x_hat = x * rms_inv.unsqueeze(1) * norm_w
        dx_hat = d_out @ qkv_w        # (M, K)
        d_qkv_w = d_out.t() @ x_hat   # (N_qkv, K)

        # RMSNorm backward (Triton — reuses _rms_norm_bwd_kernel)
        BLOCK_M_BWD, BLOCK_K_BWD = 32, 64
        num_m_tiles = triton.cdiv(M, BLOCK_M_BWD)
        dx = torch.empty_like(x)
        dnw_partials = torch.empty(
            (num_m_tiles, K), dtype=norm_w.dtype, device=x.device,
        )
        grid_bwd = (num_m_tiles, triton.cdiv(K, BLOCK_K_BWD))
        _rms_norm_bwd_kernel[grid_bwd](
            x, rms_inv, norm_w,
            dx_hat,
            dx, dnw_partials,
            M, K,
            x.stride(0), x.stride(1),
            dx_hat.stride(0), dx_hat.stride(1),
            dx.stride(0), dx.stride(1),
            BLOCK_M=BLOCK_M_BWD, BLOCK_K=BLOCK_K_BWD,
        )
        dnw = dnw_partials.sum(dim=0)

        # eps non-differentiable
        return dx, dnw, d_qkv_w, None


def norm_qkv_projection(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused RMSNorm + concatenated QKV linear projection.

    Caller stacks `c_q.weight, c_k.weight, c_v.weight` along dim 0 into
    `qkv_weight` and splits the output along dim -1. See
    `causal_self_attention_triton` for the canonical orchestration.
    """
    return NormQKVProjection.apply(x, norm_weight, qkv_weight, eps)
