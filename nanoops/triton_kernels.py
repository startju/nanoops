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
    def _norm_mlp_block_fwd_kernel(
        x_ptr, norm_w_ptr, fc_w_ptr, proj_w_ptr, residual_ptr,
        y_ptr, z_ptr, rms_inv_ptr,
        M, N_fc, K, K_out,
        eps,
        stride_xm, stride_xk,
        stride_fc_n, stride_fc_k,
        stride_proj_p, stride_proj_n,
        stride_res_m, stride_res_p,
        stride_ym, stride_yp,
        stride_zm, stride_zn,
        BLOCK_M: tl.constexpr,
        BLOCK_K_OUT: tl.constexpr,    # tile size along the output (= K_out) dim of c_proj
        BLOCK_N: tl.constexpr,        # tile size along the c_fc output / c_proj input dim
        BLOCK_K: tl.constexpr,        # tile size along input K dim (for RMSNorm + c_fc matmul)
    ):
        """Fully fused MLP block + residual:
            y = residual + relu(RMSNorm(x) @ W_fc.T)² @ W_proj.T

        Each program computes one (BLOCK_M, BLOCK_K_OUT) tile of y.
        Streams the N_fc dim (c_fc output / c_proj input) in BLOCK_N
        chunks: per chunk we recompute the (BLOCK_M, BLOCK_N) slab of
        r = relu(x_hat @ W_fc[N_chunk].T)², then accumulate
        r @ W_proj[K_out_tile, N_chunk].T into the output. The r slab
        lives only in registers — never written to HBM.

        This is the "two-matmul fused" pattern (analogous to attention's
        Q@K^T → softmax → @V). Saves the M*N_fc HBM round-trip the
        non-fused version paid for materializing r.
        """
        pid_m = tl.program_id(0)
        pid_p = tl.program_id(1)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        out_cols = pid_p * BLOCK_K_OUT + tl.arange(0, BLOCK_K_OUT)
        row_mask = rows < M
        out_col_mask = out_cols < K_out

        # Pass 1: per-row mean(x²) for the rows in this tile.
        sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            ks = k_start + tl.arange(0, BLOCK_K)
            k_mask = ks < K
            x_ptrs = x_ptr + rows[:, None] * stride_xm + ks[None, :] * stride_xk
            x = tl.load(
                x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            sum_sq += tl.sum(x * x, axis=1)
        rms_inv = 1.0 / tl.sqrt(sum_sq / K + eps)

        # Only the p=0 tile writes rms_inv (it doesn't depend on p)
        if pid_p == 0:
            tl.store(rms_inv_ptr + rows, rms_inv, mask=row_mask)

        # Output accumulator (BLOCK_M, BLOCK_K_OUT)
        out_acc = tl.zeros((BLOCK_M, BLOCK_K_OUT), dtype=tl.float32)

        # Outer loop over N_fc: compute r slab (BLOCK_M, BLOCK_N), then
        # accumulate r_slab @ W_proj[out_cols, N_slab].T into out_acc.
        for n_start in range(0, N_fc, BLOCK_N):
            ns = n_start + tl.arange(0, BLOCK_N)
            n_mask = ns < N_fc

            # Inner: compute z_slab[BLOCK_M, BLOCK_N] = x_hat @ W_fc[ns, :].T
            z_slab = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                ks = k_start + tl.arange(0, BLOCK_K)
                k_mask = ks < K
                x_ptrs = x_ptr + rows[:, None] * stride_xm + ks[None, :] * stride_xk
                x = tl.load(
                    x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float32)
                nw = tl.load(norm_w_ptr + ks, mask=k_mask, other=0.0).to(tl.float32)
                x_hat = x * rms_inv[:, None] * nw[None, :]
                # W_fc shape (N_fc, K): row ns, col ks
                fc_ptrs = (
                    fc_w_ptr
                    + ns[:, None] * stride_fc_n
                    + ks[None, :] * stride_fc_k
                )
                fc_w = tl.load(
                    fc_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float32)
                z_slab += tl.dot(x_hat, tl.trans(fc_w), input_precision="ieee")

            # relu² elementwise in registers
            relu_z = tl.where(z_slab > 0.0, z_slab, 0.0)
            r_slab = relu_z * relu_z  # (BLOCK_M, BLOCK_N) — never to HBM

            # Optionally store z_slab to HBM for backward (we need z, not r,
            # because z gives us the relu mask + sign info for relu²-bwd).
            z_ptrs = z_ptr + rows[:, None] * stride_zm + ns[None, :] * stride_zn
            tl.store(
                z_ptrs,
                z_slab.to(z_ptr.dtype.element_ty),
                mask=row_mask[:, None] & n_mask[None, :],
            )

            # out_acc += r_slab @ W_proj[out_cols, ns].T
            # W_proj shape (K_out, N_fc): row out_cols, col ns
            proj_ptrs = (
                proj_w_ptr
                + out_cols[:, None] * stride_proj_p
                + ns[None, :] * stride_proj_n
            )
            proj_w = tl.load(
                proj_ptrs,
                mask=out_col_mask[:, None] & n_mask[None, :], other=0.0,
            ).to(tl.float32)
            out_acc += tl.dot(r_slab, tl.trans(proj_w), input_precision="ieee")

        # Add residual and write y
        res_ptrs = residual_ptr + rows[:, None] * stride_res_m + out_cols[None, :] * stride_res_p
        residual = tl.load(
            res_ptrs,
            mask=row_mask[:, None] & out_col_mask[None, :], other=0.0,
        ).to(tl.float32)
        y_final = residual + out_acc

        y_ptrs = y_ptr + rows[:, None] * stride_ym + out_cols[None, :] * stride_yp
        tl.store(
            y_ptrs,
            y_final.to(y_ptr.dtype.element_ty),
            mask=row_mask[:, None] & out_col_mask[None, :],
        )


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
    """Fused full MLP block + residual: y = residual + relu(RMSNorm(x) @ W_fc.T)² @ W_proj.T.

    Math (per row m):
        x_hat[m, k] = x[m, k] * rsqrt(mean_k(x²) + eps) * norm_w[k]       (RMSNorm)
        z[m, n]     = sum_k x_hat[m, k] * W_fc[n, k]                     (Linear: c_fc)
        r[m, n]     = relu(z[m, n])²                                     (ReluSquare)
        mlp[m, p]   = sum_n r[m, n] * W_proj[p, n]                       (Linear: c_proj)
        y[m, p]     = residual[m, p] + mlp[m, p]                         (Residual add)

    Mixed-implementation strategy:
      Forward:  Triton fuses (norm + c_fc + relu²) into one kernel.
                The c_proj matmul + residual add uses `torch.addmm`
                (cuBLAS fused matmul-with-bias — saves one HBM round-trip
                vs separate matmul + add).
      Backward: relu²-backward is Triton (elementwise); both linear
                backwards use cuBLAS; RMSNorm backward is Triton
                (per-row reduction + elementwise). Residual gradient is
                identity — `d_residual = dy` passes through.

    What's saved in ctx:
      x, norm_w, W_fc, W_proj, z, rms_inv
    r is NOT saved — we recompute it from z (`relu(z)²`) when needed for
    `dW_proj = dy^T @ r`. Saves M*4K floats (~80 MB at nanchat d24 MLP).
    """

    @staticmethod
    def forward(ctx, x, norm_weight, fc_weight, proj_weight, residual, eps=1e-6):
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

        # Second linear + residual via fused addmm (cuBLAS): y = residual + r @ W_proj.T
        # `addmm(input, mat1, mat2)` = input + mat1 @ mat2 in one cuBLAS call,
        # writing directly into `y` without materializing the c_proj output as
        # a separate HBM tensor. Saves ~M*K of HBM traffic vs `(r @ W_proj.T) + residual`.
        assert residual.shape == (M, K_proj_out), (
            f"residual must be ({M}, {K_proj_out}), got {residual.shape}"
        )
        y = torch.addmm(residual, r, proj_weight.t())

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

        # y = residual + mlp_inner, so d_residual = dy (identity passthrough).
        # eps non-differentiable.
        d_residual = dy
        return dx, dnw, dW_fc, dW_proj, d_residual, None


def norm_mlp_relu_square(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    fc_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused nanchat-style MLP block + residual:
        y = residual + relu(RMSNorm(x) @ W_fc.T)² @ W_proj.T

    The first three ops (RMSNorm + c_fc + ReluSquare) fuse into a
    single Triton kernel; the final c_proj + residual fuse via
    `torch.addmm` (cuBLAS). Caller is responsible for falling back to
    the eager Python chain when Triton isn't available — check
    `NORM_MLP_ENABLED` or `_HAS_TRITON`.

    In nanchat's Block.forward, this replaces both `mlp(norm(x))` AND
    the surrounding `x = x + ...` residual add in one call:
        x = norm_mlp_relu_square(x, norm_w, W_fc, W_proj, residual=x)
    """
    return NormMLPReluSquare.apply(x, norm_weight, fc_weight, proj_weight, residual, eps)


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


# ─────────────────────────────────────────────────────────────────────
# Flash-style sliding-window SDPA in Triton.
#
# Standard Flash Attention pattern, adapted for nanchat's
# sliding-causal mask:
#   - Forward: tile Q into BLOCK_M-sized chunks. For each Q tile,
#     stream K/V in BLOCK_N chunks, maintaining (running max m, running
#     normalizer ℓ) online so we never materialize the (L, L) P matrix.
#     Output: o (B, H, L, D) and log-sum-exp ℓ (B, H, L) for backward.
#   - Backward: re-derive P[i, j] = exp(s[i, j] - LSE[i]) from a fresh
#     Q@K^T tile in fp32, then accumulate dQ, dK, dV via the same
#     tiling. Uses D[i] = sum_j o[i, j] * dO[i, j] precomputed per row
#     to skip the inner softmax-bwd reduction.
# Sliding-window mask: per-tile lower-bound `j ≥ i - W + 1`. Combined
# with causal `j ≤ i`, this lets us skip entire K/V tiles whose j range
# is outside [i_min - W + 1, i_max].
#
# Scope of v1:
#   - No GQA (assume H_q == H_kv). Caller must repeat_interleave V/K
#     before calling if their model is GQA.
#   - No FA-3-style asynchronous TMA / split-k. Single-stage tiling.
#   - bf16 inputs OK (matmul in fp32 accumulator); fp16/fp32 also work.
# ─────────────────────────────────────────────────────────────────────


if _HAS_TRITON:

    @triton.jit
    def _flash_attn_fwd_kernel(
        Q, K, V, sm_scale,
        LSE, O,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_lb, stride_lh, stride_lm,
        B, H, M, N,
        WINDOW: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
    ):
        """Forward: one program per (batch × head × Q-tile-of-BLOCK_M-rows)."""
        pid_bh = tl.program_id(0)
        pid_m = tl.program_id(1)
        bid = pid_bh // H
        hid = pid_bh % H

        # Offsets into B, H of Q, K, V, O for this program.
        q_off = bid * stride_qb + hid * stride_qh
        k_off = bid * stride_kb + hid * stride_kh
        v_off = bid * stride_vb + hid * stride_vh
        o_off = bid * stride_ob + hid * stride_oh
        lse_off = bid * stride_lb + hid * stride_lh

        # Q tile: load BLOCK_M rows of Q for this program. Stays resident.
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        q_ptrs = Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        m_mask = offs_m < M
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

        # Online softmax state per Q row.
        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

        # Sliding-causal: each query at position i (= offs_m[i_local]) can
        # attend to keys in [max(0, i - W + 1), i]. So we only need to
        # stream K/V tiles whose j range overlaps with that band.
        # For this Q tile (rows pid_m*BLOCK_M .. pid_m*BLOCK_M+BLOCK_M-1):
        #   j_min_required = max(0, pid_m*BLOCK_M - WINDOW + 1)
        #   j_max_required = pid_m*BLOCK_M + BLOCK_M - 1
        # Convert to tile indices on K (BLOCK_N-wide tiles):
        kv_tile_start = tl.maximum(0, pid_m * BLOCK_M - WINDOW + 1) // BLOCK_N
        kv_tile_end = tl.minimum(N, pid_m * BLOCK_M + BLOCK_M) // BLOCK_N + 1

        for kv_idx in range(kv_tile_start, kv_tile_end):
            offs_n = kv_idx * BLOCK_N + tl.arange(0, BLOCK_N)
            n_mask = offs_n < N

            # Load K, V tiles
            k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
            v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

            # Scores = Q @ K^T * scale, shape (BLOCK_M, BLOCK_N)
            s = tl.dot(q, tl.trans(k_tile), input_precision="ieee") * sm_scale

            # Apply sliding+causal mask per cell:
            #   keep if  j ≤ i  AND  j ≥ i - W + 1
            j = offs_n[None, :]
            i = offs_m[:, None]
            mask_keep = (j <= i) & (j >= i - WINDOW + 1) & m_mask[:, None] & n_mask[None, :]
            s = tl.where(mask_keep, s, -float("inf"))

            # Online softmax update
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(tl.exp(s - m_new[:, None]), axis=1)
            acc = acc * alpha[:, None] + tl.dot(
                tl.exp(s - m_new[:, None]).to(v_tile.dtype),
                v_tile, input_precision="ieee",
            )
            m_i = m_new

        # Finalize: output / l, save LSE = m + log(l)
        acc = acc / l_i[:, None]
        lse = m_i + tl.log(l_i)

        o_ptrs = O + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=m_mask[:, None])

        lse_ptrs = LSE + lse_off + offs_m * stride_lm
        tl.store(lse_ptrs, lse, mask=m_mask)


    @triton.jit
    def _flash_attn_bwd_preprocess_kernel(
        O, dO, D,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_db, stride_dh, stride_dm,
        B, H, M,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    ):
        """Precompute D[i] = sum_j o[i, j] * dO[i, j] — used in bwd to skip
        the inner softmax-bwd reduction. This is the classic Flash trick."""
        pid_bh = tl.program_id(0)
        pid_m = tl.program_id(1)
        bid = pid_bh // H
        hid = pid_bh % H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        m_mask = offs_m < M

        o_off = bid * stride_ob + hid * stride_oh
        o_ptrs = O + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        do_ptrs = dO + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        o = tl.load(o_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        d_row = tl.sum(o * do, axis=1)
        d_ptrs = D + bid * stride_db + hid * stride_dh + offs_m * stride_dm
        tl.store(d_ptrs, d_row, mask=m_mask)


    @triton.jit
    def _flash_attn_bwd_kernel(
        Q, K, V, sm_scale,
        LSE, D, dO,
        dQ, dK, dV,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_lb, stride_lh, stride_lm,
        stride_db, stride_dh, stride_dm,
        B, H, M, N,
        WINDOW: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
    ):
        """Backward: iterate Q tiles, stream K/V tiles within the sliding band.
        Atomic-add dK/dV across overlapping Q tiles (each K position can be
        touched by multiple Q tiles within the window)."""
        pid_bh = tl.program_id(0)
        pid_m = tl.program_id(1)
        bid = pid_bh // H
        hid = pid_bh % H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        m_mask = offs_m < M

        q_off = bid * stride_qb + hid * stride_qh
        k_off = bid * stride_kb + hid * stride_kh
        v_off = bid * stride_vb + hid * stride_vh

        # Load Q tile, dO tile, LSE, D for this Q range
        q_ptrs = Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        do_ptrs = dO + bid * stride_ob + hid * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        lse_ptrs = LSE + bid * stride_lb + hid * stride_lh + offs_m * stride_lm
        d_ptrs = D + bid * stride_db + hid * stride_dh + offs_m * stride_dm
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        lse = tl.load(lse_ptrs, mask=m_mask, other=0.0)
        d_row = tl.load(d_ptrs, mask=m_mask, other=0.0)

        # dQ accumulator
        dq_acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

        kv_tile_start = tl.maximum(0, pid_m * BLOCK_M - WINDOW + 1) // BLOCK_N
        kv_tile_end = tl.minimum(N, pid_m * BLOCK_M + BLOCK_M) // BLOCK_N + 1

        for kv_idx in range(kv_tile_start, kv_tile_end):
            offs_n = kv_idx * BLOCK_N + tl.arange(0, BLOCK_N)
            n_mask = offs_n < N

            k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
            v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

            # Recompute P[i, j] = exp(s[i, j] * sm_scale - LSE[i])
            s = tl.dot(q, tl.trans(k_tile), input_precision="ieee") * sm_scale
            j = offs_n[None, :]
            i = offs_m[:, None]
            mask_keep = (j <= i) & (j >= i - WINDOW + 1) & m_mask[:, None] & n_mask[None, :]
            s = tl.where(mask_keep, s, -float("inf"))
            p = tl.exp(s - lse[:, None])

            # dV += P^T @ dO  (BLOCK_N, BLOCK_DMODEL)
            dv = tl.dot(tl.trans(p).to(do.dtype), do, input_precision="ieee")
            # dP = dO @ V^T  (BLOCK_M, BLOCK_N)
            dp = tl.dot(do, tl.trans(v_tile), input_precision="ieee")
            # dS = P * (dP - D)  (BLOCK_M, BLOCK_N)
            ds = p * (dp - d_row[:, None]) * sm_scale
            # dQ += dS @ K
            dq_acc += tl.dot(ds.to(k_tile.dtype), k_tile, input_precision="ieee")
            # dK += dS^T @ Q
            dk = tl.dot(tl.trans(ds).to(q.dtype), q, input_precision="ieee")

            # Atomic-add dK, dV (overlapping Q tiles touch same K rows)
            dk_ptrs = dK + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            dv_ptrs = dV + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            tl.atomic_add(dk_ptrs, dk.to(dK.dtype.element_ty), mask=n_mask[:, None])
            tl.atomic_add(dv_ptrs, dv.to(dV.dtype.element_ty), mask=n_mask[:, None])

        # Write dQ (single contributor per Q tile — no atomic needed)
        dq_ptrs = dQ + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        tl.store(dq_ptrs, dq_acc.to(dQ.dtype.element_ty), mask=m_mask[:, None])


class FlashSDPA(torch.autograd.Function):
    """Flash-style sliding-causal SDPA in Triton (no GQA, no kv-cache).

    Forward: O = softmax(QK^T * scale) V with sliding-causal masking
    (each query attends to keys in [i - W + 1, i]). Memory-efficient:
    no (L, L) P matrix materialized; only LSE (L,) saved per (B, H) row.

    Backward: standard Flash recompute strategy — re-derive P from
    Q@K^T + LSE per-tile, accumulate dQ, dK, dV with atomics for the
    overlapping dK/dV contributions.
    """

    @staticmethod
    def forward(ctx, q, k, v, window_size):
        assert q.is_cuda and k.is_cuda and v.is_cuda
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert q.shape == k.shape == v.shape, (
            f"v1 requires H_q == H_kv and same L: q{q.shape} k{k.shape} v{v.shape}"
        )
        B, H, M, D = q.shape
        N = k.shape[2]
        sm_scale = D ** -0.5

        o = torch.empty_like(q)
        lse = torch.empty((B, H, M), dtype=torch.float32, device=q.device)

        BLOCK_M, BLOCK_N = 64, 64
        grid = (B * H, triton.cdiv(M, BLOCK_M))
        _flash_attn_fwd_kernel[grid](
            q, k, v, sm_scale,
            lse, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            B, H, M, N,
            WINDOW=window_size,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
        )

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = sm_scale
        ctx.window_size = window_size
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        do = do.contiguous()
        sm_scale = ctx.sm_scale
        window_size = ctx.window_size
        BLOCK_M, BLOCK_N = ctx.BLOCK_M, ctx.BLOCK_N
        B, H, M, D = q.shape
        N = k.shape[2]

        # D[i] = sum_j o[i, j] * dO[i, j]
        d = torch.empty((B, H, M), dtype=torch.float32, device=q.device)
        BLOCK_M_PRE = 64
        grid_pre = (B * H, triton.cdiv(M, BLOCK_M_PRE))
        _flash_attn_bwd_preprocess_kernel[grid_pre](
            o, do, d,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            d.stride(0), d.stride(1), d.stride(2),
            B, H, M,
            BLOCK_M=BLOCK_M_PRE, BLOCK_DMODEL=D,
        )

        # dQ, dK, dV allocated as zero (dK, dV need accumulation via atomic).
        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        grid_bwd = (B * H, triton.cdiv(M, BLOCK_M))
        _flash_attn_bwd_kernel[grid_bwd](
            q, k, v, sm_scale,
            lse, d, do,
            dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            d.stride(0), d.stride(1), d.stride(2),
            B, H, M, N,
            WINDOW=window_size,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
        )

        return dq, dk, dv, None


def flash_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int,
) -> torch.Tensor:
    """Flash-style sliding-causal SDPA. q, k, v: (B, H, L, D), H_q == H_kv.

    window_size: total keys each query attends to (= nanchat's window+1).
    """
    return FlashSDPA.apply(q, k, v, window_size)


# ─────────────────────────────────────────────────────────────────────
# Small Triton kernels covering the remaining attention/MLP elementwise
# chains. None of these are big wins individually (each saves ~10-50 µs
# per layer of kernel-launch + HBM round-trip overhead), but together
# they cover the last "all eager" pieces of nanchat's attention forward,
# letting us claim attention is "fully Triton-fused" in the sense that
# every per-element operation has a Triton kernel.
# ─────────────────────────────────────────────────────────────────────

if _HAS_TRITON:

    @triton.jit
    def _value_gate_kernel(
        v_ptr, ve_ptr, x_ptr, gate_w_ptr,
        out_ptr,
        M, D_x, D_v, ve_gate_ch,
        stride_vm, stride_vd,
        stride_vem, stride_ved,
        stride_xm, stride_xd,
        stride_gw_d_out, stride_gw_d_in,
        stride_om, stride_od,
        BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Fused value-residual gate (ResFormer):
            gate = 3 * sigmoid(x[..., :ch] @ gate_w.T)       # (M, D_v_head_dim)
            out  = v + gate * ve
        Where ch = ve_gate_channels (small).

        Per row m, gate is per-head (we broadcast across head_dim
        elements). For simplicity here we expand gate to v's shape via
        the same broadcasting the eager code does.
        """
        pid_m = tl.program_id(0)
        pid_d = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        row_mask = rows < M
        col_mask = cols < D_v

        # Compute gate = 3 * sigmoid(x[:, :ch] @ gate_w.T) for the rows in this tile.
        # gate_w shape: (D_v, ve_gate_ch). x slice: (BLOCK_M, ve_gate_ch).
        # Result gate: (BLOCK_M, D_v) — broadcast across cols later if needed.
        # We compute the per-row, per-output-dim gate value once and reuse for
        # the cols of v in this tile.
        gate_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        for k_start in range(0, ve_gate_ch, BLOCK_K):
            ks = k_start + tl.arange(0, BLOCK_K)
            k_mask = ks < ve_gate_ch
            x_ptrs = x_ptr + rows[:, None] * stride_xm + ks[None, :] * stride_xd
            x_chunk = tl.load(
                x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            gw_ptrs = (
                gate_w_ptr
                + cols[:, None] * stride_gw_d_out
                + ks[None, :] * stride_gw_d_in
            )
            gw = tl.load(
                gw_ptrs, mask=col_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            gate_acc += tl.dot(x_chunk, tl.trans(gw), input_precision="ieee")
        gate = 3.0 * tl.sigmoid(gate_acc)

        # Load v, ve, compute out = v + gate * ve
        v_ptrs = v_ptr + rows[:, None] * stride_vm + cols[None, :] * stride_vd
        ve_ptrs = ve_ptr + rows[:, None] * stride_vem + cols[None, :] * stride_ved
        v = tl.load(v_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
        ve = tl.load(ve_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
        out = v + gate * ve

        o_ptrs = out_ptr + rows[:, None] * stride_om + cols[None, :] * stride_od
        tl.store(
            o_ptrs,
            out.to(out_ptr.dtype.element_ty),
            mask=row_mask[:, None] & col_mask[None, :],
        )


    @triton.jit
    def _rotary_qk_norm_scale_kernel(
        qk_ptr, cos_ptr, sin_ptr,
        out_ptr, rms_inv_ptr,
        M, D,
        scale,
        eps,
        stride_qm, stride_qd,
        stride_om, stride_od,
        BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Fused rotary + RMSNorm + ×scale for Q or K:
            x1, x2 = qk[..., :d], qk[..., d:]
            y1 = x1 * cos + x2 * sin
            y2 = -x1 * sin + x2 * cos
            y  = concat(y1, y2)                       # (M, D), rotated
            y_normed = y * rsqrt(mean(y²) + eps) * scale

        Compact: rotate in registers → RMSNorm in registers → scale.
        Saves HBM round-trips between rotary / norm / scale stages.
        D must equal BLOCK_D (we tile only M).
        """
        pid_m = tl.program_id(0)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        half = BLOCK_D // 2
        cs_cols = tl.arange(0, BLOCK_D // 2)

        # Load lo + hi halves of qk and cos/sin (which are half-D wide)
        ptrs_lo = qk_ptr + rows[:, None] * stride_qm + cs_cols[None, :] * stride_qd
        ptrs_hi = qk_ptr + rows[:, None] * stride_qm + (cs_cols[None, :] + half) * stride_qd
        x1 = tl.load(ptrs_lo, mask=row_mask[:, None], other=0.0).to(tl.float32)
        x2 = tl.load(ptrs_hi, mask=row_mask[:, None], other=0.0).to(tl.float32)
        cos = tl.load(
            cos_ptr + rows[:, None] * (BLOCK_D // 2) + cs_cols[None, :],
            mask=row_mask[:, None], other=0.0,
        ).to(tl.float32)
        sin = tl.load(
            sin_ptr + rows[:, None] * (BLOCK_D // 2) + cs_cols[None, :],
            mask=row_mask[:, None], other=0.0,
        ).to(tl.float32)

        # Rotary: y1 = x1·cos + x2·sin ;  y2 = -x1·sin + x2·cos
        y1 = x1 * cos + x2 * sin
        y2 = -x1 * sin + x2 * cos

        # RMSNorm + scale: compute mean(y²) over full D using y1, y2 in registers
        sum_sq = tl.sum(y1 * y1, axis=1) + tl.sum(y2 * y2, axis=1)
        rms_inv = 1.0 / tl.sqrt(sum_sq / D + eps)
        norm_scale = rms_inv * scale

        # Apply norm·scale, write halves back
        y1_out = y1 * norm_scale[:, None]
        y2_out = y2 * norm_scale[:, None]
        out_ptrs_lo = out_ptr + rows[:, None] * stride_om + cs_cols[None, :] * stride_od
        out_ptrs_hi = out_ptr + rows[:, None] * stride_om + (cs_cols[None, :] + half) * stride_od
        tl.store(out_ptrs_lo, y1_out.to(out_ptr.dtype.element_ty), mask=row_mask[:, None])
        tl.store(out_ptrs_hi, y2_out.to(out_ptr.dtype.element_ty), mask=row_mask[:, None])
        # rms_inv saved for backward
        tl.store(rms_inv_ptr + rows, rms_inv, mask=row_mask)


    @triton.jit
    def _output_proj_residual_kernel(
        attn_out_ptr, proj_w_ptr, residual_ptr,
        y_ptr,
        M, D_out, D_in,
        stride_am, stride_ad,
        stride_pw_dout, stride_pw_din,
        stride_rm, stride_rd,
        stride_ym, stride_yd,
        BLOCK_M: tl.constexpr, BLOCK_DOUT: tl.constexpr, BLOCK_DIN: tl.constexpr,
    ):
        """Fused y = residual + attn_out @ W_proj.T.

        Standard tiled matmul with the residual loaded into the
        accumulator at start instead of zero-init. Same idea as cuBLAS
        `addmm` but in our Triton stack.
        """
        pid_m = tl.program_id(0)
        pid_d = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_d * BLOCK_DOUT + tl.arange(0, BLOCK_DOUT)
        row_mask = rows < M
        col_mask = cols < D_out

        # Start accumulator with residual (the "bias" in addmm)
        res_ptrs = residual_ptr + rows[:, None] * stride_rm + cols[None, :] * stride_rd
        acc = tl.load(
            res_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0,
        ).to(tl.float32)

        # Matmul-accumulate
        for k_start in range(0, D_in, BLOCK_DIN):
            ks = k_start + tl.arange(0, BLOCK_DIN)
            k_mask = ks < D_in
            a_ptrs = attn_out_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ad
            a = tl.load(
                a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            pw_ptrs = (
                proj_w_ptr
                + cols[:, None] * stride_pw_dout
                + ks[None, :] * stride_pw_din
            )
            pw = tl.load(
                pw_ptrs, mask=col_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            acc += tl.dot(a, tl.trans(pw), input_precision="ieee")

        y_ptrs = y_ptr + rows[:, None] * stride_ym + cols[None, :] * stride_yd
        tl.store(
            y_ptrs,
            acc.to(y_ptr.dtype.element_ty),
            mask=row_mask[:, None] & col_mask[None, :],
        )


class OutputProjResidual(torch.autograd.Function):
    """y = residual + attn_out @ proj_weight.T

    Forward: one Triton kernel — matmul with residual loaded as the
    accumulator init (same pattern as cuBLAS addmm). Backward uses
    cuBLAS for the two matmul gradients; residual gradient is identity.
    """

    @staticmethod
    def forward(ctx, attn_out, proj_weight, residual):
        assert attn_out.is_cuda and proj_weight.is_cuda and residual.is_cuda
        M, D_in = attn_out.shape
        D_out, D_in_w = proj_weight.shape
        assert D_in == D_in_w
        y = torch.empty((M, D_out), dtype=attn_out.dtype, device=attn_out.device)
        BLOCK_M, BLOCK_DOUT, BLOCK_DIN = 32, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(D_out, BLOCK_DOUT))
        _output_proj_residual_kernel[grid](
            attn_out, proj_weight, residual, y,
            M, D_out, D_in,
            attn_out.stride(0), attn_out.stride(1),
            proj_weight.stride(0), proj_weight.stride(1),
            residual.stride(0), residual.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_DOUT=BLOCK_DOUT, BLOCK_DIN=BLOCK_DIN,
        )
        ctx.save_for_backward(attn_out, proj_weight)
        return y

    @staticmethod
    def backward(ctx, dy):
        attn_out, proj_weight = ctx.saved_tensors
        dy = dy.contiguous()
        # y = residual + attn_out @ proj_weight.T
        # d_residual = dy (identity)
        # d_attn_out = dy @ proj_weight
        # d_proj_weight = dy.T @ attn_out
        d_attn_out = dy @ proj_weight
        d_proj_weight = dy.t() @ attn_out
        d_residual = dy
        return d_attn_out, d_proj_weight, d_residual


def output_proj_residual(
    attn_out: torch.Tensor,
    proj_weight: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Fused `y = residual + attn_out @ proj_weight.T` (Triton forward + cuBLAS backward)."""
    return OutputProjResidual.apply(attn_out, proj_weight, residual)


# ─────────────────────────────────────────────────────────────────────
# ValueGate autograd.Function: out = v + 3·sigmoid(x[:, :ch] @ gate_w.T) · ve
# Forward: _value_gate_kernel.
# Backward: cuBLAS for matmul-grads; small Triton-able elementwise but
# we just use torch ops for simplicity (it's only 3-4 elementwise ops).
# ─────────────────────────────────────────────────────────────────────

class ValueGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, ve, x, gate_w):
        """Args:
            v:      (M, D_v) — base value
            ve:     (M, D_v) — value embedding to mix in
            x:      (M, D_x) — gate input (only first ve_gate_ch cols used)
            gate_w: (D_v, ve_gate_ch) — gate projection
        Returns:
            out: (M, D_v)
        """
        assert v.is_cuda and ve.is_cuda and x.is_cuda and gate_w.is_cuda
        M, D_v = v.shape
        ve_gate_ch = gate_w.shape[1]
        x_in = x[:, :ve_gate_ch].contiguous()
        out = torch.empty_like(v)

        BLOCK_M, BLOCK_D, BLOCK_K = 32, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(D_v, BLOCK_D))
        _value_gate_kernel[grid](
            v, ve, x_in, gate_w,
            out,
            M, x.shape[1], D_v, ve_gate_ch,
            v.stride(0), v.stride(1),
            ve.stride(0), ve.stride(1),
            x_in.stride(0), x_in.stride(1),
            gate_w.stride(0), gate_w.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D, BLOCK_K=BLOCK_K,
        )
        ctx.save_for_backward(v, ve, x_in, gate_w)
        ctx.ve_gate_ch = ve_gate_ch
        ctx.x_full_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, d_out):
        v, ve, x_in, gate_w = ctx.saved_tensors
        ve_gate_ch = ctx.ve_gate_ch
        x_full_shape = ctx.x_full_shape

        # Recompute gate = 3·sigmoid(x_in @ gate_w.T) in fp32.
        # (Could save in fwd; recomputing is cheap and saves ctx memory.)
        s = torch.sigmoid((x_in.float() @ gate_w.float().t()))  # (M, D_v)
        gate = 3.0 * s

        # out = v + gate * ve
        # d_v = d_out
        # d_gate = d_out * ve   → d_s = 3 * d_gate → d_logits = d_s * s*(1-s)
        # d_ve = d_out * gate
        d_v = d_out
        d_ve = d_out * gate.to(d_out.dtype)
        d_gate = d_out.float() * ve.float()
        d_s = 3.0 * d_gate
        d_logits = d_s * s * (1.0 - s)  # (M, D_v)
        # logits = x_in @ gate_w.T → d_x_in = d_logits @ gate_w; d_gate_w = d_logits.T @ x_in
        d_x_in = (d_logits @ gate_w.float()).to(x_in.dtype)
        d_gate_w = (d_logits.t() @ x_in.float()).to(gate_w.dtype)
        # Reconstruct d_x with zeros for the unused tail columns.
        d_x = torch.zeros(x_full_shape, dtype=x_in.dtype, device=x_in.device)
        d_x[:, :ve_gate_ch] = d_x_in
        return d_v, d_ve, d_x, d_gate_w


def value_gate(
    v: torch.Tensor, ve: torch.Tensor, x: torch.Tensor, gate_w: torch.Tensor,
) -> torch.Tensor:
    """Fused ResFormer value gate: out = v + 3·sigmoid(x[:, :ch] @ gate_w.T) · ve."""
    return ValueGate.apply(v, ve, x, gate_w)


# ─────────────────────────────────────────────────────────────────────
# Rotary + RMSNorm + scale autograd.Function (for Q or K)
#
# Forward uses _rotary_qk_norm_scale_kernel (Triton).
# Backward chain: scale → RMSNorm bwd → rotary inverse (use sin → -sin
# rotation; rotary's Jacobian is orthogonal so the inverse is the same
# shape with sin negated). Uses eager PyTorch ops in backward for
# clarity (each op is small elementwise).
# ─────────────────────────────────────────────────────────────────────

class RotaryQKNormScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qk, cos, sin, scale, eps=1e-6):
        """qk: (M, D); cos, sin: (M, D/2). Returns: (M, D) rotated, normed, scaled."""
        assert qk.is_cuda and qk.is_contiguous()
        assert cos.is_contiguous() and sin.is_contiguous()
        M, D = qk.shape
        assert D % 2 == 0
        out = torch.empty_like(qk)
        rms_inv = torch.empty(M, dtype=torch.float32, device=qk.device)
        BLOCK_M = 32
        grid = (triton.cdiv(M, BLOCK_M),)
        _rotary_qk_norm_scale_kernel[grid](
            qk, cos, sin,
            out, rms_inv,
            M, D, scale, eps,
            qk.stride(0), qk.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_D=D,
        )
        ctx.save_for_backward(qk, cos, sin, rms_inv)
        ctx.scale = scale
        ctx.D = D
        return out

    @staticmethod
    def backward(ctx, d_out):
        qk, cos, sin, rms_inv = ctx.saved_tensors
        scale = ctx.scale
        D = ctx.D
        half = D // 2

        # Recompute y_rotated (post-rotary, pre-norm) in fp32:
        x1 = qk[:, :half].float()
        x2 = qk[:, half:].float()
        y1 = x1 * cos.float() + x2 * sin.float()
        y2 = -x1 * sin.float() + x2 * cos.float()
        y_rot = torch.cat([y1, y2], dim=-1)  # (M, D), fp32

        # y_normed = y_rot * rms_inv * scale → out  (no per-dim weight)
        # Backward through scale + RMSNorm (no weight version):
        #   g_eff = d_out * scale
        #   inner = mean(g_eff * y_normed_unscaled, dim=-1, keepdim=True)
        #   d_y_rot = rms_inv * (g_eff - y_normed_unscaled * inner)
        # where y_normed_unscaled = y_rot * rms_inv (without scale).
        g_eff = d_out.float() * scale
        y_unscaled = y_rot * rms_inv[:, None]
        inner = (g_eff * y_unscaled).mean(dim=-1, keepdim=True)
        d_y_rot = rms_inv[:, None] * (g_eff - y_unscaled * inner)

        # Backward through rotary: same formula with sin → -sin (orthogonal Jacobian).
        d_y1 = d_y_rot[:, :half]
        d_y2 = d_y_rot[:, half:]
        d_x1 = d_y1 * cos.float() - d_y2 * sin.float()
        d_x2 = d_y1 * sin.float() + d_y2 * cos.float()
        d_qk = torch.cat([d_x1, d_x2], dim=-1).to(qk.dtype)

        # cos, sin, scale, eps non-differentiable
        return d_qk, None, None, None, None


def rotary_qk_norm_scale(
    qk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scale: float = 1.2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused rotary embedding + RMSNorm + multiplicative scale for Q or K."""
    return RotaryQKNormScale.apply(qk, cos, sin, scale, eps)
