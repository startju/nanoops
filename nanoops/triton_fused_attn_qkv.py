"""Attention QKV-side Triton kernels for nanoops.

Contains:
  - `norm_qkv_projection`: fused outer RMSNorm + concatenated Q/K/V
    linear projection, plus its paired RMSNorm backward helper.
  - `norm_qkv_rotary_projection`: fused outer RMSNorm + Q/K/V projection;
    Q/K are immediately rotary-embedded, RMS-normalized, and scaled before
    being written.
  - `rotary_qk_norm_scale`: fused rotary + Q/K RMSNorm + scale before SDPA.

Re-exported through `nanoops.triton_kernels`.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


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
# three matmul gradients + the RMSNorm-backward Triton kernel
# (`_rms_norm_bwd_kernel`) defined right below.
# ─────────────────────────────────────────────────────────────────────


if _HAS_TRITON:

    @triton.jit
    def _rms_norm_bwd_kernel(
        x_ptr,  # (M, K) — in: original RMSNorm input
        rms_inv_ptr,  # (M,) fp32 — in: saved inverse RMS
        nw_ptr,  # (K,) — in: RMSNorm scale
        dxhat_ptr,  # (M, K) — in: grad wrt normalized/scaled activation
        dx_ptr,  # (M, K) — out: grad wrt x
        dnw_partial_ptr,  # (ceil(M/BLOCK_M), K) — out: per-row-tile dnw
        M,  # int — row count after flattening leading dims
        K,  # int — normalized hidden width
        stride_xm,  # int — x stride along M
        stride_xk,  # int — x stride along K
        stride_dxhat_m,  # int — dxhat stride along M
        stride_dxhat_k,  # int — dxhat stride along K
        stride_dx_m,  # int — dx stride along M
        stride_dx_k,  # int — dx stride along K
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
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
                x_ptrs,
                mask=row_mask[:, None] & k2_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            dxh_ptrs = (
                dxhat_ptr
                + rows[:, None] * stride_dxhat_m
                + ks2[None, :] * stride_dxhat_k
            )
            dxh = tl.load(
                dxh_ptrs,
                mask=row_mask[:, None] & k2_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            nw2 = tl.load(nw_ptr + ks2, mask=k2_mask, other=0.0).to(tl.float32)
            y_norm2 = x * rms_inv[:, None]
            g_eff2 = dxh * nw2[None, :]
            inner += tl.sum(g_eff2 * y_norm2, axis=1)
        inner = inner / K  # mean

        # Pass 2: compute dx for THIS k-tile
        x_ptrs = x_ptr + rows[:, None] * stride_xm + ks[None, :] * stride_xk
        x = tl.load(
            x_ptrs,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        dxh_ptrs = (
            dxhat_ptr + rows[:, None] * stride_dxhat_m + ks[None, :] * stride_dxhat_k
        )
        dxh = tl.load(
            dxh_ptrs,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
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
        tl.store(
            dnw_p_ptrs, dnw_partial.to(dnw_partial_ptr.dtype.element_ty), mask=k_mask
        )

    @triton.jit
    def _norm_qkv_fwd_kernel(
        x_ptr,  # (M, K) — in: activation before RMSNorm
        norm_w_ptr,  # (K,) — in: RMSNorm scale
        qkv_w_ptr,  # (N_qkv, K) — in: concatenated Q/K/V projection weight
        out_ptr,  # (M, N_qkv) — out: concatenated Q/K/V projection
        rms_inv_ptr,  # (M,) fp32 — out: saved inverse RMS for bwd
        M,  # int — row count after flattening leading dims
        N_qkv,  # int — concatenated Q/K/V output width
        K,  # int — input hidden width
        eps,  # float — RMSNorm epsilon
        stride_xm,  # int — x stride along M
        stride_xk,  # int — x stride along K
        stride_wn,  # int — qkv_weight stride along N_qkv
        stride_wk,  # int — qkv_weight stride along K
        stride_om,  # int — out stride along M
        stride_on,  # int — out stride along N_qkv
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
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
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            sum_sq += tl.sum(x * x, axis=1)
        rms_inv = tl.rsqrt(sum_sq / K + eps)

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
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            nw = tl.load(norm_w_ptr + ks, mask=k_mask, other=0.0).to(tl.float32)
            x_hat = x * rms_inv[:, None] * nw[None, :]
            w_ptrs = qkv_w_ptr + cols[:, None] * stride_wn + ks[None, :] * stride_wk
            w = tl.load(
                w_ptrs,
                mask=col_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(x_hat, tl.trans(w), input_precision="ieee")

        out_ptrs = out_ptr + rows[:, None] * stride_om + cols[None, :] * stride_on
        tl.store(
            out_ptrs,
            acc.to(out_ptr.dtype.element_ty),
            mask=row_mask[:, None] & col_mask[None, :],
        )

    @triton.jit
    def _norm_qkv_rotary_fwd_kernel(
        x_ptr,  # (M, K) — in: activation before RMSNorm
        norm_w_ptr,  # (K,) — in: outer RMSNorm scale
        qkv_w_ptr,  # (N_qkv, K) — in: concatenated Q/K/V projection weight
        cos_ptr,  # (M, D/2) — in: rotary cosine table
        sin_ptr,  # (M, D/2) — in: rotary sine table
        out_ptr,  # (M, H, D) — out: Q/K/V head tensor
        M,  # int — row count after flattening batch/time
        K,  # int — input hidden width
        D,  # int — per-head width
        weight_offset,  # int — first qkv_weight output row for this projection group
        eps,  # float — outer and QK RMSNorm epsilon
        scale,  # float — post-QK-norm scalar multiplier
        stride_xm,  # int — x stride along M
        stride_xk,  # int — x stride along K
        stride_wn,  # int — qkv_weight stride along output row
        stride_wk,  # int — qkv_weight stride along K
        stride_om,  # int — out stride along M
        stride_oh,  # int — out stride along head
        stride_od,  # int — out stride along D
        BLOCK_M: tl.constexpr,
        BLOCK_HALF: tl.constexpr,
        BLOCK_K: tl.constexpr,
        DO_ROTARY_NORM_SCALE: tl.constexpr,
    ):
        """Project one Q/K/V head tile.

        For Q/K, this folds:
            RMSNorm(x) @ W_head.T → rotary → RMSNorm(head) → * scale
        before the head is written. For V, the same projection path runs
        with DO_ROTARY_NORM_SCALE=False and writes the projected head directly.
        """
        pid_m = tl.program_id(0)
        head = tl.program_id(1)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        half_cols = tl.arange(0, BLOCK_HALF)
        row_mask = rows < M

        # Pass 1: per-row RMSNorm(x) inverse.
        sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            ks = k_start + tl.arange(0, BLOCK_K)
            k_mask = ks < K
            x_ptrs = x_ptr + rows[:, None] * stride_xm + ks[None, :] * stride_xk
            x = tl.load(
                x_ptrs,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            sum_sq += tl.sum(x * x, axis=1)
        x_rms_inv = tl.rsqrt(sum_sq / K + eps)

        # Pass 2: project the low and high halves of this head. Keeping the
        # halves separate lets the rotary + QK RMSNorm happen entirely in
        # registers before Q/K are written.
        acc_lo = tl.zeros((BLOCK_M, BLOCK_HALF), dtype=tl.float32)
        acc_hi = tl.zeros((BLOCK_M, BLOCK_HALF), dtype=tl.float32)
        out_row_lo = weight_offset + head * D + half_cols
        out_row_hi = weight_offset + head * D + half_cols + BLOCK_HALF
        for k_start in range(0, K, BLOCK_K):
            ks = k_start + tl.arange(0, BLOCK_K)
            k_mask = ks < K
            x_ptrs = x_ptr + rows[:, None] * stride_xm + ks[None, :] * stride_xk
            x = tl.load(
                x_ptrs,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            nw = tl.load(norm_w_ptr + ks, mask=k_mask, other=0.0).to(tl.float32)
            x_hat = x * x_rms_inv[:, None] * nw[None, :]

            w_lo_ptrs = qkv_w_ptr + out_row_lo[:, None] * stride_wn + ks[None, :] * stride_wk
            w_hi_ptrs = qkv_w_ptr + out_row_hi[:, None] * stride_wn + ks[None, :] * stride_wk
            w_lo = tl.load(w_lo_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)
            w_hi = tl.load(w_hi_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)
            acc_lo += tl.dot(x_hat, tl.trans(w_lo), input_precision="ieee")
            acc_hi += tl.dot(x_hat, tl.trans(w_hi), input_precision="ieee")

        if DO_ROTARY_NORM_SCALE:
            cos = tl.load(
                cos_ptr + rows[:, None] * BLOCK_HALF + half_cols[None, :],
                mask=row_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            sin = tl.load(
                sin_ptr + rows[:, None] * BLOCK_HALF + half_cols[None, :],
                mask=row_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            rot_lo = acc_lo * cos + acc_hi * sin
            rot_hi = -acc_lo * sin + acc_hi * cos
            qk_sum_sq = tl.sum(rot_lo * rot_lo, axis=1) + tl.sum(rot_hi * rot_hi, axis=1)
            qk_rms_inv = tl.rsqrt(qk_sum_sq / D + eps)
            norm_scale = qk_rms_inv * scale
            acc_lo = rot_lo * norm_scale[:, None]
            acc_hi = rot_hi * norm_scale[:, None]

        out_lo_ptrs = (
            out_ptr
            + rows[:, None] * stride_om
            + head * stride_oh
            + half_cols[None, :] * stride_od
        )
        out_hi_ptrs = (
            out_ptr
            + rows[:, None] * stride_om
            + head * stride_oh
            + (half_cols[None, :] + BLOCK_HALF) * stride_od
        )
        tl.store(out_lo_ptrs, acc_lo.to(out_ptr.dtype.element_ty), mask=row_mask[:, None])
        tl.store(out_hi_ptrs, acc_hi.to(out_ptr.dtype.element_ty), mask=row_mask[:, None])


class NormQKVProjection(torch.autograd.Function):
    """Fused RMSNorm + concatenated Q/K/V linear projection.

    Forward: one Triton kernel folds the outer RMSNorm and all three
    QKV matmuls together, writing the concatenated output (M, N_q+N_k+N_v).
    Caller slices it into q, k, v.

    Backward: three linear backwards via cuBLAS + one RMSNorm-backward
    Triton kernel (`_rms_norm_bwd_kernel`, defined above in this section).
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        qkv_weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Run fused RMSNorm + QKV projection.

        Args:
          x: (M, K) contiguous CUDA tensor.
          norm_weight: (K,) RMSNorm scale.
          qkv_weight: (N_qkv, K) concatenated projection weight.
          eps: RMSNorm epsilon.

        Returns:
          (M, N_qkv) concatenated Q/K/V projection."""
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
            x,
            norm_weight,
            qkv_weight,
            out,
            rms_inv,
            M,
            N_qkv,
            K,
            eps,
            x.stride(0),
            x.stride(1),
            qkv_weight.stride(0),
            qkv_weight.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

        ctx.save_for_backward(x, norm_weight, qkv_weight, rms_inv)
        ctx.M, ctx.N_qkv, ctx.K = M, N_qkv, K
        return out

    @staticmethod
    def backward(
        ctx: Any,
        d_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Backprop for fused RMSNorm + QKV projection.

        Args:
          d_out: (M, N_qkv) gradient of concatenated output.

        Returns:
          Gradients for (x, norm_weight, qkv_weight, eps)."""
        x, norm_w, qkv_w, rms_inv = ctx.saved_tensors
        M, N_qkv, K = ctx.M, ctx.N_qkv, ctx.K
        d_out = d_out.contiguous()

        # Linear backward (cuBLAS): out = x_hat @ qkv_w.T
        #   dx_hat = d_out @ qkv_w
        #   d_qkv_w = d_out.T @ x_hat
        x_hat = x * rms_inv.unsqueeze(1) * norm_w
        dx_hat = d_out @ qkv_w  # (M, K)
        d_qkv_w = d_out.t() @ x_hat  # (N_qkv, K)

        # RMSNorm backward (Triton — reuses _rms_norm_bwd_kernel)
        BLOCK_M_BWD, BLOCK_K_BWD = 32, 64
        num_m_tiles = triton.cdiv(M, BLOCK_M_BWD)
        dx = torch.empty_like(x)
        dnw_partials = torch.empty(
            (num_m_tiles, K),
            dtype=norm_w.dtype,
            device=x.device,
        )
        grid_bwd = (num_m_tiles, triton.cdiv(K, BLOCK_K_BWD))
        _rms_norm_bwd_kernel[grid_bwd](
            x,
            rms_inv,
            norm_w,
            dx_hat,
            dx,
            dnw_partials,
            M,
            K,
            x.stride(0),
            x.stride(1),
            dx_hat.stride(0),
            dx_hat.stride(1),
            dx.stride(0),
            dx.stride(1),
            BLOCK_M=BLOCK_M_BWD,
            BLOCK_K=BLOCK_K_BWD,
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

    Args:
      x: (M, K) contiguous CUDA tensor.
      norm_weight: (K,) RMSNorm scale.
      qkv_weight: (N_qkv, K) concatenated Q/K/V projection weight.
      eps: RMSNorm epsilon.

    Returns:
      (M, N_qkv) concatenated Q/K/V projection.
    """
    return NormQKVProjection.apply(x, norm_weight, qkv_weight, eps)


def _flatten_rotary_table(
    table: torch.Tensor,
    M: int,
    head_dim: int,
) -> torch.Tensor:
    """Return rotary table as contiguous (M, head_dim / 2)."""
    half = head_dim // 2
    if table.ndim == 2:
        assert table.shape == (M, half), (
            f"rotary table must be {(M, half)}, got {tuple(table.shape)}"
        )
        return table.contiguous()

    if table.ndim == 4:
        assert table.shape[0] == 1 and table.shape[2] == 1
        assert table.shape[-1] == half
        T = table.shape[1]
        assert M % T == 0, f"M={M} is not divisible by rotary T={T}"
        B = M // T
        return table.expand(B, T, 1, half).reshape(M, half).contiguous()

    raise AssertionError(
        f"rotary table must be 2D (M, D/2) or 4D (1, T, 1, D/2), got {table.ndim}D"
    )


class NormQKVRotaryProjection(torch.autograd.Function):
    """Fused outer RMSNorm + Q/K/V projection with Q/K rotary/QK-norm/scale.

    Forward writes:
      q: (M, n_head, head_dim) after rotary + QK RMSNorm + scale.
      k: (M, n_kv_head, head_dim) after rotary + QK RMSNorm + scale.
      v: (M, n_kv_head, head_dim) plain projected values.

    Backward currently recomputes the reference PyTorch graph and uses
    autograd for gradients. That keeps this experimental forward fusion
    correct while leaving backward-performance work isolated.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        qkv_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        n_head: int,
        n_kv_head: int,
        head_dim: int,
        scale: float = 1.2,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run fused QKV projection plus Q/K rotary, QK RMSNorm, and scale.

        Args:
          x: (M, K) contiguous CUDA tensor.
          norm_weight: (K,) outer RMSNorm scale.
          qkv_weight: ((n_head + 2*n_kv_head) * head_dim, K) concatenated weight.
          cos: (M, head_dim/2) or broadcast table (1, T, 1, head_dim/2).
          sin: (M, head_dim/2) or broadcast table (1, T, 1, head_dim/2).
          n_head: number of query heads.
          n_kv_head: number of key/value heads.
          head_dim: per-head width; must be even.
          scale: post-QK-norm scalar multiplier.
          eps: RMSNorm epsilon for both outer norm and QK norm.

        Returns:
          Tuple `(q, k, v)` with shapes `(M, n_head, head_dim)`,
          `(M, n_kv_head, head_dim)`, and `(M, n_kv_head, head_dim)`.
        """
        assert x.is_cuda and x.is_contiguous()
        assert norm_weight.is_cuda and qkv_weight.is_cuda
        assert head_dim % 2 == 0
        M, K = x.shape
        expected_n = (n_head + 2 * n_kv_head) * head_dim
        N_qkv, K_w = qkv_weight.shape
        assert K == K_w, f"x last dim {K} != qkv_weight in dim {K_w}"
        assert N_qkv == expected_n, f"qkv_weight out dim {N_qkv} != {expected_n}"
        cos_flat = _flatten_rotary_table(cos, M, head_dim)
        sin_flat = _flatten_rotary_table(sin, M, head_dim)

        q = torch.empty((M, n_head, head_dim), dtype=x.dtype, device=x.device)
        k = torch.empty((M, n_kv_head, head_dim), dtype=x.dtype, device=x.device)
        v = torch.empty((M, n_kv_head, head_dim), dtype=x.dtype, device=x.device)
        BLOCK_M, BLOCK_K = 16, 32
        block_half = head_dim // 2

        q_grid = (triton.cdiv(M, BLOCK_M), n_head)
        _norm_qkv_rotary_fwd_kernel[q_grid](
            x,
            norm_weight,
            qkv_weight,
            cos_flat,
            sin_flat,
            q,
            M,
            K,
            head_dim,
            0,
            eps,
            scale,
            x.stride(0),
            x.stride(1),
            qkv_weight.stride(0),
            qkv_weight.stride(1),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_HALF=block_half,
            BLOCK_K=BLOCK_K,
            DO_ROTARY_NORM_SCALE=True,
        )

        k_grid = (triton.cdiv(M, BLOCK_M), n_kv_head)
        _norm_qkv_rotary_fwd_kernel[k_grid](
            x,
            norm_weight,
            qkv_weight,
            cos_flat,
            sin_flat,
            k,
            M,
            K,
            head_dim,
            n_head * head_dim,
            eps,
            scale,
            x.stride(0),
            x.stride(1),
            qkv_weight.stride(0),
            qkv_weight.stride(1),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_HALF=block_half,
            BLOCK_K=BLOCK_K,
            DO_ROTARY_NORM_SCALE=True,
        )

        v_grid = (triton.cdiv(M, BLOCK_M), n_kv_head)
        _norm_qkv_rotary_fwd_kernel[v_grid](
            x,
            norm_weight,
            qkv_weight,
            cos_flat,
            sin_flat,
            v,
            M,
            K,
            head_dim,
            (n_head + n_kv_head) * head_dim,
            eps,
            scale,
            x.stride(0),
            x.stride(1),
            qkv_weight.stride(0),
            qkv_weight.stride(1),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_HALF=block_half,
            BLOCK_K=BLOCK_K,
            DO_ROTARY_NORM_SCALE=False,
        )

        ctx.save_for_backward(x, norm_weight, qkv_weight, cos_flat, sin_flat)
        ctx.n_head = n_head
        ctx.n_kv_head = n_kv_head
        ctx.head_dim = head_dim
        ctx.scale = scale
        ctx.eps = eps
        return q, k, v

    @staticmethod
    def backward(
        ctx: Any,
        d_q: torch.Tensor,
        d_k: torch.Tensor,
        d_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None, None, None]:
        """Backprop via recomputation of the reference PyTorch graph."""
        x, norm_weight, qkv_weight, cos, sin = ctx.saved_tensors
        n_head = ctx.n_head
        n_kv_head = ctx.n_kv_head
        head_dim = ctx.head_dim
        scale = ctx.scale
        eps = ctx.eps
        half = head_dim // 2

        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            nw_req = norm_weight.detach().requires_grad_(True)
            qw_req = qkv_weight.detach().requires_grad_(True)

            x_rms_inv = torch.rsqrt((x_req.float() ** 2).mean(dim=-1, keepdim=True) + eps)
            x_hat = x_req * x_rms_inv.to(x_req.dtype) * nw_req
            qkv = x_hat @ qw_req.t()
            q_flat, k_flat, v_flat = qkv.split(
                [n_head * head_dim, n_kv_head * head_dim, n_kv_head * head_dim],
                dim=-1,
            )
            q = q_flat.view(-1, n_head, head_dim)
            k = k_flat.view(-1, n_kv_head, head_dim)
            v = v_flat.view(-1, n_kv_head, head_dim)

            cos_b = cos[:, None, :]
            sin_b = sin[:, None, :]

            def _rot_norm_scale(qk: torch.Tensor) -> torch.Tensor:
                lo = qk[..., :half]
                hi = qk[..., half:]
                rot_lo = lo * cos_b + hi * sin_b
                rot_hi = -lo * sin_b + hi * cos_b
                rotated = torch.cat([rot_lo, rot_hi], dim=-1)
                qk_rms_inv = torch.rsqrt(
                    (rotated.float() ** 2).mean(dim=-1, keepdim=True) + eps
                )
                return rotated * qk_rms_inv.to(rotated.dtype) * scale

            q = _rot_norm_scale(q)
            k = _rot_norm_scale(k)
            dx, dnw, dqw = torch.autograd.grad(
                (q, k, v),
                (x_req, nw_req, qw_req),
                (d_q.contiguous(), d_k.contiguous(), d_v.contiguous()),
                allow_unused=False,
            )

        return dx, dnw, dqw, None, None, None, None, None, None, None


def norm_qkv_rotary_projection(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    scale: float = 1.2,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused RMSNorm + QKV projection with Q/K rotary, QK RMSNorm, and scale.

    Args:
      x: (M, K) contiguous CUDA tensor.
      norm_weight: (K,) RMSNorm scale for the input projection.
      qkv_weight: ((n_head + 2*n_kv_head) * head_dim, K) concatenated Q/K/V weight.
      cos: (M, head_dim/2) or broadcast table (1, T, 1, head_dim/2).
      sin: (M, head_dim/2) or broadcast table (1, T, 1, head_dim/2).
      n_head: number of query heads.
      n_kv_head: number of key/value heads.
      head_dim: per-head width.
      scale: post-QK-norm scalar multiplier.
      eps: RMSNorm epsilon.

    Returns:
      Tuple `(q, k, v)` shaped for SDPA: `(M, n_head, head_dim)`,
      `(M, n_kv_head, head_dim)`, and `(M, n_kv_head, head_dim)`.
    """
    return NormQKVRotaryProjection.apply(
        x,
        norm_weight,
        qkv_weight,
        cos,
        sin,
        n_head,
        n_kv_head,
        head_dim,
        scale,
        eps,
    )


# ─────────────────────────────────────────────────────────────────────
# Rotary + RMSNorm + scale autograd.Function (for Q or K)
#
# Forward uses _rotary_qk_norm_scale_kernel (Triton).
# Backward chain: scale → RMSNorm bwd → rotary inverse (use sin → -sin
# rotation; rotary's Jacobian is orthogonal so the inverse is the same
# shape with sin negated). Uses eager PyTorch ops in backward for
# clarity (each op is small elementwise).
# ─────────────────────────────────────────────────────────────────────

if _HAS_TRITON:

    @triton.jit
    def _rotary_qk_norm_scale_kernel(
        qk_ptr,  # (M, D) — in: Q or K before rotary
        cos_ptr,  # (M, D/2) — in: rotary cosine table
        sin_ptr,  # (M, D/2) — in: rotary sine table
        out_ptr,  # (M, D) — out: rotated, RMS-normalized, scaled tensor
        rms_inv_ptr,  # (M,) fp32 — out: saved inverse RMS for bwd
        M,  # int — row count after flattening batch/head/sequence dims
        D,  # int — even head width
        scale,  # float — post-norm scalar multiplier
        eps,  # float — RMSNorm epsilon
        stride_qm,  # int — qk stride along M
        stride_qd,  # int — qk stride along D
        stride_om,  # int — out stride along M
        stride_od,  # int — out stride along D
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
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
        ptrs_hi = (
            qk_ptr + rows[:, None] * stride_qm + (cs_cols[None, :] + half) * stride_qd
        )
        x1 = tl.load(ptrs_lo, mask=row_mask[:, None], other=0.0).to(tl.float32)
        x2 = tl.load(ptrs_hi, mask=row_mask[:, None], other=0.0).to(tl.float32)
        cos = tl.load(
            cos_ptr + rows[:, None] * (BLOCK_D // 2) + cs_cols[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        sin = tl.load(
            sin_ptr + rows[:, None] * (BLOCK_D // 2) + cs_cols[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        # Rotary: y1 = x1·cos + x2·sin ;  y2 = -x1·sin + x2·cos
        y1 = x1 * cos + x2 * sin
        y2 = -x1 * sin + x2 * cos

        # RMSNorm + scale: compute mean(y²) over full D using y1, y2 in registers
        sum_sq = tl.sum(y1 * y1, axis=1) + tl.sum(y2 * y2, axis=1)
        rms_inv = tl.rsqrt(sum_sq / D + eps)
        norm_scale = rms_inv * scale

        # Apply norm·scale, write halves back
        y1_out = y1 * norm_scale[:, None]
        y2_out = y2 * norm_scale[:, None]
        out_ptrs_lo = out_ptr + rows[:, None] * stride_om + cs_cols[None, :] * stride_od
        out_ptrs_hi = (
            out_ptr + rows[:, None] * stride_om + (cs_cols[None, :] + half) * stride_od
        )
        tl.store(
            out_ptrs_lo, y1_out.to(out_ptr.dtype.element_ty), mask=row_mask[:, None]
        )
        tl.store(
            out_ptrs_hi, y2_out.to(out_ptr.dtype.element_ty), mask=row_mask[:, None]
        )
        # rms_inv saved for backward
        tl.store(rms_inv_ptr + rows, rms_inv, mask=row_mask)


class RotaryQKNormScale(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        qk: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        scale: float,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Run rotary embedding, RMSNorm, and scalar scale.

        Args:
          qk: (M, D) contiguous CUDA tensor; D must be even.
          cos: (M, D/2) rotary cosine table.
          sin: (M, D/2) rotary sine table.
          scale: post-norm scalar multiplier.
          eps: RMSNorm epsilon.

        Returns:
          (M, D) rotated, normalized, and scaled tensor."""
        assert qk.is_cuda and qk.is_contiguous()
        assert cos.is_contiguous() and sin.is_contiguous()
        M, D = qk.shape
        assert D % 2 == 0
        out = torch.empty_like(qk)
        rms_inv = torch.empty(M, dtype=torch.float32, device=qk.device)
        BLOCK_M = 32
        grid = (triton.cdiv(M, BLOCK_M),)
        _rotary_qk_norm_scale_kernel[grid](
            qk,
            cos,
            sin,
            out,
            rms_inv,
            M,
            D,
            scale,
            eps,
            qk.stride(0),
            qk.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_D=D,
        )
        ctx.save_for_backward(qk, cos, sin, rms_inv)
        ctx.scale = scale
        ctx.D = D
        return out

    @staticmethod
    def backward(
        ctx: Any,
        d_out: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        """Backprop through rotary + unweighted RMSNorm + scalar scale.

        Args:
          d_out: (M, D) gradient of output.

        Returns:
          Gradients for (qk, cos, sin, scale, eps)."""
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
    """Fused rotary embedding + RMSNorm + multiplicative scale for Q or K.

    Args:
      qk: (M, D) contiguous CUDA tensor; D must be even.
      cos: (M, D/2) rotary cosine table.
      sin: (M, D/2) rotary sine table.
      scale: post-norm scalar multiplier.
      eps: RMSNorm epsilon.

    Returns:
      (M, D) rotated, normalized, and scaled tensor.
    """
    return RotaryQKNormScale.apply(qk, cos, sin, scale, eps)
