"""Attention QKV-side Triton kernels for nanoops.

Contains:
  - `norm_qkv_rotary_projection`: fused outer RMSNorm + Q/K/V projection;
    Q/K are immediately rotary-embedded, RMS-normalized, and scaled before
    being written.
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
# This path computes the OUTER RMSNorm inverse once per row, then shares
# it across the Q/K/V projection kernels. Q/K immediately apply rotary +
# QK RMSNorm + scale before being written.
# ─────────────────────────────────────────────────────────────────────


if _HAS_TRITON:

    @triton.jit
    def _x_rms_inv_fwd_kernel(
        x_ptr,  # (M, K) — in: activation before RMSNorm
        rms_inv_ptr,  # (M,) fp32 — out: per-row inverse RMS
        M,  # int — row count after flattening batch/time
        K,  # int — input hidden width
        eps,  # float — outer RMSNorm epsilon
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Compute the outer RMSNorm inverse once per row.

        Q/K/V projection kernels share this value instead of recomputing
        the K reduction once per projected head.
        """
        pid_m = tl.program_id(0)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M

        sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            ks = k_start + tl.arange(0, BLOCK_K)
            k_mask = ks < K
            x_ptrs = x_ptr + rows[:, None] * K + ks[None, :]
            x = tl.load(
                x_ptrs,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            sum_sq += tl.sum(x * x, axis=1)

        tl.store(rms_inv_ptr + rows, tl.rsqrt(sum_sq / K + eps), mask=row_mask)

    @triton.jit
    def _norm_qkv_rotary_fwd_kernel(
        x_ptr,  # (M, K) — in: activation before RMSNorm
        rms_inv_ptr,  # (M,) fp32 — in: precomputed inverse RMS for x
        norm_w_ptr,  # (K,) — in: outer RMSNorm scale
        qkv_w_ptr,  # (N_qkv, K) — in: concatenated Q/K/V projection weight
        cos_ptr,  # (M, D/2) — in: rotary cosine table
        sin_ptr,  # (M, D/2) — in: rotary sine table
        q_ptr,  # (M, n_head, D) — out: Q after rotary + QK norm + scale
        k_ptr,  # (M, n_kv_head, D) — out: K after rotary + QK norm + scale
        v_ptr,  # (M, n_kv_head, D) — out: projected V
        M,  # int — row count after flattening batch/time
        K,  # int — input hidden width
        D,  # int — per-head width
        n_head,  # int — number of Q heads
        n_kv_head,  # int — number of K/V heads
        rotary_T,  # int — rotary table sequence length before batch broadcast
        eps,  # float — outer and QK RMSNorm epsilon
        scale,  # float — post-QK-norm scalar multiplier
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Project one Q/K/V head tile selected by the second grid axis.

        A part is one projected head: first Q heads, then K heads, then V
        heads. V exits before rotary and QK RMSNorm.
        """
        pid_m = tl.program_id(0)
        part = tl.program_id(1)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_D)
        half_cols = tl.arange(0, BLOCK_D // 2)
        row_mask = rows < M

        x_rms_inv = tl.load(rms_inv_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        is_q = part < n_head
        is_k = (part >= n_head) & (part < n_head + n_kv_head)
        is_v = part >= n_head + n_kv_head
        head = tl.where(
            is_q,
            part,
            tl.where(is_k, part - n_head, part - n_head - n_kv_head),
        )
        weight_offset = tl.where(
            is_q,
            0,
            tl.where(is_k, n_head * D, (n_head + n_kv_head) * D),
        )

        is_hi_col = (cols % 2) == 1
        src_cols = (cols // 2) + tl.where(is_hi_col, BLOCK_D // 2, 0)
        out_rows = weight_offset + head * D + src_cols

        # Project one full head with a single dot. Weight rows are loaded in
        # [lo0, hi0, lo1, hi1, ...] order so the accumulator can be reshaped
        # into (BLOCK_M, BLOCK_D // 2, 2) and split into rotary pairs in
        # registers.
        acc_interleaved = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            ks = k_start + tl.arange(0, BLOCK_K)
            k_mask = ks < K
            x_ptrs = x_ptr + rows[:, None] * K + ks[None, :]
            x = tl.load(
                x_ptrs,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            nw = tl.load(norm_w_ptr + ks, mask=k_mask, other=0.0).to(tl.float32)
            x_hat = x * x_rms_inv[:, None] * nw[None, :]

            w_ptrs = qkv_w_ptr + out_rows[:, None] * K + ks[None, :]
            w = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)
            acc_interleaved += tl.dot(x_hat, tl.trans(w), input_precision="ieee")

        if is_v:
            v_ptrs = (
                v_ptr
                + rows[:, None] * n_kv_head * D
                + head * D
                + src_cols[None, :]
            )
            tl.store(
                v_ptrs,
                acc_interleaved.to(v_ptr.dtype.element_ty),
                mask=row_mask[:, None],
            )
            return

        pair_axis = tl.arange(0, 2)
        acc_pair = tl.reshape(acc_interleaved, (BLOCK_M, BLOCK_D // 2, 2))
        acc_lo = tl.sum(tl.where(pair_axis[None, None, :] == 0, acc_pair, 0.0), axis=2)
        acc_hi = tl.sum(tl.where(pair_axis[None, None, :] == 1, acc_pair, 0.0), axis=2)
        rotary_rows = rows % rotary_T

        cos = tl.load(
            cos_ptr + rotary_rows[:, None] * (BLOCK_D // 2) + half_cols[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        sin = tl.load(
            sin_ptr + rotary_rows[:, None] * (BLOCK_D // 2) + half_cols[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        rot_lo = acc_lo * cos + acc_hi * sin
        rot_hi = -acc_lo * sin + acc_hi * cos
        qk_sum_sq = tl.sum(rot_lo * rot_lo, axis=1) + tl.sum(rot_hi * rot_hi, axis=1)
        qk_rms_inv = tl.rsqrt(qk_sum_sq / D + eps)
        norm_scale = qk_rms_inv * scale
        qk_lo = rot_lo * norm_scale[:, None]
        qk_hi = rot_hi * norm_scale[:, None]

        if is_q:
            q_lo_ptrs = (
                q_ptr
                + rows[:, None] * n_head * D
                + head * D
                + half_cols[None, :]
            )
            q_hi_ptrs = (
                q_ptr
                + rows[:, None] * n_head * D
                + head * D
                + half_cols[None, :]
                + (BLOCK_D // 2)
            )
            tl.store(q_lo_ptrs, qk_lo.to(q_ptr.dtype.element_ty), mask=row_mask[:, None])
            tl.store(q_hi_ptrs, qk_hi.to(q_ptr.dtype.element_ty), mask=row_mask[:, None])
        else:
            k_lo_ptrs = (
                k_ptr
                + rows[:, None] * n_kv_head * D
                + head * D
                + half_cols[None, :]
            )
            k_hi_ptrs = (
                k_ptr
                + rows[:, None] * n_kv_head * D
                + head * D
                + half_cols[None, :]
                + (BLOCK_D // 2)
            )
            tl.store(k_lo_ptrs, qk_lo.to(k_ptr.dtype.element_ty), mask=row_mask[:, None])
            tl.store(k_hi_ptrs, qk_hi.to(k_ptr.dtype.element_ty), mask=row_mask[:, None])

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


def _rotary_table_T(
    table: torch.Tensor,
    M: int,
    head_dim: int,
) -> int:
    """Validate rotary table layout and return its unbroadcasted T."""
    half = head_dim // 2
    assert table.is_cuda and table.is_contiguous()
    if table.ndim == 2:
        assert table.shape == (M, half), (
            f"rotary table must be {(M, half)}, got {tuple(table.shape)}"
        )
        return M

    if table.ndim == 4:
        assert table.shape[0] == 1 and table.shape[2] == 1
        assert table.shape[-1] == half
        T = table.shape[1]
        assert M % T == 0, f"M={M} is not divisible by rotary T={T}"
        return T

    raise AssertionError(
        f"rotary table must be 2D (M, D/2) or 4D (1, T, 1, D/2), got {table.ndim}D"
    )


def _norm_qkv_rotary_projection_manual_backward(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    d_q: torch.Tensor,
    d_k: torch.Tensor,
    d_v: torch.Tensor,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    scale: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form PyTorch backward for `NormQKVRotaryProjection`.

    This helper implements the fp32 algebra used by the autograd.Function today
    and serves as scaffolding for a future Triton backward kernel. Rotary
    cos/sin are treated as constants; the returned gradients are for x,
    norm_weight, and qkv_weight.

    Forward math:
      s_x        = rsqrt(mean(x^2) + eps)
      x_norm     = x * s_x
      x_hat      = x_norm * norm_weight
      z          = x_hat @ qkv_weight.T
      q0, k0, v0 = split(z)

      r_lo = lo * cos + hi * sin
      r_hi = -lo * sin + hi * cos
      y    = scale * rmsnorm(concat(r_lo, r_hi))

    Backward math:
      rmsnorm_no_weight_bwd(r, g):
        y0 = r * s
        dr = s * (g - y0 * mean(g * y0))

      rotary_bwd:
        d_lo = d_r_lo * cos - d_r_hi * sin
        d_hi = d_r_lo * sin + d_r_hi * cos

      linear_bwd:
        d_x_hat = d_z @ qkv_weight
        dW      = d_z.T @ x_hat

      outer RMSNorm bwd:
        d_x = s_x * (d_x_norm - x_norm * mean(d_x_norm * x_norm))
    """
    M, _ = x.shape
    half = head_dim // 2
    cos_flat = _flatten_rotary_table(cos, M, head_dim).float()
    sin_flat = _flatten_rotary_table(sin, M, head_dim).float()
    cos_b = cos_flat[:, None, :]
    sin_b = sin_flat[:, None, :]

    x_f = x.float()
    norm_w_f = norm_weight.float()
    qkv_w_f = qkv_weight.float()
    x_rms_inv = torch.rsqrt((x_f * x_f).mean(dim=-1, keepdim=True) + eps)
    x_norm = x_f * x_rms_inv
    x_hat = x_norm * norm_w_f

    qkv = x_hat @ qkv_w_f.t()
    q_flat, k_flat, _ = qkv.split(
        [n_head * head_dim, n_kv_head * head_dim, n_kv_head * head_dim],
        dim=-1,
    )
    q0 = q_flat.view(M, n_head, head_dim).float()
    k0 = k_flat.view(M, n_kv_head, head_dim).float()

    def _qk_bwd(qk: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
        lo = qk[..., :half]
        hi = qk[..., half:]
        rot_lo = lo * cos_b + hi * sin_b
        rot_hi = -lo * sin_b + hi * cos_b
        rotated = torch.cat([rot_lo, rot_hi], dim=-1)

        rms_inv = torch.rsqrt((rotated * rotated).mean(dim=-1, keepdim=True) + eps)
        y0 = rotated * rms_inv
        g_eff = grad_out.contiguous().float() * scale
        inner = (g_eff * y0).mean(dim=-1, keepdim=True)
        d_rot = rms_inv * (g_eff - y0 * inner)

        d_rot_lo = d_rot[..., :half]
        d_rot_hi = d_rot[..., half:]
        d_lo = d_rot_lo * cos_b - d_rot_hi * sin_b
        d_hi = d_rot_lo * sin_b + d_rot_hi * cos_b
        return torch.cat([d_lo, d_hi], dim=-1)

    d_q0 = _qk_bwd(q0, d_q).reshape(M, n_head * head_dim)
    d_k0 = _qk_bwd(k0, d_k).reshape(M, n_kv_head * head_dim)
    d_v0 = d_v.contiguous().reshape(M, n_kv_head * head_dim).float()
    d_z = torch.cat([d_q0, d_k0, d_v0], dim=-1)

    d_x_hat = d_z @ qkv_w_f
    d_qkv_weight = d_z.t() @ x_hat

    d_norm_weight = torch.sum(d_x_hat * x_norm, dim=0)
    d_x_norm = d_x_hat * norm_w_f
    inner = (d_x_norm * x_norm).mean(dim=-1, keepdim=True)
    d_x = x_rms_inv * (d_x_norm - x_norm * inner)
    return d_x.to(x.dtype), d_norm_weight.to(norm_weight.dtype), d_qkv_weight.to(qkv_weight.dtype)


class NormQKVRotaryProjection(torch.autograd.Function):
    """Fused outer RMSNorm + Q/K/V projection with Q/K rotary/QK-norm/scale.

    Forward writes:
      q: (M, n_head, head_dim) after rotary + QK RMSNorm + scale.
      k: (M, n_kv_head, head_dim) after rotary + QK RMSNorm + scale.
      v: (M, n_kv_head, head_dim) plain projected values.

    Backward currently uses a PyTorch closed-form implementation of the same
    math as scaffolding for a future Triton backward kernel.
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
        assert norm_weight.is_contiguous() and qkv_weight.is_contiguous()
        assert head_dim % 2 == 0
        assert head_dim == 128, f"expected d24 head_dim=128, got {head_dim}"
        M, K = x.shape
        expected_n = (n_head + 2 * n_kv_head) * head_dim
        N_qkv, K_w = qkv_weight.shape
        assert K == K_w, f"x last dim {K} != qkv_weight in dim {K_w}"
        assert N_qkv == expected_n, f"qkv_weight out dim {N_qkv} != {expected_n}"
        rotary_T = _rotary_table_T(cos, M, head_dim)
        assert _rotary_table_T(sin, M, head_dim) == rotary_T

        q = torch.empty((M, n_head, head_dim), dtype=x.dtype, device=x.device)
        k = torch.empty((M, n_kv_head, head_dim), dtype=x.dtype, device=x.device)
        v = torch.empty((M, n_kv_head, head_dim), dtype=x.dtype, device=x.device)
        RMS_BLOCK_M, RMS_BLOCK_K = 32, 32
        QKV_BLOCK_M, QKV_BLOCK_K = 64, 16
        rms_inv = torch.empty((M,), dtype=torch.float32, device=x.device)
        _x_rms_inv_fwd_kernel[(triton.cdiv(M, RMS_BLOCK_M),)](
            x,
            rms_inv,
            M,
            K,
            eps,
            BLOCK_M=RMS_BLOCK_M,
            BLOCK_K=RMS_BLOCK_K,
            num_warps=4,
        )

        grid = (triton.cdiv(M, QKV_BLOCK_M), n_head + 2 * n_kv_head)
        _norm_qkv_rotary_fwd_kernel[grid](
            x,
            rms_inv,
            norm_weight,
            qkv_weight,
            cos,
            sin,
            q,
            k,
            v,
            M,
            K,
            head_dim,
            n_head,
            n_kv_head,
            rotary_T,
            eps,
            scale,
            BLOCK_M=QKV_BLOCK_M,
            BLOCK_D=head_dim,
            BLOCK_K=QKV_BLOCK_K,
            num_warps=2,
            num_stages=2,
        )

        ctx.save_for_backward(x, norm_weight, qkv_weight, cos, sin)
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
        """Backprop via the closed-form PyTorch formula."""
        x, norm_weight, qkv_weight, cos, sin = ctx.saved_tensors
        dx, dnw, dqw = _norm_qkv_rotary_projection_manual_backward(
            x,
            norm_weight,
            qkv_weight,
            cos,
            sin,
            d_q,
            d_k,
            d_v,
            ctx.n_head,
            ctx.n_kv_head,
            ctx.head_dim,
            ctx.scale,
            ctx.eps,
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
