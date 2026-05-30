"""Attention SDPA-side Triton kernels for nanoops.

Contains `flash_sdpa`: Flash-style sliding-causal SDPA with a chunked
backward. Re-exported through `nanoops.triton_kernels`.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch.library import wrap_triton

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def _pick_head_group(n_head: int) -> int:
    """Pick how many heads one Triton program handles in a static loop."""
    value = os.environ.get("NANOOPS_FLASH_SDPA_HEAD_GROUP")
    if value is None:
        return 1
    head_group = int(value)
    if head_group <= 0:
        raise ValueError(f"NANOOPS_FLASH_SDPA_HEAD_GROUP must be > 0, got {value}")
    return min(head_group, n_head)


def _pick_sdpa_tile_config(head_dim: int) -> tuple[int, int, int, int]:
    """Return `(block_m, block_n, num_warps, num_stages)` for SDPA kernels."""
    if head_dim >= 128:
        return 32, 32, 4, 1
    return 64, 64, 4, 1


# ─────────────────────────────────────────────────────────────────────
# Flash-style sliding-window SDPA in Triton.
#
# Standard Flash Attention pattern, adapted for nanchat's sliding-causal mask.
#
# Forward math for one (batch, head), with i indexing query rows and j key rows:
#   visible(i, j) = max(0, i - WINDOW + 1) <= j <= i
#   S_ij          = sm_scale * dot(Q_i, K_j)          if visible(i, j)
#                 = -inf                             otherwise
#   P_ij          = exp(S_ij - LSE_i)
#   LSE_i         = log(sum_j exp(S_ij))
#   O_i           = sum_j P_ij * V_j
#
# The kernel never materializes S or P as (L, L). It tiles Q by BLOCK_M rows and
# streams K/V by BLOCK_N rows. For each Q row it maintains an online-softmax
# triple `(m, l, acc)` where:
#   m   = running max score
#   l   = running sum exp(score - m)
#   acc = running sum exp(score - m) * V
# For a new score tile `s`:
#   m_new   = max(m, max_j s_j)
#   alpha   = exp(m - m_new)
#   p_hat_j = exp(s_j - m_new)
#   l_new   = alpha * l + sum_j p_hat_j
#   acc_new = alpha * acc + p_hat @ V_tile
# Final output is `O = acc / l`; saved backward state is
# `LSE = m + log(l)` per query row.
#
# Backward math, given G = dO:
#   Delta_i = sum_d O_id * G_id
#   P_ij    = exp(S_ij - LSE_i)              # recomputed from Q/K/LSE
#   dV_j   += sum_i P_ij * G_i
#   dP_ij   = dot(G_i, V_j)
#   dS_ij   = P_ij * (dP_ij - Delta_i)
#   dQ_i   += sm_scale * sum_j dS_ij * K_j
#   dK_j   += sm_scale * sum_i dS_ij * Q_i
#
# In code we fold `sm_scale` into the `ds` tile before the dQ/dK matmuls:
#   ds = P * (dP - Delta) * sm_scale
# so `dq += ds @ K` and `dk += ds.T @ Q`.
#
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
        Q,  # (B, M, H, D) — in: query
        K,  # (B, N, H, D) — in: key
        V,  # (B, N, H, D) — in: value
        sm_scale,  # float — attention scale, usually D**-0.5
        LSE,  # (B, M, H) fp32 — out: row log-sum-exp for bwd
        OUT,  # (B, M, H, D) — out: attention output
        B,  # int — batch size
        H,  # int — number of attention heads
        M,  # int — query sequence length
        N,  # int — key/value sequence length
        WINDOW: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        M_TILES: tl.constexpr,
        HEAD_GROUP: tl.constexpr,
        D: tl.constexpr,
    ):
        """Forward: one program per (batch × Q-tile × head-group).

        Q/K/V/OUT stay in contiguous `(B, T, H, D)` layout at the kernel
        boundary. The launch grid prioritizes `(batch, row-tile)` on axis 0
        and groups heads on axis 1. If HEAD_GROUP covers all heads, the grid
        does not split the head dimension; the program loops over heads inside.
        """
        pid_bm = tl.program_id(0)
        pid_hg = tl.program_id(1)
        bid = pid_bm // M_TILES
        pid_m = pid_bm - bid * M_TILES

        # v1 requires contiguous (B, T, H, D) Q/K/V/OUT and H_q == H_kv.
        # This keeps the kernel signature short; add explicit strides back
        # if we later support packed/non-contiguous layouts.
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        m_mask = offs_m < M
        kv_tile_start = tl.maximum(0, pid_m * BLOCK_M - WINDOW + 1) // BLOCK_N
        kv_high = tl.minimum(N, pid_m * BLOCK_M + BLOCK_M)
        kv_tile_end = (kv_high + BLOCK_N - 1) // BLOCK_N
        batch_q_base = bid * M * H * D
        batch_k_base = bid * N * H * D
        batch_lse_base = bid * M * H

        for head_off in tl.static_range(0, HEAD_GROUP):
            hid = pid_hg * HEAD_GROUP + head_off
            head_mask = hid < H
            q_base = batch_q_base + hid * D
            k_base = batch_k_base + hid * D
            v_base = batch_k_base + hid * D
            o_base = batch_q_base + hid * D
            lse_base = batch_lse_base + hid

            # Q tile: load BLOCK_M rows of Q for this program/head.
            q_ptrs = Q + q_base + offs_m[:, None] * H * D + offs_d[None, :]
            q = tl.load(
                q_ptrs,
                mask=m_mask[:, None] & head_mask,
                other=0.0,
            ).to(tl.float32)

            # Online softmax state per Q row.
            m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

            for kv_idx in range(kv_tile_start, kv_tile_end):
                offs_n = kv_idx * BLOCK_N + tl.arange(0, BLOCK_N)
                n_mask = offs_n < N

                k_ptrs = K + k_base + offs_n[:, None] * H * D + offs_d[None, :]
                v_ptrs = V + v_base + offs_n[:, None] * H * D + offs_d[None, :]
                kv_mask = n_mask[:, None] & head_mask
                k_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
                v_tile = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

                s = tl.dot(q, tl.trans(k_tile), input_precision="ieee") * sm_scale

                j = offs_n[None, :]
                i = offs_m[:, None]
                mask_keep = (
                    (j <= i)
                    & (j >= i - WINDOW + 1)
                    & m_mask[:, None]
                    & n_mask[None, :]
                    & head_mask
                )
                s = tl.where(mask_keep, s, -float("inf"))

                m_new = tl.maximum(m_i, tl.max(s, axis=1))
                all_masked = m_new == -float("inf")
                alpha = tl.where(all_masked, 1.0, tl.exp(m_i - m_new))
                p_unscaled = tl.where(
                    all_masked[:, None],
                    0.0,
                    tl.exp(s - m_new[:, None]),
                )
                l_i = l_i * alpha + tl.sum(p_unscaled, axis=1)
                acc = acc * alpha[:, None] + tl.dot(
                    p_unscaled.to(v_tile.dtype),
                    v_tile,
                    input_precision="ieee",
                )
                m_i = m_new

            acc = acc / l_i[:, None]
            lse = m_i + tl.log(l_i)

            o_ptrs = OUT + o_base + offs_m[:, None] * H * D + offs_d[None, :]
            tl.store(
                o_ptrs,
                acc.to(OUT.dtype.element_ty),
                mask=m_mask[:, None] & head_mask,
            )

            lse_ptrs = LSE + lse_base + offs_m * H
            tl.store(lse_ptrs, lse, mask=m_mask & head_mask)

    @triton.jit
    def _flash_attn_bwd_preprocess_kernel(
        OUT,  # (B, M, H, D) — in: forward output
        dO,  # (B, M, H, D) — in: gradient of output
        DELTA,  # (B, M, H) fp32 — out: row dot(O, dO)
        B,  # int — batch size
        H,  # int — number of attention heads
        M,  # int — query sequence length
        BLOCK_M: tl.constexpr,
        M_TILES: tl.constexpr,
        HEAD_GROUP: tl.constexpr,
        D: tl.constexpr,
    ):
        """Precompute Delta[i] = sum_j o[i, j] * dO[i, j] — used in bwd to skip
        the inner softmax-bwd reduction. This is the classic Flash trick."""
        pid_bm = tl.program_id(0)
        pid_hg = tl.program_id(1)
        bid = pid_bm // M_TILES
        pid_m = pid_bm - bid * M_TILES

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        m_mask = offs_m < M
        batch_q_base = bid * M * H * D
        batch_lse_base = bid * M * H

        for head_off in tl.static_range(0, HEAD_GROUP):
            hid = pid_hg * HEAD_GROUP + head_off
            head_mask = hid < H
            o_base = batch_q_base + hid * D
            o_ptrs = OUT + o_base + offs_m[:, None] * H * D + offs_d[None, :]
            do_ptrs = dO + o_base + offs_m[:, None] * H * D + offs_d[None, :]
            mask = m_mask[:, None] & head_mask
            o = tl.load(o_ptrs, mask=mask, other=0.0).to(tl.float32)
            do = tl.load(do_ptrs, mask=mask, other=0.0).to(tl.float32)
            d_row = tl.sum(o * do, axis=1)
            d_ptrs = DELTA + batch_lse_base + offs_m * H + hid
            tl.store(d_ptrs, d_row, mask=m_mask & head_mask)

    @triton.jit
    def _flash_attn_bwd_kernel(
        Q,  # (B, M, H, D) — in: query
        K,  # (B, N, H, D) — in: key
        V,  # (B, N, H, D) — in: value
        sm_scale,  # float — attention scale
        LSE,  # (B, M, H) fp32 — in: saved row log-sum-exp
        DELTA,  # (B, M, H) fp32 — in: row dot(O, dO)
        dO,  # (B, M, H, D) — in: grad wrt output
        dQ,  # (B, M, H, D) — out: grad wrt query
        dK,  # (B, N, H, D) — out: grad wrt key, atomic accumulated
        dV,  # (B, N, H, D) — out: grad wrt value, atomic accumulated
        B,  # int — batch size
        H,  # int — number of attention heads
        M,  # int — query sequence length
        N,  # int — key/value sequence length
        WINDOW: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        M_TILES: tl.constexpr,
        HEAD_GROUP: tl.constexpr,
        D: tl.constexpr,
    ):
        """Backward: iterate Q tiles, stream K/V tiles within the sliding band.
        Atomic-add dK/dV across overlapping Q tiles (each K position can be
        touched by multiple Q tiles within the window)."""
        pid_bm = tl.program_id(0)
        pid_hg = tl.program_id(1)
        bid = pid_bm // M_TILES
        pid_m = pid_bm - bid * M_TILES

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        m_mask = offs_m < M

        kv_tile_start = tl.maximum(0, pid_m * BLOCK_M - WINDOW + 1) // BLOCK_N
        kv_high = tl.minimum(N, pid_m * BLOCK_M + BLOCK_M)
        kv_tile_end = (kv_high + BLOCK_N - 1) // BLOCK_N
        batch_q_base = bid * M * H * D
        batch_k_base = bid * N * H * D
        batch_lse_base = bid * M * H

        for head_off in tl.static_range(0, HEAD_GROUP):
            hid = pid_hg * HEAD_GROUP + head_off
            head_mask = hid < H
            q_base = batch_q_base + hid * D
            k_base = batch_k_base + hid * D
            v_base = batch_k_base + hid * D
            lse_base = batch_lse_base + hid

            # Load Q tile, dO tile, LSE, Delta for this Q range/head.
            q_ptrs = Q + q_base + offs_m[:, None] * H * D + offs_d[None, :]
            do_ptrs = dO + q_base + offs_m[:, None] * H * D + offs_d[None, :]
            lse_ptrs = LSE + lse_base + offs_m * H
            d_ptrs = DELTA + lse_base + offs_m * H
            q_mask = m_mask[:, None] & head_mask
            q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
            do = tl.load(do_ptrs, mask=q_mask, other=0.0).to(tl.float32)
            lse = tl.load(lse_ptrs, mask=m_mask & head_mask, other=0.0)
            d_row = tl.load(d_ptrs, mask=m_mask & head_mask, other=0.0)

            dq_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

            for kv_idx in range(kv_tile_start, kv_tile_end):
                offs_n = kv_idx * BLOCK_N + tl.arange(0, BLOCK_N)
                n_mask = offs_n < N

                k_ptrs = K + k_base + offs_n[:, None] * H * D + offs_d[None, :]
                v_ptrs = V + v_base + offs_n[:, None] * H * D + offs_d[None, :]
                kv_mask = n_mask[:, None] & head_mask
                k_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
                v_tile = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

                s = tl.dot(q, tl.trans(k_tile), input_precision="ieee") * sm_scale
                j = offs_n[None, :]
                i = offs_m[:, None]
                mask_keep = (
                    (j <= i)
                    & (j >= i - WINDOW + 1)
                    & m_mask[:, None]
                    & n_mask[None, :]
                    & head_mask
                )
                s = tl.where(mask_keep, s, -float("inf"))
                p = tl.exp(s - lse[:, None])

                dv = tl.dot(tl.trans(p).to(do.dtype), do, input_precision="ieee")
                dp = tl.dot(do, tl.trans(v_tile), input_precision="ieee")
                ds = p * (dp - d_row[:, None]) * sm_scale
                dq_acc += tl.dot(ds.to(k_tile.dtype), k_tile, input_precision="ieee")
                dk = tl.dot(tl.trans(ds).to(q.dtype), q, input_precision="ieee")

                dk_ptrs = dK + k_base + offs_n[:, None] * H * D + offs_d[None, :]
                dv_ptrs = dV + v_base + offs_n[:, None] * H * D + offs_d[None, :]
                tl.atomic_add(
                    dk_ptrs,
                    dk.to(dK.dtype.element_ty),
                    mask=n_mask[:, None] & head_mask,
                )
                tl.atomic_add(
                    dv_ptrs,
                    dv.to(dV.dtype.element_ty),
                    mask=n_mask[:, None] & head_mask,
                )

            dq_ptrs = dQ + q_base + offs_m[:, None] * H * D + offs_d[None, :]
            tl.store(
                dq_ptrs,
                dq_acc.to(dQ.dtype.element_ty),
                mask=m_mask[:, None] & head_mask,
            )


def _flash_sdpa_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Flash-style sliding-causal SDPA and return `(out, lse)`.

    Args:
      q: (B, M, H, D) contiguous CUDA query tensor.
      k: (B, N, H, D) contiguous CUDA key tensor; v1 requires N == M.
      v: (B, N, H, D) contiguous CUDA value tensor.
      window_size: total visible keys per query.

    Returns:
      out: (B, M, H, D), dtype=q.dtype.
      lse: (B, M, H), fp32 row log-sum-exp for backward.
    """
    if not _HAS_TRITON:
        raise RuntimeError("flash_sdpa requires triton")
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert q.shape == k.shape == v.shape, (
        f"v1 requires H_q == H_kv and same L: q{q.shape} k{k.shape} v{v.shape}"
    )
    B, M, H, D = q.shape
    N = k.shape[1]
    sm_scale = D**-0.5

    out = torch.empty_like(q)
    lse = torch.empty((B, M, H), dtype=torch.float32, device=q.device)

    block_m, block_n, num_warps, num_stages = _pick_sdpa_tile_config(D)
    # Keep tensor layout as `(B, T, H, D)` into Triton. The launch grid
    # prioritizes batch/row tiles on axis 0 and uses axis 1 for heads.
    m_tiles = triton.cdiv(M, block_m)
    head_group = _pick_head_group(H)
    grid = (B * m_tiles, triton.cdiv(H, head_group))
    wrap_triton(_flash_attn_fwd_kernel)[grid](
        q,
        k,
        v,
        sm_scale,
        lse,
        out,
        B,
        H,
        M,
        N,
        WINDOW=window_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        M_TILES=m_tiles,
        HEAD_GROUP=head_group,
        D=D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, lse


def _flash_sdpa_bwd_impl(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backprop for Flash-style sliding-causal SDPA.

    Args:
      do: (B, M, H, D), grad wrt forward output.
      q/k/v: (B, M, H, D), saved forward inputs.
      out: (B, M, H, D), saved forward output.
      lse: (B, M, H), fp32 saved forward row log-sum-exp.
      window_size: total visible keys per query.

    Returns:
      dq, dk, dv with the same shapes/dtypes as q, k, v.
    """
    if not _HAS_TRITON:
        raise RuntimeError("flash_sdpa backward requires triton")
    do = do.contiguous()
    B, M, H, D = q.shape
    N = k.shape[1]
    sm_scale = D**-0.5
    block_m, block_n, num_warps, num_stages = _pick_sdpa_tile_config(D)
    head_group = _pick_head_group(H)

    # Delta[i] = sum_j out[i, j] * dO[i, j].
    delta = torch.empty((B, M, H), dtype=torch.float32, device=q.device)
    block_m_pre = block_m
    m_tiles_pre = triton.cdiv(M, block_m_pre)
    grid_pre = (B * m_tiles_pre, triton.cdiv(H, head_group))
    wrap_triton(_flash_attn_bwd_preprocess_kernel)[grid_pre](
        out,
        do,
        delta,
        B,
        H,
        M,
        BLOCK_M=block_m_pre,
        M_TILES=m_tiles_pre,
        HEAD_GROUP=head_group,
        D=D,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # dQ is one-writer per Q tile; dK/dV need atomic accumulation across
    # overlapping sliding-window Q tiles.
    dq = torch.empty_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    m_tiles = triton.cdiv(M, block_m)
    grid_bwd = (B * m_tiles, triton.cdiv(H, head_group))
    wrap_triton(_flash_attn_bwd_kernel)[grid_bwd](
        q,
        k,
        v,
        sm_scale,
        lse,
        delta,
        do,
        dq,
        dk,
        dv,
        B,
        H,
        M,
        N,
        WINDOW=window_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        M_TILES=m_tiles,
        HEAD_GROUP=head_group,
        D=D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return dq, dk, dv


@torch.library.triton_op(
    "nanoops::flash_sdpa_fwd",
    mutates_args=(),
)
def _flash_sdpa_fwd_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton-op forward wrapper returning `(out, lse)`."""
    return _flash_sdpa_fwd_impl(q, k, v, window_size)


@torch.library.triton_op(
    "nanoops::flash_sdpa_bwd",
    mutates_args=(),
)
def _flash_sdpa_bwd_op(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton-op backward wrapper returning `(dq, dk, dv)`."""
    return _flash_sdpa_bwd_impl(do, q, k, v, out, lse, window_size)


def _flash_sdpa_setup_context(
    ctx: Any,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    output: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Save tensors for `nanoops::flash_sdpa_fwd` backward."""
    q, k, v, window_size = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)
    ctx.window_size = window_size


def _flash_sdpa_autograd_backward(
    ctx: Any,
    do: torch.Tensor,
    _d_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    """Autograd callback for Flash-style SDPA."""
    q, k, v, out, lse = ctx.saved_tensors
    dq, dk, dv = _flash_sdpa_bwd_op(do, q, k, v, out, lse, ctx.window_size)
    return dq, dk, dv, None


_flash_sdpa_fwd_op.register_autograd(
    _flash_sdpa_autograd_backward,
    setup_context=_flash_sdpa_setup_context,
)


def flash_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Flash-style sliding-causal SDPA. q, k, v: (B, L, H, D), H_q == H_kv.

    window_size: total keys each query attends to (= nanchat's window+1).

    Args:
      q: (B, L, H, D) contiguous CUDA query tensor.
      k: (B, L, H, D) contiguous CUDA key tensor.
      v: (B, L, H, D) contiguous CUDA value tensor.
      window_size: total visible keys per query.

    Returns:
      (B, L, H, D) attention output.
    """
    out, _lse = _flash_sdpa_fwd_op(q, k, v, window_size)
    return out
