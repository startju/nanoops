"""Attention SDPA-side Triton kernels for nanoops.

Contains `flash_sdpa`: Flash-style sliding-causal SDPA with a chunked
backward. Re-exported through `nanoops.triton_kernels`.
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
        Q,  # (B, H, M, D) — in: query
        K,  # (B, H, N, D) — in: key
        V,  # (B, H, N, D) — in: value
        sm_scale,  # float — attention scale, usually D**-0.5
        LSE,  # (B, H, M) fp32 — out: row log-sum-exp for bwd
        O,  # (B, H, M, D) — out: attention output
        B,  # int — batch size
        H,  # int — number of attention heads
        M,  # int — query sequence length
        N,  # int — key/value sequence length
        WINDOW: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
    ):
        """Forward: one program per (batch × head × Q-tile-of-BLOCK_M-rows)."""
        pid_bh = tl.program_id(0)
        pid_m = tl.program_id(1)
        bid = pid_bh // H
        hid = pid_bh % H

        # v1 requires contiguous Q/K/V/O and H_q == H_kv, so offsets are
        # standard row-major. This keeps the kernel signature short; add
        # explicit strides back if we later support packed/non-contiguous layouts.
        bh_base = (bid * H + hid)
        q_off = bh_base * M * BLOCK_DMODEL
        k_off = bh_base * N * BLOCK_DMODEL
        v_off = bh_base * N * BLOCK_DMODEL
        o_off = bh_base * M * BLOCK_DMODEL
        lse_off = bh_base * M

        # Q tile: load BLOCK_M rows of Q for this program. Stays resident.
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        q_ptrs = Q + q_off + offs_m[:, None] * BLOCK_DMODEL + offs_d[None, :]
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
            k_ptrs = (
                K + k_off + offs_n[:, None] * BLOCK_DMODEL + offs_d[None, :]
            )
            v_ptrs = (
                V + v_off + offs_n[:, None] * BLOCK_DMODEL + offs_d[None, :]
            )
            k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
            v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

            # Scores = Q @ K^T * scale, shape (BLOCK_M, BLOCK_N)
            s = tl.dot(q, tl.trans(k_tile), input_precision="ieee") * sm_scale

            # Apply sliding+causal mask per cell:
            #   keep if  j ≤ i  AND  j ≥ i - W + 1
            j = offs_n[None, :]
            i = offs_m[:, None]
            mask_keep = (
                (j <= i) & (j >= i - WINDOW + 1) & m_mask[:, None] & n_mask[None, :]
            )
            s = tl.where(mask_keep, s, -float("inf"))

            # Online softmax update. CRITICAL: when an entire row's scores
            # were masked to -inf (sliding-window edge where this Q row's
            # window doesn't overlap this K tile at all), m_new stays at -inf
            # for that row. Then `exp(m_i - m_new) = exp(-inf - -inf) = exp(NaN)
            # = NaN`, which contaminates the accumulator forever. Guard with
            # tl.where so a "no valid keys this tile" row leaves m_i / l_i / acc
            # unchanged (alpha=1, contribution=0).
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            all_masked = m_new == -float("inf")
            alpha = tl.where(all_masked, 1.0, tl.exp(m_i - m_new))
            p_unscaled = tl.where(all_masked[:, None], 0.0, tl.exp(s - m_new[:, None]))
            l_i = l_i * alpha + tl.sum(p_unscaled, axis=1)
            acc = acc * alpha[:, None] + tl.dot(
                p_unscaled.to(v_tile.dtype),
                v_tile,
                input_precision="ieee",
            )
            m_i = m_new

        # Finalize: output / l, save LSE = m + log(l)
        acc = acc / l_i[:, None]
        lse = m_i + tl.log(l_i)

        o_ptrs = O + o_off + offs_m[:, None] * BLOCK_DMODEL + offs_d[None, :]
        tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=m_mask[:, None])

        lse_ptrs = LSE + lse_off + offs_m
        tl.store(lse_ptrs, lse, mask=m_mask)

    @triton.jit
    def _flash_attn_bwd_preprocess_kernel(
        O,  # (B, H, M, D) — in: forward output
        dO,  # (B, H, M, D) — in: gradient of output
        D,  # (B, H, M) fp32 — out: row dot(O, dO)
        B,  # int — batch size
        H,  # int — number of attention heads
        M,  # int — query sequence length
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
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

        bh_base = bid * H + hid
        o_off = bh_base * M * BLOCK_DMODEL
        o_ptrs = O + o_off + offs_m[:, None] * BLOCK_DMODEL + offs_d[None, :]
        do_ptrs = dO + o_off + offs_m[:, None] * BLOCK_DMODEL + offs_d[None, :]
        o = tl.load(o_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        d_row = tl.sum(o * do, axis=1)
        d_ptrs = D + bh_base * M + offs_m
        tl.store(d_ptrs, d_row, mask=m_mask)

    @triton.jit
    def _flash_attn_bwd_kernel(
        Q,  # (B, H, M, D) — in: query
        K,  # (B, H, N, D) — in: key
        V,  # (B, H, N, D) — in: value
        sm_scale,  # float — attention scale
        LSE,  # (B, H, M) fp32 — in: saved row log-sum-exp
        D,  # (B, H, M) fp32 — in: row dot(O, dO)
        dO,  # (B, H, M, D) — in: grad wrt output
        dQ,  # (B, H, M, D) — out: grad wrt query
        dK,  # (B, H, N, D) — out: grad wrt key, atomic accumulated
        dV,  # (B, H, N, D) — out: grad wrt value, atomic accumulated
        B,  # int — batch size
        H,  # int — number of attention heads
        M,  # int — query sequence length
        N,  # int — key/value sequence length
        WINDOW: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
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

        bh_base = bid * H + hid
        q_off = bh_base * M * BLOCK_DMODEL
        k_off = bh_base * N * BLOCK_DMODEL
        v_off = bh_base * N * BLOCK_DMODEL
        lse_off = bh_base * M

        # Load Q tile, dO tile, LSE, D for this Q range
        q_ptrs = Q + q_off + offs_m[:, None] * BLOCK_DMODEL + offs_d[None, :]
        do_ptrs = dO + q_off + offs_m[:, None] * BLOCK_DMODEL + offs_d[None, :]
        lse_ptrs = LSE + lse_off + offs_m
        d_ptrs = D + lse_off + offs_m
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

            k_ptrs = (
                K + k_off + offs_n[:, None] * BLOCK_DMODEL + offs_d[None, :]
            )
            v_ptrs = (
                V + v_off + offs_n[:, None] * BLOCK_DMODEL + offs_d[None, :]
            )
            k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
            v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

            # Recompute P[i, j] = exp(s[i, j] * sm_scale - LSE[i])
            s = tl.dot(q, tl.trans(k_tile), input_precision="ieee") * sm_scale
            j = offs_n[None, :]
            i = offs_m[:, None]
            mask_keep = (
                (j <= i) & (j >= i - WINDOW + 1) & m_mask[:, None] & n_mask[None, :]
            )
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
            dk_ptrs = (
                dK + k_off + offs_n[:, None] * BLOCK_DMODEL + offs_d[None, :]
            )
            dv_ptrs = (
                dV + v_off + offs_n[:, None] * BLOCK_DMODEL + offs_d[None, :]
            )
            tl.atomic_add(dk_ptrs, dk.to(dK.dtype.element_ty), mask=n_mask[:, None])
            tl.atomic_add(dv_ptrs, dv.to(dV.dtype.element_ty), mask=n_mask[:, None])

        # Write dQ (single contributor per Q tile — no atomic needed)
        dq_ptrs = dQ + q_off + offs_m[:, None] * BLOCK_DMODEL + offs_d[None, :]
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
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        """Run sliding-causal scaled dot-product attention.

        Args:
          q: (B, H, M, D) contiguous CUDA query tensor.
          k: (B, H, N, D) contiguous CUDA key tensor; v1 requires N == M.
          v: (B, H, N, D) contiguous CUDA value tensor.
          window_size: number of keys visible to each query.

        Returns:
          (B, H, M, D) attention output."""
        assert q.is_cuda and k.is_cuda and v.is_cuda
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert q.shape == k.shape == v.shape, (
            f"v1 requires H_q == H_kv and same L: q{q.shape} k{k.shape} v{v.shape}"
        )
        B, H, M, D = q.shape
        N = k.shape[2]
        sm_scale = D**-0.5

        o = torch.empty_like(q)
        lse = torch.empty((B, H, M), dtype=torch.float32, device=q.device)

        BLOCK_M, BLOCK_N = 64, 64
        grid = (B * H, triton.cdiv(M, BLOCK_M))
        _flash_attn_fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            lse,
            o,
            B,
            H,
            M,
            N,
            WINDOW=window_size,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=D,
        )

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = sm_scale
        ctx.window_size = window_size
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        return o

    @staticmethod
    def backward(
        ctx: Any,
        do: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Backprop for sliding-causal attention.

        Args:
          do: (B, H, M, D) gradient of attention output.

        Returns:
          Gradients for (q, k, v, window_size)."""
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
            o,
            do,
            d,
            B,
            H,
            M,
            BLOCK_M=BLOCK_M_PRE,
            BLOCK_DMODEL=D,
        )

        # dQ, dK, dV allocated as zero (dK, dV need accumulation via atomic).
        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        grid_bwd = (B * H, triton.cdiv(M, BLOCK_M))
        _flash_attn_bwd_kernel[grid_bwd](
            q,
            k,
            v,
            sm_scale,
            lse,
            d,
            do,
            dq,
            dk,
            dv,
            B,
            H,
            M,
            N,
            WINDOW=window_size,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=D,
        )

        return dq, dk, dv, None


def flash_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Flash-style sliding-causal SDPA. q, k, v: (B, H, L, D), H_q == H_kv.

    window_size: total keys each query attends to (= nanchat's window+1).

    Args:
      q: (B, H, L, D) contiguous CUDA query tensor.
      k: (B, H, L, D) contiguous CUDA key tensor.
      v: (B, H, L, D) contiguous CUDA value tensor.
      window_size: total visible keys per query.

    Returns:
      (B, H, L, D) attention output.
    """
    return FlashSDPA.apply(q, k, v, window_size)
