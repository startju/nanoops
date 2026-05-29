"""Attention QKV-side Triton kernels for nanoops.

Contains:
  - `norm_qkv_projection`: fused outer RMSNorm + Q/K/V projection;
    Q/K are immediately rotary-embedded, RMS-normalized, and scaled before
    being written. V can optionally add a gated value-embedding lookup.
    Backward is also implemented with Triton kernels, including the optional
    value-embedding gate gradients.
Re-exported through `nanoops.triton_kernels`.
"""

from __future__ import annotations

from typing import Any

import torch

from .triton_fused_add_norm import _pick_tile_config

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
    from .triton_fused_add_norm import _fused_add_norm_fwd_kernel
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
# This path materializes the OUTER RMSNorm output once, then shares it
# across the Q/K/V projection kernels. Q/K immediately apply rotary +
# QK RMSNorm + scale before being written.
# ─────────────────────────────────────────────────────────────────────


if _HAS_TRITON:

    @triton.jit
    def _qkv_projection_fwd_kernel(
        x_ptr,  # (M, K), dtype=x.dtype — in: materialized outer RMSNorm output
        q_w_ptr,  # (n_head * D, K), dtype=q_weight.dtype — in: Q projection weight
        k_w_ptr,  # (n_kv_head * D, K), dtype=k_weight.dtype — in: K projection weight
        v_w_ptr,  # (n_kv_head * D, K), dtype=v_weight.dtype — in: V projection weight
        ve_ids_ptr,  # (M,), int64 — optional in: token ids for value embedding lookup
        ve_w_ptr,  # (vocab, n_kv_head * D), dtype=ve_weight.dtype — optional VE table
        ve_gate_w_ptr,  # (n_kv_head, VE_GATE_CH), dtype=ve_gate_weight.dtype — optional gate
        cos_ptr,  # (1, T, 1, D/2), dtype=cos.dtype — in: rotary cosine table
        sin_ptr,  # (1, T, 1, D/2), dtype=sin.dtype — in: rotary sine table
        q_ptr,  # (M, n_head, D), dtype=x.dtype — out: Q after rotary + QK norm + scale
        k_ptr,  # (M, n_kv_head, D), dtype=x.dtype — out: K after rotary + QK norm + scale
        v_ptr,  # (M, n_kv_head, D), dtype=x.dtype — out: projected V plus optional VE
        qk_rms_inv_ptr,  # (M, n_head + n_kv_head), fp32 — out: per Q/K-head RMS inverse
        M,  # int — row count after flattening batch/time
        K,  # int — input hidden width
        D,  # int — per-head width
        n_head,  # int — number of Q heads
        n_kv_head,  # int — number of K/V heads
        rotary_seq_len,  # int — rotary table sequence length before batch broadcast
        eps,  # float — outer and QK RMSNorm epsilon
        scale,  # float — post-QK-norm scalar multiplier
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_VALUE_EMBEDDING: tl.constexpr,
        VE_GATE_CH: tl.constexpr,
        VE_GATE_BLOCK: tl.constexpr,
    ):
        """Project one Q/K/V head tile selected by the second grid axis.

        A part is one projected head: first Q heads, then K heads, then V
        heads. The input is already outer-RMS-normalized; V exits before
        rotary and QK RMSNorm, after optionally adding the ResFormer value
        embedding gate.
        """
        pid_m = tl.program_id(0)
        part = tl.program_id(1)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_D)
        half_cols = tl.arange(0, BLOCK_D // 2)
        row_mask = rows < M

        is_q = part < n_head
        is_k = (part >= n_head) & (part < n_head + n_kv_head)
        is_v = part >= n_head + n_kv_head
        head = tl.where(
            is_q,
            part,
            tl.where(is_k, part - n_head, part - n_head - n_kv_head),
        )
        weight_rows = head * D + cols

        # Project one full head with a single dot in ordinary column order:
        # [lo0, lo1, ..., hi0, hi1, ...]. Q/K split the accumulator into
        # rotary halves in registers; V can store the same order directly.
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            ks = k_start + tl.arange(0, BLOCK_K)
            k_mask = ks < K
            x_ptrs = x_ptr + rows[:, None] * K + ks[None, :]
            x = tl.load(
                x_ptrs,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            if is_q:
                w_ptrs = q_w_ptr + weight_rows[:, None] * K + ks[None, :]
            else:
                if is_k:
                    w_ptrs = k_w_ptr + weight_rows[:, None] * K + ks[None, :]
                else:
                    w_ptrs = v_w_ptr + weight_rows[:, None] * K + ks[None, :]
            w = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0).to(x_ptr.dtype.element_ty)
            acc += tl.dot(x, tl.trans(w))

        if is_v:
            v_out = acc.to(x_ptr.dtype.element_ty)
            if HAS_VALUE_EMBEDDING:
                gate_cols = tl.arange(0, VE_GATE_BLOCK)
                gate_mask = gate_cols < VE_GATE_CH
                gate_x = tl.load(
                    x_ptr + rows[:, None] * K + gate_cols[None, :],
                    mask=row_mask[:, None] & gate_mask[None, :],
                    other=0.0,
                )
                gate_w = tl.load(
                    ve_gate_w_ptr + head * VE_GATE_CH + gate_cols,
                    mask=gate_mask,
                    other=0.0,
                ).to(x_ptr.dtype.element_ty)
                gate_logits = tl.sum(gate_x * gate_w[None, :], axis=1, dtype=tl.float32)
                gate = 3 * tl.sigmoid(gate_logits).to(x_ptr.dtype.element_ty)
                token_ids = tl.load(ve_ids_ptr + rows, mask=row_mask, other=0)
                ve_ptrs = (
                    ve_w_ptr
                    + token_ids[:, None] * n_kv_head * D
                    + head * D
                    + cols[None, :]
                )
                ve = tl.load(
                    ve_ptrs,
                    mask=row_mask[:, None],
                    other=0.0,
                )
                v_out += gate[:, None] * ve
            v_ptrs = (
                v_ptr
                + rows[:, None] * n_kv_head * D
                + head * D
                + cols[None, :]
            )
            tl.store(
                v_ptrs,
                v_out,
                mask=row_mask[:, None],
            )
            return

        acc_halves = tl.reshape(acc.to(x_ptr.dtype.element_ty), (BLOCK_M, 2, BLOCK_D // 2))
        acc_lo, acc_hi = tl.split(tl.trans(acc_halves, 0, 2, 1))
        rotary_rows = rows % rotary_seq_len

        cos = tl.load(
            cos_ptr + rotary_rows[:, None] * (BLOCK_D // 2) + half_cols[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        sin = tl.load(
            sin_ptr + rotary_rows[:, None] * (BLOCK_D // 2) + half_cols[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        rot_lo = acc_lo * cos + acc_hi * sin
        rot_hi = -acc_lo * sin + acc_hi * cos
        rot_pair = tl.join(rot_lo, rot_hi)
        qk_halves = tl.trans(rot_pair, 0, 2, 1)
        qk = tl.reshape(qk_halves, (BLOCK_M, BLOCK_D))
        qk_sum_sq = tl.sum(qk * qk, axis=1, dtype=tl.float32)
        qk_rms_inv = tl.rsqrt(qk_sum_sq / D + eps)
        norm_scale = qk_rms_inv * scale
        qk = qk * norm_scale[:, None]

        if is_q:
            q_ptrs = (
                q_ptr
                + rows[:, None] * n_head * D
                + head * D
                + cols[None, :]
            )
            tl.store(
                qk_rms_inv_ptr + rows * (n_head + n_kv_head) + head,
                qk_rms_inv,
                mask=row_mask,
            )
            tl.store(q_ptrs, qk.to(q_ptr.dtype.element_ty), mask=row_mask[:, None])
        else:
            k_ptrs = (
                k_ptr
                + rows[:, None] * n_kv_head * D
                + head * D
                + cols[None, :]
            )
            tl.store(
                qk_rms_inv_ptr + rows * (n_head + n_kv_head) + n_head + head,
                qk_rms_inv,
                mask=row_mask,
            )
            tl.store(k_ptrs, qk.to(k_ptr.dtype.element_ty), mask=row_mask[:, None])

    @triton.jit
    def _qkv_dx_hat_outer_rms_row_inner_ve_bwd_kernel(
        q_ptr,  # (M, n_head, D), dtype=x.dtype — in: final Q output
        k_ptr,  # (M, n_kv_head, D), dtype=x.dtype — in: final K output
        qk_rms_inv_ptr,  # (M, n_head + n_kv_head), fp32 — in
        cos_ptr,  # (1, rotary_seq_len, 1, D/2) — in
        sin_ptr,  # (1, rotary_seq_len, 1, D/2) — in
        d_q_ptr,  # (M, n_head, D) — in
        d_k_ptr,  # (M, n_kv_head, D) — in
        d_v_ptr,  # (M, n_kv_head, D) — in
        ve_ids_ptr,  # (M,), int64 — optional in: VE token ids
        ve_w_ptr,  # (vocab, n_kv_head * D) — optional in: VE table
        ve_gate_w_ptr,  # (n_kv_head, VE_GATE_CH) — optional in: gate weight
        q_w_ptr,  # (n_head * D, K) — in
        k_w_ptr,  # (n_kv_head * D, K) — in
        v_w_ptr,  # (n_kv_head * D, K) — in
        dx_hat_ptr,  # (M, K), dtype=x.dtype — out: materialized d_x_hat
        x_ptr,  # (M, K) — in: original forward input
        rms_inv_ptr,  # (M,) fp32 — in: outer RMSNorm inverse
        norm_w_ptr,  # (K,) — in when HAS_NORM_WEIGHT
        outer_rms_row_inner_ptr,  # (M,) fp32 — in/out: outer RMSNorm row-inner
        d_ve_w_ptr,  # (vocab, n_kv_head * D), dtype=ve_weight.dtype — optional out
        d_ve_gate_w_ptr,  # (n_kv_head, VE_GATE_CH), dtype=ve_gate_weight.dtype — optional out
        M,  # int
        K,  # int
        D,  # int
        rotary_seq_len,  # int
        scale,  # float
        inv_scale,  # float
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        N_HEAD: tl.constexpr,
        N_KV_HEAD: tl.constexpr,
        HAS_NORM_WEIGHT: tl.constexpr,
        HAS_VALUE_EMBEDDING: tl.constexpr,
        VE_GATE_CH: tl.constexpr,
        VE_GATE_BLOCK: tl.constexpr,
    ):
        """Compute projection d_x_hat, outer-RMSNorm inner, and optional VE bwd.

        This is the no-`dz` path for
            d_x_hat = d_q0 @ q_weight + d_k0 @ k_weight + d_v @ v_weight.
        Q/K slices are recovered from saved final Q/K plus saved QK RMS inverse.

        Grid: (ceil(M / BLOCK_M), ceil(K / BLOCK_K)).
          - writes one (M, K) d_x_hat tile materialized in x.dtype;
          - atomic-adds this K tile's outer RMSNorm row-inner contribution;
          - when VE is enabled, also computes value-embedding lookup/gate
            backward. VE table/gate-weight grads are only emitted by pid_k==0
            so K tiles do not duplicate those atomics.
        """
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        half_cols = tl.arange(0, BLOCK_D // 2)
        cols = tl.arange(0, BLOCK_D)
        row_mask = rows < M
        k_mask = ks < K
        rotary_rows = rows % rotary_seq_len

        dx_hat_tile = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        cos = tl.load(
            cos_ptr + rotary_rows[:, None] * (BLOCK_D // 2) + half_cols[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        sin = tl.load(
            sin_ptr + rotary_rows[:, None] * (BLOCK_D // 2) + half_cols[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )

        scale_for_qk = scale.to(x_ptr.dtype.element_ty)
        inv_scale_for_qk = inv_scale.to(x_ptr.dtype.element_ty)

        for head in range(0, N_HEAD):
            out_base = q_ptr + rows[:, None] * N_HEAD * D + head * D
            grad_base = d_q_ptr + rows[:, None] * N_HEAD * D + head * D

            y0 = tl.load(
                out_base + cols[None, :],
                mask=row_mask[:, None],
                other=0.0,
            ) * inv_scale_for_qk
            g = tl.load(
                grad_base + cols[None, :],
                mask=row_mask[:, None],
                other=0.0,
            ) * scale_for_qk
            qk_rms_inv = tl.load(
                qk_rms_inv_ptr + rows * (N_HEAD + N_KV_HEAD) + head,
                mask=row_mask,
                other=0.0,
            )
            qk_inner = tl.sum(g * y0, axis=1, dtype=tl.float32) / D
            d_rot = (
                qk_rms_inv[:, None] * (g - y0 * qk_inner[:, None])
            ).to(x_ptr.dtype.element_ty)
            d_rot_halves = tl.reshape(d_rot, (BLOCK_M, 2, BLOCK_D // 2))
            d_rot_lo, d_rot_hi = tl.split(tl.trans(d_rot_halves, 0, 2, 1))
            d_pre_lo = d_rot_lo * cos - d_rot_hi * sin
            d_pre_hi = d_rot_lo * sin + d_rot_hi * cos
            d_pre_halves = tl.trans(tl.join(d_pre_lo, d_pre_hi), 0, 2, 1)
            d_pre = tl.reshape(d_pre_halves, (BLOCK_M, BLOCK_D))
            w = tl.load(
                q_w_ptr + (head * D + cols)[:, None] * K + ks[None, :],
                mask=k_mask[None, :],
                other=0.0,
            ).to(x_ptr.dtype.element_ty)
            dx_hat_tile += tl.dot(d_pre, w)

        for head in range(0, N_KV_HEAD):
            out_base = k_ptr + rows[:, None] * N_KV_HEAD * D + head * D
            grad_base = d_k_ptr + rows[:, None] * N_KV_HEAD * D + head * D

            y0 = tl.load(
                out_base + cols[None, :],
                mask=row_mask[:, None],
                other=0.0,
            ) * inv_scale_for_qk
            g = tl.load(
                grad_base + cols[None, :],
                mask=row_mask[:, None],
                other=0.0,
            ) * scale_for_qk
            qk_rms_inv = tl.load(
                qk_rms_inv_ptr + rows * (N_HEAD + N_KV_HEAD) + N_HEAD + head,
                mask=row_mask,
                other=0.0,
            )
            qk_inner = tl.sum(g * y0, axis=1, dtype=tl.float32) / D
            d_rot = (
                qk_rms_inv[:, None] * (g - y0 * qk_inner[:, None])
            ).to(x_ptr.dtype.element_ty)
            d_rot_halves = tl.reshape(d_rot, (BLOCK_M, 2, BLOCK_D // 2))
            d_rot_lo, d_rot_hi = tl.split(tl.trans(d_rot_halves, 0, 2, 1))
            d_pre_lo = d_rot_lo * cos - d_rot_hi * sin
            d_pre_hi = d_rot_lo * sin + d_rot_hi * cos
            d_pre_halves = tl.trans(tl.join(d_pre_lo, d_pre_hi), 0, 2, 1)
            d_pre = tl.reshape(d_pre_halves, (BLOCK_M, BLOCK_D))
            w = tl.load(
                k_w_ptr + (head * D + cols)[:, None] * K + ks[None, :],
                mask=k_mask[None, :],
                other=0.0,
            ).to(x_ptr.dtype.element_ty)
            dx_hat_tile += tl.dot(d_pre, w)

        x = tl.load(
            x_ptr + rows[:, None] * K + ks[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        x_rms_inv = tl.load(rms_inv_ptr + rows, mask=row_mask, other=0.0)
        x_norm = (x * x_rms_inv[:, None]).to(x_ptr.dtype.element_ty)

        if HAS_VALUE_EMBEDDING:
            if pid_k * BLOCK_K < VE_GATE_CH:
                gate_cols = tl.arange(0, VE_GATE_BLOCK)
                gate_mask = gate_cols < VE_GATE_CH
                tile_gate_mask = ks < VE_GATE_CH
                token_ids = tl.load(ve_ids_ptr + rows, mask=row_mask, other=0)
                x_gate = tl.load(
                    x_ptr + rows[:, None] * K + gate_cols[None, :],
                    mask=row_mask[:, None] & gate_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                x_norm_gate = x_gate * x_rms_inv[:, None]
                if HAS_NORM_WEIGHT:
                    norm_w_gate = tl.load(
                        norm_w_ptr + gate_cols,
                        mask=gate_mask,
                        other=0.0,
                    ).to(tl.float32)
                    x_hat_gate = (x_norm_gate * norm_w_gate[None, :]).to(
                        x_ptr.dtype.element_ty
                    )
                else:
                    x_hat_gate = x_norm_gate.to(x_ptr.dtype.element_ty)

                for head in range(0, N_KV_HEAD):
                    d_v = tl.load(
                        d_v_ptr
                        + rows[:, None] * N_KV_HEAD * D
                        + head * D
                        + cols[None, :],
                        mask=row_mask[:, None],
                        other=0.0,
                    ).to(x_ptr.dtype.element_ty)
                    w = tl.load(
                        v_w_ptr + (head * D + cols)[:, None] * K + ks[None, :],
                        mask=k_mask[None, :],
                        other=0.0,
                    ).to(x_ptr.dtype.element_ty)
                    dx_hat_tile += tl.dot(d_v, w)

                    gate_w = tl.load(
                        ve_gate_w_ptr + head * VE_GATE_CH + gate_cols,
                        mask=gate_mask,
                        other=0.0,
                    ).to(x_ptr.dtype.element_ty)
                    gate_logits = tl.sum(
                        x_hat_gate * gate_w[None, :],
                        axis=1,
                        dtype=tl.float32,
                    )
                    sigmoid = tl.sigmoid(gate_logits)
                    gate = 3 * sigmoid
                    ve = tl.load(
                        ve_w_ptr
                        + token_ids[:, None] * N_KV_HEAD * D
                        + head * D
                        + cols[None, :],
                        mask=row_mask[:, None],
                        other=0.0,
                    ).to(tl.float32)

                    d_gate = tl.sum(d_v * ve, axis=1)
                    d_gate_logits = 3.0 * d_gate * sigmoid * (1.0 - sigmoid)

                    if pid_k == 0:
                        d_ve_weight_tile = d_v * gate[:, None]
                        tl.atomic_add(
                            d_ve_w_ptr
                            + token_ids[:, None] * N_KV_HEAD * D
                            + head * D
                            + cols[None, :],
                            d_ve_weight_tile,
                            sem="relaxed",
                            mask=row_mask[:, None],
                        )

                        d_ve_gate_weight_tile = tl.sum(
                            d_gate_logits[:, None] * x_hat_gate.to(tl.float32),
                            axis=0,
                        )
                        tl.atomic_add(
                            d_ve_gate_w_ptr + head * VE_GATE_CH + gate_cols,
                            d_ve_gate_weight_tile,
                            sem="relaxed",
                            mask=gate_mask,
                        )

                    gate_w_tile = tl.load(
                        ve_gate_w_ptr + head * VE_GATE_CH + ks,
                        mask=tile_gate_mask,
                        other=0.0,
                    ).to(tl.float32)
                    d_x_hat_ve_tile = d_gate_logits[:, None] * gate_w_tile[None, :]
                    dx_hat_tile += tl.where(
                        tile_gate_mask[None, :],
                        d_x_hat_ve_tile,
                        0.0,
                    )
            else:
                for head in range(0, N_KV_HEAD):
                    d_v = tl.load(
                        d_v_ptr
                        + rows[:, None] * N_KV_HEAD * D
                        + head * D
                        + cols[None, :],
                        mask=row_mask[:, None],
                        other=0.0,
                    ).to(x_ptr.dtype.element_ty)
                    w = tl.load(
                        v_w_ptr + (head * D + cols)[:, None] * K + ks[None, :],
                        mask=k_mask[None, :],
                        other=0.0,
                    ).to(x_ptr.dtype.element_ty)
                    dx_hat_tile += tl.dot(d_v, w)
        else:
            for head in range(0, N_KV_HEAD):
                d_v = tl.load(
                    d_v_ptr + rows[:, None] * N_KV_HEAD * D + head * D + cols[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                ).to(x_ptr.dtype.element_ty)
                w = tl.load(
                    v_w_ptr + (head * D + cols)[:, None] * K + ks[None, :],
                    mask=k_mask[None, :],
                    other=0.0,
                ).to(x_ptr.dtype.element_ty)
                dx_hat_tile += tl.dot(d_v, w)

        dx_hat_store = dx_hat_tile.to(dx_hat_ptr.dtype.element_ty)
        tl.store(
            dx_hat_ptr + rows[:, None] * K + ks[None, :],
            dx_hat_store,
            mask=row_mask[:, None] & k_mask[None, :],
        )
        if HAS_NORM_WEIGHT:
            nw = tl.load(norm_w_ptr + ks, mask=k_mask, other=0.0)
            d_x_norm_tile = dx_hat_store * nw[None, :]
        else:
            d_x_norm_tile = dx_hat_store
        outer_rms_row_inner_partial = (
            tl.sum(d_x_norm_tile * x_norm, axis=1, dtype=tl.float32) / K
        )
        tl.atomic_add(
            outer_rms_row_inner_ptr + rows,
            outer_rms_row_inner_partial,
            sem="relaxed",
            mask=row_mask,
        )

    @triton.jit
    def _qkv_weight_grad_bwd_kernel(
        q_ptr,  # (M, n_head, D), dtype=x.dtype — in: final Q output
        k_ptr,  # (M, n_kv_head, D), dtype=x.dtype — in: final K output
        qk_rms_inv_ptr,  # (M, n_head + n_kv_head), fp32 — in
        cos_ptr,  # (1, rotary_seq_len, 1, D/2) — in
        sin_ptr,  # (1, rotary_seq_len, 1, D/2) — in
        d_q_ptr,  # (M, n_head, D) — in
        d_k_ptr,  # (M, n_kv_head, D) — in
        d_v_ptr,  # (M, n_kv_head, D) — in
        x_ptr,  # (M, K) — in
        rms_inv_ptr,  # (M,) fp32 — in
        norm_w_ptr,  # (K,) — in when HAS_NORM_WEIGHT
        d_q_w_ptr,  # (n_head * D, K) — out
        d_k_w_ptr,  # (n_kv_head * D, K) — out
        d_v_w_ptr,  # (n_kv_head * D, K) — out
        M,  # int
        K,  # int
        D,  # int
        n_head,  # int
        n_kv_head,  # int
        rotary_seq_len,  # int
        scale,  # float
        inv_scale,  # float
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_NORM_WEIGHT: tl.constexpr,
    ):
        """Compute d_q_weight/d_k_weight/d_v_weight without a d_z buffer.

        Grid: (qkv_part, ceil(D / BLOCK_N), ceil(K / BLOCK_K)).
        `qkv_part` enumerates Q heads, then K heads, then V heads. Q/K parts
        recover d_q0/d_k0 from saved final Q/K and saved qk_rms_inv; V parts
        use d_v directly.
        """
        part = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_k = tl.program_id(2)
        ns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        full_cols = tl.arange(0, BLOCK_D)
        k_mask = ks < K
        n_mask = ns < D
        half = BLOCK_D // 2

        is_q = part < n_head
        is_k = (part >= n_head) & (part < n_head + n_kv_head)
        is_v = part >= n_head + n_kv_head
        head = tl.where(
            is_q,
            part,
            tl.where(is_k, part - n_head, part - n_head - n_kv_head),
        )

        is_lo = ns < half
        pair_ns = tl.where(is_lo, ns + half, ns - half)
        rotary_cols = ns % half

        d_weight_tile = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for m_start in range(0, M, BLOCK_M):
            rows = m_start + tl.arange(0, BLOCK_M)
            row_mask = rows < M
            x_rms_inv = tl.load(rms_inv_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
            x = tl.load(
                x_ptr + rows[:, None] * K + ks[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if HAS_NORM_WEIGHT:
                nw = tl.load(norm_w_ptr + ks, mask=k_mask, other=0.0).to(tl.float32)
                x_hat = (x * x_rms_inv[:, None] * nw[None, :]).to(x_ptr.dtype.element_ty).to(tl.float32)
            else:
                x_hat = (x * x_rms_inv[:, None]).to(x_ptr.dtype.element_ty).to(tl.float32)

            if is_v:
                d_part = tl.load(
                    d_v_ptr + rows[:, None] * n_kv_head * D + head * D + ns[None, :],
                    mask=row_mask[:, None] & n_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
            else:
                if is_q:
                    out_base = q_ptr + rows[:, None] * n_head * D + head * D
                    grad_base = d_q_ptr + rows[:, None] * n_head * D + head * D
                    inv_offset = head
                else:
                    out_base = k_ptr + rows[:, None] * n_kv_head * D + head * D
                    grad_base = d_k_ptr + rows[:, None] * n_kv_head * D + head * D
                    inv_offset = n_head + head

                y0_all = tl.load(
                    out_base + full_cols[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                ).to(tl.float32) * inv_scale
                g_all = tl.load(
                    grad_base + full_cols[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                ).to(tl.float32) * scale
                qk_rms_inv = tl.load(
                    qk_rms_inv_ptr + rows * (n_head + n_kv_head) + inv_offset,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                qk_inner = tl.sum(g_all * y0_all, axis=1) / D

                y0_col = tl.load(
                    out_base + ns[None, :],
                    mask=row_mask[:, None] & n_mask[None, :],
                    other=0.0,
                ).to(tl.float32) * inv_scale
                g_col = tl.load(
                    grad_base + ns[None, :],
                    mask=row_mask[:, None] & n_mask[None, :],
                    other=0.0,
                ).to(tl.float32) * scale
                y0_pair = tl.load(
                    out_base + pair_ns[None, :],
                    mask=row_mask[:, None] & n_mask[None, :],
                    other=0.0,
                ).to(tl.float32) * inv_scale
                g_pair = tl.load(
                    grad_base + pair_ns[None, :],
                    mask=row_mask[:, None] & n_mask[None, :],
                    other=0.0,
                ).to(tl.float32) * scale

                d_rot_col = qk_rms_inv[:, None] * (g_col - y0_col * qk_inner[:, None])
                d_rot_pair = qk_rms_inv[:, None] * (
                    g_pair - y0_pair * qk_inner[:, None]
                )
                rotary_rows = rows % rotary_seq_len
                cos = tl.load(
                    cos_ptr + rotary_rows[:, None] * half + rotary_cols[None, :],
                    mask=row_mask[:, None] & n_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                sin = tl.load(
                    sin_ptr + rotary_rows[:, None] * half + rotary_cols[None, :],
                    mask=row_mask[:, None] & n_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                d_lo_col = d_rot_col * cos - d_rot_pair * sin
                d_hi_col = d_rot_pair * sin + d_rot_col * cos
                d_part = tl.where(is_lo[None, :], d_lo_col, d_hi_col)

            d_weight_tile += tl.dot(tl.trans(d_part), x_hat)

        if is_q:
            tl.store(
                d_q_w_ptr + (head * D + ns)[:, None] * K + ks[None, :],
                d_weight_tile.to(d_q_w_ptr.dtype.element_ty),
                mask=n_mask[:, None] & k_mask[None, :],
            )
        else:
            if is_k:
                tl.store(
                    d_k_w_ptr + (head * D + ns)[:, None] * K + ks[None, :],
                    d_weight_tile.to(d_k_w_ptr.dtype.element_ty),
                    mask=n_mask[:, None] & k_mask[None, :],
                )
            else:
                tl.store(
                    d_v_w_ptr + (head * D + ns)[:, None] * K + ks[None, :],
                    d_weight_tile.to(d_v_w_ptr.dtype.element_ty),
                    mask=n_mask[:, None] & k_mask[None, :],
                )

    @triton.jit
    def _outer_rms_dx_from_dx_hat_bwd_kernel(
        x_ptr,  # (M, K) — in
        rms_inv_ptr,  # (M,) fp32 — in
        norm_w_ptr,  # (K,) — in
        dx_hat_ptr,  # (M, K), dtype=x.dtype — in: materialized d_x_hat
        outer_rms_row_inner_ptr,  # (M,) fp32 — in
        dx_ptr,  # (M, K) — out
        M,  # int
        K,  # int
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_NORM_WEIGHT: tl.constexpr,
    ):
        """Compute d_x from d_x_hat and pre-accumulated RMSNorm row inner."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = rows < M
        k_mask = ks < K

        x_rms_inv = tl.load(rms_inv_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        outer_rms_row_inner = tl.load(
            outer_rms_row_inner_ptr + rows,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        x = tl.load(
            x_ptr + rows[:, None] * K + ks[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        dx_hat = tl.load(
            dx_hat_ptr + rows[:, None] * K + ks[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        x_norm = x * x_rms_inv[:, None]
        if HAS_NORM_WEIGHT:
            nw = tl.load(norm_w_ptr + ks, mask=k_mask, other=0.0).to(tl.float32)
            d_x_norm = dx_hat * nw[None, :]
        else:
            d_x_norm = dx_hat
        dx = x_rms_inv[:, None] * (
            d_x_norm - x_norm * outer_rms_row_inner[:, None]
        )
        tl.store(
            dx_ptr + rows[:, None] * K + ks[None, :],
            dx.to(dx_ptr.dtype.element_ty),
            mask=row_mask[:, None] & k_mask[None, :],
        )

    @triton.jit
    def _outer_rms_norm_weight_grad_bwd_kernel(
        x_ptr,  # (M, K) — in
        rms_inv_ptr,  # (M,) fp32 — in
        dx_hat_ptr,  # (M, K), dtype=x.dtype — in
        d_norm_w_ptr,  # (K,) — out
        M,  # int
        K,  # int
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Compute d_norm_weight = sum_m d_x_hat * x_norm.

        nanochat usually runs without `norm_weight`; this kernel is for the
        optional affine norm path and is not optimized as aggressively.
        """
        pid_k = tl.program_id(0)
        ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = ks < K

        d_norm_weight_tile = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for m_start in range(0, M, BLOCK_M):
            rows = m_start + tl.arange(0, BLOCK_M)
            row_mask = rows < M
            x_rms_inv = tl.load(rms_inv_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
            x = tl.load(
                x_ptr + rows[:, None] * K + ks[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            dx_hat = tl.load(
                dx_hat_ptr + rows[:, None] * K + ks[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            d_norm_weight_tile += tl.sum(dx_hat * x * x_rms_inv[:, None], axis=0)

        tl.store(
            d_norm_w_ptr + ks,
            d_norm_weight_tile.to(d_norm_w_ptr.dtype.element_ty),
            mask=k_mask,
        )

def _validate_rotary_table_4d(
    table: torch.Tensor,
    T: int,
    head_dim: int,
) -> None:
    """Validate a 4D broadcast rotary table."""
    half = head_dim // 2
    assert table.is_cuda and table.is_contiguous()
    assert table.ndim == 4, (
        f"rotary table must be 4D (1, T, 1, D/2), got {table.ndim}D"
    )
    assert table.shape[0] == 1 and table.shape[2] == 1
    assert T > 0
    assert table.shape[1] == T, (
        f"rotary table sequence length must be {T}, got {table.shape[1]}"
    )
    assert table.shape[-1] == half, (
        f"rotary table last dim must be {half}, got {table.shape[-1]}"
    )


def _norm_qkv_projection_bwd_impl(
    d_q: torch.Tensor,
    d_k: torch.Tensor,
    d_v: torch.Tensor,
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    ve_ids: torch.Tensor | None,
    ve_weight: torch.Tensor | None,
    ve_gate_channels: int,
    ve_gate_weight: torch.Tensor | None,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_inv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    qk_rms_inv: torch.Tensor,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    scale: float,
    eps: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Backward implementation for `nanoops::norm_qkv_projection_bwd`.

    This function launches Triton kernels for the fp32 algebra used by the
    custom-op autograd callback. Rotary cos/sin are treated as constants.
    Q/K RMSNorm backward uses saved `q`/`k`/`qk_rms_inv` from the forward
    custom op, recovering `y0 = q_or_k / scale` without recomputing Q/K
    projection outputs. Optional VE gradients are None when VE is disabled.
    Tensor inputs are expected to be CUDA contiguous tensors.

    Args:
      d_q: (B, T, n_head, head_dim), CUDA, grad of q output.
      d_k: (B, T, n_kv_head, head_dim), CUDA, grad of k output.
      d_v: (B, T, n_kv_head, head_dim), CUDA, grad of v output.
      x: (B, T, K), dtype=x.dtype, original forward input.
      norm_weight: optional (K,), dtype=norm_weight.dtype, outer RMSNorm scale.
        nanochat normally passes None; this affine path is kept for compatibility
        and tests, not as the primary optimized path.
      ve_ids: optional (B, T), CUDA integer token ids.
      ve_weight: optional (vocab, n_kv_head * head_dim), value embedding table.
      ve_gate_channels: int, leading normalized input channels used by the VE gate.
      ve_gate_weight: optional (n_kv_head, ve_gate_channels), gate weight.
      q_weight: (n_head * head_dim, K), dtype=q_weight.dtype, Q projection weight.
      k_weight: (n_kv_head * head_dim, K), dtype=k_weight.dtype, K projection weight.
      v_weight: (n_kv_head * head_dim, K), dtype=v_weight.dtype, V projection weight.
      cos: (1, T, 1, head_dim/2), rotary cosine table.
      sin: (1, T, 1, head_dim/2), rotary sine table.
      rms_inv: (B*T,), fp32 outer RMSNorm inverse from forward.
      q: (B, T, n_head, head_dim), dtype=x.dtype, final Q output.
      k: (B, T, n_kv_head, head_dim), dtype=x.dtype, final K output.
      qk_rms_inv: (B, T, n_head + n_kv_head), fp32 Q/K RMS inverse.
      n_head: int, number of query heads.
      n_kv_head: int, number of key/value heads.
      head_dim: int, per-head width; must be even.
      scale: float, post-Q/K RMSNorm scalar multiplier.
      eps: float, kept to mirror the forward custom-op signature; saved
        inverses already include epsilon.

    Returns:
      dx: (B, T, K), dtype=x.dtype.
      d_norm_weight: optional (K,), dtype=norm_weight.dtype. Present only for
        the non-primary affine norm path.
      d_ve_weight: optional (vocab, n_kv_head * head_dim), dtype=ve_weight.dtype.
      d_ve_gate_weight: optional (n_kv_head, ve_gate_channels),
        dtype=ve_gate_weight.dtype.
      d_q_weight: (n_head * head_dim, K), dtype=q_weight.dtype.
      d_k_weight: (n_kv_head * head_dim, K), dtype=k_weight.dtype.
      d_v_weight: (n_kv_head * head_dim, K), dtype=v_weight.dtype.

    Forward math:
      Outer RMSNorm:
        s_x   = rsqrt(mean_k(x^2) + eps)
        x_n   = x * s_x
        x_hat = x_n * norm_weight, or x_n when norm_weight is None

      Projections:
        q0 = x_hat @ q_weight.T
        k0 = x_hat @ k_weight.T
        v  = x_hat @ v_weight.T
        z  = concat(q0, k0, v) conceptually; it is not materialized here.

      Optional value embedding:
        x_g       = x_hat[:, :ve_gate_channels]            # (M, ch)
        gate      = 3 * sigmoid(x_g @ ve_gate_weight.T)    # (M, n_kv_head)
        ve        = ve_weight[ve_ids].view(M, n_kv_head, head_dim)
        v        += gate[..., None] * ve

      Q/K rotary + QK RMSNorm + scale:
        lo, hi = split(q0 or k0)
        r_lo   = lo * cos + hi * sin
        r_hi   = -lo * sin + hi * cos
        r      = concat(r_lo, r_hi)
        s_qk   = rsqrt(mean_d(r^2) + eps)
        y0     = r * s_qk
        q or k = scale * y0

      Saved for custom-op backward:
        x, optional norm/VE tensors, q/k/v weights, and cos/sin are saved from
        the forward inputs. The forward outputs used only by backward are:
          rms_inv     = s_x                         # (M,), fp32
          q, k        = final scaled Q/K outputs    # x.dtype
          qk_rms_inv  = s_qk per Q/K head           # (M, n_head+n_kv_head)

        Not saved: x_hat, q0, k0, v, q/k rotary pre-norm intermediates, or dz.
        Backward recomputes x_hat from saved x and rms_inv, recovers Q/K
        pre-projection grads from saved final q/k plus qk_rms_inv, and computes
        d_z conceptually without materializing it.

    Backward math:
      Q/K RMSNorm backward receives g_y = d_q or d_k:
        y0 = q_or_k / scale
        g0 = g_y * scale
        dr = s_qk * (g0 - y0 * mean_d(g0 * y0))

      y0 comes from saved final q/k and s_qk comes from saved qk_rms_inv.

      Inverse rotary:
        d_lo = d_r_lo * cos - d_r_hi * sin
        d_hi = d_r_lo * sin + d_r_hi * cos
        d_q0 or d_k0 = concat(d_lo, d_hi)

      Projection backward:
        d_z     = concat(d_q0, d_k0, d_v) conceptually
        d_x_hat = d_q0 @ q_weight + d_k0 @ k_weight + d_v @ v_weight
        dW_q    = d_q0.T @ x_hat
        dW_k    = d_k0.T @ x_hat
        dW_v    = d_v.T @ x_hat

      In code, `d_z` is not stored. The dx_hat kernel recomputes each
      Q/K/V slice, materializes d_x_hat in x.dtype, and accumulates the
      outer RMSNorm inner contribution from that materialized value; one fused
      dW kernel recomputes the same slices and writes q/k/v weight gradients.

      Optional value-embedding backward, accumulated before outer RMSNorm
      backward because the gate also consumes x_hat:
        x_g                 = x_hat[:, :ve_gate_channels]
        ve                  = ve_weight[ve_ids].view(M, n_kv_head, head_dim)
        d_gate              = sum_d(d_v * ve)
        d_gate_logits       = 3 * d_gate * sigmoid * (1 - sigmoid)
        d_x_g               = d_gate_logits @ ve_gate_weight
        d_x_hat[:, :ve_gate_channels] += d_x_g
        d_ve_weight[token]  += d_v * gate
        d_ve_gate_weight    += d_gate_logits.T @ x_g

      Outer RMSNorm backward:
        if norm_weight is not None:
          d_x_n = d_x_hat * norm_weight
          d_norm_weight = sum_m(d_x_hat * x_n)
        else:
          d_x_n = d_x_hat
        d_x = s_x * (d_x_n - x_n * mean_k(d_x_n * x_n))
    """
    if not _HAS_TRITON:
        raise RuntimeError("norm_qkv_projection backward requires triton")
    assert x.is_cuda and x.is_contiguous() and x.ndim == 3
    assert q_weight.is_cuda and k_weight.is_cuda and v_weight.is_cuda
    B, T, K = x.shape
    M = B * T
    x_flat = x.view(M, K)
    has_norm_weight = norm_weight is not None
    norm_weight_or_x = norm_weight if has_norm_weight else x_flat
    assert norm_weight_or_x.is_cuda
    assert (
        norm_weight_or_x.is_contiguous()
        and q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and v_weight.is_contiguous()
    )
    assert head_dim % 2 == 0

    d_q = d_q.contiguous().reshape(M, n_head, head_dim)
    d_k = d_k.contiguous().reshape(M, n_kv_head, head_dim)
    d_v = d_v.contiguous().reshape(M, n_kv_head, head_dim)
    saved_q = q.view(M, n_head, head_dim)
    saved_k = k.view(M, n_kv_head, head_dim)
    saved_qk_rms_inv = qk_rms_inv.view(M, n_head + n_kv_head)
    q_n = n_head * head_dim
    kv_n = n_kv_head * head_dim
    assert q.shape == (B, T, n_head, head_dim)
    assert k.shape == (B, T, n_kv_head, head_dim)
    assert qk_rms_inv.shape == (B, T, n_head + n_kv_head)
    assert d_q.shape == (M, n_head, head_dim)
    assert d_k.shape == (M, n_kv_head, head_dim)
    assert d_v.shape == (M, n_kv_head, head_dim)
    assert q_weight.shape == (q_n, K)
    assert k_weight.shape == (kv_n, K)
    assert v_weight.shape == (kv_n, K)
    assert saved_q.is_cuda and saved_k.is_cuda and saved_qk_rms_inv.is_cuda
    assert saved_q.is_contiguous() and saved_k.is_contiguous()
    assert saved_qk_rms_inv.is_contiguous()
    assert saved_q.shape == (M, n_head, head_dim)
    assert saved_k.shape == (M, n_kv_head, head_dim)
    assert saved_qk_rms_inv.shape == (M, n_head + n_kv_head)
    assert saved_qk_rms_inv.dtype == torch.float32
    has_value_embedding = (
        ve_ids is not None or ve_weight is not None or ve_gate_weight is not None
    )
    if has_value_embedding:
        assert ve_ids is not None and ve_weight is not None and ve_gate_weight is not None
        assert ve_ids.is_cuda and ve_weight.is_cuda and ve_gate_weight.is_cuda
        assert ve_weight.is_contiguous() and ve_gate_weight.is_contiguous()
        assert ve_ids.numel() == M
        assert ve_weight.ndim == 2 and ve_weight.shape[1] == kv_n
        assert 0 < ve_gate_channels <= K
        assert ve_gate_weight.shape == (n_kv_head, ve_gate_channels)
    rotary_seq_len = cos.shape[1]
    _validate_rotary_table_4d(cos, rotary_seq_len, head_dim)
    _validate_rotary_table_4d(sin, rotary_seq_len, head_dim)
    assert M % rotary_seq_len == 0, (
        f"M={M} is not divisible by rotary T={rotary_seq_len}"
    )

    assert rms_inv.is_cuda and rms_inv.is_contiguous()
    assert rms_inv.shape == (M,) and rms_inv.dtype == torch.float32
    dx_hat = torch.empty_like(x_flat)
    outer_rms_row_inner = torch.zeros((M,), dtype=torch.float32, device=x.device)
    dx = torch.empty_like(x_flat)
    d_norm_weight = torch.empty_like(norm_weight) if has_norm_weight else None
    d_q_weight = torch.empty_like(q_weight)
    d_k_weight = torch.empty_like(k_weight)
    d_v_weight = torch.empty_like(v_weight)
    d_ve_weight = None
    d_ve_gate_weight = None

    if has_value_embedding:
        assert ve_ids is not None and ve_weight is not None and ve_gate_weight is not None
        ve_ids_for_kernel = ve_ids.reshape(M).contiguous()
        ve_weight_for_kernel = ve_weight
        ve_gate_weight_for_kernel = ve_gate_weight
        d_ve_weight = torch.zeros_like(ve_weight)
        d_ve_gate_weight = torch.zeros_like(ve_gate_weight)
        d_ve_weight_for_kernel = d_ve_weight
        d_ve_gate_weight_for_kernel = d_ve_gate_weight
    else:
        ve_ids_for_kernel = x_flat
        ve_weight_for_kernel = x_flat
        ve_gate_weight_for_kernel = x_flat
        d_ve_weight_for_kernel = x_flat
        d_ve_gate_weight_for_kernel = x_flat

    DX_HAT_BLOCK_M, DX_HAT_BLOCK_K = 32, 128
    OUTER_RMS_BLOCK_M, OUTER_RMS_BLOCK_K = 32, 32
    WEIGHT_GRAD_BLOCK_N, WEIGHT_GRAD_BLOCK_K, WEIGHT_GRAD_BLOCK_M = 32, 32, 32

    _qkv_dx_hat_outer_rms_row_inner_ve_bwd_kernel[
        (triton.cdiv(M, DX_HAT_BLOCK_M), triton.cdiv(K, DX_HAT_BLOCK_K))
    ](
        saved_q,
        saved_k,
        saved_qk_rms_inv,
        cos,
        sin,
        d_q,
        d_k,
        d_v,
        ve_ids_for_kernel,
        ve_weight_for_kernel,
        ve_gate_weight_for_kernel,
        q_weight,
        k_weight,
        v_weight,
        dx_hat,
        x_flat,
        rms_inv,
        norm_weight_or_x,
        outer_rms_row_inner,
        d_ve_weight_for_kernel,
        d_ve_gate_weight_for_kernel,
        M,
        K,
        head_dim,
        rotary_seq_len,
        scale,
        1.0 / scale,
        BLOCK_M=DX_HAT_BLOCK_M,
        BLOCK_K=DX_HAT_BLOCK_K,
        BLOCK_D=head_dim,
        N_HEAD=n_head,
        N_KV_HEAD=n_kv_head,
        HAS_NORM_WEIGHT=has_norm_weight,
        HAS_VALUE_EMBEDDING=has_value_embedding,
        VE_GATE_CH=ve_gate_channels,
        VE_GATE_BLOCK=triton.next_power_of_2(ve_gate_channels),
        num_warps=4,
        num_stages=1,
    )
    _outer_rms_dx_from_dx_hat_bwd_kernel[
        (triton.cdiv(M, OUTER_RMS_BLOCK_M), triton.cdiv(K, OUTER_RMS_BLOCK_K))
    ](
        x_flat,
        rms_inv,
        norm_weight_or_x,
        dx_hat,
        outer_rms_row_inner,
        dx,
        M,
        K,
        BLOCK_M=OUTER_RMS_BLOCK_M,
        BLOCK_K=OUTER_RMS_BLOCK_K,
        HAS_NORM_WEIGHT=has_norm_weight,
        num_warps=4,
    )
    if has_norm_weight:
        _outer_rms_norm_weight_grad_bwd_kernel[
            (triton.cdiv(K, OUTER_RMS_BLOCK_K),)
        ](
            x_flat,
            rms_inv,
            dx_hat,
            d_norm_weight,
            M,
            K,
            BLOCK_M=OUTER_RMS_BLOCK_M,
            BLOCK_K=OUTER_RMS_BLOCK_K,
            num_warps=4,
        )
    _qkv_weight_grad_bwd_kernel[
        (
            n_head + 2 * n_kv_head,
            triton.cdiv(head_dim, WEIGHT_GRAD_BLOCK_N),
            triton.cdiv(K, WEIGHT_GRAD_BLOCK_K),
        )
    ](
        saved_q,
        saved_k,
        saved_qk_rms_inv,
        cos,
        sin,
        d_q,
        d_k,
        d_v,
        x_flat,
        rms_inv,
        norm_weight_or_x,
        d_q_weight,
        d_k_weight,
        d_v_weight,
        M,
        K,
        head_dim,
        n_head,
        n_kv_head,
        rotary_seq_len,
        scale,
        1.0 / scale,
        BLOCK_N=WEIGHT_GRAD_BLOCK_N,
        BLOCK_K=WEIGHT_GRAD_BLOCK_K,
        BLOCK_M=WEIGHT_GRAD_BLOCK_M,
        BLOCK_D=head_dim,
        HAS_NORM_WEIGHT=has_norm_weight,
        num_warps=4,
        num_stages=3,
    )
    return (
        dx.reshape_as(x),
        d_norm_weight,
        d_ve_weight,
        d_ve_gate_weight,
        d_q_weight,
        d_k_weight,
        d_v_weight,
    )


def _norm_qkv_projection_fwd_impl(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    ve_ids: torch.Tensor | None,
    ve_weight: torch.Tensor | None,
    ve_gate_channels: int,
    ve_gate_weight: torch.Tensor | None,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    scale: float,
    eps: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Forward implementation for `nanoops::norm_qkv_projection_fwd`.

    Args:
      x: (B, T, K), CUDA, contiguous. Activation dtype is preserved in q/k/v.
      norm_weight: optional (K,), CUDA, contiguous RMSNorm scale. None means
        plain RMSNorm without affine weight. nanochat normally uses None, so
        the affine branch is a compatibility path rather than the main target
        for optimization.
      ve_ids: optional (B, T), CUDA integer token ids. Must be passed together
        with `ve_weight`, `ve_gate_channels`, and `ve_gate_weight`.
      ve_weight: optional (vocab, n_kv_head * head_dim), CUDA, contiguous value
        embedding table.
      ve_gate_channels: int, number of leading normalized input channels used
        for the value-embedding gate. Must satisfy 0 < ch <= K when VE is used.
      ve_gate_weight: optional (n_kv_head, ve_gate_channels), CUDA, contiguous
        value-embedding gate weight.
      q_weight: (n_head * head_dim, K), CUDA, contiguous query projection.
      k_weight: (n_kv_head * head_dim, K), CUDA, contiguous key projection.
      v_weight: (n_kv_head * head_dim, K), CUDA, contiguous value projection.
      cos: (1, T, 1, head_dim/2), CUDA, contiguous rotary cosine table.
      sin: (1, T, 1, head_dim/2), CUDA, contiguous rotary sine table.
      n_head: int, number of query heads.
      n_kv_head: int, number of key/value heads.
      head_dim: int, per-head width; must be even.
      scale: float, scalar applied after Q/K RMSNorm.
      eps: float, RMSNorm epsilon for both outer norm and Q/K norm.

    Returns:
      q: (B, T, n_head, head_dim), dtype=x.dtype.
      k: (B, T, n_kv_head, head_dim), dtype=x.dtype.
      v: (B, T, n_kv_head, head_dim), dtype=x.dtype.
      rms_inv: (B*T,), fp32 hidden output saved for backward.
      qk_rms_inv: (B, T, n_head + n_kv_head), fp32 hidden per-head Q/K RMS
        inverse saved for the reduced-precision backward fast path.
    """
    if not _HAS_TRITON:
        raise RuntimeError("norm_qkv_projection requires triton")
    assert x.is_cuda and x.ndim == 3 and x.is_contiguous()
    B, T, K = x.size()
    M = B * T
    x_2d = x.view(M, K)
    assert q_weight.is_cuda and k_weight.is_cuda and v_weight.is_cuda
    has_norm_weight = norm_weight is not None
    norm_weight_or_x = norm_weight if has_norm_weight else x_2d
    assert norm_weight_or_x.is_cuda
    assert (
        norm_weight_or_x.is_contiguous()
        and q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and v_weight.is_contiguous()
    )
    assert head_dim % 2 == 0
    q_n = n_head * head_dim
    kv_n = n_kv_head * head_dim
    assert q_weight.shape == (q_n, K), f"q_weight shape {tuple(q_weight.shape)} != {(q_n, K)}"
    assert k_weight.shape == (kv_n, K), f"k_weight shape {tuple(k_weight.shape)} != {(kv_n, K)}"
    assert v_weight.shape == (kv_n, K), f"v_weight shape {tuple(v_weight.shape)} != {(kv_n, K)}"
    _validate_rotary_table_4d(cos, T, head_dim)
    _validate_rotary_table_4d(sin, T, head_dim)

    has_value_embedding = ve_ids is not None or ve_weight is not None
    if has_value_embedding:
        assert ve_ids is not None and ve_weight is not None, (
            "ve_ids and ve_weight must be passed together"
        )
        assert ve_ids.is_cuda and ve_weight.is_cuda and ve_weight.is_contiguous()
        assert ve_ids.shape == (B, T), f"ve_ids shape {tuple(ve_ids.shape)} != {(B, T)}"
        assert ve_weight.ndim == 2 and ve_weight.shape[1] == kv_n, (
            f"ve_weight shape {tuple(ve_weight.shape)} must be (vocab, {kv_n})"
        )
        ve_ids_for_kernel = ve_ids.reshape(M).contiguous()
        ve_weight_for_kernel = ve_weight
        assert ve_gate_weight is not None, (
            "ve_gate_weight is required when value embedding is provided"
        )
        assert ve_gate_weight.is_cuda and ve_gate_weight.is_contiguous()
        assert 0 < ve_gate_channels <= K
        assert ve_gate_weight.shape == (n_kv_head, ve_gate_channels), (
            f"ve_gate_weight shape {tuple(ve_gate_weight.shape)} != "
            f"{(n_kv_head, ve_gate_channels)}"
        )
    else:
        assert ve_gate_weight is None, (
            "ve_gate_weight must be None when value embedding is disabled"
        )
        ve_ids_for_kernel = x_2d
        ve_weight_for_kernel = x_2d
        ve_gate_weight = x_2d
        ve_gate_channels = 1

    q = torch.empty((B, T, n_head, head_dim), dtype=x.dtype, device=x.device)
    k = torch.empty((B, T, n_kv_head, head_dim), dtype=x.dtype, device=x.device)
    v = torch.empty((B, T, n_kv_head, head_dim), dtype=x.dtype, device=x.device)
    qk_rms_inv = torch.empty(
        (B, T, n_head + n_kv_head),
        dtype=torch.float32,
        device=x.device,
    )
    if has_value_embedding:
        QKV_BLOCK_M, QKV_BLOCK_K, QKV_NUM_WARPS = 128, 32, 4
    else:
        QKV_BLOCK_M, QKV_BLOCK_K, QKV_NUM_WARPS = 64, 64, 4
    norm_block_d = triton.next_power_of_2(K)
    norm_cfg = _pick_tile_config(M, norm_block_d, n_live_tiles=2)
    x_hat = torch.empty_like(x_2d)
    rms_inv = torch.empty((M,), dtype=torch.float32, device=x.device)
    _fused_add_norm_fwd_kernel[(triton.cdiv(M, norm_cfg.block_m),)](
        x_2d,
        x_2d,
        norm_weight_or_x,
        x_hat,
        x_2d,
        rms_inv,
        M,
        K,
        eps,
        BLOCK_M=norm_cfg.block_m,
        BLOCK_D=norm_block_d,
        HAS_NW=has_norm_weight,
        HAS_RESIDUAL=False,
        num_warps=norm_cfg.num_warps,
    )

    grid = (triton.cdiv(M, QKV_BLOCK_M), n_head + 2 * n_kv_head)
    _qkv_projection_fwd_kernel[grid](
        x_hat,
        q_weight,
        k_weight,
        v_weight,
        ve_ids_for_kernel,
        ve_weight_for_kernel,
        ve_gate_weight,
        cos,
        sin,
        q,
        k,
        v,
        qk_rms_inv,
        M,
        K,
        head_dim,
        n_head,
        n_kv_head,
        T,
        eps,
        scale,
        BLOCK_M=QKV_BLOCK_M,
        BLOCK_D=head_dim,
        BLOCK_K=QKV_BLOCK_K,
        HAS_VALUE_EMBEDDING=has_value_embedding,
        VE_GATE_CH=ve_gate_channels,
        VE_GATE_BLOCK=triton.next_power_of_2(ve_gate_channels),
        num_warps=QKV_NUM_WARPS,
        num_stages=2,
    )
    return q, k, v, rms_inv, qk_rms_inv


@torch.library.custom_op(
    "nanoops::norm_qkv_projection_fwd",
    mutates_args=(),
    device_types="cuda",
)
def _norm_qkv_projection_fwd_op(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    ve_ids: torch.Tensor | None,
    ve_weight: torch.Tensor | None,
    ve_gate_channels: int,
    ve_gate_weight: torch.Tensor | None,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    scale: float,
    eps: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Custom-op forward wrapper.

    Input shapes/types match `_norm_qkv_projection_fwd_impl`. Returns
    `(q, k, v, rms_inv, qk_rms_inv)`: q/k/v keep `x.dtype`; `rms_inv` and
    `qk_rms_inv` are fp32 hidden saved-state tensors consumed by backward.
    """
    return _norm_qkv_projection_fwd_impl(
        x,
        norm_weight,
        ve_ids,
        ve_weight,
        ve_gate_channels,
        ve_gate_weight,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        n_head,
        n_kv_head,
        head_dim,
        scale,
        eps,
    )


@_norm_qkv_projection_fwd_op.register_fake
def _norm_qkv_projection_fwd_fake(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    ve_ids: torch.Tensor | None,
    ve_weight: torch.Tensor | None,
    ve_gate_channels: int,
    ve_gate_weight: torch.Tensor | None,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    scale: float,
    eps: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Fake/meta kernel for Dynamo shape inference.

    Returns fake tensors shaped as:
      q: (B, T, n_head, head_dim), dtype=x.dtype.
      k: (B, T, n_kv_head, head_dim), dtype=x.dtype.
      v: (B, T, n_kv_head, head_dim), dtype=x.dtype.
      rms_inv: (B*T,), dtype=torch.float32.
      qk_rms_inv: (B, T, n_head + n_kv_head), dtype=torch.float32.
    """
    B, T, _K = x.shape
    return (
        torch.empty((B, T, n_head, head_dim), dtype=x.dtype, device=x.device),
        torch.empty((B, T, n_kv_head, head_dim), dtype=x.dtype, device=x.device),
        torch.empty((B, T, n_kv_head, head_dim), dtype=x.dtype, device=x.device),
        torch.empty((B * T,), dtype=torch.float32, device=x.device),
        torch.empty(
            (B, T, n_head + n_kv_head),
            dtype=torch.float32,
            device=x.device,
        ),
    )


@torch.library.custom_op(
    "nanoops::norm_qkv_projection_bwd",
    mutates_args=(),
    device_types="cuda",
)
def _norm_qkv_projection_bwd_op(
    d_q: torch.Tensor,
    d_k: torch.Tensor,
    d_v: torch.Tensor,
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    ve_ids: torch.Tensor | None,
    ve_weight: torch.Tensor | None,
    ve_gate_channels: int,
    ve_gate_weight: torch.Tensor | None,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_inv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    qk_rms_inv: torch.Tensor,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    scale: float,
    eps: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Custom-op backward wrapper.

    Inputs mirror `_norm_qkv_projection_bwd_impl`. This op returns tensors only;
    optional gradients are represented by 1-element placeholders because
    `torch.library.custom_op` return values cannot be Optional.

    Returns:
      dx: same shape/dtype as x.
      d_norm_weight: same shape/dtype as norm_weight, or a 1-element placeholder.
      d_ve_weight: same shape/dtype as ve_weight, or a 1-element placeholder.
      d_ve_gate_weight: same shape/dtype as ve_gate_weight, or a 1-element placeholder.
      d_q_weight: same shape/dtype as q_weight.
      d_k_weight: same shape/dtype as k_weight.
      d_v_weight: same shape/dtype as v_weight.
    """
    (
        dx,
        d_norm_weight,
        d_ve_weight,
        d_ve_gate_weight,
        d_q_weight,
        d_k_weight,
        d_v_weight,
    ) = (
        _norm_qkv_projection_bwd_impl(
            d_q,
            d_k,
            d_v,
            x,
            norm_weight,
            ve_ids,
            ve_weight,
            ve_gate_channels,
            ve_gate_weight,
            q_weight,
            k_weight,
            v_weight,
            cos,
            sin,
            rms_inv,
            q,
            k,
            qk_rms_inv,
            n_head,
            n_kv_head,
            head_dim,
            scale,
            eps,
        )
    )
    if d_norm_weight is None:
        d_norm_weight = torch.empty(1, dtype=x.dtype, device=x.device)
    if d_ve_weight is None:
        d_ve_weight = torch.empty(1, dtype=x.dtype, device=x.device)
    if d_ve_gate_weight is None:
        d_ve_gate_weight = torch.empty(1, dtype=x.dtype, device=x.device)
    return (
        dx,
        d_norm_weight,
        d_ve_weight,
        d_ve_gate_weight,
        d_q_weight,
        d_k_weight,
        d_v_weight,
    )


@_norm_qkv_projection_bwd_op.register_fake
def _norm_qkv_projection_bwd_fake(
    d_q: torch.Tensor,
    d_k: torch.Tensor,
    d_v: torch.Tensor,
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    ve_ids: torch.Tensor | None,
    ve_weight: torch.Tensor | None,
    ve_gate_channels: int,
    ve_gate_weight: torch.Tensor | None,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_inv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    qk_rms_inv: torch.Tensor,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    scale: float,
    eps: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Fake/meta kernel for backward shape inference.

    Output shapes mirror `_norm_qkv_projection_bwd_op`; optional gradients use
    the same 1-element placeholder convention as the real custom op.
    """
    d_norm_weight = (
        torch.empty_like(norm_weight)
        if norm_weight is not None
        else torch.empty(1, dtype=x.dtype, device=x.device)
    )
    d_ve_weight = torch.empty_like(ve_weight) if ve_weight is not None else torch.empty(
        1, dtype=x.dtype, device=x.device
    )
    d_ve_gate_weight = (
        torch.empty_like(ve_gate_weight)
        if ve_gate_weight is not None
        else torch.empty(1, dtype=x.dtype, device=x.device)
    )
    return (
        torch.empty_like(x),
        d_norm_weight,
        d_ve_weight,
        d_ve_gate_weight,
        torch.empty_like(q_weight),
        torch.empty_like(k_weight),
        torch.empty_like(v_weight),
    )


def _norm_qkv_projection_setup_context(
    ctx: Any,
    inputs: tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        int,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
        int,
        float,
        float,
    ],
    output: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> None:
    """Save tensors and scalar metadata for custom-op autograd.

    `inputs` is the exact forward-op argument tuple:
      x (B,T,K), norm_weight (K,) or None,
      ve_ids (B,T) or None, ve_weight (vocab,n_kv_head*head_dim) or None,
      ve_gate_channels int, ve_gate_weight (n_kv_head,ve_gate_channels) or None,
      q_weight (n_head*head_dim,K), k_weight/v_weight (n_kv_head*head_dim,K),
      cos/sin (1,T,1,head_dim/2), n_head, n_kv_head, head_dim, scale, eps.

    Saved tensors are:
      - forward inputs needed for recompute: x, optional norm/VE tensors,
        q/k/v weights, and cos/sin;
      - forward outputs needed only by backward: rms_inv, final q/k, qk_rms_inv.

    `x` is saved because backward recomputes `x_hat = RMSNorm(x)` instead of
    storing it. `v` is not saved: V backward uses `grad_v` and `v_weight`, and
    VE backward recomputes its gate/lookup terms. `rms_inv` is the fp32 `(B*T,)`
    outer RMSNorm inverse; `q`/`k` plus the fp32 per-head `qk_rms_inv` let the
    bf16/fp16 backward avoid Q/K projection recompute.
    """
    (
        x,
        norm_weight,
        ve_ids,
        ve_weight,
        ve_gate_channels,
        ve_gate_weight,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        n_head,
        n_kv_head,
        head_dim,
        scale,
        eps,
    ) = inputs
    q, k, _v, rms_inv, qk_rms_inv = output
    ctx.save_for_backward(
        x,
        norm_weight,
        ve_ids,
        ve_weight,
        ve_gate_weight,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        rms_inv,
        q,
        k,
        qk_rms_inv,
    )
    ctx.n_head = n_head
    ctx.n_kv_head = n_kv_head
    ctx.head_dim = head_dim
    ctx.scale = scale
    ctx.eps = eps
    ctx.ve_gate_channels = ve_gate_channels
    ctx.has_value_embedding = ve_ids is not None or ve_weight is not None


def _norm_qkv_projection_autograd_backward(
    ctx: Any,
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    _grad_rms_inv: torch.Tensor,
    _grad_qk_rms_inv: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    None,
    torch.Tensor | None,
    None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
]:
    """Autograd callback for `nanoops::norm_qkv_projection_fwd`.

    Incoming grads:
      grad_q: (B, T, n_head, head_dim).
      grad_k: (B, T, n_kv_head, head_dim).
      grad_v: (B, T, n_kv_head, head_dim).
      _grad_rms_inv: (B*T,), ignored because `rms_inv` is hidden saved state.
      _grad_qk_rms_inv: ignored because `qk_rms_inv` is hidden saved state.

    Returns one entry per forward input:
      dx, d_norm_weight, None for ve_ids, d_ve_weight, None for
      ve_gate_channels, d_ve_gate_weight, d_q_weight, d_k_weight, d_v_weight,
      then None for cos/sin and scalar metadata.
    """
    (
        x,
        norm_weight,
        ve_ids,
        ve_weight,
        ve_gate_weight,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        rms_inv,
        q,
        k,
        qk_rms_inv,
    ) = ctx.saved_tensors
    (
        dx,
        d_norm_weight,
        d_ve_weight,
        d_ve_gate_weight,
        d_q_weight,
        d_k_weight,
        d_v_weight,
    ) = (
        _norm_qkv_projection_bwd_op(
            grad_q,
            grad_k,
            grad_v,
            x,
            norm_weight,
            ve_ids,
            ve_weight,
            ctx.ve_gate_channels,
            ve_gate_weight,
            q_weight,
            k_weight,
            v_weight,
            cos,
            sin,
            rms_inv,
            q,
            k,
            qk_rms_inv,
            ctx.n_head,
            ctx.n_kv_head,
            ctx.head_dim,
            ctx.scale,
            ctx.eps,
        )
    )
    if norm_weight is None:
        d_norm_weight = None
    if not ctx.has_value_embedding:
        d_ve_weight = None
        d_ve_gate_weight = None
    return (
        dx,
        d_norm_weight,
        None,
        d_ve_weight,
        None,
        d_ve_gate_weight,
        d_q_weight,
        d_k_weight,
        d_v_weight,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_norm_qkv_projection_fwd_op.register_autograd(
    _norm_qkv_projection_autograd_backward,
    setup_context=_norm_qkv_projection_setup_context,
)


def norm_qkv_projection(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    ve_ids: torch.Tensor | None,
    ve_weight: torch.Tensor | None,
    ve_gate_channels: int,
    ve_gate_weight: torch.Tensor | None,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
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
      x: (B, T, K), CUDA, contiguous activation tensor. q/k/v outputs use
        this dtype.
      norm_weight: optional (K,), CUDA, contiguous RMSNorm scale. None means
        plain RMSNorm without an affine weight. nanochat normally uses None,
        so the affine branch is kept for compatibility instead of heavy tuning.
      ve_ids: optional (B, T), CUDA integer token ids. Must be None together
        with `ve_weight`/`ve_gate_weight`, or all three must be provided.
      ve_weight: optional (vocab, n_kv_head * head_dim), CUDA, contiguous value
        embedding table.
      ve_gate_channels: int, number of leading normalized input channels used
        by the value gate. Use 1 when value embedding is disabled.
      ve_gate_weight: optional (n_kv_head, ve_gate_channels), CUDA, contiguous
        value gate weight.
      q_weight: (n_head * head_dim, K), CUDA, contiguous query projection.
      k_weight: (n_kv_head * head_dim, K), CUDA, contiguous key projection.
      v_weight: (n_kv_head * head_dim, K), CUDA, contiguous value projection.
      cos: (1, T, 1, head_dim/2), CUDA, contiguous rotary cosine table.
      sin: (1, T, 1, head_dim/2), CUDA, contiguous rotary sine table.
      n_head: int, number of query heads.
      n_kv_head: int, number of key/value heads.
      head_dim: int, per-head width; must be even.
      scale: float, post-Q/K RMSNorm scalar multiplier.
      eps: float, RMSNorm epsilon for both outer norm and Q/K norm.

    Returns:
      q: (B, T, n_head, head_dim), dtype=x.dtype.
      k: (B, T, n_kv_head, head_dim), dtype=x.dtype.
      v: (B, T, n_kv_head, head_dim), dtype=x.dtype.
    """
    q, k, v, _rms_inv, _qk_rms_inv = _norm_qkv_projection_fwd_op(
        x,
        norm_weight,
        ve_ids,
        ve_weight,
        ve_gate_channels,
        ve_gate_weight,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        n_head,
        n_kv_head,
        head_dim,
        scale,
        eps,
    )
    return q, k, v
