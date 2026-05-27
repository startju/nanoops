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
    def _norm_qkv_rotary_fwd_kernel(
        x_ptr,  # (M, K) — in: materialized outer RMSNorm output
        q_w_ptr,  # (n_head * D, K) — in: Q projection weight
        k_w_ptr,  # (n_kv_head * D, K) — in: K projection weight
        v_w_ptr,  # (n_kv_head * D, K) — in: V projection weight
        ve_ptr,  # (M, n_kv_head, D) — optional in: value embedding
        ve_ids_ptr,  # (M,) — optional in: token ids for fused value embedding lookup
        ve_w_ptr,  # (vocab, n_kv_head * D) — optional in: value embedding table
        ve_gate_w_ptr,  # (n_kv_head, VE_GATE_CH) — optional in: value gate weight
        cos_ptr,  # (1, T, 1, D/2) — in: rotary cosine table
        sin_ptr,  # (1, T, 1, D/2) — in: rotary sine table
        q_ptr,  # (M, n_head, D) — out: Q after rotary + QK norm + scale
        k_ptr,  # (M, n_kv_head, D) — out: K after rotary + QK norm + scale
        v_ptr,  # (M, n_kv_head, D) — out: projected V plus optional value embedding
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
        HAS_VALUE_LOOKUP: tl.constexpr,
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
        is_hi_col = (cols % 2) == 1
        src_cols = (cols // 2) + tl.where(is_hi_col, BLOCK_D // 2, 0)
        weight_rows = head * D + src_cols

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
            )

            if is_q:
                w_ptrs = q_w_ptr + weight_rows[:, None] * K + ks[None, :]
            else:
                if is_k:
                    w_ptrs = k_w_ptr + weight_rows[:, None] * K + ks[None, :]
                else:
                    w_ptrs = v_w_ptr + weight_rows[:, None] * K + ks[None, :]
            w = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0).to(x_ptr.dtype.element_ty)
            acc_interleaved += tl.dot(x, tl.trans(w))

        if is_v:
            v_out = acc_interleaved
            if HAS_VALUE_EMBEDDING:
                gate_cols = tl.arange(0, VE_GATE_BLOCK)
                gate_mask = gate_cols < VE_GATE_CH
                gate_x = tl.load(
                    x_ptr + rows[:, None] * K + gate_cols[None, :],
                    mask=row_mask[:, None] & gate_mask[None, :],
                    other=0.0,
                ).to(x_ptr.dtype.element_ty)
                gate_w = tl.load(
                    ve_gate_w_ptr + head * VE_GATE_CH + gate_cols,
                    mask=gate_mask,
                    other=0.0,
                ).to(x_ptr.dtype.element_ty)
                gate_logits = tl.sum(gate_x * gate_w[None, :], axis=1, dtype=tl.float32)
                gate = 3 * tl.sigmoid(gate_logits).to(x_ptr.dtype.element_ty)
                if HAS_VALUE_LOOKUP:
                    token_ids = tl.load(ve_ids_ptr + rows, mask=row_mask, other=0)
                    ve_ptrs = (
                        ve_w_ptr
                        + token_ids[:, None] * n_kv_head * D
                        + head * D
                        + src_cols[None, :]
                    )
                else:
                    ve_ptrs = (
                        ve_ptr
                        + rows[:, None] * n_kv_head * D
                        + head * D
                        + src_cols[None, :]
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
                + src_cols[None, :]
            )
            tl.store(
                v_ptrs,
                v_out.to(v_ptr.dtype.element_ty),
                mask=row_mask[:, None],
            )
            return

        pair_axis = tl.arange(0, 2)
        acc_pair = tl.reshape(acc_interleaved, (BLOCK_M, BLOCK_D // 2, 2))
        acc_lo = tl.sum(tl.where(pair_axis[None, None, :] == 0, acc_pair, 0.0), axis=2)
        acc_hi = tl.sum(tl.where(pair_axis[None, None, :] == 1, acc_pair, 0.0), axis=2)
        rotary_rows = rows % rotary_seq_len

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

    @triton.jit
    def _ve_weight_grad_scatter_bwd_kernel(
        ve_ids_ptr,  # (M,) int64 — token ids
        d_ve_ptr,  # (M, n_kv_head, D) — grad of looked-up value embedding
        d_ve_w_ptr,  # (vocab, n_kv_head * D) fp32 — out via atomic_add
        M,  # int
        D,  # int
        n_kv_head,  # int
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Scatter-add d_ve into d_ve_weight[token_id] with atomics."""
        pid_m = tl.program_id(0)
        head = tl.program_id(1)
        pid_d = tl.program_id(2)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        row_mask = rows < M
        col_mask = cols < D
        mask = row_mask[:, None] & col_mask[None, :]

        token_ids = tl.load(ve_ids_ptr + rows, mask=row_mask, other=0)
        d_ve = tl.load(
            d_ve_ptr + rows[:, None] * n_kv_head * D + head * D + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        dst_ptrs = (
            d_ve_w_ptr
            + token_ids[:, None] * n_kv_head * D
            + head * D
            + cols[None, :]
        )
        tl.atomic_add(dst_ptrs, d_ve, sem="relaxed", mask=mask)

    @triton.jit
    def _norm_qkv_rotary_dz_bwd_kernel(
        x_ptr,  # (M, K) — in: activation before RMSNorm
        rms_inv_ptr,  # (M,) fp32 — in: per-row inverse RMS for x
        norm_w_ptr,  # (K,) — in: outer RMSNorm scale
        q_w_ptr,  # (n_head * D, K) — in: Q projection weight
        k_w_ptr,  # (n_kv_head * D, K) — in: K projection weight
        v_w_ptr,  # (n_kv_head * D, K) — in: V projection weight
        cos_ptr,  # (1, rotary_seq_len, 1, D/2) — in: rotary cosine table
        sin_ptr,  # (1, rotary_seq_len, 1, D/2) — in: rotary sine table
        d_q_ptr,  # (M, n_head, D) — in: grad of Q output
        d_k_ptr,  # (M, n_kv_head, D) — in: grad of K output
        d_v_ptr,  # (M, n_kv_head, D) — in: grad of V output
        dz_ptr,  # (M, N_qkv) fp32 — out: grad before QKV projection
        M,  # int — row count after flattening batch/time
        K,  # int — input hidden width
        D,  # int — per-head width
        n_head,  # int — number of query heads
        n_kv_head,  # int — number of key/value heads
        rotary_seq_len,  # int — rotary table sequence length before batch broadcast
        eps,  # float — QK RMSNorm epsilon
        scale,  # float — post-QK-norm scalar multiplier
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_NORM_WEIGHT: tl.constexpr,
    ):
        """Recompute Q/K pre-activations and write projection-input grads."""
        pid_m = tl.program_id(0)
        part = tl.program_id(1)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_D)
        half_cols = tl.arange(0, BLOCK_D // 2)
        row_mask = rows < M
        N = (n_head + 2 * n_kv_head) * D

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
        weight_rows = head * D + src_cols
        out_rows = weight_offset + weight_rows

        if is_v:
            d_v = tl.load(
                d_v_ptr + rows[:, None] * n_kv_head * D + head * D + src_cols[None, :],
                mask=row_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            tl.store(
                dz_ptr + rows[:, None] * N + out_rows[None, :],
                d_v,
                mask=row_mask[:, None],
            )
            return

        x_rms_inv = tl.load(rms_inv_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        acc_interleaved = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            ks = k_start + tl.arange(0, BLOCK_K)
            k_mask = ks < K
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

            if is_q:
                w_ptrs = q_w_ptr + weight_rows[:, None] * K + ks[None, :]
            else:
                w_ptrs = k_w_ptr + weight_rows[:, None] * K + ks[None, :]
            w = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)
            acc_interleaved += tl.dot(x_hat, tl.trans(w), input_precision="ieee")

        pair_axis = tl.arange(0, 2)
        acc_pair = tl.reshape(acc_interleaved, (BLOCK_M, BLOCK_D // 2, 2))
        pre_lo = tl.sum(tl.where(pair_axis[None, None, :] == 0, acc_pair, 0.0), axis=2)
        pre_hi = tl.sum(tl.where(pair_axis[None, None, :] == 1, acc_pair, 0.0), axis=2)

        rotary_rows = rows % rotary_seq_len
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

        rot_lo = pre_lo * cos + pre_hi * sin
        rot_hi = -pre_lo * sin + pre_hi * cos
        qk_sum_sq = tl.sum(rot_lo * rot_lo, axis=1) + tl.sum(rot_hi * rot_hi, axis=1)
        qk_rms_inv = tl.rsqrt(qk_sum_sq / D + eps)
        y0_lo = rot_lo * qk_rms_inv[:, None]
        y0_hi = rot_hi * qk_rms_inv[:, None]

        if is_q:
            grad_base = d_q_ptr + rows[:, None] * n_head * D + head * D
        else:
            grad_base = d_k_ptr + rows[:, None] * n_kv_head * D + head * D
        g_lo = tl.load(
            grad_base + half_cols[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32) * scale
        g_hi = tl.load(
            grad_base + (half_cols[None, :] + BLOCK_D // 2),
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32) * scale

        inner = (
            tl.sum(g_lo * y0_lo, axis=1) + tl.sum(g_hi * y0_hi, axis=1)
        ) / D
        d_rot_lo = qk_rms_inv[:, None] * (g_lo - y0_lo * inner[:, None])
        d_rot_hi = qk_rms_inv[:, None] * (g_hi - y0_hi * inner[:, None])
        d_pre_lo = d_rot_lo * cos - d_rot_hi * sin
        d_pre_hi = d_rot_lo * sin + d_rot_hi * cos

        dz_lo_ptrs = dz_ptr + rows[:, None] * N + weight_offset + head * D + half_cols[None, :]
        dz_hi_ptrs = (
            dz_ptr
            + rows[:, None] * N
            + weight_offset
            + head * D
            + half_cols[None, :]
            + (BLOCK_D // 2)
        )
        tl.store(dz_lo_ptrs, d_pre_lo, mask=row_mask[:, None])
        tl.store(dz_hi_ptrs, d_pre_hi, mask=row_mask[:, None])

    @triton.jit
    def _qkv_dz_w_bwd_kernel(
        dz_ptr,  # (M, N_qkv) fp32 — in
        q_w_ptr,  # (Q, K) — in
        k_w_ptr,  # (KV, K) — in
        v_w_ptr,  # (KV, K) — in
        dx_hat_ptr,  # (M, K) fp32 — out
        M,  # int
        QN,  # int — n_head * D
        KVN,  # int — n_kv_head * D
        K,  # int
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Compute d_x_hat = d_z @ concat(q_weight, k_weight, v_weight)."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = rows < M
        k_mask = ks < K
        N = QN + 2 * KVN

        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for n_start in range(0, QN, BLOCK_N):
            ns = n_start + tl.arange(0, BLOCK_N)
            n_mask = ns < QN
            dz = tl.load(
                dz_ptr + rows[:, None] * N + ns[None, :],
                mask=row_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            w = tl.load(
                q_w_ptr + ns[:, None] * K + ks[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(dz, w, input_precision="ieee")

        for n_start in range(0, KVN, BLOCK_N):
            ns = n_start + tl.arange(0, BLOCK_N)
            n_mask = ns < KVN
            dz = tl.load(
                dz_ptr + rows[:, None] * N + (QN + ns)[None, :],
                mask=row_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            w = tl.load(
                k_w_ptr + ns[:, None] * K + ks[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(dz, w, input_precision="ieee")

        for n_start in range(0, KVN, BLOCK_N):
            ns = n_start + tl.arange(0, BLOCK_N)
            n_mask = ns < KVN
            dz = tl.load(
                dz_ptr + rows[:, None] * N + (QN + KVN + ns)[None, :],
                mask=row_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            w = tl.load(
                v_w_ptr + ns[:, None] * K + ks[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(dz, w, input_precision="ieee")

        tl.store(
            dx_hat_ptr + rows[:, None] * K + ks[None, :],
            acc,
            mask=row_mask[:, None] & k_mask[None, :],
        )

    @triton.jit
    def _outer_rms_inner_bwd_kernel(
        x_ptr,  # (M, K) — in
        rms_inv_ptr,  # (M,) fp32 — in
        norm_w_ptr,  # (K,) — in
        dx_hat_ptr,  # (M, K) fp32 — in
        row_inner_ptr,  # (M,) fp32 — out: mean(d_x_norm * x_norm)
        M,  # int
        K,  # int
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_NORM_WEIGHT: tl.constexpr,
    ):
        """Compute the per-row RMSNorm backward inner product."""
        pid_m = tl.program_id(0)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        x_rms_inv = tl.load(rms_inv_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            ks = k_start + tl.arange(0, BLOCK_K)
            k_mask = ks < K
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
                dx_norm = dx_hat * nw[None, :]
            else:
                dx_norm = dx_hat
            acc += tl.sum(dx_norm * x_norm, axis=1)

        tl.store(row_inner_ptr + rows, acc / K, mask=row_mask)

    @triton.jit
    def _outer_rms_dx_bwd_kernel(
        x_ptr,  # (M, K) — in
        rms_inv_ptr,  # (M,) fp32 — in
        norm_w_ptr,  # (K,) — in
        dx_hat_ptr,  # (M, K) fp32 — in
        row_inner_ptr,  # (M,) fp32 — in
        dx_ptr,  # (M, K) — out
        M,  # int
        K,  # int
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_NORM_WEIGHT: tl.constexpr,
    ):
        """Compute d_x for the outer RMSNorm."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = rows < M
        k_mask = ks < K

        x_rms_inv = tl.load(rms_inv_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        row_inner = tl.load(row_inner_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
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
            dx_norm = dx_hat * nw[None, :]
        else:
            dx_norm = dx_hat
        dx = x_rms_inv[:, None] * (dx_norm - x_norm * row_inner[:, None])
        tl.store(
            dx_ptr + rows[:, None] * K + ks[None, :],
            dx.to(dx_ptr.dtype.element_ty),
            mask=row_mask[:, None] & k_mask[None, :],
        )

    @triton.jit
    def _norm_weight_grad_bwd_kernel(
        x_ptr,  # (M, K) — in
        rms_inv_ptr,  # (M,) fp32 — in
        dx_hat_ptr,  # (M, K) fp32 — in
        dnorm_w_ptr,  # (K,) — out
        M,  # int
        K,  # int
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Compute d_norm_weight = sum_m d_x_hat * x_norm."""
        pid_k = tl.program_id(0)
        ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = ks < K

        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
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
            acc += tl.sum(dx_hat * x * x_rms_inv[:, None], axis=0)

        tl.store(dnorm_w_ptr + ks, acc.to(dnorm_w_ptr.dtype.element_ty), mask=k_mask)

    @triton.jit
    def _qkv_weight_grad_bwd_kernel(
        dz_ptr,  # (M, N_qkv) fp32 — in
        x_ptr,  # (M, K) — in
        rms_inv_ptr,  # (M,) fp32 — in
        norm_w_ptr,  # (K,) — in
        d_w_ptr,  # (N_part, K) — out
        M,  # int
        N,  # int
        N_part,  # int
        dz_offset,  # int
        K,  # int
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        HAS_NORM_WEIGHT: tl.constexpr,
    ):
        """Compute d_weight = d_z_part.T @ x_hat."""
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        ns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        n_mask = ns < N_part
        k_mask = ks < K

        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for m_start in range(0, M, BLOCK_M):
            rows = m_start + tl.arange(0, BLOCK_M)
            row_mask = rows < M
            dz = tl.load(
                dz_ptr + rows[:, None] * N + (dz_offset + ns)[None, :],
                mask=row_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
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
            acc += tl.dot(tl.trans(dz), x_hat, input_precision="ieee")

        tl.store(
            d_w_ptr + ns[:, None] * K + ks[None, :],
            acc.to(d_w_ptr.dtype.element_ty),
            mask=n_mask[:, None] & k_mask[None, :],
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


def _norm_qkv_rotary_projection_triton_backward(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
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
    saved_rms_inv: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form Triton backward for `NormQKVProjection`.

    This helper launches Triton kernels for the fp32 algebra used by the
    custom-op autograd callback. Rotary cos/sin are treated as constants; the
    returned gradients are for x, norm_weight, q_weight, k_weight, and v_weight.

    Forward math:
      s_x        = rsqrt(mean(x^2) + eps)
      x_norm     = x * s_x
      x_hat      = x_norm * norm_weight
      q0 = x_hat @ q_weight.T
      k0 = x_hat @ k_weight.T
      v0 = x_hat @ v_weight.T

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
        d_x_hat = d_q0 @ q_weight + d_k0 @ k_weight + d_v0 @ v_weight
        dW_i    = d_i.T @ x_hat

      outer RMSNorm bwd:
        d_x = s_x * (d_x_norm - x_norm * mean(d_x_norm * x_norm))
    """
    if not _HAS_TRITON:
        raise RuntimeError("norm_qkv_rotary_projection backward requires triton")
    assert x.is_cuda and x.is_contiguous()
    assert q_weight.is_cuda and k_weight.is_cuda and v_weight.is_cuda
    has_norm_weight = norm_weight is not None
    norm_weight_or_x = norm_weight if has_norm_weight else x
    assert norm_weight_or_x.is_cuda
    assert (
        norm_weight_or_x.is_contiguous()
        and q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and v_weight.is_contiguous()
    )
    assert head_dim % 2 == 0

    d_q = d_q.contiguous()
    d_k = d_k.contiguous()
    d_v = d_v.contiguous()
    M, K = x.shape
    q_n = n_head * head_dim
    kv_n = n_kv_head * head_dim
    N = q_n + 2 * kv_n
    assert q_weight.shape == (q_n, K)
    assert k_weight.shape == (kv_n, K)
    assert v_weight.shape == (kv_n, K)
    rotary_seq_len = cos.shape[1]
    _validate_rotary_table_4d(cos, rotary_seq_len, head_dim)
    _validate_rotary_table_4d(sin, rotary_seq_len, head_dim)
    assert M % rotary_seq_len == 0, (
        f"M={M} is not divisible by rotary T={rotary_seq_len}"
    )

    if saved_rms_inv is None:
        rms_inv = torch.empty((M,), dtype=torch.float32, device=x.device)
        x_hat_tmp = torch.empty_like(x)
        norm_block_d = triton.next_power_of_2(K)
        norm_cfg = _pick_tile_config(M, norm_block_d, n_live_tiles=2)
        _fused_add_norm_fwd_kernel[(triton.cdiv(M, norm_cfg.block_m),)](
            x,
            x,
            norm_weight_or_x,
            x_hat_tmp,
            x,
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
    else:
        assert saved_rms_inv.is_cuda and saved_rms_inv.is_contiguous()
        assert saved_rms_inv.shape == (M,) and saved_rms_inv.dtype == torch.float32
        rms_inv = saved_rms_inv
    dz = torch.empty((M, N), dtype=torch.float32, device=x.device)
    dx_hat = torch.empty((M, K), dtype=torch.float32, device=x.device)
    row_inner = torch.empty((M,), dtype=torch.float32, device=x.device)
    dx = torch.empty_like(x)
    d_norm_weight = torch.empty_like(norm_weight) if has_norm_weight else None
    d_q_weight = torch.empty_like(q_weight)
    d_k_weight = torch.empty_like(k_weight)
    d_v_weight = torch.empty_like(v_weight)

    RMS_BLOCK_M, RMS_BLOCK_K = 32, 32
    DZ_BLOCK_M, DZ_BLOCK_K = 32, 32
    MM_BLOCK_M, MM_BLOCK_K, MM_BLOCK_N = 16, 32, 32
    DW_BLOCK_N, DW_BLOCK_K, DW_BLOCK_M = 16, 32, 32

    _norm_qkv_rotary_dz_bwd_kernel[(triton.cdiv(M, DZ_BLOCK_M), n_head + 2 * n_kv_head)](
        x,
        rms_inv,
        norm_weight_or_x,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        d_q,
        d_k,
        d_v,
        dz,
        M,
        K,
        head_dim,
        n_head,
        n_kv_head,
        rotary_seq_len,
        eps,
        scale,
        BLOCK_M=DZ_BLOCK_M,
        BLOCK_D=head_dim,
        BLOCK_K=DZ_BLOCK_K,
        HAS_NORM_WEIGHT=has_norm_weight,
        num_warps=4,
        num_stages=2,
    )
    _qkv_dz_w_bwd_kernel[(triton.cdiv(M, MM_BLOCK_M), triton.cdiv(K, MM_BLOCK_K))](
        dz,
        q_weight,
        k_weight,
        v_weight,
        dx_hat,
        M,
        q_n,
        kv_n,
        K,
        BLOCK_M=MM_BLOCK_M,
        BLOCK_K=MM_BLOCK_K,
        BLOCK_N=MM_BLOCK_N,
        num_warps=4,
        num_stages=3,
    )
    _outer_rms_inner_bwd_kernel[(triton.cdiv(M, RMS_BLOCK_M),)](
        x,
        rms_inv,
        norm_weight_or_x,
        dx_hat,
        row_inner,
        M,
        K,
        BLOCK_M=RMS_BLOCK_M,
        BLOCK_K=RMS_BLOCK_K,
        HAS_NORM_WEIGHT=has_norm_weight,
        num_warps=4,
    )
    _outer_rms_dx_bwd_kernel[(triton.cdiv(M, RMS_BLOCK_M), triton.cdiv(K, RMS_BLOCK_K))](
        x,
        rms_inv,
        norm_weight_or_x,
        dx_hat,
        row_inner,
        dx,
        M,
        K,
        BLOCK_M=RMS_BLOCK_M,
        BLOCK_K=RMS_BLOCK_K,
        HAS_NORM_WEIGHT=has_norm_weight,
        num_warps=4,
    )
    if has_norm_weight:
        _norm_weight_grad_bwd_kernel[(triton.cdiv(K, RMS_BLOCK_K),)](
            x,
            rms_inv,
            dx_hat,
            d_norm_weight,
            M,
            K,
            BLOCK_M=RMS_BLOCK_M,
            BLOCK_K=RMS_BLOCK_K,
            num_warps=4,
        )
    _qkv_weight_grad_bwd_kernel[(triton.cdiv(q_n, DW_BLOCK_N), triton.cdiv(K, DW_BLOCK_K))](
        dz,
        x,
        rms_inv,
        norm_weight_or_x,
        d_q_weight,
        M,
        N,
        q_n,
        0,
        K,
        BLOCK_N=DW_BLOCK_N,
        BLOCK_K=DW_BLOCK_K,
        BLOCK_M=DW_BLOCK_M,
        HAS_NORM_WEIGHT=has_norm_weight,
        num_warps=4,
        num_stages=3,
    )
    _qkv_weight_grad_bwd_kernel[(triton.cdiv(kv_n, DW_BLOCK_N), triton.cdiv(K, DW_BLOCK_K))](
        dz,
        x,
        rms_inv,
        norm_weight_or_x,
        d_k_weight,
        M,
        N,
        kv_n,
        q_n,
        K,
        BLOCK_N=DW_BLOCK_N,
        BLOCK_K=DW_BLOCK_K,
        BLOCK_M=DW_BLOCK_M,
        HAS_NORM_WEIGHT=has_norm_weight,
        num_warps=4,
        num_stages=3,
    )
    _qkv_weight_grad_bwd_kernel[(triton.cdiv(kv_n, DW_BLOCK_N), triton.cdiv(K, DW_BLOCK_K))](
        dz,
        x,
        rms_inv,
        norm_weight_or_x,
        d_v_weight,
        M,
        N,
        kv_n,
        q_n + kv_n,
        K,
        BLOCK_N=DW_BLOCK_N,
        BLOCK_K=DW_BLOCK_K,
        BLOCK_M=DW_BLOCK_M,
        HAS_NORM_WEIGHT=has_norm_weight,
        num_warps=4,
        num_stages=3,
    )
    return dx, d_norm_weight, d_q_weight, d_k_weight, d_v_weight


def _norm_qkv_projection_fwd_impl(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
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
    ve: torch.Tensor | None,
    ve_gate_weight: torch.Tensor | None,
    ve_gate_channels: int,
    ve_ids: torch.Tensor | None,
    ve_weight: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward implementation for `nanoops::norm_qkv_projection_fwd`.

    Returns `(q, k, v, rms_inv)`. `rms_inv` is a hidden output used only by
    the registered autograd callback.
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

    has_value_lookup = ve_ids is not None or ve_weight is not None
    assert not (ve is not None and has_value_lookup), (
        "pass either materialized ve or ve_ids + ve_weight, not both"
    )
    has_value_embedding = ve is not None or has_value_lookup
    if has_value_embedding:
        assert ve_gate_weight is not None, (
            "ve_gate_weight is required when value embedding is provided"
        )
        assert ve_gate_weight.is_cuda and ve_gate_weight.is_contiguous()
        assert 0 < ve_gate_channels <= K
        assert ve_gate_weight.shape == (n_kv_head, ve_gate_channels), (
            f"ve_gate_weight shape {tuple(ve_gate_weight.shape)} != "
            f"{(n_kv_head, ve_gate_channels)}"
        )
        if has_value_lookup:
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
            ve_for_kernel = x_2d
        else:
            assert ve is not None and ve.is_cuda
            if ve.ndim == 3:
                assert ve.shape == (B, T, kv_n), f"ve shape {tuple(ve.shape)} != {(B, T, kv_n)}"
                ve_for_kernel = ve.view(B, T, n_kv_head, head_dim)
            else:
                assert ve.ndim == 4
                assert ve.shape == (B, T, n_kv_head, head_dim), (
                    f"ve shape {tuple(ve.shape)} != {(B, T, n_kv_head, head_dim)}"
                )
                ve_for_kernel = ve
            ve_for_kernel = ve_for_kernel.contiguous()
            ve_ids_for_kernel = x_2d
            ve_weight_for_kernel = x_2d
    else:
        ve_for_kernel = x_2d
        ve_ids_for_kernel = x_2d
        ve_weight_for_kernel = x_2d
        ve_gate_weight = x_2d
        ve_gate_channels = 1

    q = torch.empty((B, T, n_head, head_dim), dtype=x.dtype, device=x.device)
    k = torch.empty((B, T, n_kv_head, head_dim), dtype=x.dtype, device=x.device)
    v = torch.empty((B, T, n_kv_head, head_dim), dtype=x.dtype, device=x.device)
    QKV_BLOCK_M, QKV_BLOCK_K = 64, 16
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
    _norm_qkv_rotary_fwd_kernel[grid](
        x_hat,
        q_weight,
        k_weight,
        v_weight,
        ve_for_kernel,
        ve_ids_for_kernel,
        ve_weight_for_kernel,
        ve_gate_weight,
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
        T,
        eps,
        scale,
        BLOCK_M=QKV_BLOCK_M,
        BLOCK_D=head_dim,
        BLOCK_K=QKV_BLOCK_K,
        HAS_VALUE_EMBEDDING=has_value_embedding,
        HAS_VALUE_LOOKUP=has_value_lookup,
        VE_GATE_CH=ve_gate_channels,
        VE_GATE_BLOCK=triton.next_power_of_2(ve_gate_channels),
        num_warps=2,
        num_stages=2,
    )
    return q, k, v, rms_inv


def _norm_qkv_projection_bwd_impl(
    d_q: torch.Tensor,
    d_k: torch.Tensor,
    d_v: torch.Tensor,
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_inv: torch.Tensor,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    scale: float,
    eps: float,
    ve: torch.Tensor | None,
    ve_gate_weight: torch.Tensor | None,
    ve_gate_channels: int,
    ve_ids: torch.Tensor | None,
    ve_weight: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Backward implementation for `nanoops::norm_qkv_projection_bwd`."""
    assert x.ndim == 3
    B, T, K = x.shape
    M = B * T
    x_2d = x.view(M, K)
    dx, dnw, dqw, dkw, dvw = _norm_qkv_rotary_projection_triton_backward(
        x_2d,
        norm_weight,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        d_q,
        d_k,
        d_v,
        n_head,
        n_kv_head,
        head_dim,
        scale,
        eps,
        saved_rms_inv=rms_inv,
    )

    d_ve = None
    d_ve_gate_weight = None
    d_ve_weight = None
    has_value_lookup = ve_ids is not None or ve_weight is not None
    has_value_embedding = ve is not None or has_value_lookup
    if has_value_embedding:
        assert ve_gate_weight is not None
        ch = ve_gate_channels
        d_v_gate = d_v.contiguous().view(M, n_kv_head, head_dim)
        if has_value_lookup:
            assert ve_ids is not None and ve_weight is not None
            ve_ids_flat = ve_ids.reshape(M).contiguous()
            ve_3d = ve_weight.view(ve_weight.shape[0], n_kv_head, head_dim)[
                ve_ids_flat.long()
            ]
        else:
            assert ve is not None
            ve_3d = ve.reshape(M, n_kv_head, head_dim)

        x_norm = x_2d.float() * rms_inv[:, None]
        if norm_weight is not None:
            x_hat = (x_norm * norm_weight.float()[None, :]).to(x_2d.dtype)
        else:
            x_hat = x_norm.to(x_2d.dtype)
        x_gate = x_hat[:, :ch]
        gate_logits = x_gate.float() @ ve_gate_weight.float().t()
        sigmoid = torch.sigmoid(gate_logits)
        gate = 3.0 * sigmoid

        d_ve_3d = d_v_gate * gate.to(d_v_gate.dtype).view(M, n_kv_head, 1)
        d_gate = (d_v_gate.float() * ve_3d.float()).sum(dim=-1)
        d_gate_logits = 3.0 * d_gate * sigmoid * (1.0 - sigmoid)
        d_ve_gate_weight = (d_gate_logits.t() @ x_gate.float()).to(ve_gate_weight.dtype)

        d_x_hat_gate = torch.zeros((M, K), dtype=torch.float32, device=x_2d.device)
        d_x_hat_gate[:, :ch] = d_gate_logits @ ve_gate_weight.float()
        if norm_weight is not None:
            dnw_gate = (d_x_hat_gate * x_norm).sum(dim=0).to(norm_weight.dtype)
            dnw = dnw + dnw_gate if dnw is not None else dnw_gate
            d_x_norm_gate = d_x_hat_gate * norm_weight.float()[None, :]
        else:
            d_x_norm_gate = d_x_hat_gate
        row_inner = (d_x_norm_gate * x_norm).mean(dim=-1, keepdim=True)
        dx_gate = rms_inv[:, None] * (d_x_norm_gate - x_norm * row_inner)
        dx = dx + dx_gate.to(dx.dtype)
        if has_value_lookup:
            assert ve_ids is not None and ve_weight is not None
            ve_ids_flat = ve_ids.reshape(M).contiguous()
            d_ve_weight_accum = torch.zeros(
                (ve_weight.shape[0], n_kv_head * head_dim),
                dtype=torch.float32,
                device=ve_weight.device,
            )
            VE_DW_BLOCK_M, VE_DW_BLOCK_D = 16, 64
            _ve_weight_grad_scatter_bwd_kernel[
                (
                    triton.cdiv(M, VE_DW_BLOCK_M),
                    n_kv_head,
                    triton.cdiv(head_dim, VE_DW_BLOCK_D),
                )
            ](
                ve_ids_flat,
                d_ve_3d.contiguous(),
                d_ve_weight_accum,
                M,
                head_dim,
                n_kv_head,
                BLOCK_M=VE_DW_BLOCK_M,
                BLOCK_D=VE_DW_BLOCK_D,
                num_warps=4,
            )
            d_ve_weight = d_ve_weight_accum.to(ve_weight.dtype).view_as(ve_weight)
        else:
            assert ve is not None
            d_ve = d_ve_3d.reshape(ve.shape)
    return dx.reshape_as(x), dnw, dqw, dkw, dvw, d_ve, d_ve_gate_weight, d_ve_weight


@torch.library.custom_op(
    "nanoops::norm_qkv_projection_fwd",
    mutates_args=(),
    device_types="cuda",
)
def _norm_qkv_projection_fwd_op(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
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
    ve: torch.Tensor | None,
    ve_gate_weight: torch.Tensor | None,
    ve_gate_channels: int,
    ve_ids: torch.Tensor | None,
    ve_weight: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom-op forward wrapper for fused attention QKV projection."""
    return _norm_qkv_projection_fwd_impl(
        x,
        norm_weight,
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
        ve,
        ve_gate_weight,
        ve_gate_channels,
        ve_ids,
        ve_weight,
    )


@_norm_qkv_projection_fwd_op.register_fake
def _norm_qkv_projection_fwd_fake(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
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
    ve: torch.Tensor | None,
    ve_gate_weight: torch.Tensor | None,
    ve_gate_channels: int,
    ve_ids: torch.Tensor | None,
    ve_weight: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake/meta kernel for Dynamo shape inference."""
    B, T, _K = x.shape
    return (
        torch.empty((B, T, n_head, head_dim), dtype=x.dtype, device=x.device),
        torch.empty((B, T, n_kv_head, head_dim), dtype=x.dtype, device=x.device),
        torch.empty((B, T, n_kv_head, head_dim), dtype=x.dtype, device=x.device),
        torch.empty((B * T,), dtype=torch.float32, device=x.device),
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
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_inv: torch.Tensor,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    scale: float,
    eps: float,
    ve: torch.Tensor | None,
    ve_gate_weight: torch.Tensor | None,
    ve_gate_channels: int,
    ve_ids: torch.Tensor | None,
    ve_weight: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Custom-op backward wrapper.

    custom_op returns cannot be Optional, so missing optional gradients use
    1-element placeholders that the autograd callback converts back to None.
    """
    dx, dnw, dqw, dkw, dvw, d_ve, d_ve_gate_weight, d_ve_weight = (
        _norm_qkv_projection_bwd_impl(
            d_q,
            d_k,
            d_v,
            x,
            norm_weight,
            q_weight,
            k_weight,
            v_weight,
            cos,
            sin,
            rms_inv,
            n_head,
            n_kv_head,
            head_dim,
            scale,
            eps,
            ve,
            ve_gate_weight,
            ve_gate_channels,
            ve_ids,
            ve_weight,
        )
    )
    if dnw is None:
        dnw = torch.empty(1, dtype=x.dtype, device=x.device)
    if d_ve is None:
        d_ve = torch.empty(1, dtype=x.dtype, device=x.device)
    if d_ve_gate_weight is None:
        d_ve_gate_weight = torch.empty(1, dtype=x.dtype, device=x.device)
    if d_ve_weight is None:
        d_ve_weight = torch.empty(1, dtype=x.dtype, device=x.device)
    return dx, dnw, dqw, dkw, dvw, d_ve, d_ve_gate_weight, d_ve_weight


@_norm_qkv_projection_bwd_op.register_fake
def _norm_qkv_projection_bwd_fake(
    d_q: torch.Tensor,
    d_k: torch.Tensor,
    d_v: torch.Tensor,
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_inv: torch.Tensor,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    scale: float,
    eps: float,
    ve: torch.Tensor | None,
    ve_gate_weight: torch.Tensor | None,
    ve_gate_channels: int,
    ve_ids: torch.Tensor | None,
    ve_weight: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Fake/meta kernel for backward shape inference."""
    dnw = torch.empty_like(norm_weight) if norm_weight is not None else torch.empty(
        1, dtype=x.dtype, device=x.device
    )
    d_ve = torch.empty_like(ve) if ve is not None else torch.empty(
        1, dtype=x.dtype, device=x.device
    )
    d_ve_gate_weight = (
        torch.empty_like(ve_gate_weight)
        if ve_gate_weight is not None
        else torch.empty(1, dtype=x.dtype, device=x.device)
    )
    d_ve_weight = torch.empty_like(ve_weight) if ve_weight is not None else torch.empty(
        1, dtype=x.dtype, device=x.device
    )
    return (
        torch.empty_like(x),
        dnw,
        torch.empty_like(q_weight),
        torch.empty_like(k_weight),
        torch.empty_like(v_weight),
        d_ve,
        d_ve_gate_weight,
        d_ve_weight,
    )


def _norm_qkv_projection_setup_context(
    ctx: Any,
    inputs: tuple[
        torch.Tensor,
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
        torch.Tensor | None,
        torch.Tensor | None,
        int,
        torch.Tensor | None,
        torch.Tensor | None,
    ],
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Save forward inputs and hidden `rms_inv` for custom-op autograd."""
    (
        x,
        norm_weight,
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
        ve,
        ve_gate_weight,
        ve_gate_channels,
        ve_ids,
        ve_weight,
    ) = inputs
    _q, _k, _v, rms_inv = output
    ctx.save_for_backward(
        x,
        norm_weight,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        rms_inv,
        ve,
        ve_gate_weight,
        ve_ids,
        ve_weight,
    )
    ctx.n_head = n_head
    ctx.n_kv_head = n_kv_head
    ctx.head_dim = head_dim
    ctx.scale = scale
    ctx.eps = eps
    ctx.ve_gate_channels = ve_gate_channels
    ctx.has_value_embedding = ve is not None or ve_ids is not None or ve_weight is not None


def _norm_qkv_projection_autograd_backward(
    ctx: Any,
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    grad_rms_inv: torch.Tensor,
) -> tuple[
    torch.Tensor,
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
    torch.Tensor | None,
    torch.Tensor | None,
    None,
    None,
    torch.Tensor | None,
]:
    """Autograd callback for `nanoops::norm_qkv_projection_fwd`."""
    (
        x,
        norm_weight,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        rms_inv,
        ve,
        ve_gate_weight,
        ve_ids,
        ve_weight,
    ) = ctx.saved_tensors
    dx, dnw, dqw, dkw, dvw, d_ve, d_ve_gate_weight, d_ve_weight = (
        _norm_qkv_projection_bwd_op(
            grad_q,
            grad_k,
            grad_v,
            x,
            norm_weight,
            q_weight,
            k_weight,
            v_weight,
            cos,
            sin,
            rms_inv,
            ctx.n_head,
            ctx.n_kv_head,
            ctx.head_dim,
            ctx.scale,
            ctx.eps,
            ve,
            ve_gate_weight,
            ctx.ve_gate_channels,
            ve_ids,
            ve_weight,
        )
    )
    if norm_weight is None:
        dnw = None
    if not ctx.has_value_embedding:
        d_ve = None
        d_ve_gate_weight = None
        d_ve_weight = None
    else:
        if ve is None:
            d_ve = None
        if ve_gate_weight is None:
            d_ve_gate_weight = None
        if ve_weight is None:
            d_ve_weight = None
    return (
        dx,
        dnw,
        dqw,
        dkw,
        dvw,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        d_ve,
        d_ve_gate_weight,
        None,
        None,
        d_ve_weight,
    )


_norm_qkv_projection_fwd_op.register_autograd(
    _norm_qkv_projection_autograd_backward,
    setup_context=_norm_qkv_projection_setup_context,
)


def norm_qkv_projection(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
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
    ve: torch.Tensor | None = None,
    ve_gate_weight: torch.Tensor | None = None,
    ve_gate_channels: int = 12,
    ve_ids: torch.Tensor | None = None,
    ve_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused RMSNorm + QKV projection with Q/K rotary, QK RMSNorm, and scale.

    Args:
      x: (B, T, C) CUDA tensor.
      norm_weight: optional (K,) RMSNorm scale for the input projection.
      q_weight: (n_head * head_dim, C) query projection weight.
      k_weight: (n_kv_head * head_dim, C) key projection weight.
      v_weight: (n_kv_head * head_dim, C) value projection weight.
      cos: (1, T, 1, head_dim/2) rotary table.
      sin: (1, T, 1, head_dim/2) rotary table.
      n_head: number of query heads.
      n_kv_head: number of key/value heads.
      head_dim: per-head width.
      scale: post-QK-norm scalar multiplier.
      eps: RMSNorm epsilon.
      ve: optional value embedding, shaped `(B, T, n_kv_head * head_dim)`
        or `(B, T, n_kv_head, head_dim)`.
      ve_gate_weight: optional `(n_kv_head, ve_gate_channels)` gate weight.
      ve_gate_channels: number of normalized input channels used by the value gate.
      ve_ids: optional `(B, T)` token ids for fused value embedding lookup.
      ve_weight: optional `(vocab, n_kv_head * head_dim)` value embedding table.

    Returns:
      Tuple `(q, k, v)` shaped for SDPA: `(B, T, n_head, head_dim)`,
      `(B, T, n_kv_head, head_dim)`, and `(B, T, n_kv_head, head_dim)`.
    """
    q, k, v, _rms_inv = _norm_qkv_projection_fwd_op(
        x,
        norm_weight,
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
        ve,
        ve_gate_weight,
        ve_gate_channels,
        ve_ids,
        ve_weight,
    )
    return q, k, v


def norm_qkv_rotary_projection(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
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
    ve: torch.Tensor | None = None,
    ve_gate_weight: torch.Tensor | None = None,
    ve_gate_channels: int = 12,
    ve_ids: torch.Tensor | None = None,
    ve_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward-compatible alias for `norm_qkv_projection`."""
    return norm_qkv_projection(
        x,
        norm_weight,
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
        ve,
        ve_gate_weight,
        ve_gate_channels,
        ve_ids,
        ve_weight,
    )


class NormQKVProjection:
    """Compatibility namespace around the `nanoops::norm_qkv_projection_fwd` custom op.

    New code should call `norm_qkv_projection` directly.
    """

    apply = staticmethod(norm_qkv_projection)
