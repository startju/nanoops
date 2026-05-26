"""Triton kernels for nanoops (Tier 3 — opt-in CUDA kernel rewrites).

⚠️  WIP — skip for code review.

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

Known WIP items (do not block on these during review):
  - `value_gate`: kernel implements per-element gate (out shape
    (M, D_v)), but nanchat's ResFormer uses per-head gate (out shape
    (M, n_kv_head)) broadcast across head_dim. Math doesn't match
    nanchat's actual usage — needs a reshape pre/post or a different
    `gate_w` shape to drop in.
  - None of the Triton kernels are wired into `integration.py` yet —
    purely opt-in via direct import + env var.

Parity tests live in tests/test_triton_*.py — 14 tests, all green.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import torch
import torch.nn.functional as F

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
# Fused residual-add + RMSNorm — placed at block boundaries:
#     y_new = norm(x + residual)
#
# Forward: two consecutive elementwise passes that share the same
# (M, D) tile layout, so fusion is a clean win — one HBM read of
# (x, residual), one write of y, plus a residual-stream write of
# summed = x + residual (which the next block needs anyway and the
# backward kernel also consumes).
#
# Backward: two paths, dispatched in FusedAddNorm.backward based on
# whether the inline kernel's per-program tile fits in the 255 fp32
# reg/thread cap (checked via TileConfig.fits_reg_budget):
#   Primary — _fused_add_norm_bwd_inline_kernel: single-pass, full
#     row in one tile, inner reduction computed in registers. Used
#     whenever it fits (covers all nanchat shapes). Wins 1.5–2.4× over
#     the 2-kernel path on common D.
#   Fallback — _fused_add_norm_inner_kernel + _fused_add_norm_bwd_kernel:
#     Flash-Attention style 2-pass (pre-compute inner[m], then 2D-tile
#     grid over (M, D) for d_summed + dnw_partial). Engages when inline
#     would spill (HAS_NW=True at D ≥ 24K).
#   Both paths fold d_summed_external (caller's direct gradient w.r.t.
#   summed) into the kernel's d_summed store, saving an extra Python
#   `+` op + its HBM round-trip.
#
# Cheap, useful at the seam between attn-block output and mlp-block
# norm-input. See FusedAddNorm class docstring + TRITON.md Chapter 2.
# ─────────────────────────────────────────────────────────────────────


class TileConfig(NamedTuple):
    """Output of `_pick_tile_config`. The `fits_reg_budget` property is
    what callers query to decide whether to use this kernel at the
    chosen tile or fall back to a different shape (e.g. a 2-kernel
    D-split path that uses less register space per program)."""

    block_m: int
    num_warps: int
    est_regs_per_thread: int  # n_live_tiles × tile / (nw × 32)

    @property
    def fits_reg_budget(self) -> bool:
        """True if the estimated reg/thread fits within Ampere's
        register file budget. False means the caller should pick a
        different kernel shape instead — the spill cliff is real
        (~10× slowdown verified at regs≈320 in bench).

        Threshold (256) sits 1 above Ampere's 255 hard cap so configs
        whose analytic estimate lands exactly on 256 still pass.
        Justified empirically: the n_live_tiles model slightly
        overestimates real peak register use (Triton's lifetime
        analysis drops tiles after their last consumer), so the
        cluster of common shapes that hit regs=256 in the model
        actually compile without spill (verified: HAS_NW=False up to
        D=32768 — 1.13× win over 2-kernel fallback). The true spill
        cliff sits much higher (~320 regs)."""
        return self.est_regs_per_thread <= 256


def _pick_tile_config(M: int, BLOCK_D: int, n_live_tiles: int) -> TileConfig:
    """Pick (BLOCK_M, num_warps) for a (BLOCK_M × BLOCK_D)-tiled kernel.

    Register-budget model (Ampere, 255 fp32 regs/thread spill cap):

        regs/thread ≈ n_live_tiles × (BLOCK_M × BLOCK_D) / (num_warps × 32)

    Targeting the 256-reg cap:
        tile ≤ 256 × 32 × num_warps / n_live_tiles
             = (8192 / n_live_tiles) × num_warps

    Args:
      M: row count. Caps BLOCK_M so the grid stays ≳ 64 programs
        (saturates RTX 3090's 82 SMs in one wave).
      BLOCK_D: column tile size, typically next_power_of_2(D).
      n_live_tiles: peak number of (BLOCK_M × BLOCK_D) fp32 tiles alive
        simultaneously in the kernel hot path. Examples:
          fwd ≈ 2 (summed_f32, y_f32)
          bwd inline ≈ 5 with affine weight (y_norm, g_eff, dy_t, d_ext,
            d_summed), ≈ 4 without (y_norm/src and g_eff/dy_t alias)
          inner pre-compute ≈ 2 (y_norm, g_eff)

    Returns a `TileConfig`. The caller should inspect `fits_reg_budget`
    when the kernel is known to spill catastrophically at large tiles
    (e.g. the bwd inline kernel at D > 16K with HAS_NW=True) — and
    pick a different code path if it returns False. nw is capped at 16
    regardless of tile size, so for very large BLOCK_D the budget can
    be exceeded even with maxed-out nw.
    """
    tile_per_nw = max(1, 8192 // n_live_tiles)  # 4096 @ n=2, 1638 @ n=5
    base_nw = 4  # initial guess for sizing BLOCK_M; finalized after
    BLOCK_M = max(
        1,
        min(
            triton.next_power_of_2(max(1, M // 64)),
            triton.next_power_of_2(max(1, tile_per_nw * base_nw // BLOCK_D)),
        ),
    )
    tile = BLOCK_M * BLOCK_D
    # Final nw scales up if tile overshoots single-warp budget; cap at 16
    # (above that → ≤1 block/SM, occupancy collapses).
    num_warps = max(4, min(16, triton.next_power_of_2(max(1, tile // tile_per_nw))))
    est_regs = n_live_tiles * tile // (num_warps * 32)
    return TileConfig(BLOCK_M, num_warps, est_regs)


if _HAS_TRITON:

    @triton.jit
    def _fused_add_norm_fwd_kernel(
        x_ptr,
        res_ptr,
        nw_ptr,
        y_ptr,
        summed_ptr,
        rms_inv_ptr,
        M,
        D,
        eps,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_NW: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
    ):
        """y = norm(x + residual) — per-row RMSNorm of the elementwise sum
        (or just norm(x) when HAS_RESIDUAL=False). Stores `summed = x + res`
        when HAS_RESIDUAL=True; when False, res_ptr and summed_ptr are not
        touched and the caller uses x in place of summed downstream.

        Layout assumptions (asserted at autograd boundary):
          - all (M, D) tensors contiguous → row stride = D, col stride = 1
          - BLOCK_D is the smallest power of 2 ≥ D; cols beyond D are
            masked (load contributes 0, store skipped). So D can be e.g.
            1536 with BLOCK_D=2048.

        Precision: computation runs in fp32 throughout to match F.rms_norm
        bit-for-bit on bf16 inputs (PyTorch's F.rms_norm internally promotes
        bf16 → fp32 for the reduction and elementwise scale, then casts back
        — verified empirically). The bf16 form of `summed` is needed only
        briefly to write the caller's residual-stream buffer; everything
        downstream of that store lives in fp32 registers until the final
        y cast at store time. Net register pressure is the same as a
        full-fp32-internal implementation (n_regs=255 at tile=16384,
        nw=4) — bf16 is for HBM/caller dtype compatibility, not for
        register savings. See FusedAddNorm class docstring for the
        spill-aware tile sizing that keeps this within budget.

        HAS_NW=False ⇒ no per-channel affine weight; output is plain
        `summed / RMS(summed)`. nw_ptr is then not dereferenced."""
        pid_m = tl.program_id(0)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_D)
        row_mask = rows < M
        col_mask = cols < D
        mask_2d = row_mask[:, None] & col_mask[None, :]

        offs = rows[:, None] * D + cols[None, :]
        # Load in input dtype; masked-off elements get 0 (additive /
        # multiplicative identity for the downstream sum_sq).
        x = tl.load(x_ptr + offs, mask=mask_2d, other=0.0)
        if HAS_RESIDUAL:
            r = tl.load(res_ptr + offs, mask=mask_2d, other=0.0)
            summed = x + r
            # Caller's next-block residual stream + bwd y_norm reconstruction.
            tl.store(summed_ptr + offs, summed, mask=mask_2d)
        else:
            # summed = x; caller uses x directly downstream (no store).
            summed = x

        # Compute pipeline runs in fp32 (matches F.rms_norm bit-for-bit;
        # see kernel docstring). summed_f32 stays live in registers from
        # here through the y store, so register pressure is fp32-internal
        # in effect — the bf16 `summed` was only kept around for the HBM
        # write above.
        summed_f32 = summed.to(tl.float32)
        sum_sq = tl.sum(summed_f32 * summed_f32, axis=1)
        rms_inv = tl.rsqrt(sum_sq / D + eps)
        tl.store(rms_inv_ptr + rows, rms_inv, mask=row_mask)

        y_f32 = summed_f32 * rms_inv[:, None]
        if HAS_NW:
            nw = tl.load(nw_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
            y_f32 = y_f32 * nw[None, :]
        tl.store(y_ptr + offs, y_f32.to(y_ptr.dtype.element_ty), mask=mask_2d)

    # Primary bwd path: single-kernel, inline inner reduction (no
    # precompute, no D-split). Used when the (BLOCK_M × full-D) tile
    # fits in registers — covers HAS_NW=False at any D and HAS_NW=True
    # up to the size where the (M / BLOCK_M, D) dnw_partials buffer
    # becomes the HBM bottleneck. Above that, FusedAddNorm.backward
    # dispatches to the inner + 2D-tile fallback pair below.
    @triton.jit
    def _fused_add_norm_bwd_inline_kernel(
        ynorm_src_ptr,
        rms_inv_ptr,
        nw_ptr,
        dy_ptr,
        d_ext_ptr,
        d_summed_ptr,
        dnw_partial_ptr,
        M,
        D,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_NW: tl.constexpr,
    ):
        """Single-kernel RMSNorm bwd. `BLOCK_D = next_power_of_2(D)` so
        the full row lives in one tile and `inner = mean_d(g_eff * y_norm)`
        is computed inline (no precompute kernel, no inner HBM buffer).

        Same `ynorm_src_ptr` / `d_ext_ptr` semantics as the 2D-tile
        `_fused_add_norm_bwd_kernel` (below).

        BLOCK_M / num_warps come from `_pick_tile_config(M, BLOCK_D,
        n_live_tiles=N)` — N=5 for HAS_NW=True (y_norm, g_eff, dy_t,
        d_ext, d_summed alive at peak), N=4 for HAS_NW=False (y_norm
        aliases src and g_eff aliases dy_t when there's no per-channel
        weight; d_ext and d_summed still independent).

        dnw_partial layout: `(ceil(M / BLOCK_M), D)` — per-m-tile sum
        of `(dy * y_norm)`; caller does `.sum(dim=0)` to (D,)."""
        pid_m = tl.program_id(0)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_D)
        row_mask = rows < M
        col_mask = cols < D
        mask_2d = row_mask[:, None] & col_mask[None, :]
        offs = rows[:, None] * D + cols[None, :]

        rms_inv = tl.load(rms_inv_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        src = tl.load(ynorm_src_ptr + offs, mask=mask_2d, other=0.0).to(tl.float32)
        dy_t = tl.load(dy_ptr + offs, mask=mask_2d, other=0.0).to(tl.float32)

        if HAS_NW:
            nw = tl.load(nw_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
            y_norm = src * rms_inv[:, None]
            g_eff = dy_t * nw[None, :]
        else:
            y_norm = src
            g_eff = dy_t

        # Inner reduction over the full D (BLOCK_D ≥ D, masked elements
        # contributed 0). Result is (BLOCK_M,) — one scalar per row.
        inner = tl.sum(g_eff * y_norm, axis=1) / D

        d_ext = tl.load(d_ext_ptr + offs, mask=mask_2d, other=0.0)
        d_summed = (rms_inv[:, None] * (g_eff - y_norm * inner[:, None])).to(
            d_summed_ptr.dtype.element_ty
        ) + d_ext
        tl.store(
            d_summed_ptr + offs,
            d_summed,
            mask=mask_2d,
        )

        if HAS_NW:
            dnw_partial = tl.sum(dy_t * y_norm, axis=0)
            dnw_p_ptrs = dnw_partial_ptr + pid_m * D + cols
            tl.store(
                dnw_p_ptrs,
                dnw_partial.to(dnw_partial_ptr.dtype.element_ty),
                mask=col_mask,
            )

    @triton.jit
    def _fused_add_norm_inner_kernel(
        ynorm_src_ptr,
        rms_inv_ptr,
        nw_ptr,
        dy_ptr,
        inner_ptr,
        M,
        D,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_NW: tl.constexpr,
    ):
        """Pre-computes inner[m] = mean_d(g_eff * y_norm) for the bwd kernel,
        so the bwd kernel doesn't redundantly recompute this per d_tile
        (D/BLOCK_D times per m_tile).

        Same `ynorm_src_ptr` semantics as the bwd kernel:
          - HAS_NW=True : ynorm_src is `summed`, reconstruct y_norm = summed * rms_inv
          - HAS_NW=False: ynorm_src is `y` itself which equals y_norm directly

        Grid is 1D over m_tiles — each program does one (BLOCK_M, D)
        row tile reduction. BLOCK_D = next_power_of_2(D) so the whole D
        fits in a single tile column-wise (same pattern as fwd kernel).

        Writes one fp32 per row to `inner_ptr` (shape (M,))."""
        pid_m = tl.program_id(0)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_D)
        row_mask = rows < M
        col_mask = cols < D
        mask_2d = row_mask[:, None] & col_mask[None, :]
        offs = rows[:, None] * D + cols[None, :]

        rms_inv = tl.load(rms_inv_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        src = tl.load(ynorm_src_ptr + offs, mask=mask_2d, other=0.0).to(tl.float32)
        dy_t = tl.load(dy_ptr + offs, mask=mask_2d, other=0.0).to(tl.float32)

        if HAS_NW:
            y_norm = src * rms_inv[:, None]
            nw = tl.load(nw_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
            g_eff = dy_t * nw[None, :]
        else:
            y_norm = src
            g_eff = dy_t

        inner = tl.sum(g_eff * y_norm, axis=1) / D
        tl.store(inner_ptr + rows, inner, mask=row_mask)

    # Fixed-config kernel (no @triton.autotune) for CUDA Graph compatibility.
    # Triton autotune's dispatch path retains some non-capture-friendly
    # operations even with cache hit, so wrapping this kernel inside
    # `torch.cuda.make_graphed_callables` for fwd+bwd fails with
    # `cudaErrorStreamCaptureInvalidated`. Using a fixed config makes
    # the call path purely capture-friendly.
    #
    # Config (BLOCK_M=32, BLOCK_D=64, num_warps=4, num_stages=2) is a
    # conservative middle-ground from the autotune sweep that worked
    # across all nanchat shapes (D ≤ 4096). Caller must pass these as
    # kwargs to keep them visible at the call site rather than buried
    # in autotune state.
    @triton.jit
    def _fused_add_norm_bwd_kernel(
        ynorm_src_ptr,
        rms_inv_ptr,
        nw_ptr,
        dy_ptr,
        d_ext_ptr,
        inner_ptr,
        d_summed_ptr,
        dnw_partial_ptr,
        M,
        D,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_NW: tl.constexpr,
    ):
        """RMSNorm backward second-stage kernel (paired with
        `_fused_add_norm_inner_kernel` which pre-computes inner[m]).

        The `ynorm_src_ptr` tensor is interpreted based on HAS_NW:
          - HAS_NW=True : pointer is to `summed`; kernel reconstructs
            y_norm = summed * rms_inv inline (one multiply per loaded
            tile, runs on idle FP32 cores). Used so fwd doesn't need a
            separate 2·M·D HBM write for y_norm.
          - HAS_NW=False: pointer is to `y` itself, which equals y_norm
            (since `y = summed * rms_inv * 1` when there's no per-channel
            affine). Kernel uses it directly — saves the multiply.

        Grid is 2D (M_tile × D_tile): each program produces one
        (BLOCK_M × BLOCK_D) slice of d_summed and writes one
        (BLOCK_M × BLOCK_D) partial of dnw. Splitting along D keeps
        per-program register pressure manageable (vs a 1D-over-M grid
        which would force the whole D into a single tile and spill).

        `inner_ptr` is the per-row reduction `inner[m] = mean_d(g_eff *
        y_norm)`, pre-computed by `_fused_add_norm_inner_kernel`. This
        avoids each d_tile program recomputing the same full-D reduction
        (would be D/BLOCK_D × redundant otherwise).

        `d_ext_ptr` is the gradient w.r.t. `summed` coming from outside
        (caller's direct consumption of summed as residual stream). The
        kernel adds it to the RMSNorm-bwd-computed gradient in pass 2
        so the final `d_summed` already contains the total — saves an
        extra Python torch elementwise add + its HBM round-trip
        (~50 μs/call at d24 shape).

        Math (when HAS_NW=True; `dy` is gradient of the fwd-output `y`):
          y_norm[m, d]   = summed[m, d] * rms_inv[m]
          g_eff[m, d]    = dy[m, d] * nw[d]
          d_summed[m, d] = rms_inv[m] * (g_eff - y_norm * inner[m]) + d_ext[m, d]
          dnw_partial[m_tile, d] = sum_{m in tile} (dy * y_norm)

        When HAS_NW=False: y_norm[m, d] = ynorm_src[m, d] directly,
        g_eff collapses to dy, dnw_partial is not produced; nw_ptr
        and dnw_partial_ptr are not dereferenced.
        """
        pid_m = tl.program_id(0)
        pid_d = tl.program_id(1)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ds = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        row_mask = rows < M
        d_mask = ds < D

        rms_inv = tl.load(rms_inv_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        if HAS_NW:
            nw = tl.load(nw_ptr + ds, mask=d_mask, other=0.0).to(tl.float32)
        # Pre-computed per-row inner — replaces the previous pass-1
        # full-D reduction loop. One scalar load per row, not redone
        # per d_tile program.
        inner = tl.load(inner_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        # Compute d_summed for this (m_tile, d_tile) slice.
        # All (M, D) tensors are contiguous (autograd boundary ensures
        # this), so row stride = D, col stride = 1 — no per-tensor
        # stride params needed.
        offs = rows[:, None] * D + ds[None, :]
        mask_2d = row_mask[:, None] & d_mask[None, :]
        src = tl.load(ynorm_src_ptr + offs, mask=mask_2d, other=0.0).to(tl.float32)
        if HAS_NW:
            y_norm = src * rms_inv[:, None]
        else:
            y_norm = src
        dy_t = tl.load(dy_ptr + offs, mask=mask_2d, other=0.0).to(tl.float32)
        if HAS_NW:
            g_eff = dy_t * nw[None, :]
        else:
            g_eff = dy_t
        # Fold external d_summed (caller's direct consumption of summed)
        # into d_summed in-kernel — saves an extra Python `+` op + its
        # HBM round-trip outside.
        d_ext = tl.load(d_ext_ptr + offs, mask=mask_2d, other=0.0)
        d_summed_tile = (rms_inv[:, None] * (g_eff - y_norm * inner[:, None])).to(
            d_summed_ptr.dtype.element_ty
        ) + d_ext
        tl.store(
            d_summed_ptr + offs,
            d_summed_tile,
            mask=mask_2d,
        )

        # Per-m-tile partial dnw[d] = sum over tile rows of (dy * y_norm)
        # — only when affine weight exists.
        if HAS_NW:
            dnw_partial = tl.sum(dy_t * y_norm, axis=0)
            dnw_p_ptrs = dnw_partial_ptr + pid_m * D + ds
            tl.store(
                dnw_p_ptrs,
                dnw_partial.to(dnw_partial_ptr.dtype.element_ty),
                mask=d_mask,
            )


def _fused_add_norm_fwd_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward kernel call for FusedAddNorm. Returns (y, summed, rms_inv);
    rms_inv (and one of y/summed as ynorm_src) is saved for backward.

    One Triton kernel does add + RMSNorm and writes y + summed (no separate
    y_norm tensor — it stays in registers during fwd). The kernel's
    constexpr HAS_NW branch skips the per-channel scale entirely when
    norm_weight is None."""
    assert x.is_cuda and x.is_contiguous()
    assert residual.is_cuda and residual.is_contiguous()
    M, D = x.shape
    assert residual.shape == (M, D)
    has_nw = norm_weight is not None
    if has_nw:
        assert norm_weight.is_cuda
        assert norm_weight.shape == (D,)
    y = torch.empty_like(x)
    summed = torch.empty_like(x)
    rms_inv = torch.empty((M,), dtype=torch.float32, device=x.device)

    # Triton's tl.arange requires power-of-2 BLOCK_D; non-pow2 D
    # (e.g. d24's 1536) gets padded and the kernel's col_mask zeroes
    # the trailing lanes so they don't affect the reduction.
    BLOCK_D = triton.next_power_of_2(D)

    # Tile sizing via the shared _pick_tile_config helper. fwd's
    # hot path holds ~2 fp32 tiles alive simultaneously (summed_f32
    # through the reduction; y_f32 during the final multiply). The
    # helper translates that to BLOCK_M and num_warps under the
    # Ampere 255 fp32 reg/thread spill cap (see helper docstring
    # for the formula).
    cfg = _pick_tile_config(M, BLOCK_D, n_live_tiles=2)
    BLOCK_M, num_warps = cfg.block_m, cfg.num_warps

    # HAS_NW=False path doesn't dereference nw_ptr; pass `x` as a
    # valid placeholder pointer (Triton still requires the arg).
    grid = (triton.cdiv(M, BLOCK_M),)
    nw_arg = norm_weight if has_nw else x
    _fused_add_norm_fwd_kernel[grid](
        x,
        residual,
        nw_arg,
        y,
        summed,
        rms_inv,
        M,
        D,
        eps,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        HAS_NW=has_nw,
        HAS_RESIDUAL=True,
        num_warps=num_warps,
    )
    return y, summed, rms_inv


def _fused_add_norm_bwd_impl(
    dy: torch.Tensor,
    d_summed_external: torch.Tensor,
    ynorm_src: torch.Tensor,
    norm_weight: torch.Tensor | None,
    rms_inv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Backward kernel call for FusedAddNorm. Returns (d_summed, dnw);
    caller fans out d_summed → (d_x, d_residual) (both grads alias since
    `summed = x + residual`). dnw is None when norm_weight is None.

    Dispatches between two paths based on inline-tile reg budget:
      (A) inline single-kernel `_fused_add_norm_bwd_inline_kernel` —
          full D in one tile, inner reduction in registers, no precompute.
          Wins 1.5–2.4× on most shapes (kernel-only).
      (B) 2-kernel D-split fallback `_fused_add_norm_inner_kernel` +
          `_fused_add_norm_bwd_kernel` — used when inline tile would
          exceed Ampere's 255 fp32 reg/thread cap (HAS_NW=True at D ≥ 16K).

    Both paths reconstruct y_norm = ynorm_src · rms_inv (when HAS_NW=True;
    when HAS_NW=False ynorm_src IS y_norm). The fwd→bwd save strategy is
    handled by setup_context: ynorm_src = summed when HAS_NW=True, y when
    HAS_NW=False (since y == y_norm in the latter case)."""
    M, D = ynorm_src.shape
    has_nw = norm_weight is not None
    dy = dy.contiguous()
    d_summed_external = d_summed_external.contiguous()
    d_summed = torch.empty_like(ynorm_src)

    # Dispatch between (A) inline and (B) 2-kernel D-split paths.
    # n_live_tiles=5 for HAS_NW=True (y_norm, g_eff, dy_t, d_ext,
    # d_summed alive at peak), 4 for HAS_NW=False (y_norm and g_eff
    # alias src and dy_t respectively when there's no per-channel
    # weight). The `fits_reg_budget` check on the inline TileConfig
    # tells us whether the chosen (BLOCK_M, num_warps) would spill
    # past Ampere's 255 fp32 reg/thread hard cap — if so, fall back to
    # the 2-kernel D-split path which uses a much smaller fixed tile.
    BLOCK_D = triton.next_power_of_2(D)
    inline_n_live = 5 if has_nw else 4
    inline_cfg = _pick_tile_config(M, BLOCK_D, n_live_tiles=inline_n_live)
    use_inline = inline_cfg.fits_reg_budget

    if use_inline:
        BLOCK_M, num_warps = inline_cfg.block_m, inline_cfg.num_warps
        num_m_tiles = triton.cdiv(M, BLOCK_M)

        if has_nw:
            dnw_partials = torch.empty(
                (num_m_tiles, D), dtype=norm_weight.dtype, device=ynorm_src.device
            )
            nw_arg, dnw_arg = norm_weight, dnw_partials
        else:
            nw_arg = dnw_arg = ynorm_src  # dummy ptrs; kernel skips deref

        _fused_add_norm_bwd_inline_kernel[(num_m_tiles,)](
            ynorm_src,
            rms_inv,
            nw_arg,
            dy,
            d_summed_external,
            d_summed,
            dnw_arg,
            M,
            D,
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
            HAS_NW=has_nw,
            num_warps=num_warps,
        )
    else:
        # 2-kernel path (inline would spill). Fixed config for
        # CUDA Graph capturability.
        BLOCK_M_BWD, BLOCK_D_BWD, NUM_WARPS_BWD = 32, 64, 4
        num_m_tiles = triton.cdiv(M, BLOCK_M_BWD)
        inner_buf = torch.empty((M,), dtype=torch.float32, device=ynorm_src.device)
        if has_nw:
            dnw_partials = torch.empty(
                (num_m_tiles, D), dtype=norm_weight.dtype, device=ynorm_src.device
            )
            nw_arg, dnw_arg = norm_weight, dnw_partials
        else:
            nw_arg = dnw_arg = ynorm_src  # dummy ptrs; kernel skips deref

        # Stage 1: pre-compute inner[m]. fwd-style sizing
        # (n_live_tiles=2: y_norm and g_eff alive together briefly).
        INNER_BLOCK_D = triton.next_power_of_2(D)
        inner_cfg = _pick_tile_config(M, INNER_BLOCK_D, n_live_tiles=2)
        _fused_add_norm_inner_kernel[(triton.cdiv(M, inner_cfg.block_m),)](
            ynorm_src,
            rms_inv,
            nw_arg,
            dy,
            inner_buf,
            M,
            D,
            BLOCK_M=inner_cfg.block_m,
            BLOCK_D=INNER_BLOCK_D,
            HAS_NW=has_nw,
            num_warps=inner_cfg.num_warps,
        )

        # Stage 2: bwd reads pre-computed inner; 2D grid splits D.
        _fused_add_norm_bwd_kernel[(num_m_tiles, triton.cdiv(D, BLOCK_D_BWD))](
            ynorm_src,
            rms_inv,
            nw_arg,
            dy,
            d_summed_external,
            inner_buf,
            d_summed,
            dnw_arg,
            M,
            D,
            BLOCK_M=BLOCK_M_BWD,
            BLOCK_D=BLOCK_D_BWD,
            HAS_NW=has_nw,
            num_warps=NUM_WARPS_BWD,
        )

    dnw = dnw_partials.sum(dim=0) if has_nw else None
    return d_summed, dnw


# ── torch.library.custom_op wrapping — opaque to dynamo, same rationale
# as FusedMLPBlock above. Without this, calling fused_add_norm under
# torch.compile would either graph-break (autograd.Function) or attempt
# to trace into the Triton kernels with FakeTensors and crash on
# .data_ptr() (allow_in_graph). custom_op tells dynamo "opaque op,
# here's its shape via register_fake, here's its autograd". ──

@torch.library.custom_op(
    "nanoops::fused_add_norm_fwd",
    mutates_args=(),
    device_types="cuda",
)
def _fused_add_norm_fwd_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _fused_add_norm_fwd_impl(x, residual, norm_weight, eps)


@_fused_add_norm_fwd_op.register_fake
def _fused_add_norm_fwd_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, _D = x.shape
    return (
        torch.empty_like(x),  # y
        torch.empty_like(x),  # summed
        torch.empty((M,), dtype=torch.float32, device=x.device),  # rms_inv
    )


# custom_op return types can't be Optional[Tensor], so we always return
# two tensors and use a 1-elem placeholder for dnw when norm_weight is
# None. The autograd wrapper below substitutes that placeholder back to
# None before returning to autograd (autograd requires None grad for a
# None input).

@torch.library.custom_op(
    "nanoops::fused_add_norm_bwd",
    mutates_args=(),
    device_types="cuda",
)
def _fused_add_norm_bwd_op(
    dy: torch.Tensor,
    d_summed_external: torch.Tensor,
    ynorm_src: torch.Tensor,
    norm_weight: torch.Tensor | None,
    rms_inv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_summed, dnw = _fused_add_norm_bwd_impl(
        dy, d_summed_external, ynorm_src, norm_weight, rms_inv
    )
    if dnw is None:
        dnw = torch.empty(1, dtype=ynorm_src.dtype, device=ynorm_src.device)
    return d_summed, dnw


@_fused_add_norm_bwd_op.register_fake
def _fused_add_norm_bwd_fake(
    dy: torch.Tensor,
    d_summed_external: torch.Tensor,
    ynorm_src: torch.Tensor,
    norm_weight: torch.Tensor | None,
    rms_inv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if norm_weight is not None:
        dnw = torch.empty_like(norm_weight)
    else:
        dnw = torch.empty(1, dtype=ynorm_src.dtype, device=ynorm_src.device)
    return torch.empty_like(ynorm_src), dnw


def _fused_add_norm_setup_context(ctx, inputs, output):
    _x, _residual, norm_weight, _eps = inputs
    y, summed, rms_inv = output
    # ynorm_src is `summed` when norm_weight exists (bwd needs to multiply
    # by rms_inv to reconstruct y_norm), else `y` directly (which equals
    # y_norm in the no-affine case). Saving only one tensor avoids an
    # extra M·D ref-count for the duration of the bwd graph.
    ynorm_src = summed if norm_weight is not None else y
    ctx.save_for_backward(ynorm_src, norm_weight, rms_inv)


def _fused_add_norm_op_backward(ctx, grad_y, grad_summed, grad_rms_inv):
    # grad_rms_inv is always None — no downstream consumer.
    # grad_summed IS the d_summed_external from the residual stream.
    ynorm_src, norm_weight, rms_inv = ctx.saved_tensors
    d_summed, dnw = _fused_add_norm_bwd_op(
        grad_y, grad_summed, ynorm_src, norm_weight, rms_inv
    )
    if norm_weight is None:
        dnw = None
    # d_x = d_residual = d_summed (both alias — autograd accumulates correctly).
    return d_summed, d_summed, dnw, None


_fused_add_norm_fwd_op.register_autograd(
    _fused_add_norm_op_backward,
    setup_context=_fused_add_norm_setup_context,
)


def fused_add_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor | None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused `y = norm(x + residual)`. Also returns `summed = x + residual`
    so the caller can plug it directly into the next block's residual
    stream (no re-add needed).

    Pass `norm_weight=None` for RMSNorm without a per-channel affine
    scale (plain `summed / RMS(summed)`).

    The canonical block-boundary fusion: between an attn block's output
    and the mlp block's norm-input (or symmetrically between mlp output
    and the next layer's attn norm-input).

    Implemented as a `torch.library.custom_op` (with register_fake +
    register_autograd) so torch.compile keeps the op as an opaque FX node
    instead of breaking the graph at the call — see
    `_fused_add_norm_fwd_impl` / `_fused_add_norm_bwd_impl` for the actual
    kernel call sequences."""
    # custom_op returns (y, summed, rms_inv); rms_inv is saved-for-backward
    # only, so we drop it here and return the original (y, summed) shape.
    y, summed, _rms_inv = _fused_add_norm_fwd_op(x, residual, norm_weight, eps)
    return y, summed


# ─────────────────────────────────────────────────────────────────────
# Fused MLP block + outer residual — standard transformer mlp side:
#     y = x + relu(RMSNorm(x) @ W_fc.T)² @ W_proj.T
#
# Per row m, in math:
#     y_norm[m, k] = x[m, k] * rsqrt(mean_k(x[m, k]²) + eps)
#     x_hat[m, k]  = y_norm[m, k] * norm_weight[k]                    (RMSNorm)
#     z[m, n]      = sum_k x_hat[m, k] * W_fc[n, k]                   (Linear: c_fc)
#     r[m, n]      = max(z[m, n], 0)²                                 (ReluSquare)
#     mlp[m, p]    = sum_n r[m, n] * W_proj[p, n]                     (Linear: c_proj)
#     y[m, p]      = x[m, p] + mlp[m, p]                              (Residual add)
#
# See `class FusedMLPBlock` for the kernel breakdown (3-step fwd, 4-step
# all-Triton bwd) and ctx contents. The class docstring is the source of
# truth — do not duplicate the flow description here.
# ─────────────────────────────────────────────────────────────────────


if _HAS_TRITON:
    # c_fc matmul with inline weight cast: z = x @ W_fc.T, but W_fc is
    # loaded in its native dtype (fp32 master) and cast to x's dtype on
    # load — avoids materializing a cast weight tile in HBM. Replaces
    # `torch.matmul(x_hat, fc_weight.t())` in fwd step 1.
    # Trade: lose cuBLAS's tensor-core efficiency (~70% peak) for
    # Triton's (~60% peak), gain 1 launch + ~75 μs HBM round-trip
    # (36 MB write+read at d24).
    # d24 config locked: (BLOCK_M=256, BLOCK_N=64, BLOCK_K=32, nw=8, st=2).
    @triton.jit
    def _cast_matmul_kernel(
        x_ptr,  # (M, K) bf16
        w_ptr,  # (N, K) fp32 (cast to x's dtype on load)
        z_ptr,  # (M, N) bf16 — output
        M,
        N,
        K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        IEEE_PRECISION: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        ms = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = ms < M
        n_mask = ns < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            ks = k_start + tl.arange(0, BLOCK_K)
            k_mask = ks < K
            x_tile = tl.load(
                x_ptr + ms[:, None] * K + ks[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            w_tile = tl.load(
                w_ptr + ns[:, None] * K + ks[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            if IEEE_PRECISION:
                acc += tl.dot(
                    x_tile.to(tl.float32),
                    tl.trans(w_tile.to(tl.float32)),
                    input_precision="ieee",
                )
            else:
                acc += tl.dot(x_tile, tl.trans(w_tile.to(x_tile.dtype)))

        tl.store(
            z_ptr + ms[:, None] * N + ns[None, :],
            acc.to(z_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    # relu² + c_proj + residual_add in one Triton pass — r stays in
    # registers (saves M·N HBM round-trip). Caller passes fixed config
    # (autotune dispatch isn't CUDA Graph capture-friendly); d24 winner
    # locked: (BLOCK_M=128, BLOCK_K_OUT=64, BLOCK_N=32, nw=8, st=2).
    @triton.jit
    def _relu_sq_linear_residual_fwd_kernel(
        z_ptr,
        proj_w_ptr,
        residual_ptr,
        y_ptr,
        M,
        N,
        K_out,
        BLOCK_M: tl.constexpr,
        BLOCK_K_OUT: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IEEE_PRECISION: tl.constexpr,
    ):
        """IEEE_PRECISION: False (bf16 path) → r stays bf16 → Ampere
        bf16 tensor cores. True (fp32 path) → r stays fp32 + pass
        `input_precision="ieee"` to tl.dot, disabling TF32 downcast.
        Slower but bit-tight vs PyTorch `@` for fp32 parity tests."""
        pid_m = tl.program_id(0)
        pid_k_out = tl.program_id(1)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        k_outs = pid_k_out * BLOCK_K_OUT + tl.arange(0, BLOCK_K_OUT)
        row_mask = rows < M
        k_out_mask = k_outs < K_out

        # c_proj matmul, inner loop over N.
        acc = tl.zeros((BLOCK_M, BLOCK_K_OUT), dtype=tl.float32)
        for n_start in range(0, N, BLOCK_N):
            n_cols = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_cols < N

            z = tl.load(
                z_ptr + rows[:, None] * N + n_cols[None, :],
                mask=row_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            relu_z = tl.where(z > 0.0, z, 0.0)
            r = relu_z * relu_z
            # Cast weight to z's dtype on load — handles fp32 master +
            # bf16 activation. `.to(z.dtype)` is a no-op when matched.
            proj_w = tl.load(
                proj_w_ptr + k_outs[:, None] * N + n_cols[None, :],
                mask=k_out_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            if IEEE_PRECISION:
                acc += tl.dot(r, tl.trans(proj_w), input_precision="ieee")
            else:
                acc += tl.dot(r, tl.trans(proj_w).to(z.dtype))

        # Add residual, write y. Keep residual in native dtype; cast the
        # matmul acc to output dtype first, then add — saves a bf16→fp32
        # conversion on the load + skips the final store cast.
        offs = rows[:, None] * K_out + k_outs[None, :]
        mask_2d = row_mask[:, None] & k_out_mask[None, :]
        residual = tl.load(residual_ptr + offs, mask=mask_2d, other=0.0)
        y = acc.to(y_ptr.dtype.element_ty) + residual
        tl.store(y_ptr + offs, y, mask=mask_2d)

    # dz bwd for the c_proj + relu² + residual_add fwd op
    # (`_relu_sq_linear_residual_fwd_kernel`). The other bwd outputs of
    # that fwd are split off:
    #   dW_proj          → `_mlp_dW_proj_bwd_kernel` (different reduction axis)
    #   d_residual = dy  → folded into dx by `_mlp_dx_bwd_kernel`'s outer-residual pass
    # This kernel produces only dz, plus the inner_buf side-output for D.
    #
    # Math:
    #   dr = dy @ W_proj           (matmul; dr kept in registers, never to HBM,
    #                               saves ~50 MB round-trip at d24)
    #   dz = 2·relu(z) · dr        (relu² bwd, applied inline)
    #
    # Side-output: inner_buf[m] += Σ_n(dz·z)/norm_dim via atomic_add (= inner).
    # D uses it via the identity Σ_k(dx_hat·x_hat) = Σ_n(dz·z) to skip its
    # per-row reduction over norm_dim. Division folded in here (we already
    # have K_out, which equals norm_dim in MLP by construction — asserted
    # in forward as K_proj_out == K). Free — dz and z are both already in
    # registers.
    #
    # Why atomic_add (and not a scratchpad + downstream reduce)?
    # Our grid here is (M/BM, N/BN), so the N-axis reduction needed for
    # inner is split across multiple programs per m_tile. None of the
    # downstream kernels (B, C, D) has a free natural N-axis reduction
    # we could piggyback on — D reduces over N for its dx_hat matmul,
    # but its (M/BM, K/BK) grid duplicates each m_tile K/BK times, which
    # would either duplicate the inner work or require inter-program sync.
    # A scratchpad-then-`torch.sum` route works (benched ~equal) but adds
    # an extra (num_n_tiles, M) buffer + a separate reduce launch. Using
    # atomic_add here keeps everything self-contained in this kernel.
    # d24 config locked: (BLOCK_M=128, BLOCK_N=128, BLOCK_K_OUT=32, nw=8, st=2).
    @triton.jit
    def _mlp_dz_bwd_kernel(
        dy_ptr,  # (M, K_out) bf16 — gradient w.r.t. y
        z_ptr,  # (M, N) bf16 — saved from fwd
        proj_w_ptr,  # (K_out, N) fp32 master or bf16 — W_proj (cast to dy.dtype on load)
        dz_ptr,  # (M, N) bf16 — output (gradient w.r.t. z)
        inner_buf_ptr,  # (M,) fp32 — side-output Σ_n(dz·z)/norm_dim; caller zero-inits
        M,
        N,
        K_out,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K_OUT: tl.constexpr,
        IEEE_PRECISION: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < M
        n_mask = n_cols < N

        # dr matmul: dr[m, n] = sum_{kp} dy[m, kp] * W_proj[kp, n]
        dr = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kp_start in range(0, K_out, BLOCK_K_OUT):
            kps = kp_start + tl.arange(0, BLOCK_K_OUT)
            kp_mask = kps < K_out
            dy = tl.load(
                dy_ptr + rows[:, None] * K_out + kps[None, :],
                mask=row_mask[:, None] & kp_mask[None, :],
                other=0.0,
            )
            # Cast fp32 master weight to dy's dtype on load (see Step 2 fwd).
            proj_w = tl.load(
                proj_w_ptr + kps[:, None] * N + n_cols[None, :],
                mask=kp_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            if IEEE_PRECISION:
                dr += tl.dot(
                    dy.to(tl.float32), proj_w.to(tl.float32), input_precision="ieee"
                )
            else:
                dr += tl.dot(dy, proj_w.to(dy.dtype))

        # relu² bwd applied to dr in registers — dr never materialized to HBM.
        z = tl.load(
            z_ptr + rows[:, None] * N + n_cols[None, :],
            mask=row_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        relu_z = tl.where(z > 0.0, z, 0.0)
        dz = dr.to(dz_ptr.dtype.element_ty) * 2 * relu_z

        tl.store(
            dz_ptr + rows[:, None] * N + n_cols[None, :],
            dz,
            mask=row_mask[:, None] & n_mask[None, :],
        )

        # Side-output: partial inner = Σ_n(dz·z) / norm_dim (see kernel header).
        # Force fp32 accumulator — dz and z are bf16 in the bf16 path, and
        # summing BLOCK_N bf16 products in bf16 would lose precision.
        # K_out == norm_dim in MLP, so dividing here saves D a div-on-load.
        inner_partial = tl.sum(dz * z, axis=1, dtype=tl.float32) / K_out
        tl.atomic_add(inner_buf_ptr + rows, inner_partial, mask=row_mask)

    # dW_proj = dy.T @ relu²(z), r recomputed inline (no materialization).
    # Pairs with `_mlp_dz_bwd_kernel` to cover both bwd outputs of the
    # fwd kernel `_relu_sq_linear_residual_fwd_kernel`; reduction axis
    # differs (M here vs K_out there) so they're separate kernels.
    # d24 config locked: (BLOCK_K_OUT=64, BLOCK_N=128, BLOCK_M=64, nw=4, st=2).
    @triton.jit
    def _mlp_dW_proj_bwd_kernel(
        dy_ptr,  # (M, K_out) bf16
        z_ptr,  # (M, N) bf16 — saved from fwd
        dW_proj_ptr,  # (K_out, N) — output (dtype = W_proj.dtype, typically fp32 master)
        M,
        N,
        K_out,
        BLOCK_K_OUT: tl.constexpr,  # output tile along K_out (c_proj's output dim)
        BLOCK_N: tl.constexpr,  # output tile along N (= N_fc)
        BLOCK_M: tl.constexpr,  # reduction tile along M
        IEEE_PRECISION: tl.constexpr,
    ):
        pid_k_out = tl.program_id(0)
        pid_n = tl.program_id(1)

        k_outs = pid_k_out * BLOCK_K_OUT + tl.arange(0, BLOCK_K_OUT)
        ns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_out_mask = k_outs < K_out
        n_mask = ns < N

        acc = tl.zeros((BLOCK_K_OUT, BLOCK_N), dtype=tl.float32)
        for m_start in range(0, M, BLOCK_M):
            ms = m_start + tl.arange(0, BLOCK_M)
            m_mask = ms < M

            # dy[ms, k_outs] — needs transpose for dW_proj = dy.T @ r
            dy = tl.load(
                dy_ptr + ms[:, None] * K_out + k_outs[None, :],
                mask=m_mask[:, None] & k_out_mask[None, :],
                other=0.0,
            )  # (BLOCK_M, BLOCK_K_OUT)

            # Load z and compute r = relu²(z) inline — never to HBM.
            z = tl.load(
                z_ptr + ms[:, None] * N + ns[None, :],
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            relu_z = tl.where(z > 0.0, z, 0.0)
            r = relu_z * relu_z

            # acc[k_out, n] += sum_m dy[m, k_out] * r[m, n]
            if IEEE_PRECISION:
                acc += tl.dot(tl.trans(dy.to(tl.float32)), r, input_precision="ieee")
            else:
                acc += tl.dot(tl.trans(dy), r)

        tl.store(
            dW_proj_ptr + k_outs[:, None] * N + ns[None, :],
            acc.to(dW_proj_ptr.dtype.element_ty),
            mask=k_out_mask[:, None] & n_mask[None, :],
        )

    # dW_fc = dz.T @ x_hat with x_hat reconstructed from (x, rms_inv, nw)
    # inside the GEMM inner loop — no x_hat materialization. Saves
    # M·K HBM write+read + one launch vs the eager (cuBLAS) chain.
    # d24 config locked: (BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, nw=4, st=2).
    # Other shapes prefer (BLOCK_N=128, BLOCK_K=64), within 2% on 3090.
    @triton.jit
    def _mlp_dW_fc_bwd_kernel(
        dz_ptr,  # (M, N_fc) bf16
        x_ptr,  # (M, K) bf16 — fwd input, source for x_hat recompute
        rms_inv_ptr,  # (M,) fp32
        nw_ptr,  # (K,) bf16 — unused when HAS_NW=False (placeholder)
        dW_fc_ptr,  # (N_fc, K) — output (dtype = W_fc.dtype, typically fp32 master)
        M,
        N_fc,
        K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        IEEE_PRECISION: tl.constexpr,
        HAS_NW: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        ns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        n_mask = ns < N_fc
        k_mask = ks < K

        if HAS_NW:
            nw = tl.load(nw_ptr + ks, mask=k_mask, other=0.0)

        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for m_start in range(0, M, BLOCK_M):
            ms = m_start + tl.arange(0, BLOCK_M)
            m_mask = ms < M

            # Reconstruct x_hat tile (BLOCK_M, BLOCK_K) in registers.
            x = tl.load(
                x_ptr + ms[:, None] * K + ks[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            rms_inv = tl.load(rms_inv_ptr + ms, mask=m_mask, other=0.0)
            if HAS_NW:
                x_hat = x * rms_inv[:, None] * nw[None, :]
            else:
                x_hat = x * rms_inv[:, None]

            # dz tile (BLOCK_M, BLOCK_N)
            dz_tile = tl.load(
                dz_ptr + ms[:, None] * N_fc + ns[None, :],
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

            # dW_fc[n, k] += sum_m(dz[m, n] * x_hat[m, k])
            # => acc[n, k] += dz.T[n, m] @ x_hat[m, k]
            if IEEE_PRECISION:
                acc += tl.dot(
                    tl.trans(dz_tile).to(tl.float32), x_hat, input_precision="ieee"
                )
            else:
                acc += tl.dot(tl.trans(dz_tile), x_hat.to(dz_tile.dtype))

        tl.store(
            dW_fc_ptr + ns[:, None] * K + ks[None, :],
            acc.to(dW_fc_ptr.dtype.element_ty),
            mask=n_mask[:, None] & k_mask[None, :],
        )

    # Fused (c_fc bwd dx_hat) + (RMSNorm bwd) — dx_hat = dz @ W_fc computed
    # in-kernel then immediately consumed by the dx formula, never to HBM.
    # (Safe because dW_fc uses x_hat, not dx_hat.) Step A's pre-computed
    # `inner` makes dx purely elementwise — unlocks big tensor-core tiles
    # vs the BLOCK_M=4 a K-reduction would force.
    #
    # Math (per element; K below = norm_dim, the kernel's `K` param):
    #   y_norm   = x · rms_inv
    #   g_eff    = dx_hat · nw                  (= dx_hat if HAS_NW=False)
    #   inner    = inner_buf                    (A already divided by norm_dim)
    #   dx       = rms_inv · (g_eff - y_norm · inner) + dy  (outer residual passthrough)
    #   dnw_partial[m_tile, k] = Σ_{m∈m_tile} (dx_hat · y_norm)  [HAS_NW only]
    @triton.jit
    def _mlp_dx_bwd_kernel(
        dz_ptr,  # (M, N_fc) bf16
        W_fc_ptr,  # (N_fc, K) fp32 master or bf16 — cast to dz.dtype on load
        x_ptr,  # (M, K) bf16 — fwd input, used for y_norm = x·rms_inv
        rms_inv_ptr,  # (M,) fp32
        nw_ptr,  # (K,) bf16 — unused when HAS_NW=False (placeholder)
        dy_ptr,  # (M, K) bf16 — outer residual passthrough, folded into dx
        inner_buf_ptr,  # (M,) fp32 — Σ_n(dz·z) / norm_dim from A (= inner)
        dx_ptr,  # (M, K) bf16 — output
        dnw_partial_ptr,  # (num_m_tiles, K) bf16 — output (only used if HAS_NW)
        M,
        N_fc,
        K,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IEEE_PRECISION: tl.constexpr,
        HAS_NW: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = rows < M
        k_mask = ks < K

        # Matmul: dx_hat_tile = dz @ W_fc[:, k_tile], reduce over N_fc.
        dx_hat = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for n_start in range(0, N_fc, BLOCK_N):
            ns = n_start + tl.arange(0, BLOCK_N)
            n_mask = ns < N_fc
            dz_tile = tl.load(
                dz_ptr + rows[:, None] * N_fc + ns[None, :],
                mask=row_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            # Cast fp32 master weight to dz's dtype on load.
            W_fc_tile = tl.load(
                W_fc_ptr + ns[:, None] * K + ks[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            if IEEE_PRECISION:
                dx_hat += tl.dot(
                    dz_tile.to(tl.float32),
                    W_fc_tile.to(tl.float32),
                    input_precision="ieee",
                )
            else:
                dx_hat += tl.dot(dz_tile, W_fc_tile.to(dz_tile.dtype))

        # Apply RMSNorm bwd inline using pre-computed inner; fold outer
        # residual gradient (dy) into dx in-kernel.
        offs = rows[:, None] * K + ks[None, :]
        mask_2d = row_mask[:, None] & k_mask[None, :]
        rms_inv = tl.load(rms_inv_ptr + rows, mask=row_mask, other=0.0)
        # inner = (1/norm_dim) · Σ_n(dz·z); A already did the division.
        inner = tl.load(inner_buf_ptr + rows, mask=row_mask, other=0.0)
        x = tl.load(x_ptr + offs, mask=mask_2d, other=0.0)
        # dy kept in native dtype — cast the fp32 norm-path result to
        # output dtype first, then add dy. Saves a bf16→fp32 conversion
        # on dy load + skips the final cast on store (small but free).
        dy = tl.load(dy_ptr + offs, mask=mask_2d, other=0.0)

        y_norm = x * rms_inv[:, None]
        if HAS_NW:
            nw = tl.load(nw_ptr + ks, mask=k_mask, other=0.0)
            g_eff = dx_hat * nw[None, :]
        else:
            g_eff = dx_hat
        dx = (rms_inv[:, None] * (g_eff - y_norm * inner[:, None])).to(
            dx_ptr.dtype.element_ty
        ) + dy

        tl.store(dx_ptr + offs, dx, mask=mask_2d)

        # dnw_partial per (m_tile, k_tile) — only when norm_w exists.
        if HAS_NW:
            dnw_partial = tl.sum(dx_hat * y_norm, axis=0)
            tl.store(
                dnw_partial_ptr + pid_m * K + ks,
                dnw_partial.to(dnw_partial_ptr.dtype.element_ty),
                mask=k_mask,
            )


def _fused_mlp_block_fwd_impl(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    fc_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward kernel sequence for FusedMLPBlock. Returns (y, rms_inv, z);
    rms_inv and z are saved for the backward pass.

    Standard transformer mlp sub-block + outer residual:
        x_hat = norm(x) * norm_weight              (RMSNorm)
        z     = x_hat @ W_fc.T                      (Linear: c_fc)
        r     = relu(z)²                            (ReluSquare)
        mlp   = r @ W_proj.T                        (Linear: c_proj)
        y     = x + mlp                             (Residual add)

    Three steps:
      0. `_fused_add_norm_fwd_kernel` with HAS_RESIDUAL=False — produces
         x_hat + rms_inv (the add path is skipped; kernel reused for its
         rms_inv side-output).
      1. `_cast_matmul_kernel` — z = x_hat @ W_fc.T with the fp32→x.dtype
         weight cast fused into the matmul load (no bf16 weight tile to
         HBM).
      2. `_relu_sq_linear_residual_fwd_kernel` — relu² + c_proj +
         residual_add(x) → y in one Triton pass."""
    assert x.is_cuda and x.is_contiguous()
    assert fc_weight.is_cuda and proj_weight.is_cuda
    has_nw = norm_weight is not None
    if has_nw:
        assert norm_weight.is_cuda
    M, K = x.shape
    N_fc, K_w = fc_weight.shape
    K_proj_out, N_proj_in = proj_weight.shape
    assert K == K_w, f"x last dim {K} != fc_weight in dim {K_w}"
    assert N_fc == N_proj_in, f"fc out dim {N_fc} != proj in dim {N_proj_in}"
    assert K_proj_out == K, (
        f"c_proj out dim {K_proj_out} must equal x's K {K} (residual stream width)"
    )
    ieee = x.dtype == torch.float32

    # Step 0: RMSNorm. Reuse `_fused_add_norm_fwd_kernel` with
    # HAS_RESIDUAL=False (no plain-norm variant exists, and we need
    # this kernel's rms_inv side-output for bwd). The kernel skips
    # the residual load + summed store; x is passed as a placeholder
    # for res_ptr/summed_ptr (untouched). Step 2 / bwd consume x
    # directly as the residual stream.
    BLOCK_D_NORM = triton.next_power_of_2(K)
    norm_cfg = _pick_tile_config(M, BLOCK_D_NORM, n_live_tiles=2)
    x_hat = torch.empty_like(x)
    rms_inv = torch.empty((M,), dtype=torch.float32, device=x.device)
    nw_arg = norm_weight if has_nw else x  # placeholder when HAS_NW=False
    _fused_add_norm_fwd_kernel[(triton.cdiv(M, norm_cfg.block_m),)](
        x,
        x,
        nw_arg,
        x_hat,
        x,
        rms_inv,
        M,
        K,
        eps,
        BLOCK_M=norm_cfg.block_m,
        BLOCK_D=BLOCK_D_NORM,
        HAS_NW=has_nw,
        HAS_RESIDUAL=False,
        num_warps=norm_cfg.num_warps,
    )

    # Step 1: c_fc via `_cast_matmul_kernel` — Triton matmul that
    # loads fc_weight (fp32 master typical) in native dtype and casts
    # inline. Replaces `torch.matmul(x_hat, fc_weight.t())` to avoid
    # materializing a bf16 cast weight tile.
    # d24 manual sweep winner (bf16): (BM=256, BN=64, BK=32, nw=8, st=2)
    # gives 639 us — ~2× faster than (64,64,64,nw=4,st=3) and slightly
    # beats cuBLAS+cast (654 us effective). Per-stage shared mem in
    # fp32 IEEE path: (256·32 + 64·32)·4 = 40 KB; ×2 stages = 80 KB,
    # within 100 KB budget, so safe for parity tests too.
    BLOCK_M_S1, BLOCK_N_S1, BLOCK_K_S1 = 256, 64, 32
    z = torch.empty((M, N_fc), dtype=x.dtype, device=x.device)
    grid_s1 = (triton.cdiv(M, BLOCK_M_S1), triton.cdiv(N_fc, BLOCK_N_S1))
    _cast_matmul_kernel[grid_s1](
        x_hat,
        fc_weight,
        z,
        M,
        N_fc,
        K,
        BLOCK_M=BLOCK_M_S1,
        BLOCK_N=BLOCK_N_S1,
        BLOCK_K=BLOCK_K_S1,
        IEEE_PRECISION=ieee,
        num_warps=8,
        num_stages=2,
    )

    # Step 2: Triton kernel for relu² + c_proj + outer residual (= x).
    # d24 sweep winner: (BM=128, BKO=64, BN=32, nw=8, st=2) at 737 us
    # (vs (64,128,32,4,3) 750 us). Marginal but free.
    BLOCK_M_FWD, BLOCK_K_OUT_FWD, BLOCK_N_FWD = 128, 64, 32
    y = torch.empty((M, K_proj_out), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(M, BLOCK_M_FWD), triton.cdiv(K_proj_out, BLOCK_K_OUT_FWD))
    _relu_sq_linear_residual_fwd_kernel[grid](
        z,
        proj_weight,
        x,
        y,
        M,
        N_fc,
        K_proj_out,
        BLOCK_M=BLOCK_M_FWD,
        BLOCK_K_OUT=BLOCK_K_OUT_FWD,
        BLOCK_N=BLOCK_N_FWD,
        IEEE_PRECISION=ieee,
        num_warps=8,
        num_stages=2,
    )
    return y, rms_inv, z


def _fused_mlp_block_bwd_impl(
    dy: torch.Tensor,
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    fc_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    rms_inv: torch.Tensor,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Backward kernel sequence for FusedMLPBlock. Returns
    (dx, dnw, dW_fc, dW_proj); dnw is None when norm_weight is None.

    Four Triton kernels (no cuBLAS):
      A. `_mlp_dz_bwd_kernel` — dz = 2·relu(z) · (dy @ W_proj). Side-output
         `inner_buf[m] = Σ_n(dz·z) / norm_dim` via atomic_add (free —
         dz/z in registers; division folded in since K_out == norm_dim in
         MLP). Step D uses it via the identity Σ_k(dx_hat·x_hat) =
         Σ_n(dz·z) to skip its K-reduction.
      B. `_mlp_dW_proj_bwd_kernel` — dW_proj = dy.T @ relu²(z) (r
         recomputed inline; output dtype = W_proj.dtype master).
      C. `_mlp_dW_fc_bwd_kernel` — dW_fc = dz.T @ x_hat (x_hat
         reconstructed from x, rms_inv, norm_w inline; output dtype =
         W_fc.dtype master).
      D. `_mlp_dx_bwd_kernel` — dx (and dnw_partial) in one fused pass.
         x has TWO gradient paths in `y = x + mlp(norm(x))`:
           (i)  outer-residual: dx ← dy   (direct passthrough)
           (ii) norm path:      dx ← RMSNorm_bwd(dx_hat),
                                dx_hat ← dz @ W_fc
         RMSNorm_bwd math (per element):
             g_eff = dx_hat · nw    (= dx_hat if no affine)
             dx_path_ii = rms_inv · (g_eff - y_norm · inner)
         where inner is the one A pre-divided by norm_dim. dx_hat is
         never written to HBM; (i)'s `+ dy` is folded into the store.
         Safe because kernel C uses x_hat, not dx_hat."""
    M, K = x.shape
    N_fc = fc_weight.shape[0]
    K_out = proj_weight.shape[0]
    dy = dy.contiguous()
    ieee = x.dtype == torch.float32
    has_nw = norm_weight is not None
    nw_arg = norm_weight if has_nw else x  # placeholder when HAS_NW=False

    # A: dz + inner_buf side-output via atomic_add.
    # d24 sweep winner: (BM=128, BN=128, BKO=32, nw=8, st=2) at 721 us
    # (vs prior (64,128,32,8,3) 800 us — ~10% speedup).
    BLOCK_M_A, BLOCK_N_A, BLOCK_K_OUT_A = 128, 128, 32
    dz = torch.empty_like(z)
    inner_buf = torch.zeros((M,), dtype=torch.float32, device=z.device)
    grid_a = (triton.cdiv(M, BLOCK_M_A), triton.cdiv(N_fc, BLOCK_N_A))
    _mlp_dz_bwd_kernel[grid_a](
        dy,
        z,
        proj_weight,
        dz,
        inner_buf,
        M,
        N_fc,
        K_out,
        BLOCK_M=BLOCK_M_A,
        BLOCK_N=BLOCK_N_A,
        BLOCK_K_OUT=BLOCK_K_OUT_A,
        IEEE_PRECISION=ieee,
        num_warps=8,
        num_stages=2,
    )

    # B: dW_proj = dy.T @ relu²(z), r recomputed inline.
    # Output dtype = proj_weight.dtype (fp32 master typical), so the
    # gradient lands directly on the master weight without a downstream
    # .to() cast. d24 sweep winner: (BKO=64, BN=128, BM=64, nw=4, st=2)
    # at 590 us.
    BLOCK_K_OUT_B, BLOCK_N_B, BLOCK_M_B = 64, 128, 64
    dW_proj = torch.empty((K_out, N_fc), dtype=proj_weight.dtype, device=z.device)
    grid_b = (triton.cdiv(K_out, BLOCK_K_OUT_B), triton.cdiv(N_fc, BLOCK_N_B))
    _mlp_dW_proj_bwd_kernel[grid_b](
        dy,
        z,
        dW_proj,
        M,
        N_fc,
        K_out,
        BLOCK_K_OUT=BLOCK_K_OUT_B,
        BLOCK_N=BLOCK_N_B,
        BLOCK_M=BLOCK_M_B,
        IEEE_PRECISION=ieee,
        num_warps=4,
        num_stages=2,
    )

    # C: dW_fc = dz.T @ x_hat, x_hat reconstructed in registers from
    # (x, rms_inv, norm_w) — no x_hat HBM materialization. Output dtype
    # = fc_weight.dtype (fp32 master typical).
    BLOCK_M_C, BLOCK_N_C, BLOCK_K_C = 64, 64, 128
    dW_fc = torch.empty((N_fc, K), dtype=fc_weight.dtype, device=x.device)
    grid_c = (triton.cdiv(N_fc, BLOCK_N_C), triton.cdiv(K, BLOCK_K_C))
    _mlp_dW_fc_bwd_kernel[grid_c](
        dz,
        x,
        rms_inv,
        nw_arg,
        dW_fc,
        M,
        N_fc,
        K,
        BLOCK_M=BLOCK_M_C,
        BLOCK_N=BLOCK_N_C,
        BLOCK_K=BLOCK_K_C,
        IEEE_PRECISION=ieee,
        HAS_NW=has_nw,
        num_warps=4,
        num_stages=2,
    )

    # D: dx_hat matmul + RMSNorm bwd + outer-residual fold in one kernel.
    # d24 bf16 sweep winner: (BK=64, BN=128, nw=8, st=2) at 1043 us
    # (vs (64,64,nw=4,st=3) 1456 us — 28% speedup). For fp32 IEEE path
    # that config would exceed 100 KB SM shared-mem budget
    # ((64·128+128·64)·4 = 64 KB/stage × 2 = 128 KB), so use a safer
    # config for parity tests. BLOCK_M=64 fixed (dnw_partials shape
    # depends on it).
    BLOCK_M_D = 64
    if ieee:
        BLOCK_K_D, BLOCK_N_D, NW_D, ST_D = 64, 64, 8, 3  # fp32-safe
    else:
        BLOCK_K_D, BLOCK_N_D, NW_D, ST_D = 64, 128, 8, 2  # bf16 winner
    num_m_tiles = triton.cdiv(M, BLOCK_M_D)
    dx = torch.empty_like(x)
    if has_nw:
        dnw_partials = torch.empty(
            (num_m_tiles, K),
            dtype=norm_weight.dtype,
            device=x.device,
        )
    else:
        # 1-elem placeholder — kernel won't touch it when HAS_NW=False.
        dnw_partials = torch.empty(1, dtype=x.dtype, device=x.device)
    grid_d = (num_m_tiles, triton.cdiv(K, BLOCK_K_D))
    _mlp_dx_bwd_kernel[grid_d](
        dz,
        fc_weight,
        x,
        rms_inv,
        nw_arg,
        dy,
        inner_buf,
        dx,
        dnw_partials,
        M,
        N_fc,
        K,
        BLOCK_M=BLOCK_M_D,
        BLOCK_K=BLOCK_K_D,
        BLOCK_N=BLOCK_N_D,
        IEEE_PRECISION=ieee,
        HAS_NW=has_nw,
        num_warps=NW_D,
        num_stages=ST_D,
    )
    dnw = dnw_partials.sum(dim=0) if has_nw else None
    return dx, dnw, dW_fc, dW_proj


# ── torch.library.custom_op wrapping — opaque FX node under torch.compile ──
# Without this, the fused_mlp_block call would be either (a) a plain
# autograd.Function.apply (graph-break under torch.compile, killing
# cross-op fusion across the mlp boundary) or (b) a function dynamo
# tries to trace into (which would hit `.data_ptr()` on FakeTensor and
# explode, since we launch Triton kernels). custom_op + register_fake +
# register_autograd tells dynamo: "this is an opaque op, here's how its
# shape/dtype works for tracing, here's how its autograd works." Inductor
# keeps fusing on both sides of the call.

@torch.library.custom_op(
    "nanoops::fused_mlp_block_fwd",
    mutates_args=(),
    device_types="cuda",
)
def _fused_mlp_block_fwd_op(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    fc_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _fused_mlp_block_fwd_impl(x, norm_weight, fc_weight, proj_weight, eps)


@_fused_mlp_block_fwd_op.register_fake
def _fused_mlp_block_fwd_fake(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    fc_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Shape/dtype inference for dynamo — must mirror _fwd_impl's outputs.
    M, _K = x.shape
    N_fc = fc_weight.shape[0]
    K_proj_out = proj_weight.shape[0]
    return (
        torch.empty((M, K_proj_out), dtype=x.dtype, device=x.device),
        torch.empty((M,), dtype=torch.float32, device=x.device),
        torch.empty((M, N_fc), dtype=x.dtype, device=x.device),
    )


# The backward needs to be a custom_op too — otherwise dynamo traces
# into _fused_mlp_block_bwd_impl during the compiled backward pass and
# hits `.data_ptr()` on FakeTensors when launching the Triton kernels.

# custom_op return types can't be Optional[Tensor], so we always return
# 4 tensors and use a 1-elem placeholder for dnw when norm_weight is None.
# The autograd-side wrapper below substitutes that placeholder back to None
# (autograd convention: gradient for None input must be None).

@torch.library.custom_op(
    "nanoops::fused_mlp_block_bwd",
    mutates_args=(),
    device_types="cuda",
)
def _fused_mlp_block_bwd_op(
    dy: torch.Tensor,
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    fc_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    rms_inv: torch.Tensor,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dx, dnw, dW_fc, dW_proj = _fused_mlp_block_bwd_impl(
        dy, x, norm_weight, fc_weight, proj_weight, rms_inv, z
    )
    if dnw is None:
        dnw = torch.empty(1, dtype=x.dtype, device=x.device)  # placeholder
    return dx, dnw, dW_fc, dW_proj


@_fused_mlp_block_bwd_op.register_fake
def _fused_mlp_block_bwd_fake(
    dy: torch.Tensor,
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    fc_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    rms_inv: torch.Tensor,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N_fc, K = fc_weight.shape
    if norm_weight is not None:
        dnw = torch.empty_like(norm_weight)
    else:
        dnw = torch.empty(1, dtype=x.dtype, device=x.device)
    return (
        torch.empty_like(x),
        dnw,
        torch.empty((N_fc, K), dtype=fc_weight.dtype, device=x.device),
        torch.empty_like(proj_weight),
    )


def _fused_mlp_block_setup_context(ctx, inputs, output):
    _x, norm_weight, fc_weight, proj_weight, _eps = inputs
    _y, rms_inv, z = output
    # Save the same tensors the eager class did, plus the inputs we need
    # again in bwd. x is an input so we save it via save_for_backward too.
    x = inputs[0]
    ctx.save_for_backward(norm_weight, fc_weight, proj_weight, x, rms_inv, z)


def _fused_mlp_block_op_backward(ctx, grad_y, grad_rms_inv, grad_z):
    # grad_rms_inv / grad_z are unused: rms_inv and z exist only to
    # plumb fwd→bwd state inside this op, no downstream consumer.
    norm_w, W_fc, W_proj, x, rms_inv, z = ctx.saved_tensors
    dx, dnw, dW_fc, dW_proj = _fused_mlp_block_bwd_op(
        grad_y, x, norm_w, W_fc, W_proj, rms_inv, z
    )
    # When norm_w was None, _bwd_op returns a placeholder dnw — substitute
    # back to None so autograd sees the correct "no-grad-for-this-input".
    if norm_w is None:
        dnw = None
    # 5 inputs → 5 grads. eps is a Python float, no grad.
    return dx, dnw, dW_fc, dW_proj, None


_fused_mlp_block_fwd_op.register_autograd(
    _fused_mlp_block_op_backward,
    setup_context=_fused_mlp_block_setup_context,
)


def fused_mlp_block(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    fc_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Full MLP sub-block + residual add with pre-norm:
        y = x + relu²(norm(x)·norm_weight @ W_fc.T) @ W_proj.T

    Standard transformer mlp side: `y = x + mlp(norm(x))`. If the caller
    needs to pre-sum with an attention residual, do it outside.
    `norm_weight=None` ⇒ plain RMSNorm without the per-channel affine.

    fc_weight / proj_weight are loaded in their native dtype inside the
    fwd/bwd Triton kernels and cast inline to the activation dtype before
    each tensor-core matmul (handles the fp32-master + bf16-activation
    case typical in nanchat). dW_fc / dW_proj are allocated with the
    master weight's dtype, so the gradient lands directly on the master
    weight — no wrapper-level `.to()` and no autograd routing needed.

    Implemented as a `torch.library.custom_op` (with register_fake +
    register_autograd) so torch.compile keeps the op as an opaque FX
    node instead of breaking the graph at the call. See the impl helpers
    `_fused_mlp_block_fwd_impl` / `_fused_mlp_block_bwd_impl` for the
    actual kernel call sequences."""
    # custom_op returns (y, rms_inv, z); rms_inv/z are saved-for-backward
    # only, so we drop them here and return just y.
    y, _rms_inv, _z = _fused_mlp_block_fwd_op(
        x, norm_weight, fc_weight, proj_weight, eps
    )
    return y


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
        x_ptr,
        rms_inv_ptr,
        nw_ptr,
        dxhat_ptr,
        dx_ptr,
        dnw_partial_ptr,
        M,
        K,
        stride_xm,
        stride_xk,
        stride_dxhat_m,
        stride_dxhat_k,
        stride_dx_m,
        stride_dx_k,
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
        x_ptr,
        norm_w_ptr,
        qkv_w_ptr,
        out_ptr,
        rms_inv_ptr,
        M,
        N_qkv,
        K,
        eps,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
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


class NormQKVProjection(torch.autograd.Function):
    """Fused RMSNorm + concatenated Q/K/V linear projection.

    Forward: one Triton kernel folds the outer RMSNorm and all three
    QKV matmuls together, writing the concatenated output (M, N_q+N_k+N_v).
    Caller slices it into q, k, v.

    Backward: three linear backwards via cuBLAS + one RMSNorm-backward
    Triton kernel (`_rms_norm_bwd_kernel`, defined above in this section).
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
    def backward(ctx, d_out):
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
        Q,
        K,
        V,
        sm_scale,
        LSE,
        O,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        stride_lb,
        stride_lh,
        stride_lm,
        B,
        H,
        M,
        N,
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
            k_ptrs = (
                K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            )
            v_ptrs = (
                V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
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

        o_ptrs = O + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=m_mask[:, None])

        lse_ptrs = LSE + lse_off + offs_m * stride_lm
        tl.store(lse_ptrs, lse, mask=m_mask)

    @triton.jit
    def _flash_attn_bwd_preprocess_kernel(
        O,
        dO,
        D,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        stride_db,
        stride_dh,
        stride_dm,
        B,
        H,
        M,
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
        Q,
        K,
        V,
        sm_scale,
        LSE,
        D,
        dO,
        dQ,
        dK,
        dV,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        stride_lb,
        stride_lh,
        stride_lm,
        stride_db,
        stride_dh,
        stride_dm,
        B,
        H,
        M,
        N,
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

        q_off = bid * stride_qb + hid * stride_qh
        k_off = bid * stride_kb + hid * stride_kh
        v_off = bid * stride_vb + hid * stride_vh

        # Load Q tile, dO tile, LSE, D for this Q range
        q_ptrs = Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        do_ptrs = (
            dO
            + bid * stride_ob
            + hid * stride_oh
            + offs_m[:, None] * stride_om
            + offs_d[None, :] * stride_od
        )
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

            k_ptrs = (
                K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            )
            v_ptrs = (
                V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
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
                dK + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            )
            dv_ptrs = (
                dV + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            )
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
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
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
            o,
            do,
            d,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            d.stride(0),
            d.stride(1),
            d.stride(2),
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
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            d.stride(0),
            d.stride(1),
            d.stride(2),
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
        v_ptr,
        ve_ptr,
        x_ptr,
        gate_w_ptr,
        out_ptr,
        M,
        D_x,
        D_v,
        ve_gate_ch,
        stride_vm,
        stride_vd,
        stride_vem,
        stride_ved,
        stride_xm,
        stride_xd,
        stride_gw_d_out,
        stride_gw_d_in,
        stride_om,
        stride_od,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
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
                x_ptrs,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            gw_ptrs = (
                gate_w_ptr
                + cols[:, None] * stride_gw_d_out
                + ks[None, :] * stride_gw_d_in
            )
            gw = tl.load(
                gw_ptrs,
                mask=col_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            gate_acc += tl.dot(x_chunk, tl.trans(gw), input_precision="ieee")
        gate = 3.0 * tl.sigmoid(gate_acc)

        # Load v, ve, compute out = v + gate * ve
        v_ptrs = v_ptr + rows[:, None] * stride_vm + cols[None, :] * stride_vd
        ve_ptrs = ve_ptr + rows[:, None] * stride_vem + cols[None, :] * stride_ved
        v = tl.load(v_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(
            tl.float32
        )
        ve = tl.load(ve_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(
            tl.float32
        )
        out = v + gate * ve

        o_ptrs = out_ptr + rows[:, None] * stride_om + cols[None, :] * stride_od
        tl.store(
            o_ptrs,
            out.to(out_ptr.dtype.element_ty),
            mask=row_mask[:, None] & col_mask[None, :],
        )

    @triton.jit
    def _rotary_qk_norm_scale_kernel(
        qk_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        rms_inv_ptr,
        M,
        D,
        scale,
        eps,
        stride_qm,
        stride_qd,
        stride_om,
        stride_od,
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

    @triton.jit
    def _output_proj_residual_kernel(
        attn_out_ptr,
        proj_w_ptr,
        residual_ptr,
        y_ptr,
        M,
        D_out,
        D_in,
        stride_am,
        stride_ad,
        stride_pw_dout,
        stride_pw_din,
        stride_rm,
        stride_rd,
        stride_ym,
        stride_yd,
        BLOCK_M: tl.constexpr,
        BLOCK_DOUT: tl.constexpr,
        BLOCK_DIN: tl.constexpr,
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
        out_mask = row_mask[:, None] & col_mask[None, :]

        # Matmul-accumulate
        acc = tl.zeros((BLOCK_M, BLOCK_DOUT), dtype=tl.float32)
        for k_start in range(0, D_in, BLOCK_DIN):
            ks = k_start + tl.arange(0, BLOCK_DIN)
            k_mask = ks < D_in
            a_ptrs = attn_out_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ad
            a = tl.load(
                a_ptrs,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            pw_ptrs = (
                proj_w_ptr
                + cols[:, None] * stride_pw_dout
                + ks[None, :] * stride_pw_din
            )
            pw = tl.load(
                pw_ptrs,
                mask=col_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(a, tl.trans(pw), input_precision="ieee")

        # Add residual at the end in native dtype (saves a bf16→fp32
        # conversion on residual load + skips the final store cast).
        res_ptrs = residual_ptr + rows[:, None] * stride_rm + cols[None, :] * stride_rd
        residual = tl.load(res_ptrs, mask=out_mask, other=0.0)
        y = acc.to(y_ptr.dtype.element_ty) + residual

        y_ptrs = y_ptr + rows[:, None] * stride_ym + cols[None, :] * stride_yd
        tl.store(y_ptrs, y, mask=out_mask)


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
            attn_out,
            proj_weight,
            residual,
            y,
            M,
            D_out,
            D_in,
            attn_out.stride(0),
            attn_out.stride(1),
            proj_weight.stride(0),
            proj_weight.stride(1),
            residual.stride(0),
            residual.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_DOUT=BLOCK_DOUT,
            BLOCK_DIN=BLOCK_DIN,
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
            v,
            ve,
            x_in,
            gate_w,
            out,
            M,
            x.shape[1],
            D_v,
            ve_gate_ch,
            v.stride(0),
            v.stride(1),
            ve.stride(0),
            ve.stride(1),
            x_in.stride(0),
            x_in.stride(1),
            gate_w.stride(0),
            gate_w.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
            BLOCK_K=BLOCK_K,
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
    v: torch.Tensor,
    ve: torch.Tensor,
    x: torch.Tensor,
    gate_w: torch.Tensor,
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
