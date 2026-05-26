"""FusedAddNorm Triton kernels + shared TileConfig sizing utility.

`y = norm(x + residual)` with `summed = x + residual` returned as a side
output for the next block's residual stream. See `fused_add_norm` and
TRITON_zh.md Chapter 2.

Also hosts the shared `TileConfig` / `_pick_tile_config` helper used by
this file and `triton_fused_mlp_block.py` (which reuses
`_fused_add_norm_fwd_kernel` for its Step 0 RMSNorm). Other Triton
modules in this package (`triton_fused_mlp_block`, `triton_attn`)
import what they need from here.

Re-exported through `nanoops.triton_kernels` for backward-compat callers.
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


