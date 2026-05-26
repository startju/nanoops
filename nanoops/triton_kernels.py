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

The actual code lives in three feature-split sibling modules:
  - `triton_fused_add_norm.py` — FusedAddNorm + shared TileConfig helper
  - `triton_fused_mlp_block.py` — FusedMLPBlock (reuses Step 0 kernel
    from triton_fused_add_norm)
  - `triton_attn.py` — the 5 attention-side classes (NormQKVProjection,
    FlashSDPA, OutputProjResidual, ValueGate, RotaryQKNormScale)

This module is a thin re-export shim so existing
`from nanoops.triton_kernels import …` callers keep working unchanged.

Parity tests live in tests/test_triton_*.py.
"""

from __future__ import annotations

# ── FusedAddNorm + shared utilities ─────────────────────────────────
from .triton_fused_add_norm import (
    NORM_MLP_ENABLED,
    TileConfig,
    _fused_add_norm_bwd_impl,
    _fused_add_norm_bwd_op,
    _fused_add_norm_fwd_impl,
    _fused_add_norm_fwd_op,
    _pick_tile_config,
    fused_add_norm,
)

# ── FusedMLPBlock ───────────────────────────────────────────────────
from .triton_fused_mlp_block import (
    _fused_mlp_block_bwd_impl,
    _fused_mlp_block_bwd_op,
    _fused_mlp_block_fwd_impl,
    _fused_mlp_block_fwd_op,
    fused_mlp_block,
)

# ── Attention-side kernels ──────────────────────────────────────────
from .triton_attn import (
    FlashSDPA,
    NormQKVProjection,
    OutputProjResidual,
    RotaryQKNormScale,
    ValueGate,
    flash_sdpa,
    norm_qkv_projection,
    output_proj_residual,
    rotary_qk_norm_scale,
    value_gate,
)

# Private @triton.jit kernels — re-exported for direct benchmark scripts
# (/tmp/sweep_*.py, /tmp/bench_*.py) that import them by name. Guarded
# because they only exist when triton is installed; on a no-triton
# environment NORM_MLP_ENABLED is False and these names just aren't
# bound — same behavior as the pre-split file.
try:
    from .triton_fused_add_norm import (
        _fused_add_norm_bwd_inline_kernel,
        _fused_add_norm_bwd_kernel,
        _fused_add_norm_fwd_kernel,
        _fused_add_norm_inner_kernel,
    )
    from .triton_fused_mlp_block import (
        _cast_matmul_kernel,
        _mlp_dW_fc_bwd_kernel,
        _mlp_dW_proj_bwd_kernel,
        _mlp_dx_bwd_kernel,
        _mlp_dz_bwd_kernel,
        _relu_sq_linear_residual_fwd_kernel,
    )
except ImportError:
    pass
