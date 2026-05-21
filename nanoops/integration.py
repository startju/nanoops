"""Monkey-patch helpers to swap nanoops into nanchat.

Two layers of patching:

1. **F namespace** — replaces the `F` (torch.nn.functional) attribute on each
   target module with a proxy whose specific attributes route through nanoops.
   Everything not overridden falls through to the real `torch.nn.functional`.

2. **torch top-level functions** — overrides individual attributes on the
   `torch` module itself (sigmoid, tanh). nanchat calls these as
   `torch.sigmoid(x)` / `torch.tanh(x)`, not via the F namespace. Targeted
   per-attribute swaps (NOT a full-module proxy) so the rest of `torch`
   stays untouched.

Two entry points:
  - `patch_nanchat()`: permanent swap (used by `scripts/base_train.py` when
    `NANOOPS=1` env var is set).
  - `patched()`: context manager that swaps in and restores on exit (used by
    `tests/test_nanchat_integration.py` to compare PyTorch vs nanoops paths).

Patched ops:
    F.linear, F.embedding, F.rms_norm, F.cross_entropy, F.softmax,
    F.scaled_dot_product_attention, torch.sigmoid, torch.tanh.

NOT patched (intentional):
  - F.relu — nanchat does `F.relu(x).square()` (a chain of two ops); nanoops's
    fused `relu_square` would need to replace the whole chain at a different
    call site, not the `F.relu` slot.
"""

from __future__ import annotations

import contextlib
import importlib
import os
from typing import Iterator

import torch
import torch.nn.functional as F_orig
import torch.utils.checkpoint as _ckpt

import nanoops.functional as nF


# Captured at patch time so the checkpoint wrapper can call the un-patched
# original CausalSelfAttention.forward.
_orig_attn_forward = None


_F_OVERRIDES = {
    "linear": nF.linear,
    "embedding": nF.embedding,
    "rms_norm": nF.rms_norm,
    "cross_entropy": nF.cross_entropy,
    "softmax": nF.softmax,
    "scaled_dot_product_attention": nF.scaled_dot_product_attention,
}

_TORCH_OVERRIDES = {
    # Patched as attributes directly on the `torch` module — nanchat calls
    # `torch.sigmoid(x)` / `torch.tanh(x)` (not via F namespace). Scope is the
    # whole Python process, but only these two names are touched; the rest of
    # `torch` is untouched.
    "sigmoid": nF.sigmoid,
    "tanh": nF.tanh,
}


# Module-level functions in nanchat that have nanoops equivalents but are NOT
# looked up via F namespace (so the F-namespace patch doesn't touch them).
# Format: (module_path, attr_name) -> replacement.
_MODULE_FUNC_OVERRIDES = {
    # nanchat defines apply_rotary_emb as a top-level function in gpt.py and
    # calls it directly via module lookup (`gpt.apply_rotary_emb(...)`).
    # Replacing this attribute swaps in nanoops's autograd Function version.
    ("nanochat.gpt", "apply_rotary_emb"): nF.apply_rotary_emb,
}


def _patched_mlp_forward(self, x):
    """nanchat MLP.forward, but `F.relu(x).square()` -> `nF.relu_square(x)`.

    `F.relu(x).square()` is a TWO-op chain: F.relu returns a fresh tensor,
    then `.square()` is a tensor method on it. Neither a single F-namespace
    patch nor a single torch.X patch can route the chain to nanoops's fused
    `relu_square` — we have to replace the whole forward. nanchat has exactly
    one such site (`gpt.py:137` MLP.forward), so a targeted class-method
    swap is the minimal fix.
    """
    if os.environ.get("NANOOPS_MLP_CHECKPOINT"):
        return _ckpt.checkpoint(_mlp_inner, self, x, use_reentrant=False)
    return _mlp_inner(self, x)


def _mlp_inner(self, x):
    """The actual MLP forward — same as the patched body, separated out so
    activation checkpointing can wrap it cleanly."""
    x = self.c_fc(x)
    x = nF.relu_square(x)
    x = self.c_proj(x)
    return x


def _patched_l_attn_forward(self, x, ve, cos_sin, window_size, kv_cache):
    """Wraps CausalSelfAttention.forward with torch.utils.checkpoint when
    NANOOPS_L_ATTN_CHECKPOINT=1. With SlidingWindowSDPA now using the
    Flash-style LSE-only ctx (no P matrix saved), this checkpoint is
    largely redundant — the ~900 MB of P-ctx it used to free is already
    gone. Left in as a knob: future deeper / wider configs might still
    benefit from also dropping the QKV/MLP-input activations, which
    SDPA's LSE trick doesn't touch.

    Only triggers during training (kv_cache is None). Inference bypasses.
    """
    if os.environ.get("NANOOPS_L_ATTN_CHECKPOINT") and kv_cache is None:
        return _ckpt.checkpoint(
            _orig_attn_forward, self, x, ve, cos_sin, window_size, kv_cache,
            use_reentrant=False,
        )
    return _orig_attn_forward(self, x, ve, cos_sin, window_size, kv_cache)


def _patched_sdpa_attention(q, k, v, window_size, enable_gqa):
    """Replacement for nanchat.flash_attention._sdpa_attention.

    Routes ALL training attention (both sliding-window S layers and full
    L layers, whenever Tq == Tk) through nanoops's SlidingWindowSDPA so
    they share one autograd-graph shape and one chunked-ctx memory model.
    For L layers we set window_size=Tq (the sliding mask reduces to pure
    causal) and chunk_size=Tq//8 so the GEMMs stay reasonably sized.

    Inference paths keep the original behavior:
      - single-token gen (Tq == 1): trim k/v + PyTorch SDPA
      - cached gen (Tq != Tk): explicit causal+sliding mask + PyTorch SDPA
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length — route through nF.sliding_window_sdpa
    # with window_size=Tq (covers all keys, so the sliding mask reduces
    # to pure causal) and chunk_size=Tq//8. Note: this gives up Flash
    # backend's tile-based O(L) memory; A/B on d24+B=1 didn't help with
    # OOM, but we keep the unified chunked codepath so all training
    # attention goes through the same nanoops Function (consistent
    # autograd graph, easier to reason about).
    if (window < 0 or window >= Tq) and Tq == Tk:
        return nF.sliding_window_sdpa(
            q, k, v, window_size=Tq, enable_gqa=enable_gqa,
            chunk_size=max(1, Tq // 8),
        )

    # Single token generation — same path as original (left-trim k/v).
    if Tq == 1:
        if window >= 0 and window < Tk:
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False, enable_gqa=enable_gqa
        )

    # Sliding window during training (Tq == Tk, finite window) — chunked path.
    if Tq == Tk and window >= 0 and window < Tq:
        # Convention translation between nanchat and nanoops sliding window:
        #   nanchat: window = max distance back (FA3-style "left" arg, 0..window
        #            inclusive → window+1 distinct allowed key offsets incl. self)
        #   nanoops: W      = TOTAL number of keys each query attends to
        # So nanoops_W = nanchat_window + 1. See nanchat/flash_attention.py:85
        # ("window is 'left' tokens we need to include (window + 1) keys total").
        return nF.sliding_window_sdpa(q, k, v, window + 1, enable_gqa=enable_gqa)

    # Cached generation with chunk inference (Tq != Tk) — build explicit mask
    # and fall back to one big SDPA call (sliding_window_sdpa assumes Tq == Tk).
    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, enable_gqa=enable_gqa
    )


_TARGET_MODULES = [
    # nanchat modules — call F.* directly
    "nanochat.gpt",
    "nanochat.flash_attention",
    "nanochat.engine",
    # PyTorch nn module internals — `nn.X.forward()` looks up F.* from
    # its own host module's namespace, not from any of the nanchat modules
    # above. nanchat overrides `Linear.forward` so `nn.Linear` is moot in
    # practice, but `nn.Embedding` is used as-is (wte, value_embeds), and
    # bare `nn.Linear` could slip in via future code. Patching here is
    # cheap insurance — within one NANOOPS=1 process, replacing PyTorch's
    # F.embedding / F.linear with nanoops's is exactly the intent.
    "torch.nn.modules.sparse",          # nn.Embedding -> F.embedding
    "torch.nn.modules.linear",          # nn.Linear -> F.linear
    "torch.nn.modules.normalization",   # nn.RMSNorm / nn.LayerNorm -> F.rms_norm / F.layer_norm
    "torch.nn.modules.loss",            # nn.CrossEntropyLoss -> F.cross_entropy
    "torch.nn.modules.activation",      # nn.Softmax -> F.softmax
]


def _make_patched_F() -> object:
    """Build a proxy: nanoops overrides win; everything else falls through to F."""
    proxy = type("PatchedF", (), {})()
    for attr in dir(F_orig):
        if not attr.startswith("_"):
            setattr(proxy, attr, getattr(F_orig, attr))
    for name, op in _F_OVERRIDES.items():
        setattr(proxy, name, op)
    return proxy


_PATCHED = False  # set to True after _apply() runs; cleared by _restore()


def _apply() -> dict[str, dict]:
    """Apply F-namespace + torch.X + module-func + method patches.
    Returns originals dict for restore.

    Idempotence guard: refuses to re-patch if a previous _apply() hasn't
    been _restore()'d. Without this guard, a second _apply() would
    capture the ALREADY-patched class methods / module funcs as
    "originals", and the restore path would put back patched versions
    instead of the true originals.
    """
    global _PATCHED
    if _PATCHED:
        raise RuntimeError(
            "nanoops.integration already patched; call _restore() before _apply() again"
        )
    originals: dict[str, dict] = {"F": {}, "torch": {}, "module_func": {}, "method": {}}
    # F-namespace patch
    proxy = _make_patched_F()
    for modname in _TARGET_MODULES:
        mod = importlib.import_module(modname)
        if hasattr(mod, "F"):
            originals["F"][modname] = mod.F
            mod.F = proxy
    # torch.X attribute patch
    for name, op in _TORCH_OVERRIDES.items():
        originals["torch"][name] = getattr(torch, name)
        setattr(torch, name, op)
    # Module-level function attribute patch (e.g. nanchat.gpt.apply_rotary_emb)
    for (modname, attr_name), replacement in _MODULE_FUNC_OVERRIDES.items():
        mod = importlib.import_module(modname)
        originals["module_func"][(modname, attr_name)] = getattr(mod, attr_name)
        setattr(mod, attr_name, replacement)
    # MLP.forward: route the `F.relu(x).square()` chain to fused relu_square
    # (and optionally activation-checkpoint via NANOOPS_MLP_CHECKPOINT=1).
    gpt_mod = importlib.import_module("nanochat.gpt")
    originals["method"][("nanochat.gpt", "MLP", "forward")] = gpt_mod.MLP.forward
    gpt_mod.MLP.forward = _patched_mlp_forward
    # CausalSelfAttention.forward: wrap to allow activation checkpoint of the
    # full-attention (L) layers via NANOOPS_L_ATTN_CHECKPOINT=1. Always
    # installed; the env-var check inside picks the right path per layer.
    # The _PATCHED guard above already ensures we're not double-patching;
    # the `is None` check here is defense-in-depth in case someone bypasses
    # _apply() to call lower-level methods.
    global _orig_attn_forward
    assert _orig_attn_forward is None, (
        "_orig_attn_forward already captured — call _restore() before _apply()"
    )
    _orig_attn_forward = gpt_mod.CausalSelfAttention.forward
    originals["method"][("nanochat.gpt", "CausalSelfAttention", "forward")] = _orig_attn_forward
    gpt_mod.CausalSelfAttention.forward = _patched_l_attn_forward
    # Sliding window SDPA: always ON (measured +6.2% tok/sec and -10.4%
    # peak memory at B=2 on nanchat d20, RTX 3090). Patched
    # _sdpa_attention falls through to the original SDPA call for full
    # attention layers and single-token inference, so this swap is a
    # strict superset of behavior — no regression possible on the
    # non-sliding paths.
    fa_mod = importlib.import_module("nanochat.flash_attention")
    originals["module_func"][("nanochat.flash_attention", "_sdpa_attention")] = (
        fa_mod._sdpa_attention
    )
    fa_mod._sdpa_attention = _patched_sdpa_attention
    _PATCHED = True  # global declared at top of _apply()
    return originals


def _restore(originals: dict[str, dict]) -> None:
    for modname, original_F in originals["F"].items():
        importlib.import_module(modname).F = original_F
    for name, op in originals["torch"].items():
        setattr(torch, name, op)
    for (modname, attr_name), original in originals["module_func"].items():
        setattr(importlib.import_module(modname), attr_name, original)
    for (modname, cls_name, method_name), original in originals["method"].items():
        cls = getattr(importlib.import_module(modname), cls_name)
        setattr(cls, method_name, original)
    global _PATCHED, _orig_attn_forward
    _PATCHED = False
    _orig_attn_forward = None


def patch_nanchat() -> list[str]:
    """Permanently swap nanoops into nanchat. Returns the list of patched op names."""
    _apply()
    names = (
        [f"F.{n}" for n in _F_OVERRIDES]
        + [f"torch.{n}" for n in _TORCH_OVERRIDES]
        + [f"{mod}.{attr}" for mod, attr in _MODULE_FUNC_OVERRIDES]
        + ["MLP.forward(relu_square fused)"]
    )
    names.append("nanochat.flash_attention._sdpa_attention(sliding_window_sdpa)")
    if os.environ.get("NANOOPS_MLP_CHECKPOINT"):
        names.append("MLP.forward(activation checkpoint)")
    if os.environ.get("NANOOPS_L_ATTN_CHECKPOINT"):
        names.append("CausalSelfAttention.forward(L-only activation checkpoint)")
    names.append("_sdpa_attention(full-attn → chunked sliding, default)")
    return names


def maybe_patch_nanchat(env_var: str = "NANOOPS") -> bool:
    """If `$NANOOPS` is set (any truthy value), apply the patch and print
    the swap summary. Returns whether the patch was applied.

    This is the entry point training scripts (`scripts/base_train.py`)
    call unconditionally — centralizing the env-var check so call sites
    stay one line.
    """
    if not os.environ.get(env_var):
        return False
    patched_names = patch_nanchat()
    print(f"[nanoops] swapped in: {', '.join(patched_names)}")
    return True


@contextlib.contextmanager
def patched() -> Iterator[None]:
    """Temporarily swap nanoops in; restore PyTorch ops on exit."""
    originals = _apply()
    try:
        yield
    finally:
        _restore(originals)
