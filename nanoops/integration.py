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

import nanoops.functional as nF


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


def _patched_mlp_forward(self, x):
    """nanchat MLP.forward, but `F.relu(x).square()` -> `nF.relu_square(x)`.

    `F.relu(x).square()` is a TWO-op chain: F.relu returns a fresh tensor,
    then `.square()` is a tensor method on it. Neither a single F-namespace
    patch nor a single torch.X patch can route the chain to nanoops's fused
    `relu_square` — we have to replace the whole forward. nanchat has exactly
    one such site (`gpt.py:137` MLP.forward), so a targeted class-method
    swap is the minimal fix.
    """
    x = self.c_fc(x)
    x = nF.relu_square(x)
    x = self.c_proj(x)
    return x

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


def _apply() -> dict[str, dict]:
    """Apply F-namespace + torch.X + MLP.forward patches. Returns originals dict."""
    originals: dict[str, dict] = {"F": {}, "torch": {}, "method": {}}
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
    # MLP.forward: route the `F.relu(x).square()` chain to fused relu_square
    gpt_mod = importlib.import_module("nanochat.gpt")
    originals["method"][("nanochat.gpt", "MLP", "forward")] = gpt_mod.MLP.forward
    gpt_mod.MLP.forward = _patched_mlp_forward
    return originals


def _restore(originals: dict[str, dict]) -> None:
    for modname, original_F in originals["F"].items():
        importlib.import_module(modname).F = original_F
    for name, op in originals["torch"].items():
        setattr(torch, name, op)
    for (modname, cls_name, method_name), original in originals["method"].items():
        cls = getattr(importlib.import_module(modname), cls_name)
        setattr(cls, method_name, original)


def patch_nanchat() -> list[str]:
    """Permanently swap nanoops into nanchat. Returns the list of patched op names."""
    _apply()
    return (
        [f"F.{n}" for n in _F_OVERRIDES]
        + [f"torch.{n}" for n in _TORCH_OVERRIDES]
        + ["MLP.forward(relu_square fused)"]
    )


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
