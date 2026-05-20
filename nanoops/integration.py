"""Monkey-patch helpers to swap nanoops into nanchat.

Replaces the `F` (torch.nn.functional) namespace in each nanchat module with
a proxy whose specific attributes route through nanoops. Everything not
overridden falls through to the real `torch.nn.functional`.

Two entry points:
  - `patch_nanchat()`: permanent swap (used by `scripts/base_train.py` when
    `NANOOPS=1` env var is set).
  - `patched()`: context manager that swaps in and restores on exit (used by
    `tests/test_nanchat_integration.py` to compare PyTorch vs nanoops paths).

Patched ops:
    F.linear, F.embedding, F.rms_norm, F.cross_entropy, F.softmax,
    F.scaled_dot_product_attention.

NOT patched (intentional):
  - F.relu — nanchat does `F.relu(x).square()` (a chain of two ops); nanoops's
    fused `relu_square` would need to replace the whole chain at a different
    call site, not the `F.relu` slot.
  - torch.sigmoid / torch.tanh — accessed via `torch.X`, not `F.X`. Patching
    `torch` globally is too invasive; module-level proxies aren't worth the
    complexity for these two ops.
"""

from __future__ import annotations

import contextlib
import importlib
from typing import Iterator

import torch.nn.functional as F_orig

import nanoops.functional as nF


_OVERRIDES = {
    "linear": nF.linear,
    "embedding": nF.embedding,
    "rms_norm": nF.rms_norm,
    "cross_entropy": nF.cross_entropy,
    "softmax": nF.softmax,
    "scaled_dot_product_attention": nF.scaled_dot_product_attention,
}

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
    for name, op in _OVERRIDES.items():
        setattr(proxy, name, op)
    return proxy


def _apply() -> dict[str, object]:
    """Patch the `F` attribute of every target module. Returns originals dict."""
    originals: dict[str, object] = {}
    proxy = _make_patched_F()
    for modname in _TARGET_MODULES:
        mod = importlib.import_module(modname)
        if hasattr(mod, "F"):
            originals[modname] = mod.F
            mod.F = proxy
    return originals


def _restore(originals: dict[str, object]) -> None:
    for modname, original_F in originals.items():
        importlib.import_module(modname).F = original_F


def patch_nanchat() -> list[str]:
    """Permanently swap nanoops into nanchat. Returns the list of patched op names."""
    _apply()
    return list(_OVERRIDES.keys())


@contextlib.contextmanager
def patched() -> Iterator[None]:
    """Temporarily swap nanoops in; restore PyTorch ops on exit."""
    originals = _apply()
    try:
        yield
    finally:
        _restore(originals)
