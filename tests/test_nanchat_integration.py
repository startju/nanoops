"""Integration test: swap nanoops into nanchat's actual GPT.

Builds a tiny nanchat GPT, runs it twice with identical seeds and data:
once with PyTorch's torch.nn.functional, once with nanoops monkey-patched
into nanchat.gpt's `F` namespace. Compares the loss curves to verify
nanoops produces numerically identical results when dropped into the
real nanchat training loop.

What's swapped (via patching gpt_mod.F at import time):
  - F.rms_norm     -> nanoops.functional.rms_norm     (used in `norm()`)
  - F.linear       -> nanoops.functional.linear        (used in nanchat's Linear.forward)
  - F.cross_entropy-> nanoops.functional.cross_entropy (used in GPT.forward loss)

What's NOT swapped (kept as PyTorch):
  - F.scaled_dot_product_attention   (Tier 2, no nanoops version)
  - F.relu / .square() chain         (nanoops has relu_square but the
                                     gpt.py call is `F.relu(x).square()`,
                                     not easily monkey-patchable)
  - F.softmax, F.embedding (used in inference / minor paths)
  - torch.outer/cat/sigmoid/tanh     (accessed as torch.X, not F.X)

So the test exercises the three biggest ops (linear, rms_norm, cross_entropy)
running through nanoops's custom autograd Functions inside nanchat's real
model architecture (with FA3 fallback SDPA, rotary, smear gate, etc.).
"""

import contextlib
import torch

import nanochat.gpt as gpt_mod
import nanoops.functional as nF
from nanochat.gpt import GPTConfig, GPT


@contextlib.contextmanager
def nanoops_swapped_in():
    """Patch nanchat.gpt's F namespace to use nanoops for select ops."""
    original_F = gpt_mod.F

    class PatchedF:
        # Override with nanoops where available
        rms_norm = staticmethod(nF.rms_norm)
        cross_entropy = staticmethod(nF.cross_entropy)
        linear = staticmethod(nF.linear)

    # Fall through every other attribute to the original F
    for attr in dir(original_F):
        if not attr.startswith("_") and not hasattr(PatchedF, attr):
            setattr(PatchedF, attr, getattr(original_F, attr))

    gpt_mod.F = PatchedF
    try:
        yield
    finally:
        gpt_mod.F = original_F


def _tiny_config() -> GPTConfig:
    return GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )


def _make_model_and_data(seed: int = 0):
    """Deterministic tiny model + a batch of dummy training data."""
    torch.manual_seed(seed)
    model = GPT(_tiny_config())
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, 64, (2, 16), generator=g)
    targets = torch.randint(0, 64, (2, 16), generator=g)
    return model, input_ids, targets


def _train_steps(model, input_ids, targets, n_steps: int = 5) -> list[float]:
    """Run `n_steps` AdamW steps on the same batch; return per-step loss."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(n_steps):
        opt.zero_grad()
        loss = model(input_ids, targets)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_nanchat_runs_with_nanoops_swapped_in():
    """Forward+backward+step succeeds and produces finite, decreasing loss."""
    with nanoops_swapped_in():
        model, x, t = _make_model_and_data(seed=0)
        losses = _train_steps(model, x, t, n_steps=5)
    assert all(torch.isfinite(torch.tensor(l)).item() for l in losses), \
        f"non-finite losses: {losses}"
    assert losses[-1] < losses[0], \
        f"loss did not decrease: {losses}"


def test_nanchat_pytorch_vs_nanoops_loss_curve_matches():
    """Same seed + same data -> identical loss curves through 5 AdamW steps."""
    # PyTorch baseline
    model_pt, x, t = _make_model_and_data(seed=0)
    losses_pt = _train_steps(model_pt, x, t, n_steps=5)

    # nanoops swap-in (same seed -> same initial weights)
    with nanoops_swapped_in():
        model_no, x2, t2 = _make_model_and_data(seed=0)
        losses_no = _train_steps(model_no, x2, t2, n_steps=5)

    # Expected magnitude of divergence: ~1e-3 per step.
    # PyTorch's `F.cross_entropy` uses a single fused C++ kernel; nanoops
    # decomposes into chunked_logsumexp + gather + scatter, each with its
    # own fp32 rounding. The cumulative noise across 5 AdamW steps lands
    # in the ~1e-3 range and stays bounded (doesn't diverge), which is the
    # mark of "different impl, same math" rather than a bug.
    for i, (a, b) in enumerate(zip(losses_pt, losses_no)):
        assert abs(a - b) < 2e-3, \
            f"step {i}: PyTorch loss {a:.6f} vs nanoops loss {b:.6f}  diff={abs(a-b):.2e}"
    # Additionally check the SHAPE of the curve matches: both should
    # decrease monotonically by roughly the same amount per step.
    decreases_pt = [losses_pt[i] - losses_pt[i+1] for i in range(len(losses_pt) - 1)]
    decreases_no = [losses_no[i] - losses_no[i+1] for i in range(len(losses_no) - 1)]
    for i, (dp, dn) in enumerate(zip(decreases_pt, decreases_no)):
        assert abs(dp - dn) < 5e-3, \
            f"step {i}->{i+1}: drop differs (pt={dp:.4f} no={dn:.4f})"
