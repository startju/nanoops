"""End-to-end integration test for nanoops.

Builds a tiny "transformer-style" model twice — once with PyTorch ops and
once with nanoops — and verifies forward + backward produce identical
results (within fp32 numerical noise). Exercises Tier 1 ops in realistic
composition: embedding + rms_norm + linear + relu_square + cross_entropy,
plus a gating block using sigmoid/tanh.

Attention is intentionally omitted — `F.scaled_dot_product_attention` is
a Tier 2 op not yet implemented in nanoops.
"""

import pytest
import torch
import torch.nn.functional as F

import nanoops.functional as nF


def make_params(V: int, D: int) -> dict[str, torch.Tensor]:
    """Random parameters for the mini-model, scaled for stable forward."""
    torch.manual_seed(0)
    return {
        "emb": torch.randn(V, D) * 0.5,
        # block 1 (MLP)
        "norm1_w": torch.ones(D),
        "fc1_w": torch.randn(D * 4, D) * (3 / D) ** 0.5,
        "fc2_w": torch.randn(D, D * 4) * (3 / (D * 4)) ** 0.5,
        # block 2 (gated MLP — exercises sigmoid + tanh + mul-via-PyTorch)
        "norm2_w": torch.ones(D),
        "gate_w": torch.randn(D, D) * (3 / D) ** 0.5,
        "value_w": torch.randn(D, D) * (3 / D) ** 0.5,
        # output
        "norm_final_w": torch.ones(D),
        "lm_head_w": torch.randn(V, D) * 0.01,
    }


def model_pytorch(input_ids, p, target):
    D = p["emb"].size(-1)
    x = F.embedding(input_ids, p["emb"])
    # block 1: pre-norm + MLP + residual (relu^2 activation, nanchat-style)
    h = F.rms_norm(x, (D,), p["norm1_w"])
    h = F.linear(h, p["fc1_w"])
    h = F.relu(h).square()
    h = F.linear(h, p["fc2_w"])
    x = x + h
    # block 2: gated path using sigmoid + tanh
    h = F.rms_norm(x, (D,), p["norm2_w"])
    gate = torch.sigmoid(F.linear(h, p["gate_w"]))
    value = torch.tanh(F.linear(h, p["value_w"]))
    x = x + gate * value
    # output
    x = F.rms_norm(x, (D,), p["norm_final_w"])
    logits = F.linear(x, p["lm_head_w"])
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target.view(-1),
        ignore_index=-1,
        reduction="mean",
    )


def model_nanoops(input_ids, p, target):
    D = p["emb"].size(-1)
    x = nF.embedding(input_ids, p["emb"])
    # block 1
    h = nF.rms_norm(x, (D,), p["norm1_w"])
    h = nF.linear(h, p["fc1_w"])
    h = nF.relu_square(h)
    h = nF.linear(h, p["fc2_w"])
    x = x + h
    # block 2 (gated)
    h = nF.rms_norm(x, (D,), p["norm2_w"])
    gate = nF.sigmoid(nF.linear(h, p["gate_w"]))
    value = nF.tanh(nF.linear(h, p["value_w"]))
    x = x + gate * value
    # output
    x = nF.rms_norm(x, (D,), p["norm_final_w"])
    logits = nF.linear(x, p["lm_head_w"])
    return nF.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target.view(-1),
        ignore_index=-1,
        reduction="mean",
    )


def _clone_params(p: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone().requires_grad_(True) for k, v in p.items()}


def test_e2e_forward_parity():
    """Same inputs + same weights → losses match within fp32 noise."""
    p = make_params(V=128, D=32)
    input_ids = torch.randint(0, 128, (4, 16))
    target = torch.randint(0, 128, (4, 16))
    target[0, 0] = -1  # exercise ignore_index path

    loss_pt = model_pytorch(input_ids, _clone_params(p), target)
    loss_no = model_nanoops(input_ids, _clone_params(p), target)
    assert torch.allclose(loss_pt, loss_no, atol=1e-5), (
        f"forward mismatch: pt={loss_pt.item()} no={loss_no.item()}"
    )


def test_e2e_backward_parity():
    """Per-parameter gradients match within fp32 noise."""
    p = make_params(V=128, D=32)
    input_ids = torch.randint(0, 128, (4, 16))
    target = torch.randint(0, 128, (4, 16))
    target[0, 0] = -1

    p_pt = _clone_params(p)
    p_no = _clone_params(p)
    model_pytorch(input_ids, p_pt, target).backward()
    model_nanoops(input_ids, p_no, target).backward()

    for name in p_pt:
        g_pt, g_no = p_pt[name].grad, p_no[name].grad
        assert torch.allclose(g_pt, g_no, atol=1e-5), (
            f"grad mismatch on {name}: max_diff={(g_pt - g_no).abs().max().item()}"
        )


def test_e2e_training_loop_loss_curve_matches():
    """Run several optimizer steps; loss trajectories must coincide exactly."""
    p = make_params(V=128, D=32)
    p_pt = _clone_params(p)
    p_no = _clone_params(p)
    opt_pt = torch.optim.AdamW(list(p_pt.values()), lr=1e-3)
    opt_no = torch.optim.AdamW(list(p_no.values()), lr=1e-3)

    torch.manual_seed(42)
    losses_pt, losses_no = [], []
    for step in range(10):
        input_ids = torch.randint(0, 128, (4, 16))
        target = torch.randint(0, 128, (4, 16))
        target[0, 0] = -1  # vary nothing about ignore pattern

        opt_pt.zero_grad()
        loss_pt = model_pytorch(input_ids, p_pt, target)
        loss_pt.backward()
        opt_pt.step()
        losses_pt.append(loss_pt.item())

        opt_no.zero_grad()
        loss_no = model_nanoops(input_ids, p_no, target)
        loss_no.backward()
        opt_no.step()
        losses_no.append(loss_no.item())

    for i, (a, b) in enumerate(zip(losses_pt, losses_no)):
        assert abs(a - b) < 1e-4, f"step {i}: pt={a} no={b}"
