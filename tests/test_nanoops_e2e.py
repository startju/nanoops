"""End-to-end integration test for nanoops.

Builds a "near-nanchat" GPT-style model twice — once with PyTorch ops
and once with nanoops — and verifies forward + backward + multi-step
training produce identical results (within fp32 numerical noise).

The model mirrors nanchat's architecture as closely as Tier 1 ops allow:
  - Token embedding
  - Rotary-position-aware attention block (causal, multi-head)
  - MLP block with relu^2 activation (nanchat's choice)
  - Gated path using sigmoid + tanh (mirrors nanchat's smear_gate idea)
  - RmsNorm pre-norm on every sub-block
  - Logit softcap via tanh
  - cross_entropy with ignore_index=-1

`F.scaled_dot_product_attention` is the only op kept from PyTorch in
both versions (nanoops SDPA is Tier 2). Everything else — linear,
embedding, rms_norm, relu_square, sigmoid, tanh, cross_entropy, outer,
cat — comes from nanoops in the nanoops variant.
"""

import torch
import torch.nn.functional as F

import nanoops.functional as nF


# ----- shared shape config -----
V = 128
D = 32
N_HEAD = 4
HEAD_DIM = D // N_HEAD
SEQ_LEN = 16
BATCH = 4
SOFTCAP = 15.0


def make_params() -> dict[str, torch.Tensor]:
    """Random parameters scaled (a=1 / sqrt(3*fan_in)) for stable forward."""
    torch.manual_seed(0)
    g = lambda *s: torch.randn(*s) * (3 / s[-1]) ** 0.5
    return {
        "emb": torch.randn(V, D) * 0.5,
        # block 1 (attention)
        "norm_attn_w": torch.ones(D),
        "q_w": g(D, D),
        "k_w": g(D, D),
        "v_w": g(D, D),
        "o_w": g(D, D),
        # block 1 (mlp)
        "norm_mlp_w": torch.ones(D),
        "fc1_w": g(D * 4, D),
        "fc2_w": g(D, D * 4),
        # block 2 (gated path — exercises sigmoid + tanh)
        "norm_gate_w": torch.ones(D),
        "gate_w": g(D, D),
        "value_w": g(D, D),
        # output
        "norm_final_w": torch.ones(D),
        "lm_head_w": torch.randn(V, D) * 0.01,
    }


def precompute_rotary(ops, device):
    """Generate rotary cos/sin tables via outer + arange (Tier 1)."""
    half = HEAD_DIM // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(SEQ_LEN, device=device, dtype=torch.float32)
    freqs = ops["outer"](t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    # broadcast shape (1, T, 1, half_dim) for (B, T, n_head, head_dim) inputs
    return cos[None, :, None, :], sin[None, :, None, :]


def apply_rotary(x, cos, sin, ops):
    """Rotate the last dim in two halves, then re-cat — nanchat's exact recipe."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = -x1 * sin + x2 * cos
    return ops["cat"]([y1, y2], dim=-1)


def attn_block(x, p, cos, sin, ops):
    """Pre-norm attention with rotary + causal SDPA (PyTorch op — Tier 2)."""
    h = ops["rms_norm"](x, (D,), p["norm_attn_w"])
    B, T, _ = h.shape
    q = ops["linear"](h, p["q_w"]).view(B, T, N_HEAD, HEAD_DIM)
    k = ops["linear"](h, p["k_w"]).view(B, T, N_HEAD, HEAD_DIM)
    v = ops["linear"](h, p["v_w"]).view(B, T, N_HEAD, HEAD_DIM)
    q = apply_rotary(q, cos, sin, ops)
    k = apply_rotary(k, cos, sin, ops)
    # (B, T, H, D_head) -> (B, H, T, D_head)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    # SDPA stays PyTorch in both variants (nanoops Tier 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = out.transpose(1, 2).reshape(B, T, D)
    return x + ops["linear"](out, p["o_w"])


def mlp_block(x, p, ops):
    """Pre-norm MLP with relu^2 activation (nanchat's choice)."""
    h = ops["rms_norm"](x, (D,), p["norm_mlp_w"])
    h = ops["linear"](h, p["fc1_w"])
    h = ops["relu_square"](h)
    h = ops["linear"](h, p["fc2_w"])
    return x + h


def gated_block(x, p, ops):
    """Sigmoid-gated tanh value (mirrors nanchat's smear_gate pattern)."""
    h = ops["rms_norm"](x, (D,), p["norm_gate_w"])
    gate = ops["sigmoid"](ops["linear"](h, p["gate_w"]))
    value = ops["tanh"](ops["linear"](h, p["value_w"]))
    return x + gate * value


def forward(input_ids, target, p, ops):
    cos, sin = precompute_rotary(ops, device=input_ids.device)
    x = ops["embedding"](input_ids, p["emb"])
    x = attn_block(x, p, cos, sin, ops)
    x = mlp_block(x, p, ops)
    x = gated_block(x, p, ops)
    x = ops["rms_norm"](x, (D,), p["norm_final_w"])
    logits = ops["linear"](x, p["lm_head_w"])
    # logit softcap via tanh (nanchat gpt.py:472)
    logits = SOFTCAP * ops["tanh"](logits / SOFTCAP)
    return ops["cross_entropy"](
        logits.view(-1, V), target.view(-1), ignore_index=-1, reduction="mean"
    )


# ----- two op tables -----
PT_OPS = {
    "embedding": F.embedding,
    "linear": F.linear,
    "rms_norm": F.rms_norm,
    "relu_square": lambda x: F.relu(x).square(),
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "outer": torch.outer,
    "cat": torch.cat,
    "cross_entropy": F.cross_entropy,
}
NO_OPS = {
    "embedding": nF.embedding,
    "linear": nF.linear,
    "rms_norm": nF.rms_norm,
    "relu_square": nF.relu_square,
    "sigmoid": nF.sigmoid,
    "tanh": nF.tanh,
    "outer": nF.outer,
    "cat": nF.cat,
    "cross_entropy": nF.cross_entropy,
}


def _clone(p):
    return {k: v.detach().clone().requires_grad_(True) for k, v in p.items()}


def test_e2e_forward_parity():
    p = make_params()
    input_ids = torch.randint(0, V, (BATCH, SEQ_LEN))
    target = torch.randint(0, V, (BATCH, SEQ_LEN))
    target.view(-1)[0] = -1  # exercise ignore_index

    l_pt = forward(input_ids, target, _clone(p), PT_OPS)
    l_no = forward(input_ids, target, _clone(p), NO_OPS)
    assert torch.allclose(l_pt, l_no, atol=1e-5), (
        f"forward mismatch: pt={l_pt.item()} no={l_no.item()}"
    )


def test_e2e_backward_parity():
    p = make_params()
    input_ids = torch.randint(0, V, (BATCH, SEQ_LEN))
    target = torch.randint(0, V, (BATCH, SEQ_LEN))
    target.view(-1)[0] = -1

    p_pt, p_no = _clone(p), _clone(p)
    forward(input_ids, target, p_pt, PT_OPS).backward()
    forward(input_ids, target, p_no, NO_OPS).backward()

    for name in p_pt:
        g_pt, g_no = p_pt[name].grad, p_no[name].grad
        diff = (g_pt - g_no).abs().max().item()
        assert torch.allclose(g_pt, g_no, atol=1e-4), (
            f"grad mismatch on {name}: max_diff={diff}"
        )


def test_e2e_training_loop_loss_curve_matches():
    """Multi-step AdamW: loss curve match per step (no weight drift)."""
    p = make_params()
    p_pt, p_no = _clone(p), _clone(p)
    opt_pt = torch.optim.AdamW(list(p_pt.values()), lr=1e-3)
    opt_no = torch.optim.AdamW(list(p_no.values()), lr=1e-3)

    torch.manual_seed(42)
    for step in range(10):
        input_ids = torch.randint(0, V, (BATCH, SEQ_LEN))
        target = torch.randint(0, V, (BATCH, SEQ_LEN))
        target.view(-1)[0] = -1

        opt_pt.zero_grad()
        l_pt = forward(input_ids, target, p_pt, PT_OPS)
        l_pt.backward()
        opt_pt.step()

        opt_no.zero_grad()
        l_no = forward(input_ids, target, p_no, NO_OPS)
        l_no.backward()
        opt_no.step()

        assert abs(l_pt.item() - l_no.item()) < 1e-4, (
            f"step {step}: pt={l_pt.item()} no={l_no.item()}"
        )
