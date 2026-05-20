"""Parity tests for nanoops.scaled_dot_product_attention vs PyTorch.

We force PyTorch's MATH backend (not Flash/MemEfficient) as the reference,
because non-math backends handle `-inf` in float `attn_mask` and `is_causal`
with `L != S` in subtly non-canonical ways (numerically close but not bitwise).
nanoops follows the math reference, so we compare against that.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import nanoops.functional as nF


def _rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float64)


def _check_fwd_bwd(q, k, v, **kw):
    """Run nanoops SDPA and PyTorch math-backend SDPA; assert forward & grads match."""
    q1, k1, v1 = q.clone().requires_grad_(True), k.clone().requires_grad_(True), v.clone().requires_grad_(True)
    q2, k2, v2 = q.clone().requires_grad_(True), k.clone().requires_grad_(True), v.clone().requires_grad_(True)
    o1 = nF.scaled_dot_product_attention(q1, k1, v1, **kw)
    with sdpa_kernel([SDPBackend.MATH]):
        o2 = F.scaled_dot_product_attention(q2, k2, v2, **kw)
    assert torch.allclose(o1, o2, atol=1e-10), \
        f"forward mismatch: max={((o1-o2).abs().max()).item()}"
    g = torch.randn_like(o1)
    o1.backward(g)
    o2.backward(g)
    for name, a, b in [("q", q1.grad, q2.grad), ("k", k1.grad, k2.grad), ("v", v1.grad, v2.grad)]:
        assert torch.allclose(a, b, atol=1e-10), \
            f"grad_{name} mismatch: max={((a-b).abs().max()).item()}"


def test_sdpa_basic():
    q, k, v = _rand(2, 4, 8, 16), _rand(2, 4, 8, 16), _rand(2, 4, 8, 16)
    _check_fwd_bwd(q, k, v)


def test_sdpa_causal():
    q, k, v = _rand(2, 4, 8, 16), _rand(2, 4, 8, 16), _rand(2, 4, 8, 16)
    _check_fwd_bwd(q, k, v, is_causal=True)


def test_sdpa_bool_mask():
    q, k, v = _rand(2, 4, 8, 16), _rand(2, 4, 8, 16), _rand(2, 4, 8, 16)
    mask = torch.ones(8, 8, dtype=torch.bool).tril()
    _check_fwd_bwd(q, k, v, attn_mask=mask)


def test_sdpa_float_mask():
    q, k, v = _rand(2, 4, 8, 16), _rand(2, 4, 8, 16), _rand(2, 4, 8, 16)
    mask = torch.where(torch.ones(8, 8).tril().bool(), 0.0, float("-inf"))
    _check_fwd_bwd(q, k, v, attn_mask=mask)


def test_sdpa_custom_scale():
    q, k, v = _rand(2, 4, 8, 16), _rand(2, 4, 8, 16), _rand(2, 4, 8, 16)
    _check_fwd_bwd(q, k, v, is_causal=True, scale=0.123)


def test_sdpa_gqa():
    # H_q=4, H_kv=2 -> G=2
    q = _rand(2, 4, 8, 16)
    k = _rand(2, 2, 8, 16)
    v = _rand(2, 2, 8, 16)
    _check_fwd_bwd(q, k, v, is_causal=True, enable_gqa=True)


def test_sdpa_gqa_diff_kv_len():
    # cached-gen-style: Tq != Tk, with mask
    q = _rand(2, 4, 1, 16)
    k = _rand(2, 2, 6, 16)
    v = _rand(2, 2, 6, 16)
    mask = torch.ones(1, 6, dtype=torch.bool)  # attend to all 6
    _check_fwd_bwd(q, k, v, attn_mask=mask, enable_gqa=True)


def test_sdpa_causal_unequal_lens():
    # is_causal with L != S (cached gen, right-aligned causal)
    q = _rand(2, 4, 3, 16)
    k = _rand(2, 4, 8, 16)
    v = _rand(2, 4, 8, 16)
    _check_fwd_bwd(q, k, v, is_causal=True)
