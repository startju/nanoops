"""Parity tests for SlidingWindowSDPA: vs full SDPA with sliding mask,
across no-GQA and GQA configurations.

Both implementations compute the same math (sliding causal attention);
their outputs/grads should match modulo bf16 rounding noise.
"""

import math
import torch

import nanoops.functional as nF
from nanoops.functional import SlidingWindowSDPA


def _full_with_sliding_mask(q, k, v, window_size, enable_gqa=False):
    """Reference: compute via the regular ScaledDotProductAttention with
    an L×L sliding+causal mask."""
    B, H_q, L, D = q.shape
    H_kv = k.shape[-3]
    if enable_gqa and H_q != H_kv:
        G = H_q // H_kv
        k = k.repeat_interleave(G, dim=-3)
        v = v.repeat_interleave(G, dim=-3)
    i_idx = torch.arange(L, device=q.device).unsqueeze(1)
    j_idx = torch.arange(L, device=q.device).unsqueeze(0)
    mask = (j_idx <= i_idx) & (j_idx >= i_idx - window_size + 1)
    return nF.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def _check(q_shape, k_shape, W, dtype=torch.float64, atol=1e-10):
    """Run both, compare forward + backward."""
    torch.manual_seed(0)
    enable_gqa = q_shape[1] != k_shape[1]

    q0 = torch.randn(*q_shape, dtype=dtype)
    k0 = torch.randn(*k_shape, dtype=dtype)
    v0 = torch.randn(*k_shape, dtype=dtype)

    # Reference
    q1, k1, v1 = q0.clone().requires_grad_(True), k0.clone().requires_grad_(True), v0.clone().requires_grad_(True)
    out_ref = _full_with_sliding_mask(q1, k1, v1, W, enable_gqa=enable_gqa)
    g = torch.randn_like(out_ref)
    out_ref.backward(g)

    # SlidingWindowSDPA
    q2, k2, v2 = q0.clone().requires_grad_(True), k0.clone().requires_grad_(True), v0.clone().requires_grad_(True)
    out_sw = nF.sliding_window_sdpa(q2, k2, v2, W, enable_gqa=enable_gqa)
    out_sw.backward(g)

    assert torch.allclose(out_ref, out_sw, atol=atol), \
        f"forward mismatch (max diff {(out_ref - out_sw).abs().max()})"
    assert torch.allclose(q1.grad, q2.grad, atol=atol), \
        f"q.grad mismatch (max diff {(q1.grad - q2.grad).abs().max()})"
    assert torch.allclose(k1.grad, k2.grad, atol=atol), \
        f"k.grad mismatch (max diff {(k1.grad - k2.grad).abs().max()})"
    assert torch.allclose(v1.grad, v2.grad, atol=atol), \
        f"v.grad mismatch (max diff {(v1.grad - v2.grad).abs().max()})"


def test_no_gqa_window_smaller_than_L():
    """W < L: real sliding window, multiple chunks."""
    _check(q_shape=(2, 4, 32, 16), k_shape=(2, 4, 32, 16), W=8)


def test_no_gqa_window_equals_L():
    """W == L: degenerates to full causal attention (single chunk)."""
    _check(q_shape=(2, 4, 16, 16), k_shape=(2, 4, 16, 16), W=16)


def test_no_gqa_window_larger_than_L():
    """W > L: same as full causal attention."""
    _check(q_shape=(2, 4, 16, 16), k_shape=(2, 4, 16, 16), W=32)


def test_gqa_2x():
    """H_q=8, H_kv=4: each kv head shared by 2 q heads."""
    _check(q_shape=(2, 8, 32, 16), k_shape=(2, 4, 32, 16), W=8)


def test_gqa_4x():
    """H_q=8, H_kv=2: each kv head shared by 4 q heads."""
    _check(q_shape=(2, 8, 32, 16), k_shape=(2, 2, 32, 16), W=8)


def test_gqa_non_chunked():
    """GQA + W >= L (no chunking) — still needs to repeat_interleave correctly."""
    _check(q_shape=(2, 8, 16, 16), k_shape=(2, 2, 16, 16), W=16)
