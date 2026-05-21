"""Parity tests: LookupSorted vs Lookup (and both vs PyTorch reference).

LookupSorted has the same forward as Lookup and a mathematically-equivalent
backward (segmented sum instead of atomic index_add_). These tests pin that
contract so future tweaks to either don't drift them apart.
"""

import torch
import torch.nn.functional as F

from nanoops.functional import Lookup, LookupSorted


def _check(indices, V, D, dtype=torch.float64):
    """Run forward + backward through PyTorch, Lookup, LookupSorted; compare."""
    torch.manual_seed(0)
    w0 = torch.randn(V, D, dtype=dtype)

    # PyTorch reference
    w_pt = w0.clone().requires_grad_(True)
    out_pt = F.embedding(indices, w_pt)
    g = torch.randn_like(out_pt)
    out_pt.backward(g)

    # nanoops Lookup (naive)
    w_naive = w0.clone().requires_grad_(True)
    out_naive = Lookup.apply(indices, w_naive)
    out_naive.backward(g)

    # nanoops LookupSorted
    w_sorted = w0.clone().requires_grad_(True)
    out_sorted = LookupSorted.apply(indices, w_sorted)
    out_sorted.backward(g)

    assert torch.allclose(out_pt, out_naive), "Lookup forward mismatch vs PyTorch"
    assert torch.allclose(out_pt, out_sorted), "LookupSorted forward mismatch vs PyTorch"
    assert torch.allclose(w_pt.grad, w_naive.grad), "Lookup grad mismatch vs PyTorch"
    assert torch.allclose(w_pt.grad, w_sorted.grad, atol=1e-10), \
        f"LookupSorted grad mismatch vs PyTorch (max diff: {(w_pt.grad - w_sorted.grad).abs().max()})"
    assert torch.allclose(w_naive.grad, w_sorted.grad, atol=1e-10), \
        f"LookupSorted grad mismatch vs Lookup (max diff: {(w_naive.grad - w_sorted.grad).abs().max()})"


def test_no_duplicates():
    """Each token appears at most once."""
    indices = torch.tensor([3, 1, 4, 0, 2])
    _check(indices, V=8, D=4)


def test_heavy_duplicates():
    """Same token repeated many times — stresses the segmented-sum path."""
    indices = torch.tensor([2] * 50 + [5] * 30 + [0] * 20)
    _check(indices, V=8, D=16)


def test_realistic_token_distribution():
    """LLM-like: large vocab, batch_tokens << vocab, heavy-tailed indices."""
    torch.manual_seed(42)
    V = 1024
    N = 256
    # Zipf-ish: a few common tokens, many rare
    indices = torch.cat([
        torch.randint(0, 10, (N // 2,)),         # common
        torch.randint(0, V, (N // 2,)),           # rest spread out
    ])
    _check(indices, V=V, D=32)


def test_single_unique():
    """All indices point to the same row."""
    indices = torch.tensor([7, 7, 7, 7])
    _check(indices, V=10, D=8)


def test_empty():
    """Empty indices — backward should produce all-zero grad."""
    indices = torch.tensor([], dtype=torch.int64)
    _check(indices, V=8, D=4)


def test_bf16_no_accumulation_drift():
    """bf16 with many duplicates per row exercises the fp32 cumsum promotion.

    Without the promotion, cumsum's bf16 partial sums accumulate >10%
    relative error vs Lookup at this N, and group_sum = cumsum[end] -
    cumsum[start] propagates that error into every grad row.
    """
    torch.manual_seed(0)
    V, D, N = 256, 64, 1024
    indices = torch.randint(0, 16, (N,))  # only 16 unique rows → ~64 dups each
    weight = torch.randn(V, D, dtype=torch.bfloat16)
    g = torch.randn(N, D, dtype=torch.bfloat16)

    w_naive = weight.clone().requires_grad_(True)
    w_sorted = weight.clone().requires_grad_(True)
    Lookup.apply(indices, w_naive).backward(g)
    LookupSorted.apply(indices, w_sorted).backward(g)

    # bf16 has 7-bit mantissa; per-row sums of ~64 bf16 grads can differ by
    # a few ULPs between the two impls (Lookup atomic-adds, LookupSorted
    # fp32 cumsum then casts). atol=0.01 catches the "no fp32 promotion"
    # bug which gave ~1.7 max abs diff at this scale, while still allowing
    # normal rounding noise.
    max_diff = (w_naive.grad - w_sorted.grad).abs().max().item()
    assert max_diff < 0.01, (
        f"LookupSorted bf16 grad drifted from Lookup by {max_diff:.4f}; "
        f"likely the cumsum fp32-promotion was lost."
    )
