"""Parity tests: nanoops vs torch."""
import pytest
import torch
import torch.nn as tnn
import torch.nn.functional as tF

import nanoops.nn as nnn
import nanoops.functional as nF


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("shape", [(4, 8), (2, 3, 8), (16,)])
def test_linear_functional_matches_torch(shape, bias):
    torch.manual_seed(0)
    in_features, out_features = 8, 5
    x = torch.randn(*shape)
    w = torch.randn(out_features, in_features)
    b = torch.randn(out_features) if bias else None

    expected = tF.linear(x, w, b)
    got = nF.linear(x, w, b)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-6)


@pytest.mark.parametrize("bias", [True, False])
def test_linear_module_matches_torch(bias):
    torch.manual_seed(0)
    in_features, out_features = 8, 5

    mine = nnn.Linear(in_features, out_features, bias=bias)
    theirs = tnn.Linear(in_features, out_features, bias=bias)
    # share weights so outputs must match exactly
    with torch.no_grad():
        theirs.weight.copy_(mine.weight)
        if bias:
            theirs.bias.copy_(mine.bias)

    x = torch.randn(4, in_features)
    assert torch.allclose(mine(x), theirs(x), atol=1e-6)


def test_linear_module_backward_matches_torch():
    torch.manual_seed(0)
    in_features, out_features = 8, 5

    mine = nnn.Linear(in_features, out_features)
    theirs = tnn.Linear(in_features, out_features)
    with torch.no_grad():
        theirs.weight.copy_(mine.weight)
        theirs.bias.copy_(mine.bias)

    x = torch.randn(4, in_features, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)

    mine(x).sum().backward()
    theirs(x2).sum().backward()

    assert torch.allclose(x.grad, x2.grad, atol=1e-6)
    assert torch.allclose(mine.weight.grad, theirs.weight.grad, atol=1e-6)
    assert torch.allclose(mine.bias.grad, theirs.bias.grad, atol=1e-6)
