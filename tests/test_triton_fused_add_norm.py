"""Parity tests for fused_add_norm: y = norm(x + residual)."""

import pytest
import torch

triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("triton kernels require CUDA", allow_module_level=True)

from nanoops.triton_kernels import fused_add_norm


def _reference(x, residual, nw, eps=1e-6):
    summed = x + residual
    rms_inv = torch.rsqrt((summed.float() ** 2).mean(dim=-1, keepdim=True) + eps)
    y = (summed.float() * rms_inv * nw.float()).to(x.dtype)
    return y, summed


def test_fused_add_norm_forward():
    torch.manual_seed(0)
    M, D = 32, 128
    x = torch.randn(M, D, dtype=torch.float32, device="cuda")
    res = torch.randn(M, D, dtype=torch.float32, device="cuda")
    nw = torch.randn(D, dtype=torch.float32, device="cuda")

    y_ref, s_ref = _reference(x, res, nw)
    y_triton, s_triton = fused_add_norm(x, res, nw)
    assert torch.allclose(y_ref, y_triton, atol=1e-3), \
        f"y max diff {(y_ref - y_triton).abs().max():.4e}"
    assert torch.allclose(s_ref, s_triton, atol=1e-6), \
        f"summed max diff {(s_ref - s_triton).abs().max():.4e}"


def test_fused_add_norm_backward():
    torch.manual_seed(0)
    M, D = 32, 128
    x0 = torch.randn(M, D, dtype=torch.float32, device="cuda")
    r0 = torch.randn(M, D, dtype=torch.float32, device="cuda")
    nw0 = torch.randn(D, dtype=torch.float32, device="cuda")
    gy = torch.randn(M, D, dtype=torch.float32, device="cuda")
    gs = torch.randn(M, D, dtype=torch.float32, device="cuda")

    # Reference
    x1, r1, nw1 = (t.clone().requires_grad_() for t in (x0, r0, nw0))
    y_ref, s_ref = _reference(x1, r1, nw1)
    (y_ref * gy + s_ref * gs).sum().backward()

    # Triton
    x2, r2, nw2 = (t.clone().requires_grad_() for t in (x0, r0, nw0))
    y_triton, s_triton = fused_add_norm(x2, r2, nw2)
    (y_triton * gy + s_triton * gs).sum().backward()

    atol = 5e-3
    for name, ref, got in [
        ("x.grad", x1.grad, x2.grad),
        ("res.grad", r1.grad, r2.grad),
        ("nw.grad", nw1.grad, nw2.grad),
    ]:
        assert torch.allclose(ref, got, atol=atol), \
            f"{name} max diff {(ref - got).abs().max():.4e}"
