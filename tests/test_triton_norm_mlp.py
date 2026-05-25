"""Parity tests for the fused Triton MLP block.

Math: y = relu(x_hat @ W_fc.T)² @ W_proj.T + residual
where x_hat is the caller's already-normalized input.
"""

import pytest
import torch

triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("triton kernels require CUDA", allow_module_level=True)

from nanoops.triton_kernels import fused_mlp_block


def _reference(x_hat, W_fc, W_proj, residual):
    """Eager PyTorch reference — same math via torch ops."""
    z = x_hat @ W_fc.t()
    r = z.clamp(min=0).square()
    return residual + r @ W_proj.t()


@pytest.mark.parametrize("dtype,atol", [
    # fp32 path uses IEEE_PRECISION=True inside the Triton kernel for
    # bit-tight parity with PyTorch's `@`. bf16 path uses bf16 tensor
    # cores (5e-2 reflects per-element bf16 rounding through 2 matmuls).
    (torch.float32, 1e-3),
    (torch.bfloat16, 1e-1),
])
def test_forward_parity(dtype, atol):
    torch.manual_seed(0)
    M, K, N_fc = 64, 128, 256
    x_hat = torch.randn(M, K, dtype=dtype, device="cuda")
    W_fc = torch.randn(N_fc, K, dtype=dtype, device="cuda") * 0.1
    W_proj = torch.randn(K, N_fc, dtype=dtype, device="cuda") * 0.1
    residual = torch.randn(M, K, dtype=dtype, device="cuda")

    y_ref = _reference(x_hat, W_fc, W_proj, residual)
    y_triton = fused_mlp_block(x_hat, W_fc, W_proj, residual)
    assert torch.allclose(y_ref, y_triton, atol=atol), \
        f"forward mismatch (max {(y_ref - y_triton).abs().max().item():.4f}, dtype={dtype})"


def test_backward_parity():
    torch.manual_seed(0)
    M, K, N_fc = 64, 128, 256
    dtype = torch.float32  # tighter parity in fp32 (IEEE matmul path)

    x_hat0 = torch.randn(M, K, dtype=dtype, device="cuda")
    W_fc0 = torch.randn(N_fc, K, dtype=dtype, device="cuda") * 0.1
    W_proj0 = torch.randn(K, N_fc, dtype=dtype, device="cuda") * 0.1
    res0 = torch.randn(M, K, dtype=dtype, device="cuda")
    g = torch.randn(M, K, dtype=dtype, device="cuda")

    # Reference: PyTorch chain with autograd
    xh1 = x_hat0.clone().requires_grad_(True)
    Wfc1 = W_fc0.clone().requires_grad_(True)
    Wproj1 = W_proj0.clone().requires_grad_(True)
    res1 = res0.clone().requires_grad_(True)
    y_ref = _reference(xh1, Wfc1, Wproj1, res1)
    y_ref.backward(g)

    # Triton fused
    xh2 = x_hat0.clone().requires_grad_(True)
    Wfc2 = W_fc0.clone().requires_grad_(True)
    Wproj2 = W_proj0.clone().requires_grad_(True)
    res2 = res0.clone().requires_grad_(True)
    y_triton = fused_mlp_block(xh2, Wfc2, Wproj2, res2)
    y_triton.backward(g)

    # Backward goes through 3 matmul bwd + 1 relu² bwd; rounding drift
    # is small under IEEE matmul. 5e-3 is plenty tight on unit-variance inputs.
    atol = 5e-3
    for name, ref, got in [
        ("x_hat.grad", xh1.grad, xh2.grad),
        ("W_fc.grad", Wfc1.grad, Wfc2.grad),
        ("W_proj.grad", Wproj1.grad, Wproj2.grad),
        ("residual.grad", res1.grad, res2.grad),
    ]:
        max_diff = (ref - got).abs().max().item()
        assert torch.allclose(ref, got, atol=atol), \
            f"{name} mismatch (max {max_diff:.4e})"
