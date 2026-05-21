"""Parity tests for the fused Triton MLP block.

Math: y = relu(RMSNorm(x) @ W_fc.T)² @ W_proj.T
"""

import pytest
import torch

triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("triton kernels require CUDA", allow_module_level=True)

from nanoops.triton_kernels import norm_mlp_relu_square


def _reference(x, norm_w, W_fc, W_proj, eps=1e-6):
    """Eager PyTorch reference — same math via torch ops."""
    # RMSNorm
    rms_inv = torch.rsqrt((x.float() ** 2).mean(dim=-1, keepdim=True) + eps)
    x_hat = (x.float() * rms_inv * norm_w.float()).to(x.dtype)
    # c_fc
    z = x_hat @ W_fc.t()
    # ReluSquare
    r = z.clamp(min=0).square()
    # c_proj
    return r @ W_proj.t()


@pytest.mark.parametrize("dtype,atol", [
    # Tolerances reflect pure matmul summation-order rounding. The
    # kernel uses `tl.dot(..., input_precision="ieee")` so fp32 inputs
    # actually use fp32 matmul (Triton's default would downcast to
    # TF32 on Ampere — only 10-bit mantissa — and that produced ~1%
    # drift before the explicit IEEE flag).
    (torch.float32, 1e-3),
    (torch.bfloat16, 1e-1),
])
def test_forward_parity(dtype, atol):
    torch.manual_seed(0)
    M, K, N_fc = 64, 128, 256
    x = torch.randn(M, K, dtype=dtype, device="cuda")
    norm_w = torch.randn(K, dtype=dtype, device="cuda")
    W_fc = torch.randn(N_fc, K, dtype=dtype, device="cuda") * 0.1
    W_proj = torch.randn(K, N_fc, dtype=dtype, device="cuda") * 0.1

    y_ref = _reference(x, norm_w, W_fc, W_proj)
    y_triton = norm_mlp_relu_square(x, norm_w, W_fc, W_proj)
    assert torch.allclose(y_ref, y_triton, atol=atol), \
        f"forward mismatch (max {(y_ref - y_triton).abs().max().item():.4f}, dtype={dtype})"


def test_backward_parity():
    torch.manual_seed(0)
    M, K, N_fc = 64, 128, 256
    dtype = torch.float32  # tighter parity in fp32

    x0 = torch.randn(M, K, dtype=dtype, device="cuda")
    norm_w0 = torch.randn(K, dtype=dtype, device="cuda")
    W_fc0 = torch.randn(N_fc, K, dtype=dtype, device="cuda") * 0.1
    W_proj0 = torch.randn(K, N_fc, dtype=dtype, device="cuda") * 0.1
    g = torch.randn(M, K, dtype=dtype, device="cuda")

    # Reference: PyTorch chain with autograd
    x1 = x0.clone().requires_grad_(True)
    nw1 = norm_w0.clone().requires_grad_(True)
    Wfc1 = W_fc0.clone().requires_grad_(True)
    Wproj1 = W_proj0.clone().requires_grad_(True)
    y_ref = _reference(x1, nw1, Wfc1, Wproj1)
    y_ref.backward(g)

    # Triton fused
    x2 = x0.clone().requires_grad_(True)
    nw2 = norm_w0.clone().requires_grad_(True)
    Wfc2 = W_fc0.clone().requires_grad_(True)
    Wproj2 = W_proj0.clone().requires_grad_(True)
    y_triton = norm_mlp_relu_square(x2, nw2, Wfc2, Wproj2)
    y_triton.backward(g)

    # Backward goes through 4 matmuls + RMSNorm bwd reduction; the
    # rounding drift cascades slightly more than the forward 2-matmul
    # chain. 5e-3 is plenty tight given inputs of unit variance.
    atol = 5e-3
    for name, ref, got in [
        ("x.grad", x1.grad, x2.grad),
        ("norm_w.grad", nw1.grad, nw2.grad),
        ("W_fc.grad", Wfc1.grad, Wfc2.grad),
        ("W_proj.grad", Wproj1.grad, Wproj2.grad),
    ]:
        max_diff = (ref - got).abs().max().item()
        assert torch.allclose(ref, got, atol=atol), \
            f"{name} mismatch (max {max_diff:.4e})"
