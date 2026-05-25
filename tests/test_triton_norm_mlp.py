"""Parity tests for the fused Triton MLP block.

Math: summed = x + residual_in
      y = summed + relu²(norm(summed) @ W_fc.T) @ W_proj.T
"""

import pytest
import torch
import torch.nn.functional as F

triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("triton kernels require CUDA", allow_module_level=True)

from nanoops.triton_kernels import fused_mlp_block


def _reference(x, residual_in, norm_w, W_fc, W_proj, eps=1e-6):
    """Eager PyTorch reference."""
    summed = x + residual_in
    x_hat = F.rms_norm(summed, (summed.shape[-1],), weight=norm_w, eps=eps)
    z = x_hat @ W_fc.t()
    r = z.clamp(min=0).square()
    return summed + r @ W_proj.t()


@pytest.mark.parametrize("dtype,atol", [
    (torch.float32, 1e-3),
    (torch.bfloat16, 1e-1),
])
def test_forward_parity(dtype, atol):
    torch.manual_seed(0)
    M, K, N_fc = 64, 128, 256
    x = torch.randn(M, K, dtype=dtype, device="cuda")
    res = torch.randn(M, K, dtype=dtype, device="cuda")
    norm_w = torch.randn(K, dtype=dtype, device="cuda")
    W_fc = torch.randn(N_fc, K, dtype=dtype, device="cuda") * 0.1
    W_proj = torch.randn(K, N_fc, dtype=dtype, device="cuda") * 0.1

    y_ref = _reference(x, res, norm_w, W_fc, W_proj)
    y_triton = fused_mlp_block(x, res, norm_w, W_fc, W_proj)
    assert torch.allclose(y_ref, y_triton, atol=atol), \
        f"forward mismatch (max {(y_ref - y_triton).abs().max().item():.4f}, dtype={dtype})"


def test_backward_parity():
    torch.manual_seed(0)
    M, K, N_fc = 64, 128, 256
    dtype = torch.float32  # IEEE path for bit-tight parity

    x0 = torch.randn(M, K, dtype=dtype, device="cuda")
    res0 = torch.randn(M, K, dtype=dtype, device="cuda")
    norm_w0 = torch.randn(K, dtype=dtype, device="cuda")
    W_fc0 = torch.randn(N_fc, K, dtype=dtype, device="cuda") * 0.1
    W_proj0 = torch.randn(K, N_fc, dtype=dtype, device="cuda") * 0.1
    g = torch.randn(M, K, dtype=dtype, device="cuda")

    # Reference
    x1, res1, nw1, Wfc1, Wproj1 = (t.clone().requires_grad_(True)
                                    for t in (x0, res0, norm_w0, W_fc0, W_proj0))
    y_ref = _reference(x1, res1, nw1, Wfc1, Wproj1)
    y_ref.backward(g)

    # Triton fused
    x2, res2, nw2, Wfc2, Wproj2 = (t.clone().requires_grad_(True)
                                    for t in (x0, res0, norm_w0, W_fc0, W_proj0))
    y_triton = fused_mlp_block(x2, res2, nw2, Wfc2, Wproj2)
    y_triton.backward(g)

    atol = 5e-3
    for name, ref, got in [
        ("x.grad", x1.grad, x2.grad),
        ("residual_in.grad", res1.grad, res2.grad),
        ("norm_w.grad", nw1.grad, nw2.grad),
        ("W_fc.grad", Wfc1.grad, Wfc2.grad),
        ("W_proj.grad", Wproj1.grad, Wproj2.grad),
    ]:
        max_diff = (ref - got).abs().max().item()
        assert torch.allclose(ref, got, atol=atol), \
            f"{name} mismatch (max {max_diff:.4e})"
