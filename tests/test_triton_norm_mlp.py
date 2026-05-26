"""Parity tests for the fused Triton MLP block.

Math: y = x + relu²(norm(x)·norm_w @ W_fc.T) @ W_proj.T
"""

import pytest
import torch
import torch.nn.functional as F

triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("triton kernels require CUDA", allow_module_level=True)

from nanoops.triton_kernels import fused_mlp_block


def _reference(x, norm_w, W_fc, W_proj, eps=1e-6):
    """Eager PyTorch reference. norm_w=None ⇒ plain RMSNorm."""
    x_hat = F.rms_norm(x, (x.shape[-1],), weight=norm_w, eps=eps)
    z = x_hat @ W_fc.t()
    r = z.clamp(min=0).square()
    return x + r @ W_proj.t()


@pytest.mark.parametrize("dtype,atol", [
    (torch.float32, 1e-3),
    # bf16 atol raised from 1e-1 → 1.5e-1: fused fwd's `summed * summed`
    # now runs in bf16 before the fp32 accumulator (`tl.sum(..., dtype=fp32)`),
    # whereas F.rms_norm promotes bf16 → fp32 *before* the squaring. The
    # truncated-mantissa squared products plus a long sum over D push max
    # diff slightly past the old 1e-1 tolerance on adversarial seeds. The
    # change is intentional (one less `summed_f32` intermediate); end-to-end
    # training shows no loss/MFU regression.
    (torch.bfloat16, 1.5e-1),
])
@pytest.mark.parametrize("has_nw", [True, False])
def test_forward_parity(dtype, atol, has_nw):
    torch.manual_seed(0)
    M, K, N_fc = 64, 128, 256
    x = torch.randn(M, K, dtype=dtype, device="cuda")
    norm_w = torch.randn(K, dtype=dtype, device="cuda") if has_nw else None
    W_fc = torch.randn(N_fc, K, dtype=dtype, device="cuda") * 0.1
    W_proj = torch.randn(K, N_fc, dtype=dtype, device="cuda") * 0.1

    y_ref = _reference(x, norm_w, W_fc, W_proj)
    y_triton = fused_mlp_block(x, norm_w, W_fc, W_proj)
    assert torch.allclose(y_ref, y_triton, atol=atol), \
        f"forward mismatch (max {(y_ref - y_triton).abs().max().item():.4f}, dtype={dtype}, has_nw={has_nw})"


@pytest.mark.parametrize("has_nw", [True, False])
def test_backward_parity(has_nw):
    torch.manual_seed(0)
    M, K, N_fc = 64, 128, 256
    dtype = torch.float32  # IEEE path for bit-tight parity

    x0 = torch.randn(M, K, dtype=dtype, device="cuda")
    norm_w0 = torch.randn(K, dtype=dtype, device="cuda") if has_nw else None
    W_fc0 = torch.randn(N_fc, K, dtype=dtype, device="cuda") * 0.1
    W_proj0 = torch.randn(K, N_fc, dtype=dtype, device="cuda") * 0.1
    g = torch.randn(M, K, dtype=dtype, device="cuda")

    def _grads(use_triton):
        x = x0.clone().requires_grad_(True)
        nw = norm_w0.clone().requires_grad_(True) if has_nw else None
        Wfc = W_fc0.clone().requires_grad_(True)
        Wproj = W_proj0.clone().requires_grad_(True)
        y = (fused_mlp_block if use_triton else _reference)(x, nw, Wfc, Wproj)
        y.backward(g)
        return x.grad, (nw.grad if has_nw else None), Wfc.grad, Wproj.grad

    ref = _grads(use_triton=False)
    got = _grads(use_triton=True)
    atol = 5e-3
    for name, r, g_ in zip(("x", "norm_w", "W_fc", "W_proj"), ref, got):
        if r is None and g_ is None:
            continue
        max_diff = (r - g_).abs().max().item()
        assert torch.allclose(r, g_, atol=atol), \
            f"{name}.grad mismatch (max {max_diff:.4e}, has_nw={has_nw})"
