"""Parity tests for the small Triton kernels covering the
remaining elementwise pieces of nanchat's CausalSelfAttention forward.
"""

import pytest
import torch

triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("triton kernels require CUDA", allow_module_level=True)

from nanoops.triton_kernels import (
    output_proj_residual,
    value_gate,
    rotary_qk_norm_scale,
)


# ─────────────────────────────────────────────────────────────────────
# output_proj_residual:  y = residual + attn_out @ proj_weight.T
# ─────────────────────────────────────────────────────────────────────

def test_output_proj_residual_forward():
    torch.manual_seed(0)
    M, D_in, D_out = 32, 64, 48
    attn_out = torch.randn(M, D_in, dtype=torch.float32, device="cuda")
    proj_w = torch.randn(D_out, D_in, dtype=torch.float32, device="cuda") * 0.1
    res = torch.randn(M, D_out, dtype=torch.float32, device="cuda")
    y_ref = torch.addmm(res, attn_out, proj_w.t())
    y_triton = output_proj_residual(attn_out, proj_w, res)
    assert torch.allclose(y_ref, y_triton, atol=1e-3), \
        f"fwd max diff {(y_ref - y_triton).abs().max():.4e}"


def test_output_proj_residual_backward():
    torch.manual_seed(0)
    M, D_in, D_out = 32, 64, 48
    a0 = torch.randn(M, D_in, dtype=torch.float32, device="cuda")
    w0 = torch.randn(D_out, D_in, dtype=torch.float32, device="cuda") * 0.1
    r0 = torch.randn(M, D_out, dtype=torch.float32, device="cuda")
    g = torch.randn(M, D_out, dtype=torch.float32, device="cuda")

    a1, w1, r1 = a0.clone().requires_grad_(), w0.clone().requires_grad_(), r0.clone().requires_grad_()
    torch.addmm(r1, a1, w1.t()).backward(g)

    a2, w2, r2 = a0.clone().requires_grad_(), w0.clone().requires_grad_(), r0.clone().requires_grad_()
    output_proj_residual(a2, w2, r2).backward(g)

    for name, ref, got in [("a", a1.grad, a2.grad), ("w", w1.grad, w2.grad), ("r", r1.grad, r2.grad)]:
        assert torch.allclose(ref, got, atol=5e-3), \
            f"{name}.grad max diff {(ref - got).abs().max():.4e}"


# ─────────────────────────────────────────────────────────────────────
# value_gate:  out = v + 3 * sigmoid(x[:, :ch] @ gate_w.T) * ve
# ─────────────────────────────────────────────────────────────────────

def _value_gate_ref(v, ve, x, gate_w):
    ve_gate_ch = gate_w.shape[1]
    gate = 3.0 * torch.sigmoid(x[:, :ve_gate_ch] @ gate_w.t())
    return v + gate * ve


def test_value_gate_forward():
    torch.manual_seed(0)
    M, D_v, D_x, ch = 32, 64, 128, 16
    v = torch.randn(M, D_v, dtype=torch.float32, device="cuda")
    ve = torch.randn(M, D_v, dtype=torch.float32, device="cuda")
    x = torch.randn(M, D_x, dtype=torch.float32, device="cuda")
    gate_w = torch.randn(D_v, ch, dtype=torch.float32, device="cuda") * 0.1
    out_ref = _value_gate_ref(v, ve, x, gate_w)
    out_triton = value_gate(v, ve, x, gate_w)
    assert torch.allclose(out_ref, out_triton, atol=1e-3), \
        f"max diff {(out_ref - out_triton).abs().max():.4e}"


def test_value_gate_backward():
    torch.manual_seed(0)
    M, D_v, D_x, ch = 32, 64, 128, 16
    v0 = torch.randn(M, D_v, dtype=torch.float32, device="cuda")
    ve0 = torch.randn(M, D_v, dtype=torch.float32, device="cuda")
    x0 = torch.randn(M, D_x, dtype=torch.float32, device="cuda")
    gw0 = torch.randn(D_v, ch, dtype=torch.float32, device="cuda") * 0.1
    g = torch.randn(M, D_v, dtype=torch.float32, device="cuda")

    v1, ve1, x1, gw1 = (t.clone().requires_grad_() for t in (v0, ve0, x0, gw0))
    _value_gate_ref(v1, ve1, x1, gw1).backward(g)

    v2, ve2, x2, gw2 = (t.clone().requires_grad_() for t in (v0, ve0, x0, gw0))
    value_gate(v2, ve2, x2, gw2).backward(g)

    atol = 5e-3
    for name, ref, got in [
        ("v", v1.grad, v2.grad),
        ("ve", ve1.grad, ve2.grad),
        ("x[:, :ch]", x1.grad[:, :ch], x2.grad[:, :ch]),
        ("gate_w", gw1.grad, gw2.grad),
    ]:
        assert torch.allclose(ref, got, atol=atol), \
            f"{name}.grad max diff {(ref - got).abs().max():.4e}"


# ─────────────────────────────────────────────────────────────────────
# rotary_qk_norm_scale: rotary + RMSNorm + scale
# ─────────────────────────────────────────────────────────────────────

def _rotary_qk_norm_scale_ref(qk, cos, sin, scale, eps):
    half = qk.shape[-1] // 2
    x1, x2 = qk[..., :half], qk[..., half:]
    y1 = x1 * cos + x2 * sin
    y2 = -x1 * sin + x2 * cos
    y = torch.cat([y1, y2], dim=-1)
    rms_inv = torch.rsqrt((y.float() ** 2).mean(dim=-1, keepdim=True) + eps)
    return (y * rms_inv * scale).to(qk.dtype)


def test_rotary_qk_norm_scale_forward():
    torch.manual_seed(0)
    M, D = 32, 64
    qk = torch.randn(M, D, dtype=torch.float32, device="cuda")
    cos = torch.randn(M, D // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(M, D // 2, dtype=torch.float32, device="cuda")
    scale = 1.2
    out_ref = _rotary_qk_norm_scale_ref(qk, cos, sin, scale, eps=1e-6)
    out_triton = rotary_qk_norm_scale(qk, cos, sin, scale, eps=1e-6)
    assert torch.allclose(out_ref, out_triton, atol=1e-3), \
        f"max diff {(out_ref - out_triton).abs().max():.4e}"


def test_rotary_qk_norm_scale_backward():
    torch.manual_seed(0)
    M, D = 32, 64
    qk0 = torch.randn(M, D, dtype=torch.float32, device="cuda")
    cos = torch.randn(M, D // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(M, D // 2, dtype=torch.float32, device="cuda")
    g = torch.randn(M, D, dtype=torch.float32, device="cuda")

    q1 = qk0.clone().requires_grad_(True)
    _rotary_qk_norm_scale_ref(q1, cos, sin, 1.2, 1e-6).backward(g)

    q2 = qk0.clone().requires_grad_(True)
    rotary_qk_norm_scale(q2, cos, sin, 1.2, 1e-6).backward(g)

    assert torch.allclose(q1.grad, q2.grad, atol=5e-3), \
        f"qk.grad max diff {(q1.grad - q2.grad).abs().max():.4e}"
