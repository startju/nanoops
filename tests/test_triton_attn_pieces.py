"""Parity tests for the small Triton kernels covering the
remaining elementwise pieces of nanchat's CausalSelfAttention forward.
"""

import pytest
import torch

triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("triton kernels require CUDA", allow_module_level=True)

from nanoops.triton_kernels import (
    norm_qkv_rotary_projection,
    output_proj_residual,
    value_gate,
    rotary_qk_norm_scale,
)


# ─────────────────────────────────────────────────────────────────────
# norm_qkv_rotary_projection:
#   RMSNorm(x) @ W_qkv.T, with Q/K rotary + QK RMSNorm + scale fused before writeback
# ─────────────────────────────────────────────────────────────────────

def _norm_qkv_rotary_projection_ref(
    x,
    norm_weight,
    qkv_weight,
    cos,
    sin,
    n_head,
    n_kv_head,
    head_dim,
    scale,
    eps,
):
    x_rms_inv = torch.rsqrt((x.float() ** 2).mean(dim=-1, keepdim=True) + eps)
    x_hat = x * x_rms_inv.to(x.dtype) * norm_weight
    qkv = x_hat @ qkv_weight.t()
    q_flat, k_flat, v_flat = qkv.split(
        [n_head * head_dim, n_kv_head * head_dim, n_kv_head * head_dim],
        dim=-1,
    )
    q = q_flat.view(-1, n_head, head_dim)
    k = k_flat.view(-1, n_kv_head, head_dim)
    v = v_flat.view(-1, n_kv_head, head_dim)

    half = head_dim // 2
    cos_b = cos[:, None, :]
    sin_b = sin[:, None, :]

    def _rot_norm_scale(qk):
        lo, hi = qk[..., :half], qk[..., half:]
        rot_lo = lo * cos_b + hi * sin_b
        rot_hi = -lo * sin_b + hi * cos_b
        rotated = torch.cat([rot_lo, rot_hi], dim=-1)
        qk_rms_inv = torch.rsqrt((rotated.float() ** 2).mean(dim=-1, keepdim=True) + eps)
        return rotated * qk_rms_inv.to(rotated.dtype) * scale

    return _rot_norm_scale(q), _rot_norm_scale(k), v


def test_norm_qkv_rotary_projection_forward():
    torch.manual_seed(0)
    M, K, n_head, n_kv_head, head_dim = 32, 64, 4, 2, 32
    n_qkv = (n_head + 2 * n_kv_head) * head_dim
    x = torch.randn(M, K, dtype=torch.float32, device="cuda")
    norm_weight = torch.randn(K, dtype=torch.float32, device="cuda")
    qkv_weight = torch.randn(n_qkv, K, dtype=torch.float32, device="cuda") * 0.1
    cos = torch.randn(M, head_dim // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(M, head_dim // 2, dtype=torch.float32, device="cuda")

    q_ref, k_ref, v_ref = _norm_qkv_rotary_projection_ref(
        x, norm_weight, qkv_weight, cos, sin, n_head, n_kv_head, head_dim, 1.2, 1e-6
    )
    q_tri, k_tri, v_tri = norm_qkv_rotary_projection(
        x, norm_weight, qkv_weight, cos, sin, n_head, n_kv_head, head_dim, 1.2, 1e-6
    )

    for name, ref, got in [("q", q_ref, q_tri), ("k", k_ref, k_tri), ("v", v_ref, v_tri)]:
        assert torch.allclose(ref, got, atol=2e-3), \
            f"{name} max diff {(ref - got).abs().max():.4e}"


def test_norm_qkv_rotary_projection_backward():
    torch.manual_seed(0)
    M, K, n_head, n_kv_head, head_dim = 16, 32, 2, 1, 16
    n_qkv = (n_head + 2 * n_kv_head) * head_dim
    x0 = torch.randn(M, K, dtype=torch.float32, device="cuda")
    nw0 = torch.randn(K, dtype=torch.float32, device="cuda")
    qw0 = torch.randn(n_qkv, K, dtype=torch.float32, device="cuda") * 0.1
    cos = torch.randn(M, head_dim // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(M, head_dim // 2, dtype=torch.float32, device="cuda")

    qg = torch.randn(M, n_head, head_dim, dtype=torch.float32, device="cuda")
    kg = torch.randn(M, n_kv_head, head_dim, dtype=torch.float32, device="cuda")
    vg = torch.randn(M, n_kv_head, head_dim, dtype=torch.float32, device="cuda")

    x1, nw1, qw1 = (t.clone().requires_grad_() for t in (x0, nw0, qw0))
    q1, k1, v1 = _norm_qkv_rotary_projection_ref(
        x1, nw1, qw1, cos, sin, n_head, n_kv_head, head_dim, 1.2, 1e-6
    )
    torch.autograd.backward((q1, k1, v1), (qg, kg, vg))

    x2, nw2, qw2 = (t.clone().requires_grad_() for t in (x0, nw0, qw0))
    q2, k2, v2 = norm_qkv_rotary_projection(
        x2, nw2, qw2, cos, sin, n_head, n_kv_head, head_dim, 1.2, 1e-6
    )
    torch.autograd.backward((q2, k2, v2), (qg, kg, vg))

    for name, ref, got in [("x", x1.grad, x2.grad), ("norm_weight", nw1.grad, nw2.grad), ("qkv_weight", qw1.grad, qw2.grad)]:
        assert torch.allclose(ref, got, atol=5e-3), \
            f"{name}.grad max diff {(ref - got).abs().max():.4e}"


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
