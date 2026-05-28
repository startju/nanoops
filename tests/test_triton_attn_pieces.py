"""Parity tests for the small Triton kernels covering the
remaining elementwise pieces of nanchat's CausalSelfAttention forward.
"""

import pytest
import torch

triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("triton kernels require CUDA", allow_module_level=True)

from nanoops.triton_fused_attn_qkv import (
    _norm_qkv_projection_fwd_op,
    _norm_qkv_projection_backward,
)
from nanoops.triton_kernels import (
    norm_qkv_projection,
    output_proj_residual,
    value_gate,
)


# ─────────────────────────────────────────────────────────────────────
# norm_qkv_projection:
#   RMSNorm(x) @ W_qkv.T, with Q/K rotary + QK RMSNorm + scale fused before writeback
# ─────────────────────────────────────────────────────────────────────

def _norm_qkv_projection_ref(
    x,
    norm_weight,
    ve_ids,
    ve_weight,
    ve_gate_channels,
    ve_gate_weight,
    q_weight,
    k_weight,
    v_weight,
    cos,
    sin,
    n_head,
    n_kv_head,
    head_dim,
    scale,
    eps,
):
    prefix_shape = x.shape[:-1]
    C = x.shape[-1]
    M = x.numel() // C
    x_2d = x.reshape(M, C)
    x_rms_inv = torch.rsqrt((x_2d.float() ** 2).mean(dim=-1, keepdim=True) + eps)
    x_hat = x_2d * x_rms_inv.to(x.dtype)
    if norm_weight is not None:
        x_hat = x_hat * norm_weight
    q = (x_hat @ q_weight.t()).view(-1, n_head, head_dim)
    k = (x_hat @ k_weight.t()).view(-1, n_kv_head, head_dim)
    v = (x_hat @ v_weight.t()).view(-1, n_kv_head, head_dim)
    if ve_ids is not None or ve_weight is not None:
        assert ve_gate_weight is not None
        assert ve_ids is not None and ve_weight is not None
        ve = ve_weight[ve_ids.reshape(-1)].reshape(M, n_kv_head, head_dim)
        gate = 3.0 * torch.sigmoid(
            x_hat[:, :ve_gate_channels].float() @ ve_gate_weight.float().t()
        )
        v = v + gate.to(v.dtype).view(M, n_kv_head, 1) * ve

    half = head_dim // 2
    assert cos.ndim == 4 and sin.ndim == 4
    assert cos.shape == sin.shape
    assert cos.shape[0] == 1 and cos.shape[2] == 1 and cos.shape[-1] == half
    T = cos.shape[1]
    assert M % T == 0
    B = M // T
    cos_b = cos.expand(B, T, 1, half).reshape(M, 1, half)
    sin_b = sin.expand(B, T, 1, half).reshape(M, 1, half)

    def _rot_norm_scale(qk):
        lo, hi = qk[..., :half], qk[..., half:]
        rot_lo = lo * cos_b + hi * sin_b
        rot_hi = -lo * sin_b + hi * cos_b
        rotated = torch.cat([rot_lo, rot_hi], dim=-1)
        qk_rms_inv = torch.rsqrt((rotated.float() ** 2).mean(dim=-1, keepdim=True) + eps)
        return rotated * qk_rms_inv.to(rotated.dtype) * scale

    return (
        _rot_norm_scale(q).view(*prefix_shape, n_head, head_dim),
        _rot_norm_scale(k).view(*prefix_shape, n_kv_head, head_dim),
        v.view(*prefix_shape, n_kv_head, head_dim),
    )


def test_norm_qkv_projection_forward():
    torch.manual_seed(0)
    B, T, K, n_head, n_kv_head, head_dim = 2, 16, 128, 4, 2, 128
    x = torch.randn(B, T, K, dtype=torch.float32, device="cuda")
    norm_weight = torch.randn(K, dtype=torch.float32, device="cuda")
    q_weight = torch.randn(n_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    k_weight = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    v_weight = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    cos = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")

    q_ref, k_ref, v_ref = _norm_qkv_projection_ref(
        x,
        norm_weight,
        None,
        None,
        1,
        None,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        n_head,
        n_kv_head,
        head_dim,
        1.2,
        1e-6,
    )
    q_tri, k_tri, v_tri = norm_qkv_projection(
        x,
        norm_weight,
        None,
        None,
        1,
        None,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        n_head,
        n_kv_head,
        head_dim,
        1.2,
        1e-6,
    )

    for name, ref, got in [
        ("q", q_ref, q_tri),
        ("k", k_ref, k_tri),
        ("v", v_ref, v_tri),
    ]:
        assert torch.allclose(ref, got, atol=5e-3), \
            f"{name} max diff {(ref - got).abs().max():.4e}"


def test_norm_qkv_projection_broadcast_rotary_table():
    torch.manual_seed(0)
    B, T, K, n_head, n_kv_head, head_dim = 2, 16, 128, 4, 2, 128
    M = B * T
    x = torch.randn(B, T, K, dtype=torch.float32, device="cuda")
    norm_weight = torch.randn(K, dtype=torch.float32, device="cuda")
    q_weight = torch.randn(n_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    k_weight = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    v_weight = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    cos = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")

    q_ref, k_ref, v_ref = _norm_qkv_projection_ref(
        x,
        norm_weight,
        None,
        None,
        1,
        None,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        n_head,
        n_kv_head,
        head_dim,
        1.2,
        1e-6,
    )
    q_tri, k_tri, v_tri = norm_qkv_projection(
        x,
        norm_weight,
        None,
        None,
        1,
        None,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        n_head,
        n_kv_head,
        head_dim,
        1.2,
        1e-6,
    )

    for name, ref, got in [
        ("q", q_ref, q_tri),
        ("k", k_ref, k_tri),
        ("v", v_ref, v_tri),
    ]:
        assert torch.allclose(ref, got, atol=5e-3), \
            f"{name} max diff {(ref - got).abs().max():.4e}"


def test_norm_qkv_projection_value_embedding_lookup_forward_backward():
    torch.manual_seed(0)
    B, T, K, n_head, n_kv_head, head_dim = 2, 8, 128, 2, 1, 128
    vocab_size = 11
    ve_gate_channels = 12
    x0 = torch.randn(B, T, K, dtype=torch.float32, device="cuda")
    nw0 = torch.randn(K, dtype=torch.float32, device="cuda")
    qw0 = torch.randn(n_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    kw0 = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    vw0 = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    ve_w0 = torch.randn(vocab_size, n_kv_head * head_dim, dtype=torch.float32, device="cuda")
    gw0 = torch.randn(n_kv_head, ve_gate_channels, dtype=torch.float32, device="cuda") * 0.1
    ve_ids = torch.randint(0, vocab_size, (B, T), dtype=torch.long, device="cuda")
    ve_ids[0, :4] = ve_ids[1, :4]
    cos = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")

    qg = torch.randn(B, T, n_head, head_dim, dtype=torch.float32, device="cuda")
    kg = torch.randn(B, T, n_kv_head, head_dim, dtype=torch.float32, device="cuda")
    vg = torch.randn(B, T, n_kv_head, head_dim, dtype=torch.float32, device="cuda")

    x1, nw1, qw1, kw1, vw1, ve_w1, gw1 = (
        t.clone().requires_grad_() for t in (x0, nw0, qw0, kw0, vw0, ve_w0, gw0)
    )
    q1, k1, v1 = _norm_qkv_projection_ref(
        x1,
        nw1,
        ve_ids,
        ve_w1,
        ve_gate_channels,
        gw1,
        qw1,
        kw1,
        vw1,
        cos,
        sin,
        n_head,
        n_kv_head,
        head_dim,
        1.2,
        1e-6,
    )
    torch.autograd.backward((q1, k1, v1), (qg, kg, vg))

    x2, nw2, qw2, kw2, vw2, ve_w2, gw2 = (
        t.clone().requires_grad_() for t in (x0, nw0, qw0, kw0, vw0, ve_w0, gw0)
    )
    q2, k2, v2 = norm_qkv_projection(
        x2,
        nw2,
        ve_ids,
        ve_w2,
        ve_gate_channels,
        gw2,
        qw2,
        kw2,
        vw2,
        cos,
        sin,
        n_head,
        n_kv_head,
        head_dim,
        1.2,
        1e-6,
    )
    torch.autograd.backward((q2, k2, v2), (qg, kg, vg))

    for name, ref, got, atol in [
        ("q", q1, q2, 5e-3),
        ("k", k1, k2, 5e-3),
        ("v", v1, v2, 5e-3),
        ("x.grad", x1.grad, x2.grad, 2e-2),
        ("norm_weight.grad", nw1.grad, nw2.grad, 2e-2),
        ("q_weight.grad", qw1.grad, qw2.grad, 4e-2),
        ("k_weight.grad", kw1.grad, kw2.grad, 4e-2),
        ("v_weight.grad", vw1.grad, vw2.grad, 4e-2),
        ("ve_weight.grad", ve_w1.grad, ve_w2.grad, 1e-2),
        ("ve_gate_weight.grad", gw1.grad, gw2.grad, 1e-2),
    ]:
        assert torch.allclose(ref, got, atol=atol), \
            f"{name} max diff {(ref - got).abs().max():.4e}"


def test_norm_qkv_projection_backward():
    torch.manual_seed(0)
    B, T, K, n_head, n_kv_head, head_dim = 2, 8, 128, 2, 1, 128
    x0 = torch.randn(B, T, K, dtype=torch.float32, device="cuda")
    nw0 = torch.randn(K, dtype=torch.float32, device="cuda")
    qw0 = torch.randn(n_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    kw0 = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    vw0 = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    cos = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")

    qg = torch.randn(B, T, n_head, head_dim, dtype=torch.float32, device="cuda")
    kg = torch.randn(B, T, n_kv_head, head_dim, dtype=torch.float32, device="cuda")
    vg = torch.randn(B, T, n_kv_head, head_dim, dtype=torch.float32, device="cuda")

    x1, nw1, qw1, kw1, vw1 = (t.clone().requires_grad_() for t in (x0, nw0, qw0, kw0, vw0))
    q1, k1, v1 = _norm_qkv_projection_ref(
        x1, nw1, None, None, 1, None, qw1, kw1, vw1, cos, sin, n_head, n_kv_head, head_dim, 1.2, 1e-6
    )
    torch.autograd.backward((q1, k1, v1), (qg, kg, vg))

    x2, nw2, qw2, kw2, vw2 = (t.clone().requires_grad_() for t in (x0, nw0, qw0, kw0, vw0))
    q2, k2, v2 = norm_qkv_projection(
        x2, nw2, None, None, 1, None, qw2, kw2, vw2, cos, sin, n_head, n_kv_head, head_dim, 1.2, 1e-6
    )
    torch.autograd.backward((q2, k2, v2), (qg, kg, vg))

    for name, ref, got in [
        ("x", x1.grad, x2.grad),
        ("norm_weight", nw1.grad, nw2.grad),
        ("q_weight", qw1.grad, qw2.grad),
        ("k_weight", kw1.grad, kw2.grad),
        ("v_weight", vw1.grad, vw2.grad),
    ]:
        assert torch.allclose(ref, got, atol=4e-2), \
            f"{name}.grad max diff {(ref - got).abs().max():.4e}"


def test_norm_qkv_projection_backward_broadcast_rotary_table():
    torch.manual_seed(0)
    B, T, K, n_head, n_kv_head, head_dim = 2, 8, 128, 2, 1, 128
    M = B * T
    x0 = torch.randn(B, T, K, dtype=torch.float32, device="cuda")
    nw0 = torch.randn(K, dtype=torch.float32, device="cuda")
    qw0 = torch.randn(n_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    kw0 = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    vw0 = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    cos = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")

    qg = torch.randn(B, T, n_head, head_dim, dtype=torch.float32, device="cuda")
    kg = torch.randn(B, T, n_kv_head, head_dim, dtype=torch.float32, device="cuda")
    vg = torch.randn(B, T, n_kv_head, head_dim, dtype=torch.float32, device="cuda")

    x1, nw1, qw1, kw1, vw1 = (t.clone().requires_grad_() for t in (x0, nw0, qw0, kw0, vw0))
    q1, k1, v1 = _norm_qkv_projection_ref(
        x1, nw1, None, None, 1, None, qw1, kw1, vw1, cos, sin, n_head, n_kv_head, head_dim, 1.2, 1e-6
    )
    torch.autograd.backward((q1, k1, v1), (qg, kg, vg))

    x2, nw2, qw2, kw2, vw2 = (t.clone().requires_grad_() for t in (x0, nw0, qw0, kw0, vw0))
    q2, k2, v2 = norm_qkv_projection(
        x2, nw2, None, None, 1, None, qw2, kw2, vw2, cos, sin, n_head, n_kv_head, head_dim, 1.2, 1e-6
    )
    torch.autograd.backward((q2, k2, v2), (qg, kg, vg))

    for name, ref, got in [
        ("x", x1.grad, x2.grad),
        ("norm_weight", nw1.grad, nw2.grad),
        ("q_weight", qw1.grad, qw2.grad),
        ("k_weight", kw1.grad, kw2.grad),
        ("v_weight", vw1.grad, vw2.grad),
    ]:
        assert torch.allclose(ref, got, atol=4e-2), \
            f"{name}.grad max diff {(ref - got).abs().max():.4e}"


def test_norm_qkv_projection_backward_bf16_smoke():
    torch.manual_seed(0)
    B, T, K, n_head, n_kv_head, head_dim = 2, 8, 128, 2, 1, 128
    x = torch.randn(B, T, K, dtype=torch.bfloat16, device="cuda").requires_grad_()
    norm_weight = torch.randn(K, dtype=torch.bfloat16, device="cuda").requires_grad_()
    q_weight = (
        torch.randn(n_head * head_dim, K, dtype=torch.bfloat16, device="cuda") * 0.1
    ).requires_grad_()
    k_weight = (
        torch.randn(n_kv_head * head_dim, K, dtype=torch.bfloat16, device="cuda") * 0.1
    ).requires_grad_()
    v_weight = (
        torch.randn(n_kv_head * head_dim, K, dtype=torch.bfloat16, device="cuda") * 0.1
    ).requires_grad_()
    cos = torch.randn(1, T, 1, head_dim // 2, dtype=torch.bfloat16, device="cuda")
    sin = torch.randn(1, T, 1, head_dim // 2, dtype=torch.bfloat16, device="cuda")

    q, k, v = norm_qkv_projection(
        x,
        norm_weight,
        None,
        None,
        1,
        None,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        n_head,
        n_kv_head,
        head_dim,
        1.2,
        1e-6,
    )
    (q.float().sum() + k.float().sum() + v.float().sum()).backward()

    for name, grad in [
        ("x", x.grad),
        ("norm_weight", norm_weight.grad),
        ("q_weight", q_weight.grad),
        ("k_weight", k_weight.grad),
        ("v_weight", v_weight.grad),
    ]:
        assert torch.isfinite(grad.float()).all(), f"{name}.grad contains non-finite values"


def test_norm_qkv_projection_no_norm_weight_backward():
    torch.manual_seed(0)
    B, T, K, n_head, n_kv_head, head_dim = 2, 8, 128, 2, 1, 64
    x0 = torch.randn(B, T, K, dtype=torch.float32, device="cuda")
    qw0 = torch.randn(n_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    kw0 = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    vw0 = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    cos = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")

    qg = torch.randn(B, T, n_head, head_dim, dtype=torch.float32, device="cuda")
    kg = torch.randn(B, T, n_kv_head, head_dim, dtype=torch.float32, device="cuda")
    vg = torch.randn(B, T, n_kv_head, head_dim, dtype=torch.float32, device="cuda")

    x1, qw1, kw1, vw1 = (t.clone().requires_grad_() for t in (x0, qw0, kw0, vw0))
    q1, k1, v1 = _norm_qkv_projection_ref(
        x1, None, None, None, 1, None, qw1, kw1, vw1, cos, sin, n_head, n_kv_head, head_dim, 1.2, 1e-6
    )
    torch.autograd.backward((q1, k1, v1), (qg, kg, vg))

    x2, qw2, kw2, vw2 = (t.clone().requires_grad_() for t in (x0, qw0, kw0, vw0))
    q2, k2, v2 = norm_qkv_projection(
        x2, None, None, None, 1, None, qw2, kw2, vw2, cos, sin, n_head, n_kv_head, head_dim, 1.2, 1e-6
    )
    torch.autograd.backward((q2, k2, v2), (qg, kg, vg))

    for name, ref, got in [
        ("x", x1.grad, x2.grad),
        ("q_weight", qw1.grad, qw2.grad),
        ("k_weight", kw1.grad, kw2.grad),
        ("v_weight", vw1.grad, vw2.grad),
    ]:
        assert torch.allclose(ref, got, atol=4e-2), \
            f"{name}.grad max diff {(ref - got).abs().max():.4e}"


def test_norm_qkv_projection_backward_formula():
    torch.manual_seed(0)
    M, K, n_head, n_kv_head, head_dim = 16, 128, 2, 1, 128
    x0 = torch.randn(M, K, dtype=torch.float32, device="cuda")
    nw0 = torch.randn(K, dtype=torch.float32, device="cuda")
    qw0 = torch.randn(n_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    kw0 = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    vw0 = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    cos = torch.randn(1, M, 1, head_dim // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(1, M, 1, head_dim // 2, dtype=torch.float32, device="cuda")

    qg = torch.randn(M, n_head, head_dim, dtype=torch.float32, device="cuda")
    kg = torch.randn(M, n_kv_head, head_dim, dtype=torch.float32, device="cuda")
    vg = torch.randn(M, n_kv_head, head_dim, dtype=torch.float32, device="cuda")

    x1, nw1, qw1, kw1, vw1 = (t.clone().requires_grad_() for t in (x0, nw0, qw0, kw0, vw0))
    q1, k1, v1 = _norm_qkv_projection_ref(
        x1, nw1, None, None, 1, None, qw1, kw1, vw1, cos, sin, n_head, n_kv_head, head_dim, 1.2, 1e-6
    )
    torch.autograd.backward((q1, k1, v1), (qg, kg, vg))

    q_saved, k_saved, _v_saved, rms_inv, qk_rms_inv = _norm_qkv_projection_fwd_op(
        x0.view(1, M, K),
        nw0,
        None,
        None,
        1,
        None,
        qw0,
        kw0,
        vw0,
        cos,
        sin,
        n_head,
        n_kv_head,
        head_dim,
        1.2,
        1e-6,
    )
    (
        dx,
        d_norm_weight,
        d_q_weight,
        d_k_weight,
        d_v_weight,
        d_ve_weight,
        d_ve_gate_weight,
    ) = _norm_qkv_projection_backward(
        x0,
        nw0,
        qw0,
        kw0,
        vw0,
        cos,
        sin,
        qg,
        kg,
        vg,
        n_head,
        n_kv_head,
        head_dim,
        1.2,
        saved_rms_inv=rms_inv,
        saved_q=q_saved.view(M, n_head, head_dim),
        saved_k=k_saved.view(M, n_kv_head, head_dim),
        saved_qk_rms_inv=qk_rms_inv.view(M, n_head + n_kv_head),
    )
    assert d_ve_weight is None
    assert d_ve_gate_weight is None

    for name, ref, got in [
        ("x", x1.grad, dx),
        ("norm_weight", nw1.grad, d_norm_weight),
        ("q_weight", qw1.grad, d_q_weight),
        ("k_weight", kw1.grad, d_k_weight),
        ("v_weight", vw1.grad, d_v_weight),
    ]:
        assert torch.allclose(ref, got, atol=4e-2), \
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
