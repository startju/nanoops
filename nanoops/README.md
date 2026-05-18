# nanoops

> 中文版：[README_zh.md](README_zh.md)

A from-scratch reimplementation of the PyTorch operators used by nanochat.

**All operators here are for teaching purposes.** They prioritize readability and
showing the math over performance, fused kernels, or edge-case coverage. For real
training, use `torch.nn` / `torch.nn.functional` — this package exists so you can
read the implementation, step through it in a debugger, and compare its behavior
to PyTorch side by side.

## Layout

The public API mirrors PyTorch so nanochat code can swap implementations by
changing imports only.

| File | Mirrors | Contents |
| --- | --- | --- |
| `nn.py` | `torch.nn` | Module-style ops (`Linear`, ...) |
| `functional.py` | `torch.nn.functional` | Functional ops + `autograd.Function` subclasses |

## Conventions

- Module init schemes match `torch.nn` exactly, so weights from a `nanoops`
  module and the corresponding `torch.nn` module are statistically interchangeable.
- `autograd.Function` subclasses (e.g. `Matmul`) use the legacy
  `forward(ctx, ...)` signature for clarity — the math and the cached tensors
  sit next to each other.
- Shape restrictions (e.g. 2D-only `Matmul`) are intentional: they keep the
  implementation small enough to read in one sitting. Higher-rank versions are
  left as an exercise.

## Parity tests

`tests/test_nanoops.py` checks each op against its `torch` counterpart on both
forward and backward. Run with:

```
pytest tests/test_nanoops.py
```

When adding a new op, add a parity test alongside it.

## TODO

Sequenced by what nanochat actually depends on. Tier 1 is enough to run a forward
pass of the model on toy weights; later tiers unlock training, sampling, and the
fast-path optimizations.

### Tier 1 — core forward pass

- [x] `nn.Linear` / `F.linear`
- [ ] `nn.Embedding`
- [ ] `F.rms_norm` (used as the only normalization)
- [ ] `F.relu` (MLP uses `relu(x) ** 2`)
- [ ] `F.softmax`
- [ ] `F.cross_entropy` (with `ignore_index`)
- [ ] `torch.arange`, `torch.outer`, `torch.cat`, `torch.stack`
- [ ] `torch.sigmoid`, `torch.tanh` (gates + logit softcap)

### Tier 2 — attention & generation

- [ ] Rotary embeddings: cos/sin precompute + `apply_rotary_emb`
- [ ] `F.scaled_dot_product_attention` (start with the naive `softmax(QK/√d) V`)
- [ ] `torch.topk`, `torch.multinomial`, `torch.argmax` (for `engine.py` sampling)
- [ ] `torch.where`, `torch.roll` (eval / loss masking)

### Tier 3 — training loop

- [ ] `nn.init.normal_`, `uniform_`, `zeros_`, `constant_`
- [ ] AdamW step (fused-style, no `torch.compile`)
- [ ] Muon optimizer (matrices only; embeddings stay on AdamW)
- [ ] Param-group routing (matrix → Muon, embedding/scalar → AdamW)

### Tier 4 — performance / advanced (optional)

- [ ] FP8 matmul wrapper around `torch._scaled_mm` + custom `autograd.Function`
- [ ] FlashAttention-3 shim with SDPA fallback (mirrors `nanochat/flash_attention.py`)
- [ ] DDP variant of the Muon+AdamW optimizer
- [ ] `torch.compile` compatibility pass

### Conventions for each new op

1. Implement in `nn.py` or `functional.py` to match the PyTorch import path.
2. Add a parity test in `tests/test_nanoops.py` covering forward **and** backward.
3. Keep the implementation small enough to read in one screen — no fused
   kernels, no shape-generalization beyond what nanochat actually needs.
