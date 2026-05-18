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

- Module init schemes are deliberate, not always matching `torch.nn`. Where
  PyTorch's default has a historical wart (e.g. `Linear`'s
  `kaiming_uniform_(a=sqrt(5))`), nanoops picks the principled value (`a=1`)
  and documents the divergence in the class docstring.
- `autograd.Function` subclasses (e.g. `Mm`, `Add`, `Lookup`) use the legacy
  `forward(ctx, ...)` signature for clarity — the math and the cached tensors
  sit next to each other.
- Shape restrictions (2D-only `Mm`, 1D-only `Lookup`) are intentional: they
  keep the autograd primitive small enough to read in one sitting. Higher-rank
  handling is done by the caller (see `linear` / `embedding` in
  `functional.py`, which flatten + unflatten around the 2D/1D core).

## Parity tests

`tests/test_nanoops.py` checks each op against its `torch` counterpart on both
forward and backward. Run with:

```
pytest tests/test_nanoops.py
```

When adding a new op, add a parity test alongside it.

## TODO

**Scope: only ops with a meaningful autograd backward.** Optimizers
(AdamW/Muon), parameter init, discrete sampling (`topk`/`argmax`/`multinomial`),
constant generators (`arange`, rotary cos/sin tables), DDP, and `torch.compile`
all use PyTorch directly — nanoops is about teaching backward, not replicating
every utility.

Sequenced by what nanochat actually depends on. Tier 1 is enough to run a
forward + backward pass through the core blocks; Tier 2 adds attention; Tier 3
adds optional fast-path variants.

### Tier 1 — core blocks

- [x] `nn.Linear` / `F.linear`
- [ ] `nn.Embedding`
- [ ] `F.rms_norm` (used as the only normalization)
- [ ] `F.relu` (MLP uses `relu(x) ** 2`)
- [ ] `F.softmax`
- [ ] `F.cross_entropy` (with `ignore_index`)
- [ ] `torch.outer`, `torch.cat`, `torch.stack`
- [ ] `torch.sigmoid`, `torch.tanh` (gates + logit softcap)

### Tier 2 — attention

- [ ] `apply_rotary_emb` (cos/sin tables stay on PyTorch)
- [ ] `F.scaled_dot_product_attention` (start with the naive `softmax(QK/√d) V`)
- [ ] `torch.where`, `torch.roll` (eval / loss masking)

### Tier 3 — performance / advanced (optional)

- [ ] FP8 matmul wrapper around `torch._scaled_mm` + custom `autograd.Function`
- [ ] FlashAttention-3 shim with SDPA fallback (mirrors `nanochat/flash_attention.py`)

### Conventions for each new op

1. Implement in `nn.py` or `functional.py` to match the PyTorch import path.
2. Add a parity test in `tests/test_nanoops.py` covering forward **and** backward.
3. Keep the implementation small enough to read in one screen — no fused
   kernels, no shape-generalization beyond what nanochat actually needs.
