# nanoops

从零手写 nanochat 用到的 PyTorch 算子。

**本目录下所有算子仅用于教学。** 实现风格优先可读性、把数学过程摆在明面上，而
不是追求性能、融合 kernel 或边界覆盖。生产训练请直接用 `torch.nn` /
`torch.nn.functional`——这个包的存在意义是让你能读源码、用调试器单步走、并
与 PyTorch 行为做对照。

> English version: [README.md](README.md)

## 目录结构

公共 API 与 PyTorch 对齐，nanochat 切换实现只需改 import。

| 文件 | 镜像 | 内容 |
| --- | --- | --- |
| `nn.py` | `torch.nn` | Module 风格算子（`Linear`, …） |
| `functional.py` | `torch.nn.functional` | 函数式算子 + `autograd.Function` 子类 |

## 约定

- Module 的初始化方案是经过推敲的选择，不必然与 `torch.nn` 一致。当 PyTorch
  的默认值有历史包袱（例如 `Linear` 的 `kaiming_uniform_(a=sqrt(5))`），
  nanoops 选择更有理论依据的值（`a=1`），并在类的 docstring 里说明差异。
- `autograd.Function` 子类（例如 `Mm`、`Add`、`Lookup`）刻意使用 legacy 的
  `forward(ctx, ...)` 签名——把数学和缓存张量写在一起更直观。
- shape 限制（`Mm` 只接 2D、`Lookup` 只接 1D）是有意为之，目的是让 autograd
  原语一屏读完。更高维的处理由调用方负责（见 `functional.py` 里的 `linear`
  和 `embedding`，在 2D/1D 核心算子外做 flatten + unflatten）。

## Parity 测试

`tests/test_nanoops.py` 对每个算子做与 `torch` 的前向 + 反向对拍，运行方式：

```
pytest tests/test_nanoops.py
```

新增算子时请在同一文件里补一条 parity 测试。

## TODO

**范围：只实现有意义反向传播的算子。** 优化器（AdamW/Muon）、参数初始化、
离散采样（`topk`/`argmax`/`multinomial`）、常量生成器（`arange`、rotary 的
cos/sin 表）、DDP、`torch.compile` 等**不可导**的部分一律走 PyTorch——
nanoops 的目的是教 backward，不是复刻所有工具函数。

按 nanochat 真实依赖排序。Tier 1 完成即可跑通核心 block 的前向 + 反向；
Tier 2 加入注意力；Tier 3 是可选的性能优化版本。

### Tier 1 —— 核心 block

- [x] `nn.Linear` / `F.linear`
- [ ] `nn.Embedding`
- [ ] `F.rms_norm`（模型里唯一用到的 normalization）
- [ ] `F.relu`（MLP 用的是 `relu(x) ** 2`）
- [ ] `F.softmax`
- [ ] `F.cross_entropy`（带 `ignore_index`）
- [ ] `torch.outer` / `torch.cat` / `torch.stack`
- [ ] `torch.sigmoid` / `torch.tanh`（gate 与 logit softcap）

### Tier 2 —— 注意力

- [ ] `apply_rotary_emb`（cos/sin 表继续用 PyTorch）
- [ ] `F.scaled_dot_product_attention`（先用朴素的 `softmax(QK/√d) V`）
- [ ] `torch.where` / `torch.roll`（eval 与 loss masking 用）

### Tier 3 —— 性能 / 进阶（可选）

- [ ] FP8 matmul：包一层 `torch._scaled_mm` + 自定义 `autograd.Function`
- [ ] FlashAttention-3 兼容层 + SDPA fallback（对齐 `nanochat/flash_attention.py`）

### 新增算子流程

1. 按 PyTorch 的 import 路径放到 `nn.py` 或 `functional.py`。
2. 在 `tests/test_nanoops.py` 中补一条 parity 测试，**前向和反向都要覆盖**。
3. 实现保持一屏可读——不要融合 kernel，shape 也不要做超出 nanochat 实际
   需要的泛化。
