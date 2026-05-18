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

- Module 的初始化方案与 `torch.nn` 完全一致，nanoops 的 module 与对应的
  `torch.nn` module 在统计意义上权重可互换。
- `autograd.Function` 子类（例如 `Matmul`）刻意使用 legacy 的
  `forward(ctx, ...)` 签名——把数学和缓存张量写在一起更直观。
- shape 限制（例如 `Matmul` 只接受 2D）是有意为之，目的是让实现一屏读完。
  更高维版本留作练习。

## Parity 测试

`tests/test_nanoops.py` 对每个算子做与 `torch` 的前向 + 反向对拍，运行方式：

```
pytest tests/test_nanoops.py
```

新增算子时请在同一文件里补一条 parity 测试。

## TODO

按 nanochat 真实依赖排序。Tier 1 完成即可跑通模型前向（玩具权重）；后续 tier
逐步解锁训练、采样、以及性能优化。

### Tier 1 —— 模型前向

- [x] `nn.Linear` / `F.linear`
- [ ] `nn.Embedding`
- [ ] `F.rms_norm`（模型里唯一用到的 normalization）
- [ ] `F.relu`（MLP 用的是 `relu(x) ** 2`）
- [ ] `F.softmax`
- [ ] `F.cross_entropy`（带 `ignore_index`）
- [ ] `torch.arange` / `torch.outer` / `torch.cat` / `torch.stack`
- [ ] `torch.sigmoid` / `torch.tanh`（gate 与 logit softcap）

### Tier 2 —— 注意力与生成

- [ ] Rotary embedding：cos/sin 预计算 + `apply_rotary_emb`
- [ ] `F.scaled_dot_product_attention`（先用朴素的 `softmax(QK/√d) V`）
- [ ] `torch.topk` / `torch.multinomial` / `torch.argmax`（`engine.py` 采样用）
- [ ] `torch.where` / `torch.roll`（eval 与 loss masking 用）

### Tier 3 —— 训练循环

- [ ] `nn.init.normal_` / `uniform_` / `zeros_` / `constant_`
- [ ] AdamW step（fused 风格，不依赖 `torch.compile`）
- [ ] Muon 优化器（仅矩阵参数；embedding 仍走 AdamW）
- [ ] 参数分组路由（matrix → Muon，embedding/scalar → AdamW）

### Tier 4 —— 性能 / 进阶（可选）

- [ ] FP8 matmul：包一层 `torch._scaled_mm` + 自定义 `autograd.Function`
- [ ] FlashAttention-3 兼容层 + SDPA fallback（对齐 `nanochat/flash_attention.py`）
- [ ] Muon+AdamW 的 DDP 变体
- [ ] `torch.compile` 兼容性

### 新增算子流程

1. 按 PyTorch 的 import 路径放到 `nn.py` 或 `functional.py`。
2. 在 `tests/test_nanoops.py` 中补一条 parity 测试，**前向和反向都要覆盖**。
3. 实现保持一屏可读——不要融合 kernel，shape 也不要做超出 nanochat 实际
   需要的泛化。
