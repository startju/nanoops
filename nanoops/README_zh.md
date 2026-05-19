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
- [x] `nn.Embedding` / `F.embedding`
- [x] `nn.RMSNorm` / `F.rms_norm`（模型里唯一用到的 normalization）
- [x] `F.relu`（MLP 用的是 `relu(x) ** 2`）
- [ ] `F.softmax`
- [ ] `F.cross_entropy`（带 `ignore_index`）
- [x] `torch.outer`
- [x] `torch.cat`
- [x] `torch.stack`
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

## 附录：反向推导

nanoops 大多数算子反向是平凡的（gradient routing、elementwise scaling）。
少数算子的 Jacobian 非平凡，化简后的形式对可读性和显存都有意义。这些推导
放在这里。

### RmsNorm

最后一维长度 $D$，一个 slice 内 forward（不带 `weight`）：

$$
s = \text{mean}(x^2) + \epsilon, \qquad n = \sqrt{s}, \qquad y_i = \frac{x_i}{n} \quad (i = 0, \dots, D-1)
$$

**记号说明.** $a \odot b$ 表示**按元素相乘**（Hadamard 积），**不是**内积/点积——
结果是和输入同形状的向量，$(a \odot b)_i = a_i b_i$。需要 reduce 成标量时会显式写出
（例如 $\text{mean}(g \odot y) = \tfrac{1}{D} \sum_i g_i y_i$）。

给定 upstream $g_i = \partial L / \partial y_i$，推 $\partial L / \partial x_j$。

**Jacobian.** 对 $y_i = x_i / n$ 套商法则。回忆一下：$f = u/v$ 时，

$$
\frac{\partial f}{\partial t} = \frac{(\partial u / \partial t) \cdot v - u \cdot (\partial v / \partial t)}{v^2}
$$

这里有两个关键点让推导不平凡：

- $\partial x_i / \partial x_j = \delta_{ij}$（Kronecker delta）—— 输入向量的每个元素是**独立变量**，所以 $x_5$ 对 $x_7$ 的偏导是 0，对自己的偏导是 1。
- $\partial n / \partial x_j \neq 0$ —— $n$ 通过 $s = \text{mean}(x^2)$ 依赖**所有** $x_k$，不能当常数。这正是 RmsNorm 反向"非对角"的根源。

代入：

$$
\frac{\partial y_i}{\partial x_j} = \frac{\delta_{ij} \cdot n - x_i \cdot (\partial n / \partial x_j)}{n^2} = \frac{\delta_{ij}}{n} - \frac{x_i}{n^2} \cdot \frac{\partial n}{\partial x_j}
$$

第一项是"假设 $n$ 是常数"的对角缩放。第二项是 $n$ 也会动带来的修正——改任意一个 $x_j$ 都会让 $n$ 变，而 $n$ 又出现在**每个** $y_i$ 的分母里（大家共享同一个分母）。第二项**稠密**——每一对 $(i, j)$ 都贡献非零值——这就是 Jacobian 不是对角矩阵的原因，也是所有 normalization-class 算子反向里都有"减去某种投影"的根本来源。

下面用链式法则算 $\partial n / \partial x_j$。$\sqrt{\,}$ 的导数来自幂法则
（对 $s^{1/2}$ 求导）：

$$
\frac{\partial n}{\partial s} = \frac{1}{2} s^{-1/2} = \frac{1}{2 \sqrt{s}} = \frac{1}{2n}
$$

（直觉：$s = n^2$ 在 $n$ 处的斜率是 $2n$；$n = \sqrt{s}$ 是它的反函数，所以
斜率取倒数 $1/(2n)$。）再配上 $\partial s / \partial x_j = 2 x_j / D$：

$$
\frac{\partial n}{\partial x_j} = \frac{\partial n}{\partial s} \cdot \frac{\partial s}{\partial x_j} = \frac{1}{2n} \cdot \frac{2 x_j}{D} = \frac{x_j}{D n}
$$

$\sqrt{\,}$ 带的 $\tfrac{1}{2}$ 正好和 $x^2$ 带的 $2$ 抵消——一个小巧合，
让 RmsNorm 的反向公式看起来干净。

代回：

$$
\frac{\partial y_i}{\partial x_j} = \frac{\delta_{ij}}{n} - \frac{x_i x_j}{D n^3}
$$

**链式法则展开 $\partial L / \partial x_j$：**

$$
\frac{\partial L}{\partial x_j} = \sum_i g_i \frac{\partial y_i}{\partial x_j} = \frac{g_j}{n} - \frac{x_j}{D n^3} \sum_i g_i x_i
$$

**用 $y = x/n$ 化简。** $x_i = y_i n$，所以 $\sum_i g_i x_i = n \sum_i g_i y_i$：

$$
\frac{\partial L}{\partial x_j} = \frac{g_j}{n} - \frac{y_j n}{D n^3} \cdot n \sum_i g_i y_i = \frac{1}{n} \left[ g_j - y_j \cdot \text{mean}(g \odot y) \right]
$$

**最终形式**（向量，per slice）：

$$
\boxed{\ \frac{\partial L}{\partial x} = \frac{1}{n} \left( g - y \cdot \text{mean}(g \odot y) \right)\ }
$$

**带 weight $w$.** Forward 变成 $z_i = y_i \cdot w_i$（输出是 $z$，不是 $y$），
upstream 现在是 $g_i = \partial L / \partial z_i$。需要算两个梯度：
$\partial L / \partial x$ 和 $\partial L / \partial w$。

**先算 $\partial L / \partial x_j$**：链式法则先穿过 $z$。因为 $w_i$ 不依赖 $x_j$，

$$
\frac{\partial z_i}{\partial x_j} = w_i \cdot \frac{\partial y_i}{\partial x_j}
$$

所以

$$
\frac{\partial L}{\partial x_j} = \sum_i g_i \cdot \frac{\partial z_i}{\partial x_j} = \sum_i (g_i w_i) \cdot \frac{\partial y_i}{\partial x_j}
$$

这就是无 weight 推导里把 $g$ 整体替换成 $g \odot w$ ——"替换 $g \rightarrow g \odot w$"
不是黑魔法，**就是链式法则的一步**。把替换代入之前的最终形式：

$$
\boxed{\ \frac{\partial L}{\partial x} = \frac{1}{n} \left( g \odot w - y \cdot \text{mean}(g \odot w \odot y) \right)\ }
$$

**再算 $\partial L / \partial w_k$**：从 $z_i = y_i \cdot w_i$ 可知，$z_i$ 只在
$i = k$ 时依赖 $w_k$，所以

$$
\frac{\partial z_i}{\partial w_k} = y_i \cdot \delta_{ik}
$$

per-slice 算：

$$
\frac{\partial L}{\partial w_k} = \sum_i g_i \cdot y_i \cdot \delta_{ik} = g_k \cdot y_k
$$

但 $w$ 的 shape 是 $(D,)$，在 forward 时被**广播**到每个 batch 位置——所有
$(B \cdot T)$ 个 slice 共用同一份 $w$。按反向广播规则（和 `Add` 里的
`unbroadcast` 同一套机制），梯度要在广播过的维度上 sum：

$$
\frac{\partial L}{\partial w} = \sum_{\text{batch}} g \odot y
$$

（具体来说：如果 $g, y$ 形状是 $(B, T, D)$，那 `dL/dw = (g * y).sum(dim=(0,1))` 得到一个 $(D,)$ 张量。）

**对 nanoops 的意义。** 反向**只需要 $y$ 和 $n$**（或者等价的 $\text{rsqrt} = 1/n$），
**不需要原始 $x$**。autograd 自动追踪的版本必须保存 $x$（因为底层每个
`mul`/`div` 都需要两个输入），而自定义 Function 每层 RmsNorm 省一份
$(\dots, D)$ 的 tensor——LLM 规模下是真金白银的显存。

**几何直觉。** $\text{mean}(g \odot y)$ 是 $g$ 在归一化方向 $y$ 上的投影系数。
RmsNorm 把这个方向"压平了"（任何沿 $y$ 方向的拉伸都被 $n$ 自动除掉），
所以这个方向上的 $g$ 不能传回去——必须先减掉再 scale。同样的"减去
归一化方向上的投影"结构在 softmax 和 LayerNorm 的反向里会再次出现。
