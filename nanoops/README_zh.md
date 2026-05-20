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
- [x] `relu_square` —— 融合 `relu(x)**2`，对应 nanchat 的 `F.relu(x).square()`
- [x] `F.softmax`
- [x] `F.cross_entropy`（带 `ignore_index`）
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

### Softmax

最后一维长度 $D$，一个 slice 内 forward：

$$
y_i = \frac{e^{x_i}}{\sum_j e^{x_j}}, \qquad \sum_i y_i = 1
$$

（实际实现里 $x$ 先减去 $\max(x)$ 再取 exp——纯数值稳定，不影响导数。）

**Jacobian.** 对 $y_i = e^{x_i} / Z$ 用商法则（$Z = \sum_k e^{x_k}$）：

$$
\frac{\partial y_i}{\partial x_j} = \frac{(\partial_j e^{x_i}) \cdot Z - e^{x_i} \cdot (\partial_j Z)}{Z^2}
$$

两个分量：

- $\partial_j e^{x_i} = \delta_{ij} \cdot e^{x_i}$（$e^{x_i}$ 只依赖 $x_i$）。
- $\partial_j Z = \partial_j \sum_k e^{x_k} = e^{x_j}$（求和导数挑出第 $j$ 项）。

代入：

$$
\frac{\partial y_i}{\partial x_j} = \frac{e^{x_i}}{Z} \delta_{ij} - \frac{e^{x_i} \cdot e^{x_j}}{Z^2} = y_i (\delta_{ij} - y_j)
$$

第一项是"按 $y_i$ 对角缩放"；第二项 $y_i y_j$ 是稠密的外积，**耦合每对
$(i, j)$**——和 RmsNorm 一样是非对角 Jacobian。

**链式法则展开 $\partial L / \partial x_j$：**

$$
\frac{\partial L}{\partial x_j} = \sum_i g_i \cdot y_i (\delta_{ij} - y_j) = y_j g_j - y_j \sum_i g_i y_i = y_j \left( g_j - \langle g, y \rangle \right)
$$

其中 $\langle g, y \rangle = \sum_i g_i y_i$ 是沿 softmax dim 的内积。

**最终形式**（向量，per slice）：

$$
\boxed{\ \frac{\partial L}{\partial x} = y \odot \left( g - \langle g, y \rangle \right)\ }
$$

代码上：`(g * y).sum(dim=dim, keepdim=True)` 算内积，然后 `y * (g - inner)`。

**对 nanoops 的意义。** 反向**只需要 $y$**（forward 输出），**不需要 $x$**——
和 RmsNorm 一样的省内存技巧。完整 $(D, D)$ Jacobian **从未被物化**：
化简后只剩一次 sum-reduction + 一次 elementwise mul 链。

**插一段：什么是 "null direction"？** 算子的 null direction 是这样一种输入
扰动方向——沿它移动输入，输出**完全不变**（数学定义：Jacobian 的零空间里
的向量）。因为 $L$ 通过 $y$ 依赖 $x$，沿 null direction 移动 $x$ 不改 $y$，
就不可能改 $L$，所以 $\partial L / \partial x$ 沿这个方向**必须严格为 0**。
backward 公式必须减掉 upstream grad 在 null direction 上的投影，**否则链式
法则会无中生有地编造梯度信号**——和"输出不变"这个事实矛盾。RmsNorm 的
`mean(g ⊙ y)` 和 Softmax 的 `⟨g, y⟩` 减法做的正是这个修正。

**和 ctx 内存的关联：什么时候 backward 能 "save y" 而不存 x？** 能不能只存
`y`（加上 `1/n` 这种标量元数据）取决于反向公式是否需要 forward $x \to y$
过程中**丢掉的信息**：

| Forward 形状 | Save `y` 可行？ | 原因 |
|---|---|---|
| Bijective（sigmoid, tanh） | ✓ | $y$ 唯一确定 $x$，没丢信息 |
| 有 null direction 且 grad 沿其为 0（RmsNorm, Softmax, ReLU²） | ✓ | 丢的信息在 null direction 上；backward 对它不变 |
| Backward 显式需要 $x$（如 Linear 的 $\partial L/\partial W = g \otimes x$） | ✗ | 必须存 $x$；只看 $y$ 推不出 backward 需要的量 |

**新 autograd Function 的设计配方**：先推 backward，然后检查公式里**还有没有 $x$**。
如果有，看能不能用 $y$ 表达（比如 sigmoid backward 把 $\sigma(x)(1-\sigma(x))$
改写成 $y(1-y)$）。如果实在改不掉，就必须存 $x$。

**和 RmsNorm 的对比。** 二者都是"减去归一化方向上的投影"结构：

| 算子 | scale factor | projection | reduction |
|---|---|---|---|
| RmsNorm | $1/n$（标量，per slice） | $y \cdot \text{mean}(g \odot y)$ | mean（除以 $D$） |
| Softmax | $y$（elementwise） | $\langle g, y \rangle$（标量，per slice） | sum（不除） |

scale 不同、reduction 不同，但**反向的 pattern 完全一致**：**沿算子 null
方向的梯度分量不传递；减掉后再 scale。**

Softmax 的 null 方向是常数向量 $\mathbf{1}$：所有 $x_i$ 同加一个常数，
$y$ 不变（分子分母里常数抵消）。$\langle g, y \rangle$ 这一步正好把
$g$ 在这个 null 方向上的投影减掉。

### Cross-entropy（融合 log-softmax + NLL）

单样本（类别维长度 $C$ 的一个 slice）：logits $x \in \mathbb{R}^C$，整数
target $t \in \{0, \dots, C-1\}$：

$$
L = -\log(\text{softmax}(x)_t) = -x_t + \log \sum_j e^{x_j}
$$

第二种形式是 **log-sum-exp 减去 target 的 logit**——这是 PyTorch
`F.cross_entropy` 用的融合视角，forward 里**根本不显式算 softmax**（LSE
里同样做 max-subtract 保证数值稳定）。

**Backward.** 逐项求导：

- $\partial(-x_t)/\partial x_j = -\delta_{jt}$（target 位置 −1，其余为 0）。
- $\partial(\log \sum_k e^{x_k})/\partial x_j = e^{x_j}/Z = y_j$（链式法则：$\log \to 1/Z$，求和导数挑出 $e^{x_j}$）。

两项相加：

$$
\boxed{\ \frac{\partial L}{\partial x} = \text{softmax}(x) - \text{one\_hot}(t)\ }
$$

完整的 $(C, C)$ softmax Jacobian **完全不需要物化**——$\log$ 贡献的 $1/Z$
和 softmax Jacobian **互相抵消**，整个反向化简成**一次 elementwise 减法**。
这是深度学习里最优雅的化简之一。

**为什么 fusion 抵消得这么干净。** 如果按 `nll(log(softmax(x)), t)` 朴素
组合：

- $\log$ backward：$g_y = -\frac{1}{y_t}\,\delta_{it}$（target 处有 $1/y_t$ 的奇异！）
- Softmax backward：$y \odot (g_y - \langle g_y, y \rangle)$

代入：$\langle g_y, y \rangle = (-1/y_t) \cdot y_t = -1$，于是
$g_x = y \odot g_y + y = y - \text{one\_hot}(t)$ —— target 位置的 $y_t$
正好抵消 $1/y_t$。结果完全相同，但朴素路径：

- 产生中间量 $-1/y_t$，**bf16 下 $y_t$ 很小时会下溢**；
- 物化 softmax Jacobian 的 $\langle g, y \rangle$ 内积；
- 走 3 个独立的 backward function（log + softmax + indexing）。

融合推导**预先看穿这个抵消**，不仅性能好，还避开了数值陷阱。

**内存和 ctx.** 只存 $y$（softmax 输出）和 $t$（target 索引）。整个反向是
**一次 elementwise 减法 + 在 target 位置做 scatter**：

```python
grad_x = y                  # 复制 softmax 输出
grad_x[range(N), t] -= 1    # 每个 target 位置减 1
```

**和前面算子的对比。**

| 算子 | Backward 化简 | "消失"的是什么 |
|---|---|---|
| RmsNorm | $(1/n)(g - y \cdot \text{mean}(g \odot y))$ | sqrt + 除法链 |
| Softmax | $y \odot (g - \langle g, y \rangle)$ | $(D, D)$ Jacobian 的物化 |
| ReLU² | $2 y g$ | mask op + 乘法链 |
| **Cross-entropy** | $y - \text{one\_hot}(t)$ | **softmax Jacobian 和 log 的 $1/y$ 一起抵消** |

Cross-entropy 是**最戏剧化的**：两个非平凡算子（$\log$ 和 $\text{softmax}$）
组合后化简成**一次减法**。这个抵消不是巧合——$\log \circ \text{softmax}$
正是 "log-likelihood" 的规范形式，它对原始 logits 的梯度**总是**"prediction
minus target"，对任何分类损失都成立。这就是 cross-entropy + softmax 成为
通用分类损失的根本原因。

**`ignore_index`**（nanchat 在 `gpt.py:477` 用 `ignore_index=-1`）：
$t = $ `ignore_index` 的位置对 loss 和 grad **贡献都为 0**。在 reduction
之前把对应行的 `grad_x` 归零即可。

**Reduction**（`'mean'` / `'sum'` / `'none'`）：`'mean'` 时把 `grad_x`
除以 $N_{\text{valid}}$（不算 `ignore_index` 位置），`'sum'` 不变，`'none'`
不缩放。
