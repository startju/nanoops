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
- [x] `nn.Embedding` / `F.embedding`
- [x] `nn.RMSNorm` / `F.rms_norm` (used as the only normalization)
- [x] `relu_square` — fused `relu(x)**2`, mirrors nanchat's `F.relu(x).square()`
- [x] `F.softmax`
- [x] `F.cross_entropy` (with `ignore_index`)
- [x] `torch.outer`
- [x] `torch.cat`
- [x] `torch.stack`
- [x] `torch.sigmoid`, `torch.tanh` (gates + logit softcap)

### Tier 2 — attention

- [x] `apply_rotary_emb` (cos/sin tables stay on PyTorch)
- [ ] `F.scaled_dot_product_attention` (start with the naive `softmax(QK/√d) V`)
- [x] `torch.where`, `torch.roll` (eval / loss masking)

### Tier 3 — performance / advanced (optional)

- ~~FP8 matmul wrapper around `torch._scaled_mm` + custom `autograd.Function`~~ — **skip**: nanchat already has this in `nanochat/fp8.py` with custom autograd (saving FP8 tensors in ctx instead of fp32, per commit d9678ff). Re-implementing in nanoops would duplicate working production code.
- [ ] FlashAttention-3 shim with SDPA fallback (mirrors `nanochat/flash_attention.py`)

### Conventions for each new op

1. Implement in `nn.py` or `functional.py` to match the PyTorch import path.
2. Add a parity test in `tests/test_nanoops.py` covering forward **and** backward.
3. Keep the implementation small enough to read in one screen — no fused
   kernels, no shape-generalization beyond what nanochat actually needs.

## Appendix: backward derivations

Most ops in nanoops have trivial backward (gradient routing, elementwise
scaling). A few have non-trivial Jacobians where the analytical simplification
matters for both readability and memory. Those derivations live here.

### RmsNorm

Per slice of the last dim (length $D$), forward (without `weight`):

$$
s = \text{mean}(x^2) + \epsilon, \qquad n = \sqrt{s}, \qquad y_i = \frac{x_i}{n} \quad (i = 0, \dots, D-1)
$$

**Notation.** $a \odot b$ denotes **element-wise (Hadamard) multiplication**,
not the inner product — it returns a vector of the same shape with
$(a \odot b)_i = a_i b_i$. Reductions to scalars are written explicitly
(e.g. $\text{mean}(g \odot y) = \tfrac{1}{D} \sum_i g_i y_i$).

Given upstream $g_i = \partial L / \partial y_i$, derive $\partial L / \partial x_j$.

**Jacobian.** Apply the quotient rule to $y_i = x_i / n$. Recall: for $f = u/v$,

$$
\frac{\partial f}{\partial t} = \frac{(\partial u / \partial t) \cdot v - u \cdot (\partial v / \partial t)}{v^2}
$$

Two facts make this non-trivial here:

- $\partial x_i / \partial x_j = \delta_{ij}$ (Kronecker delta) — the inputs are independent variables, so $x_5$'s derivative w.r.t. $x_7$ is zero, w.r.t. itself is one.
- $\partial n / \partial x_j \neq 0$ — $n$ depends on **all** $x_k$ through $s = \text{mean}(x^2)$, so we cannot treat it as a constant. This is exactly what makes RmsNorm's backward non-diagonal.

Plugging in:

$$
\frac{\partial y_i}{\partial x_j} = \frac{\delta_{ij} \cdot n - x_i \cdot (\partial n / \partial x_j)}{n^2} = \frac{\delta_{ij}}{n} - \frac{x_i}{n^2} \cdot \frac{\partial n}{\partial x_j}
$$

The first term is the "if $n$ were constant" diagonal scaling. The second is the correction because changing any single $x_j$ moves $n$, which in turn affects **every** $y_i$ (they all share that one denominator). This second term is dense — every $(i, j)$ pair contributes — which is why the Jacobian isn't diagonal and why normalization-class ops always have a "subtract the projection of something" pattern in their backward.

Now compute $\partial n / \partial x_j$ via the chain rule. The square-root
derivative comes from the power rule applied to $s^{1/2}$:

$$
\frac{\partial n}{\partial s} = \frac{1}{2} s^{-1/2} = \frac{1}{2 \sqrt{s}} = \frac{1}{2n}
$$

(intuition: $s = n^2$ has slope $2n$ at $n$; $n = \sqrt{s}$ is its inverse, so
its slope is the reciprocal $1/(2n)$.) Combined with $\partial s / \partial x_j = 2 x_j / D$:

$$
\frac{\partial n}{\partial x_j} = \frac{\partial n}{\partial s} \cdot \frac{\partial s}{\partial x_j} = \frac{1}{2n} \cdot \frac{2 x_j}{D} = \frac{x_j}{D n}
$$

The $\tfrac{1}{2}$ from $\sqrt{\,}$ cancels the $2$ from $x^2$ — a small
coincidence that keeps RmsNorm's backward formulas tidy.

Substituting:

$$
\frac{\partial y_i}{\partial x_j} = \frac{\delta_{ij}}{n} - \frac{x_i x_j}{D n^3}
$$

**Chain through to $\partial L / \partial x_j$:**

$$
\frac{\partial L}{\partial x_j} = \sum_i g_i \frac{\partial y_i}{\partial x_j} = \frac{g_j}{n} - \frac{x_j}{D n^3} \sum_i g_i x_i
$$

**Simplify via $y = x/n$.** Using $x_i = y_i n$, so $\sum_i g_i x_i = n \sum_i g_i y_i$:

$$
\frac{\partial L}{\partial x_j} = \frac{g_j}{n} - \frac{y_j n}{D n^3} \cdot n \sum_i g_i y_i = \frac{1}{n} \left[ g_j - y_j \cdot \text{mean}(g \odot y) \right]
$$

**Final form** (vector, per slice):

$$
\boxed{\ \frac{\partial L}{\partial x} = \frac{1}{n} \left( g - y \cdot \text{mean}(g \odot y) \right)\ }
$$

**With weight $w$.** Forward becomes $z_i = y_i \cdot w_i$ (the output is $z$,
not $y$), so upstream is now $g_i = \partial L / \partial z_i$. We need two
gradients: $\partial L / \partial x$ and $\partial L / \partial w$.

For $\partial L / \partial x_j$: chain through $z$ first. Since $w_i$ does not
depend on $x_j$,

$$
\frac{\partial z_i}{\partial x_j} = w_i \cdot \frac{\partial y_i}{\partial x_j}
$$

so

$$
\frac{\partial L}{\partial x_j} = \sum_i g_i \cdot \frac{\partial z_i}{\partial x_j} = \sum_i (g_i w_i) \cdot \frac{\partial y_i}{\partial x_j}
$$

This is the no-weight derivation with $g$ replaced by $g \odot w$ — that's
where the "substitute $g \rightarrow g \odot w$" shortcut comes from; it's
literally one step of the chain rule. Plugging the substitution into the
no-weight final form:

$$
\boxed{\ \frac{\partial L}{\partial x} = \frac{1}{n} \left( g \odot w - y \cdot \text{mean}(g \odot w \odot y) \right)\ }
$$

For $\partial L / \partial w_k$: from $z_i = y_i \cdot w_i$, $z_i$ depends on
$w_k$ only when $i = k$, so

$$
\frac{\partial z_i}{\partial w_k} = y_i \cdot \delta_{ik}
$$

Per slice:

$$
\frac{\partial L}{\partial w_k} = \sum_i g_i \cdot y_i \cdot \delta_{ik} = g_k \cdot y_k
$$

But $w$ has shape $(D,)$ and is **broadcast** across every batch position in
forward — the same $w$ is shared by all $(B \cdot T)$ slices. The reverse-
broadcast rule (same machinery as `Add`'s `unbroadcast`) says we sum the
gradient over the broadcast dimensions:

$$
\frac{\partial L}{\partial w} = \sum_{\text{batch}} g \odot y
$$

(concretely: if $g, y$ have shape $(B, T, D)$, then `dL/dw = (g * y).sum(dim=(0,1))` gives a $(D,)$ tensor.)

**Why this matters for nanoops.** The backward needs only $y$ and $n$ (or
equivalently $\text{rsqrt} = 1/n$), **not** the original $x$. Autograd-traced
backward saves $x$ because each underlying `mul`/`div` op needs both inputs;
the custom Function saves one $(\dots, D)$ tensor per layer — a real memory
win at LLM scale.

**Geometric intuition.** $\text{mean}(g \odot y)$ is the projection of $g$
onto the normalization direction. RmsNorm flattens that direction (any
rescaling of $x$ gets undone by $n$), so the corresponding component of $g$
doesn't propagate back — we subtract it before scaling. The same "subtract
the projection along the normalized direction" pattern recurs in softmax and
LayerNorm backward.

### Softmax

Per slice of length $D$ along the softmax dim, forward:

$$
y_i = \frac{e^{x_i}}{\sum_j e^{x_j}}, \qquad \sum_i y_i = 1
$$

(In practice subtract $\max(x)$ from $x$ before $\exp$ — pure numerical
stability, doesn't change derivatives.)

**Jacobian.** Quotient rule on $y_i = e^{x_i} / Z$ where $Z = \sum_k e^{x_k}$:

$$
\frac{\partial y_i}{\partial x_j} = \frac{(\partial_j e^{x_i}) \cdot Z - e^{x_i} \cdot (\partial_j Z)}{Z^2}
$$

Two pieces:

- $\partial_j e^{x_i} = \delta_{ij} \cdot e^{x_i}$ ($e^{x_i}$ depends only on $x_i$).
- $\partial_j Z = \partial_j \sum_k e^{x_k} = e^{x_j}$ (the sum's derivative picks the $j$-th term).

Plugging in:

$$
\frac{\partial y_i}{\partial x_j} = \frac{e^{x_i}}{Z} \delta_{ij} - \frac{e^{x_i} \cdot e^{x_j}}{Z^2} = y_i (\delta_{ij} - y_j)
$$

The first term is "diagonal scaling by $y_i$"; the second is a dense outer
product $y_i y_j$ that couples every $(i, j)$ — softmax has a fully
non-diagonal Jacobian like RmsNorm.

**Chain through to $\partial L / \partial x_j$:**

$$
\frac{\partial L}{\partial x_j} = \sum_i g_i \cdot y_i (\delta_{ij} - y_j) = y_j g_j - y_j \sum_i g_i y_i = y_j \left( g_j - \langle g, y \rangle \right)
$$

where $\langle g, y \rangle = \sum_i g_i y_i$ is the inner product along the
softmax dim.

**Final form** (vector, per slice):

$$
\boxed{\ \frac{\partial L}{\partial x} = y \odot \left( g - \langle g, y \rangle \right)\ }
$$

In code: `(g * y).sum(dim=dim, keepdim=True)` for the inner product, then
`y * (g - inner)`.

**Why this matters for nanoops.** Backward needs only $y$ (the output), not
$x$ — same memory trick as RmsNorm. The full $(D, D)$ Jacobian is never
materialized; the analytic simplification reduces it to one sum-reduction
plus one elementwise multiply chain.

**Aside: what's a "null direction"?** A null direction of an op is an input
perturbation that leaves the output unchanged — formally, a vector in the
Jacobian's null space. Because $L$ depends on $x$ only through $y$, moving
$x$ along a null direction can't change $y$, so it can't change $L$ either,
so $\partial L / \partial x$ must be exactly zero along it. The backward
formula has to subtract any component of the upstream gradient that would
project onto a null direction — otherwise the chain rule starts inventing
gradient signal where the math says there is none. The `mean(g ⊙ y)` (in
RmsNorm) and `⟨g, y⟩` (in Softmax) subtractions are exactly these
corrections.

**Connection to ctx memory: when can backward `save y` instead of `x`?** Whether
you can save just `y` (plus scalar metadata like `1/n`) depends on whether the
backward formula needs information that was *lost* in the forward $x \to y$:

| Forward shape | Save `y` works? | Why |
|---|---|---|
| Bijective (sigmoid, tanh) | ✓ | $y$ uniquely determines $x$ — no info lost |
| Has null direction with grad-zero along it (RmsNorm, Softmax, ReLU²) | ✓ | The lost info lives in the null direction; backward is invariant to it |
| Backward formula explicitly needs $x$ (e.g. Linear's $\partial L/\partial W = g \otimes x$) | ✗ | Must save $x$; $y$ alone underdetermines what backward needs |

**Recipe for a new autograd Function**: derive the backward, then check whether
the resulting formula still contains $x$. If yes, see if $x$ can be replaced
by an expression in $y$ (e.g. `sigmoid_backward` rewrites $\sigma(x)(1-\sigma(x))$
as $y(1-y)$). If you can't eliminate $x$, you must save it.

**Comparison with RmsNorm.** Both share the "subtract a projection along the
normalized direction" structure:

| Op | scale factor | projection | reduction |
|---|---|---|---|
| RmsNorm | $1/n$ (scalar per slice) | $y \cdot \text{mean}(g \odot y)$ | mean (divide by $D$) |
| Softmax | $y$ (elementwise) | $\langle g, y \rangle$ (scalar per slice) | sum (no divide) |

Different scale, different reduction — but **the same backward pattern**:
the gradient component lying along the op's *null direction* doesn't
propagate; subtract it before scaling.

For softmax specifically, the null direction is the constant vector
$\mathbf{1}$: adding a constant to every $x_i$ leaves $y$ unchanged (the
constant cancels in numerator and denominator). The $\langle g, y \rangle$
subtraction removes exactly the component of $g$ aligned with that null
direction.

### Cross-entropy (fused log-softmax + NLL)

For a single sample (one slice along the class dim of length $C$) with logits
$x \in \mathbb{R}^C$ and integer target $t \in \{0, \dots, C-1\}$:

$$
L = -\log(\text{softmax}(x)_t) = -x_t + \log \sum_j e^{x_j}
$$

The second form is **log-sum-exp minus the target logit** — the fused view
that PyTorch's `F.cross_entropy` uses, avoiding any explicit softmax tensor
in forward (with the same max-subtract trick inside LSE for stability as
Softmax).

**Backward.** Differentiate term by term:

- $\partial(-x_t)/\partial x_j = -\delta_{jt}$ (−1 at the target index, 0 elsewhere).
- $\partial(\log \sum_k e^{x_k})/\partial x_j = e^{x_j}/Z = y_j$ (chain rule: $\log \to 1/Z$, sum derivative picks $e^{x_j}$).

Adding the two:

$$
\boxed{\ \frac{\partial L}{\partial x} = \text{softmax}(x) - \text{one\_hot}(t)\ }
$$

The full $(C, C)$ softmax Jacobian **never gets materialized** — log's $1/Z$
and softmax's Jacobian cancel into a single elementwise subtraction. This is
one of the most elegant simplifications in deep learning.

**Why the fusion cancels so cleanly.** If you composed naively as
`nll(log(softmax(x)), t)`:

- $\log$ backward: $g_y = -\frac{1}{y_t}\,\delta_{it}$ (a $1/y_t$ singularity at the target!)
- Softmax backward: $y \odot (g_y - \langle g_y, y \rangle)$

Substituting: $\langle g_y, y \rangle = (-1/y_t) \cdot y_t = -1$, so
$g_x = y \odot g_y + y = y - \text{one\_hot}(t)$ — the $y_t$ at the target
position cancels the $1/y_t$. Same answer, but the naive path:

- creates an intermediate $-1/y_t$ that **underflows in bf16** when $y_t$ is small,
- materializes the softmax Jacobian's $\langle g, y \rangle$ inner product,
- runs through 3 separate backward functions instead of one.

The fused derivation makes the cancellation obvious upfront and avoids the
numerical hazard.

**Memory and ctx.** Save only $y$ (the softmax output) and $t$ (the target
indices). The full backward is one elementwise subtraction plus a scatter
into the target positions:

```python
grad_x = y                  # copy softmax output
grad_x[range(N), t] -= 1    # subtract 1 at each target
```

**Comparison with what we've seen so far.**

| Op | Backward simplification | What "disappears" |
|---|---|---|
| RmsNorm | $(1/n)(g - y \cdot \text{mean}(g \odot y))$ | sqrt + division chain |
| Softmax | $y \odot (g - \langle g, y \rangle)$ | full $(D, D)$ Jacobian materialization |
| ReLU² | $2 y g$ | mask op + multiply chain |
| **Cross-entropy** | $y - \text{one\_hot}(t)$ | **softmax Jacobian AND log's $1/y$ — both cancel** |

Cross-entropy is the **most dramatic** of these: two non-trivial ops ($\log$
and $\text{softmax}$) compose into a single subtraction. The cancellation is
no accident — $\log \circ \text{softmax}$ is the canonical "log-likelihood"
function whose gradient w.r.t. unnormalized logits is *always* "prediction
minus target" for any classification-style loss. That's what makes
cross-entropy + softmax the universal classification loss.

**`ignore_index`** (nanchat uses `ignore_index=-1` at `gpt.py:477`): positions
where $t = $ `ignore_index` contribute 0 to the loss AND 0 to the gradient.
Zero out the corresponding rows of `grad_x` before any reduction.

**Reduction** (`'mean'` / `'sum'` / `'none'`): scale `grad_x` by
$1 / N_{\text{valid}}$ for `'mean'` (where $N_{\text{valid}}$ excludes
`ignore_index` positions), by $1$ for `'sum'`, or no scaling for `'none'`.

### Sigmoid

The simplest "save y" example — a bijective elementwise op whose backward
formula can be written purely in terms of the output.

Forward:

$$
y = \sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Backward.** Direct chain rule via the quotient form. Let $u = 1 + e^{-x}$,
so $y = 1/u$:

$$
\frac{dy}{dx} = -u^{-2} \cdot \frac{du}{dx} = -\frac{1}{(1+e^{-x})^2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^2}
$$

This is correct but still expressed in $x$. To save the per-op memory, rewrite
in terms of $y$. From $y = 1/(1+e^{-x})$ we get $e^{-x} = (1-y)/y$, so

$$
\frac{dy}{dx} = \frac{(1-y)/y}{(1/y)^2} = (1-y) \cdot y = y(1-y)
$$

**Final form:**

$$
\boxed{\ \frac{\partial L}{\partial x} = g \cdot y \cdot (1 - y)\ }
$$

**Why this is the cleanest "save y" example.** Sigmoid is **bijective**:
given $y \in (0, 1)$, you can solve for $x = \log(y/(1-y))$ uniquely. So
$y$ contains all the information needed for backward; saving $x$ would be
redundant. The same memory (one $(...,)$ tensor of the input's shape) holds
either $x$ or $y$ — but backward formula is cleaner in $y$.

Connecting to the "Connection to ctx memory" recipe earlier in this
appendix: sigmoid is exactly **case A** (bijective). No null direction, no
quirky cancellation — just substitute $y$ wherever $x$ appears in the
backward and you're done.

**Tanh** follows the same pattern. Forward $y = \tanh(x)$; backward derives
to $1 - y^2$ via quotient rule on $(e^x - e^{-x})/(e^x + e^{-x})$:

$$
\frac{d \tanh}{dx} = \frac{(e^x + e^{-x})^2 - (e^x - e^{-x})^2}{(e^x + e^{-x})^2} = 1 - \tanh^2(x) = 1 - y^2
$$

So $\partial L / \partial x = g \cdot (1 - y^2)$, again purely in $y$. Same
"bijective / save y" story as sigmoid.

### Rotary positional embedding

The most symmetric op in the codebase: **backward is the same forward
formula with $\sin$ negated**. Comes from the rotation matrix being
orthogonal.

**Forward.** For a 4D input $x \in \mathbb{R}^{B \times T \times H \times d}$,
split the last dim into halves $x_1 = x[\dots, :d/2]$ and $x_2 = x[\dots, d/2:]$.
Each pair $(x_1, x_2)$ is rotated by angle $\theta$ encoded in $(\cos, \sin)$:

$$
\begin{pmatrix} y_1 & y_2 \end{pmatrix} = \begin{pmatrix} x_1 & x_2 \end{pmatrix} \cdot R(\theta), \qquad R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
$$

(PyTorch convention: features sit on the last dim, so we treat $(x_1, x_2)$
as a row vector and apply $R(\theta)$ from the right.)

Component form:

$$
y_1 = x_1 \cos\theta + x_2 \sin\theta, \qquad y_2 = -x_1 \sin\theta + x_2 \cos\theta
$$

The output is $\text{cat}(y_1, y_2)$ along the last dim.

**Backward.** Two derivations side by side — the elementary chain rule, and
the matrix-algebra shortcut.

*Elementary path.* From the forward formulas, the four partial derivatives are:

$$
\frac{\partial y_1}{\partial x_1} = \cos\theta, \quad
\frac{\partial y_1}{\partial x_2} = \sin\theta, \quad
\frac{\partial y_2}{\partial x_1} = -\sin\theta, \quad
\frac{\partial y_2}{\partial x_2} = \cos\theta
$$

Chain rule (write $g_1 = \partial L / \partial y_1$, $g_2 = \partial L / \partial y_2$):

$$
\frac{\partial L}{\partial x_1} = g_1 \cdot \frac{\partial y_1}{\partial x_1} + g_2 \cdot \frac{\partial y_2}{\partial x_1} = g_1 \cos\theta + g_2 (-\sin\theta) = g_1 \cos\theta - g_2 \sin\theta
$$

$$
\frac{\partial L}{\partial x_2} = g_1 \cdot \frac{\partial y_1}{\partial x_2} + g_2 \cdot \frac{\partial y_2}{\partial x_2} = g_1 \sin\theta + g_2 \cos\theta
$$

*Matrix-algebra path (same result, different angle).* Forward is linear in $x$:
$y = x \cdot R(\theta)$. With row vectors, chain rule gives
$\partial L / \partial x = g \cdot R(\theta)^T$. For a rotation matrix
$R(\theta)^T = R(\theta)^{-1} = R(-\theta)$:

$$
R(\theta)^T = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix}
$$

Both paths give the same boxed result:

$$
\boxed{\ g_x^1 = g_1 \cos\theta - g_2 \sin\theta, \qquad g_x^2 = g_1 \sin\theta + g_2 \cos\theta\ }
$$

**This is literally the forward formula with $\sin \to -\sin$.** Same kernel,
opposite rotation direction. Backward = "un-rotate".

**Why $R^T = R^{-1}$.** Rotation matrices are orthogonal, which you can verify
directly:

$$
R(\theta) R(\theta)^T = \begin{pmatrix} \cos & -\sin \\ \sin & \cos \end{pmatrix} \begin{pmatrix} \cos & \sin \\ -\sin & \cos \end{pmatrix} = \begin{pmatrix} \cos^2 + \sin^2 & \cos\sin - \sin\cos \\ \sin\cos - \cos\sin & \sin^2 + \cos^2 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
$$

The $\cos^2 + \sin^2 = 1$ identity does all the work.

**Memory and ctx.** Backward depends only on $g$, $\cos$, $\sin$ — **not on
$x$ or $y$**. ctx saves only $(\cos, \sin)$, which are precomputed lookup
tables shared across the whole attention call (a reference, not a fresh
allocation). nanoops's `ApplyRotaryEmb.forward` does not save $x$ at all,
making this one of the cheapest ctx footprints in the codebase.

$\cos$ and $\sin$ are computed once from `arange` + `outer` + `cos()`/`sin()`
(see `nanchat/gpt.py:_precompute_rotary_embeddings`) and are not
differentiable, so backward returns `None` for both.

**Why this is so much cleaner than RmsNorm / Softmax.** Those ops have
*data-dependent* normalization — backward needs $y$ (or $x$) because the
Jacobian depends on the input values. Rotary's Jacobian is just $R(\theta)$,
a function of position only — completely independent of the input tensor's
values. So there's nothing to save from $x$ at all.

### Trivial-backward ops: Where and Roll

Two ops worth noting for completeness — their backwards are one-liners,
but they're the cleanest possible illustrations of **case A** and **case B**
from the "Connection to ctx memory" table earlier in this appendix.

**Where**: $y = \text{where}(\text{cond}, a, b)$ — pick $a$ where `cond` is
true, else $b$.

Backward routes the upstream grad by the mask:

$$
\frac{\partial L}{\partial a} = g \odot \mathbf{1}[\text{cond}], \qquad
\frac{\partial L}{\partial b} = g \odot \mathbf{1}[\neg\text{cond}]
$$

Grad flows only where that operand was actually picked; the other operand
gets zero at that position. ctx saves only `cond` (bool, 1 byte/elem).

This is the **case B** pattern in its purest form: there's a "null direction"
(perturbing $a$ at positions where cond=False, or perturbing $b$ at positions
where cond=True, doesn't change $y$), and gradient along it is zero by
construction. The bool mask carries all the information backward needs.

Same gradient-gating idea as ReLU's `g * (x > 0)` — but **more general**, since
the mask can come from any condition (`x > threshold`, `mask_token != PAD`,
attention causality mask, etc.), not just $x > 0$.

**Roll**: $y = \text{roll}(x, \text{shifts}, \text{dims})$ — cyclic shift along
given dim(s).

Backward applies the inverse permutation:

$$
\frac{\partial L}{\partial x} = \text{roll}(g, -\text{shifts}, \text{dims})
$$

ctx saves **no tensor at all** — only the integer `(shifts, dims)` params as
plain attributes. Roll is bijective (a permutation always has an inverse),
so $y$ fully determines $x$ — but we don't even need $y$, since the backward
formula only references the shift parameters. This is the **case A** pattern
with an extra flourish: zero tensor footprint in ctx.

`shifts` and `dims` are Python ints / tuples, non-differentiable; backward
returns `None` for both.

### Pattern: orthogonal transformations have "free" backwards

The Rotary and Roll sections above share a structural identity worth
spelling out. Both forwards are *linear* in $x$ (matrix multiplication by
some $M$, with PyTorch's row-vector convention where features sit on the
last dim):

- Rotary: $y = x \cdot R(\theta)$ where $R(\theta)$ is a 2×2 rotation matrix
- Roll:   $y = x \cdot P$ where $P$ is a permutation matrix

Both matrices are **orthogonal**: $M M^T = I$, equivalently $M^T = M^{-1}$.
The chain rule for $y = x M$ is $\partial L / \partial x = g \cdot M^T$,
so for orthogonal $M$:

$$
\frac{\partial L}{\partial x} = g \cdot M^{-1}
$$

In words: **backward is the inverse forward applied to grad_output**.

| Op | Forward matrix $M$ | $M^{-1}$ | Backward in code |
|---|---|---|---|
| Rotary | $R(\theta)$ | $R(-\theta)$ | run forward formula with $\sin \to -\sin$ |
| Roll | $P_k$ (shift by $k$) | $P_{-k}$ (shift by $-k$) | run forward op with $k \to -k$ |

What "free" means here: **no Jacobian materialization, no saved tensors of
$x$ or $y$, no closed-form simplification needed**. The same forward kernel
with the single transformation parameter negated computes the backward.

Why this works: an orthogonal matrix's transpose IS its inverse — they
coincide. So "Jacobian transpose" (which chain rule demands) equals
"inverse Jacobian" (which corresponds to a familiar inverse operation).
For non-orthogonal linear ops (like a generic matmul with non-square or
non-orthogonal $W$), $W^T \neq W^{-1}$ and you have to actually carry out
the transpose-matmul; that's why $W^T$ shows up explicitly in `Mm`'s
backward (`grad_left = grad_output @ right.T`) and *isn't* expressible as
"forward with one parameter flipped".

If you ever add a new linear op to nanoops and notice its matrix is
orthogonal — DCT, the real-valued FFT (up to a scaling factor),
Walsh-Hadamard, signed permutations, etc. — its backward fits this template
out of the box.
