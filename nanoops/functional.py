"""Functional ops, mirroring `torch.nn.functional`."""

from __future__ import annotations

import math

import torch


# Mark every public wrapper as `allow_in_graph` so torch.compile (Dynamo)
# treats each call as an opaque node instead of graph-breaking when it
# can't trace through our Python autograd.Function. Doesn't enable fusion
# into the surrounding compiled region (the node stays a black box), but
# eliminates the cache-invalidation / recompile cost of a graph break and
# lets PyTorch ops on either side keep compiling normally.
try:
    _allow_in_graph = torch.compiler.allow_in_graph
except AttributeError:  # PyTorch < 2.0
    def _allow_in_graph(fn):
        return fn


class Mm(torch.autograd.Function):
    """2D-only matrix multiply, mirroring `torch.mm` semantics.

    Higher-rank inputs must be flattened to 2D by the caller (see `linear`).

    Complexity (left (M,K) @ right (K,N) -> out (M,N)):
      forward:  O(M*K*N) FLOPs; allocates out (M*N). ctx holds refs to
                left and right (O(M*K + K*N) memory) until backward runs.
      backward: O(M*K*N) FLOPs each for grad_left and grad_right -> 2x forward;
                allocates grad_left (M*K) and grad_right (K*N).
      total:    ~3x forward FLOPs end-to-end (standard matmul rule of thumb).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        assert left.ndim == 2, f"left must be 2D, got {left.ndim}D"
        assert right.ndim == 2, f"right must be 2D, got {right.ndim}D"
        ctx.save_for_backward(left, right)
        output = left @ right
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left, right = ctx.saved_tensors
        grad_left = grad_output @ right.T
        grad_right = left.T @ grad_output
        return grad_left, grad_right


def unbroadcast(grad: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    # Sum out broadcasted dimensions to match target shape. Two phases:
    #   (1) any extra leading dims (grad.ndim > len(target_shape)) get
    #       summed away without keepdim.
    #   (2) any size-1 target dim that got broadcast up gets summed with
    #       keepdim=True.
    # Both phases are batched into a SINGLE sum() call each (tuple of dims),
    # so the total dispatch cost is at most 2 kernel launches regardless of
    # how many dims need reducing. The original loop-form did one launch per
    # dim — Add/Mul/Where backward called from every elementwise op in the
    # graph, so the savings add up.
    n_extra = grad.ndim - len(target_shape)
    if n_extra > 0:
        grad = grad.sum(dim=list(range(n_extra)))
    # grad.ndim == len(target_shape) now.
    keepdim_axes = [
        i for i, (g, t) in enumerate(zip(grad.shape, target_shape)) if g != t
    ]
    if keepdim_axes:
        grad = grad.sum(dim=keepdim_axes, keepdim=True)
    return grad


class Add(torch.autograd.Function):
    """Elementwise add with NumPy/PyTorch broadcasting semantics."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        # Fail fast with a clear error if shapes are not broadcastable,
        # instead of relying on the downstream `+` op's message.
        torch.broadcast_shapes(left.shape, right.shape)
        ctx.left_shape = left.shape
        ctx.right_shape = right.shape
        return left + right

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return unbroadcast(grad_output, ctx.left_shape), unbroadcast(
            grad_output, ctx.right_shape
        )


class Mul(torch.autograd.Function):
    """Elementwise multiply with NumPy/PyTorch broadcasting semantics.

    Memory note: unlike `Add` whose backward only needs operand shapes,
    `Mul`'s backward needs each operand multiplied by the OTHER one
    (dL/da = g * b, dL/db = g * a) — so we must save both tensors. For
    (..., D) bf16 inputs that's 2 x (..., D) tensors held in ctx until
    backward, roughly 4 bytes/elem. This is the cost of a non-constant
    Jacobian even for an op as simple as elementwise multiply.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        # Fail fast with a clear error if shapes are not broadcastable.
        torch.broadcast_shapes(left.shape, right.shape)
        ctx.save_for_backward(left, right)
        return left * right

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left, right = ctx.saved_tensors
        # "Multiply by the other operand", then unbroadcast for shape.
        grad_left = unbroadcast(grad_output * right, left.shape)
        grad_right = unbroadcast(grad_output * left, right.shape)
        return grad_left, grad_right


@_allow_in_graph
def mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Mirrors `torch.mul` / the `*` operator (elementwise with broadcasting)."""
    return Mul.apply(left, right)


@_allow_in_graph
def linear(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    # weight: (out, in); input: (..., in) -> (..., out)
    # Layout: storing weight as (out, in) keeps the contracted dim (`in`) as
    # the fast axis in BOTH operands of the GEMM below. `weight.T` is a free
    # view (stride flip, no copy); BLAS then sees both operands with `in`
    # contiguous — the cache-friendliest pattern for inner dot products.
    # With tuned BLAS this is mostly absorbed by internal packing, but the
    # advantage becomes real in tensor-parallel sharding (row-blocks of W
    # are physically contiguous) and in any non-BLAS / small-matrix path.
    input_shape = input.shape
    new_input = input.reshape(-1, input_shape[-1])
    out = Mm.apply(new_input, weight.T)
    out = Add.apply(out, bias) if bias is not None else out
    return out.reshape(*input_shape[:-1], -1)


class Lookup(torch.autograd.Function):
    """1D-only index select, mirroring `torch.nn.functional.embedding` semantics.

    Complexity (indices (N,), weight (V,D) -> out (N,D)):
      forward:  O(N*D) - copy N rows. No V factor: this is the whole point of
                using indexing over `one_hot(indices) @ weight`, which would
                cost O(N*V*D) compute and O(N*V) memory for the one-hot matrix.
                Allocates out (N*D).
      backward: O(V*D) to zero-init grad_weight + O(N*D) for index_add_,
                i.e. O(V*D) when V >> N (the typical case: V=50k, N=batch*seq).
                The dense (V*D) grad allocation is what `sparse=True` avoids
                in production paths - only N rows are actually touched.

    Key takeaway: embedding's asymptotic win is on the forward pass; backward
    still materializes a full (V*D) gradient tensor unless sparse.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        indices: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        # (N,E)
        assert indices.ndim == 1, f"indices must be 1D, got {indices.ndim}D"
        # (E,D)
        assert weight.ndim == 2, f"weight must be 2D, got {weight.ndim}D"
        ctx.save_for_backward(indices, weight)
        # (N, D)
        return weight[indices]

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[None, torch.Tensor]:
        indices, weight = ctx.saved_tensors
        grad_weight = torch.zeros_like(weight)
        #  for i in range(N):
        #      grad_weight[indices[i]] += grad_output[i]
        grad_weight.index_add_(0, indices, grad_output)
        return None, grad_weight


@_allow_in_graph
def embedding(
    indices: torch.Tensor,
    weight: torch.Tensor,
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    # Signature matches torch.nn.functional.embedding so that
    # nn.Embedding.forward() (which always passes all 7 args from its config)
    # can route through this function via monkey-patching. nanoops doesn't
    # implement padding_idx / max_norm / scale_grad_by_freq / sparse — accept
    # only the (None/False/default) values that nanchat actually uses.
    if padding_idx is not None:
        raise NotImplementedError("nanoops.embedding: padding_idx not supported")
    if max_norm is not None:
        raise NotImplementedError("nanoops.embedding: max_norm not supported")
    if scale_grad_by_freq:
        raise NotImplementedError("nanoops.embedding: scale_grad_by_freq not supported")
    if sparse:
        raise NotImplementedError("nanoops.embedding: sparse not supported")
    # weight: (V, D); input: (...) -> (..., D)
    # Layout (why (V, D), not (D, V)):
    #   Under row-major storage, `weight[i]` reads D contiguous floats
    #   (~1-2 cache lines). `index_add_` along dim 0 likewise writes D
    #   contiguous floats. A (D, V) layout would do strided column gathers
    #   with V*4-byte stride (~200KB jumps at V=50k), turning each lookup
    #   into ~D cache misses instead of ~D/16. This op doesn't go through
    #   BLAS, so the layout-locality advantage shows up directly —
    #   typically a 10-100x gap, not a wash.
    indices_shape = indices.shape
    new_indices = indices.reshape(-1)
    out = Lookup.apply(new_indices, weight)
    return out.reshape(*indices_shape, -1)


class RMSNorm(torch.autograd.Function):
    """Root-mean-square normalization, mirroring `torch.nn.functional.rms_norm` semantics."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        assert input.ndim == 2, f"input must be 2D, got {input.ndim}D"
        if weight is not None:
            assert weight.ndim == 1, f"weight must be 1D, got {weight.ndim}D"
            assert input.shape[-1] == weight.shape[0], (
                f"last dim of input ({input.shape[-1]}) must match weight shape ({weight.shape[0]})"
            )
        # Use rsqrt + multiply instead of sqrt + divide:
        # - rsqrt() is a single fused op (one CUDA instruction on NVIDIA HW)
        # - fp32 mul throughput is ~30x fp32 div on Ampere/Hopper
        # - more numerically stable in fp16/bf16
        rsqrt = input.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
        y = input * rsqrt
        ctx.save_for_backward(weight, rsqrt, y)
        output = y * weight if weight is not None else y
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        weight, rsqrt, y = ctx.saved_tensors
        # grad_weight uses original grad_output (= g); grad_input uses
        # g_eff = g·w (or just g when weight is None) — see README appendix.
        if weight is not None:
            grad_weight = (grad_output * y).sum(dim=0)
            g_eff = grad_output * weight
        else:
            grad_weight = None
            g_eff = grad_output
        grad_input = rsqrt * (g_eff - y * (g_eff * y).mean(dim=-1, keepdim=True))
        return grad_input, grad_weight, None


@_allow_in_graph
def rms_norm(
    input: torch.Tensor,
    normalized_shape: tuple[int, ...],
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    assert len(normalized_shape) == 1, (
        f"nanoops rms_norm only supports 1D normalized_shape, got {normalized_shape}"
    )
    D = normalized_shape[0]
    assert input.shape[-1] == D, f"input last dim {input.shape[-1]} != normalized_shape {D}"
    input_shape = input.shape
    input_flat = input.reshape(-1, D)
    out_flat = RMSNorm.apply(input_flat, weight, eps)
    return out_flat.reshape(input_shape)

@_allow_in_graph
def outer(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Mirrors `torch.outer`. Both inputs must be 1D.

    Implemented as a 2D matmul of column x row: (M, 1) @ (1, N) -> (M, N).
    No new autograd Function needed — Mm's backward plus autograd's view
    backward together give the right (M,) and (N,) gradients.
    """
    assert left.ndim == 1, f"left must be 1D, got {left.ndim}D"
    assert right.ndim == 1, f"right must be 1D, got {right.ndim}D"
    return Mm.apply(left.unsqueeze(-1), right.unsqueeze(0))


class Cat(torch.autograd.Function):
    """Concatenation along a dim, mirroring `torch.cat` semantics.

    Signature note: `dim` is the FIRST positional parameter (before `*tensors`)
    because Python requires `*args` to be the last positional parameter. The
    functional `cat()` wrapper re-orders to PyTorch's `(tensors, dim)` convention.

    Why `*tensors` and not `tensors: list[Tensor] | tuple[Tensor, ...]`:
    `autograd.Function.apply()` only tracks gradients through tensors that are
    individual positional arguments — it does NOT recurse into list/tuple
    containers to find tensors. Passing a list as a single arg makes autograd
    treat it as an opaque non-differentiable constant, so the backward graph
    never gets built and the output ends up with no grad_fn. Splatting the
    tensors via `*tensors` makes each one its own positional arg, which is
    what autograd needs to set up the chain. (Type hints are static-analysis
    only; they don't influence apply's runtime behavior.)
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        dim: int,
        *tensors: torch.Tensor,
    ) -> torch.Tensor:
        ctx.dim = dim
        ctx.sizes = [t.shape[dim] for t in tensors]
        return torch.cat(tensors, dim=dim)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> "tuple[None, *tuple[torch.Tensor, ...]]":
        grads = torch.split(grad_output, ctx.sizes, dim=ctx.dim)
        return (None, *grads)  # None for dim (int, non-differentiable)


@_allow_in_graph
def cat(
    tensors: list[torch.Tensor] | tuple[torch.Tensor, ...],
    dim: int = 0,
) -> torch.Tensor:
    """Mirrors `torch.cat`. Reorders args so dim goes to Cat's first slot."""
    return Cat.apply(dim, *tensors)


@_allow_in_graph
def stack(
    tensors: list[torch.Tensor] | tuple[torch.Tensor, ...],
    dim: int = 0,
) -> torch.Tensor:
    """Mirrors `torch.stack` using `cat` and `unsqueeze`."""
    assert all(t.shape == tensors[0].shape for t in tensors), (
        f"all tensors must have the same shape; got {[tuple(t.shape) for t in tensors]}"
    )
    unsqueezed = [t.unsqueeze(dim) for t in tensors]
    return cat(unsqueezed, dim=dim)


class ReluSquare(torch.autograd.Function):
    """Fused ReLU-squared: y = max(x, 0)**2 — nanchat's MLP activation.

    Backward simplifies beautifully:
        d(relu(x)^2) / dx = 2 * relu(x) * 1[x > 0] = 2 * relu(x)
    The mask is redundant because relu(x) is already 0 wherever x ≤ 0,
    so we don't need to save or apply it — backward is just `2 * y * g`,
    one fused mul chain. Saves the relu output y (same memory as input).

    Compared to composing `square(relu(x))`: this fuses both ops' backwards
    into one analytic expression, ~3x fewer backward FLOPs and one fewer
    intermediate tensor. The classic "fusion-as-optimization" pattern,
    minus the GPU kernel — same idea as fused GELU / SwiGLU in production.

    Subgradient at x=0 taken as 0 (PyTorch convention).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor
    ) -> torch.Tensor:
        y = x.clamp(min=0)
        ctx.save_for_backward(y)
        return y * y

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> torch.Tensor:
        (y,) = ctx.saved_tensors
        return grad_output * 2 * y  # = 2 * relu(x) * g; mask absorbed into y


@_allow_in_graph
def relu_square(input: torch.Tensor) -> torch.Tensor:
    """Fused relu(x)**2 — nanchat's MLP activation."""
    return ReluSquare.apply(input)

class Softmax(torch.autograd.Function):
    """Softmax along a dim, mirroring `torch.nn.functional.softmax` semantics."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        input_max = input.amax(dim=dim, keepdim=True)
        input_exp = (input - input_max).exp()
        sum_exp = input_exp.sum(dim=dim, keepdim=True)
        # In-place divide reuses input_exp's storage as the output. Safe because
        # input_exp is a fresh tensor (just created by .exp()) with no other
        # references — saves one input-sized allocation vs `input_exp / sum_exp`.
        output = input_exp.div_(sum_exp)
        ctx.save_for_backward(output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (output,) = ctx.saved_tensors
        dim = ctx.dim
        # Compact: y * (g - <g,y>) = g*y - y*<g,y>. Compute g*y once into gy,
        # then in-place subtract y * <g,y> via addcmul_(... value=-1), which
        # is a single fused kernel — no `output * inner` intermediate tensor.
        # Saves one input-sized transient + one kernel launch vs the naive form.
        gy = grad_output * output                                  # 1 alloc
        inner = gy.sum(dim=dim, keepdim=True)                       # tiny <g,y>
        gy.addcmul_(output, inner, value=-1)                        # in-place: gy -= y * inner
        return gy, None  # None for dim (int, non-differentiable)


@_allow_in_graph
def softmax(
    input: torch.Tensor,
    dim: int = -1,
    _stacklevel: int = 3,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Mirrors `torch.nn.functional.softmax`. Default dim=-1 (PyTorch's
    default is None and warns; we pick the common case).

    `_stacklevel` and `dtype` are accepted for signature compatibility with
    `torch.nn.functional.softmax` (so `nn.Softmax.forward()` can route here
    via monkey-patching); `_stacklevel` is ignored, `dtype` triggers a cast
    of the input before softmax (matching PyTorch's behavior).
    """
    if dtype is not None:
        input = input.to(dtype)
    return Softmax.apply(input, dim)


def chunked_logsumexp(
    input: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Memory-efficient logsumexp via online softmax.

    `torch.logsumexp(x, dim)` materializes a full `exp(x - max)` tensor the
    same shape as `x` during compute. For (B*T, V) logits at LLM scale this
    is a transient ~13 GB allocation. This chunked version processes `x`
    along `dim` in slices of `chunk_size`, maintaining a running
    `(max, sum_of_exp_shifted_by_max)` pair. Transient peak memory drops to
    roughly `chunk_size / size_along_dim` times what `torch.logsumexp` uses.

    Algorithm (online softmax — the same trick at the heart of Flash Attention):

        for each chunk c along dim:
            m'      = max(running_max, c.amax)
            new_sum = running_sum * exp(running_max - m')        # rebase to m'
            new_sum += (c - m').exp().sum                         # add chunk
            running_max, running_sum = m', new_sum
        return running_max + log(running_sum)

    Numerically identical to `torch.logsumexp` up to fp rounding (each rebase
    preserves the running LSE in shifted form).

    Performance: roughly memory-bandwidth bound; smaller chunks add only
    kernel-launch overhead. On a 3090, chunk_size=4096 is ~5% slower than
    `torch.logsumexp` with 1/4 the transient peak; chunk_size=1024 cuts the
    peak by 16x at ~14% extra time. Anything in [1024, 8192] is reasonable.
    """
    chunks = input.split(chunk_size, dim)
    # Initialize running stats from the first chunk: no rebase needed yet.
    running_max = chunks[0].amax(dim, keepdim=True)
    running_sum = (chunks[0] - running_max).exp().sum(dim, keepdim=True)
    # Fold in each remaining chunk: shift the basis to the new max, then add
    # the chunk's own exp-sum contribution computed in that same basis.
    for chunk in chunks[1:]:
        new_max = torch.maximum(running_max, chunk.amax(dim, keepdim=True))
        rebase = (running_max - new_max).exp()                       # rescale old sum
        chunk_sum = (chunk - new_max).exp().sum(dim, keepdim=True)   # this chunk's contribution
        running_sum = running_sum * rebase + chunk_sum
        running_max = new_max
    result = running_max + running_sum.log()
    return result if keepdim else result.squeeze(dim)


class CrossEntropy(torch.autograd.Function):
    """Cross-entropy loss, mirroring `torch.nn.functional.cross_entropy` semantics.

    Fused log-softmax + NLL with memory optimizations in both directions.
    Empirical: at NT=16K, V=32K, bf16 (logits ~1 GB), the total peak GPU
    memory is roughly HALF of PyTorch's `F.cross_entropy` (2x input vs 4x
    input).

    Forward optimizations:
      1. ctx saves `(input, log_sum_exp, target)`, NOT softmax — `input`
         is already alive as a function argument (saved reference is free)
         and `log_sum_exp` is `(..., 1)`. Backward recomputes softmax via
         `(input - log_sum_exp).exp_()`. Saves ~1x logits of long-lived
         memory vs PyTorch (which keeps log_softmax in ctx until backward).
      2. LSE itself uses `chunked_logsumexp` (online softmax), so the
         transient exp-temp during forward is bounded by `chunk_size / V`
         of logits size (~1/8 of what `torch.logsumexp` would allocate).

    Backward optimizations (all in-place to keep peak at 1x softmax):
      3. `(input - log_sum_exp).exp_()` — in-place exp on the sub temp;
         using `.exp()` would briefly hold both sub_temp and exp_result
         simultaneously (2x peak).
      4. `grad_input *= ...` — in-place scaling by upstream * mask; `a*b`
         would create a *new* (B*T, V) tensor, another 2x peak transient.

    See README appendix for the derivation; the boxed result is
    `dL/dx = softmax(x) - one_hot(target)`.

    Honest performance note: nanoops wins on memory but loses on time.
    At the same NT=16K, V=32K, bf16 benchmark on a 3090, one fwd+bwd takes:

        PyTorch F.cross_entropy (eager)        :  9.2 ms   (baseline)
        torch.compile (default, fused Triton)  :  4.1 ms   (2.3x faster)
        nanoops CrossEntropy (this class)      : 16.3 ms   (1.8x SLOWER)

    The slowness is structural: nanoops's Python-level autograd Function
    dispatches ~15-20 individual PyTorch ops per fwd+bwd, each paying
    ~30-50 us of dispatch + autograd machinery (the chunked LSE loop alone
    is ~8 op-dispatches per chunk). PyTorch eager calls a single fused
    C++ `cross_entropy_loss`; `torch.compile` further fuses everything
    into one Triton kernel. Closing this gap requires kernel-level fusion
    (Triton / CUDA), which is out of scope for nanoops.

    So: read this class to understand what fusion buys you in memory and
    why production code (Liger Kernel, Flash CE, `torch.compile`) does it
    at the kernel level. Don't use this class as a drop-in replacement
    expecting the same throughput.

    Returns per-sample loss (shape matches `target`); `'mean'` / `'sum'`
    reduction is handled by the `cross_entropy()` functional wrapper.

    See the README appendix for the full derivation (the boxed result is
    `dL/dx = softmax(x) - one_hot(target)`).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        target: torch.Tensor,
        dim: int,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        # Use chunked_logsumexp (online softmax) so even the transient exp
        # tensor is broken into chunk_size-sized pieces. With torch.logsumexp
        # the temp briefly hits 1x input size (~13 GB at LLM scale); chunked
        # version peaks at chunk_size / V of that. ctx still doesn't end up
        # holding any (..., V) tensor of its own.
        log_sum_exp = chunked_logsumexp(input, dim=dim, keepdim=True)
        # ignore_index handling: gather() would crash on out-of-range target
        # values (e.g. -1), so replace those with 0 as a safe placeholder,
        # compute the per-sample loss as usual, then zero out those positions.
        valid_mask = target != ignore_index
        safe_target = torch.where(valid_mask, target, 0)
        per_sample = (log_sum_exp - input.gather(dim, safe_target.unsqueeze(dim))).squeeze(dim)
        per_sample = per_sample * valid_mask  # 0 at ignored positions
        # Save `input` (already alive — just a reference, no new allocation)
        # and the tiny log_sum_exp; recompute softmax in backward.
        ctx.save_for_backward(input, log_sum_exp, target)
        ctx.dim = dim
        ctx.ignore_index = ignore_index
        return per_sample

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        input, log_sum_exp, target = ctx.saved_tensors
        dim = ctx.dim
        ignore_index = ctx.ignore_index
        # Recompute softmax = exp(input - LSE); stable since input - LSE <= 0.
        # `.exp_()` (in-place on the sub temp) keeps peak at 1x (..., V); using
        # `.exp()` would briefly hold sub_temp AND exp_result simultaneously
        # → 2x peak. Same one extra mul+exp we pay for the forward savings.
        grad_input = (input - log_sum_exp).exp_()  # = softmax(input)
        # softmax - one_hot(target): subtract 1 at the target column per row.
        # For ignored positions we use safe_target=0 here (the row gets zeroed
        # out by valid_mask below, so the spurious -1 at col 0 doesn't matter).
        valid_mask = target != ignore_index
        safe_target = torch.where(valid_mask, target, 0)
        target_idx = safe_target.unsqueeze(dim)
        grad_input.scatter_add_(
            dim,
            target_idx,
            torch.full_like(target_idx, -1.0, dtype=grad_input.dtype),
        )
        # Scale by upstream grad_output AND zero out ignored rows in one step.
        # Use `*=` (in-place) — `a * b` would create a *new* (B*T, V) tensor,
        # briefly doubling backward peak to 2x the softmax size before the old
        # one becomes unreferenced. `*=` modifies grad_input directly.
        #   - grad_output (per-sample) broadcasts across the class dim
        #   - valid_mask (bool) → 0.0 at ignored positions kills those rows
        grad_input *= (grad_output * valid_mask).unsqueeze(dim)
        # Return one grad per forward input: (input, target, dim, ignore_index).
        # Only `input` is differentiable; the rest are ints / non-tensor.
        return grad_input, None, None, None


@_allow_in_graph
def cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    size_average=None,  # deprecated in PyTorch
    ignore_index: int = -100,
    reduce=None,  # deprecated in PyTorch
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    dim: int = -1,
) -> torch.Tensor:
    """Mirrors `torch.nn.functional.cross_entropy`.

    Wraps the fused `CrossEntropy` autograd Function and applies reduction.
    See `CrossEntropy` for the algorithm and the memory / time trade-offs;
    this wrapper just adds reduction.

    Args:
      input:        Logits of shape (..., C); class dim defaults to -1.
      target:       Integer class indices, shape matching input minus dim.
                    Positions equal to `ignore_index` contribute 0 to both
                    loss and gradient.
      ignore_index: Target value to skip. Defaults to -100 (PyTorch default);
                    nanchat passes -1 explicitly.
      reduction:    "mean" (default) / "sum" / "none". "mean" divides by the
                    number of *non-ignored* positions, matching PyTorch
                    (returns NaN if all positions are ignored).
      dim:          Class dim. Defaults to -1 (PyTorch hardcodes class to
                    dim 1; we expose `dim` for flexibility).

    Not supported (vs PyTorch):
      - `weight` (per-class loss weighting).
      - `label_smoothing`.
      - Deprecated `size_average` / `reduce` flags.
    Signatures are accepted (for nn.CrossEntropyLoss.forward routing) but
    non-default values raise NotImplementedError.
    """
    if weight is not None:
        raise NotImplementedError("nanoops.cross_entropy: weight not supported")
    if label_smoothing != 0.0:
        raise NotImplementedError("nanoops.cross_entropy: label_smoothing not supported")
    if size_average is not None or reduce is not None:
        raise NotImplementedError(
            "nanoops.cross_entropy: deprecated size_average/reduce not supported"
        )
    per_sample = CrossEntropy.apply(input, target, dim, ignore_index)
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        # PyTorch convention: divide by N_valid (excluding ignore_index), not
        # by total. Cast to float to avoid integer division. If everything is
        # ignored, valid_count is 0 and we return NaN (matching PyTorch).
        valid_count = (target != ignore_index).sum().to(per_sample.dtype)
        return per_sample.sum() / valid_count
    raise ValueError(f"unknown reduction: {reduction!r} (expected 'mean'/'sum'/'none')")


class Sigmoid(torch.autograd.Function):
    """Elementwise sigmoid: y = 1 / (1 + exp(-x)).

    Backward: dL/dx = g * y * (1 - y) — uses only y. Sigmoid is bijective,
    so y uniquely determines x; saving y (the output) is the natural choice
    (same memory as saving x, but lets backward use the y(1-y) form directly).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor
    ) -> torch.Tensor:
        # Use the tensor method `x.sigmoid()` rather than `torch.sigmoid(x)`:
        # `nanoops.integration` monkey-patches `torch.sigmoid` to *this*
        # function, so calling `torch.sigmoid(x)` here would infinitely
        # recurse. Tensor methods aren't shadowed by that patch.
        y = x.sigmoid()
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> torch.Tensor:
        (y,) = ctx.saved_tensors
        return grad_output * y * (1 - y)


@_allow_in_graph
def sigmoid(input: torch.Tensor) -> torch.Tensor:
    """Mirrors `torch.sigmoid`."""
    return Sigmoid.apply(input)


class Tanh(torch.autograd.Function):
    """Elementwise tanh: y = tanh(x).

    Backward: dL/dx = g * (1 - y^2) — uses only y. Like sigmoid, tanh is
    bijective; saving the output y is the natural choice.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor
    ) -> torch.Tensor:
        # See Sigmoid.forward for why we use the tensor method here instead
        # of the top-level `torch.tanh` function.
        y = x.tanh()
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> torch.Tensor:
        (y,) = ctx.saved_tensors
        return grad_output * (1 - y * y)


@_allow_in_graph
def tanh(input: torch.Tensor) -> torch.Tensor:
    """Mirrors `torch.tanh`."""
    return Tanh.apply(input)


class ApplyRotaryEmb(torch.autograd.Function):
    """Apply rotary positional embedding to a 4D tensor's last-dim halves.

    Mirrors `nanchat/gpt.py:57` apply_rotary_emb. Treats the last dim as
    two halves (x1, x2) and rotates each pair (x1[i], x2[i]) by the angle
    encoded in (cos[i], sin[i]):

        y1 = x1 * cos + x2 * sin
        y2 = -x1 * sin + x2 * cos

    In matrix form this is R(θ) applied to a 2-vector. Rotation matrices
    are orthogonal: R(θ)^T = R(-θ). So backward is the SAME forward shape
    with sin negated:

        grad_x1 = g1 * cos - g2 * sin
        grad_x2 = g1 * sin + g2 * cos

    Memory: ctx saves only (cos, sin) — precomputed lookup tables, usually
    shared across the entire attention call. NO x or y is saved (backward
    is purely linear in grad_output, with cos/sin as multipliers).

    cos and sin come from non-differentiable arange/outer/cos/sin
    precomputation (see `nanchat/gpt.py:_precompute_rotary_embeddings`),
    so backward returns None for both.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        assert x.ndim == 4, f"x must be 4D (B, T, n_head, head_dim), got {x.ndim}D"
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        # Pre-allocate output and write each half via slice assignment.
        # The original `torch.cat([y1, y2], dim=-1)` had to hold y1, y2, AND
        # the cat output alive at the same time (peak ~2x input). Slicing
        # into a pre-allocated buffer drops y1, y2 immediately after the
        # copy, lowering peak to ~1.5x input. Called 40+ times per d20 iter
        # (every Q and K of every layer), so the peak savings compound.
        out = torch.empty_like(x)
        out[..., :d] = x1 * cos + x2 * sin
        out[..., d:] = -x1 * sin + x2 * cos
        ctx.save_for_backward(cos, sin)
        return out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        cos, sin = ctx.saved_tensors
        d = grad_output.shape[-1] // 2
        g1, g2 = grad_output[..., :d], grad_output[..., d:]
        # Same pre-allocate + slice-assign trick as forward; R(-θ) means
        # the formula is the forward with sin → -sin.
        grad_x = torch.empty_like(grad_output)
        grad_x[..., :d] = g1 * cos - g2 * sin
        grad_x[..., d:] = g1 * sin + g2 * cos
        # cos and sin are precomputed constants, not differentiable.
        return grad_x, None, None


@_allow_in_graph
def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotary positional embedding, mirroring `nanchat/gpt.py:57`."""
    return ApplyRotaryEmb.apply(x, cos, sin)


class Where(torch.autograd.Function):
    """Elementwise `torch.where(cond, a, b)` with grad routing by `cond`.

    Forward picks `a` where cond is True, else `b`. Backward routes the
    upstream grad to whichever operand was selected: dL/da is g masked by
    cond, dL/db is g masked by ~cond. `cond` is bool and non-differentiable.

    Broadcasting: cond/a/b can have different broadcastable shapes; the
    backward unbroadcasts each per-operand grad back to the operand's
    original shape (same `unbroadcast` helper as `Add` / `Mul`).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        condition: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(condition)
        ctx.a_shape = a.shape
        ctx.b_shape = b.shape
        return torch.where(condition, a, b)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[None, torch.Tensor, torch.Tensor]:
        (condition,) = ctx.saved_tensors
        # Bool * float promotes False->0, True->1 — gradient routing without
        # allocating intermediate zero tensors.
        grad_a = unbroadcast(grad_output * condition, ctx.a_shape)
        grad_b = unbroadcast(grad_output * (~condition), ctx.b_shape)
        return None, grad_a, grad_b  # condition is bool, non-differentiable


@_allow_in_graph
def where(
    condition: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Mirrors `torch.where(condition, a, b)` (3-arg form)."""
    return Where.apply(condition, a, b)


class Roll(torch.autograd.Function):
    """Cyclic shift along given dim(s), mirroring `torch.roll`.

    Forward applies the shift; backward applies the inverse shift
    (negate shifts, keep dims). Bijective op — backward saves no tensor,
    just the integer shift/dim params on ctx.

    `shifts` can be `int` or tuple; `dims` can be `int`, tuple, or `None`
    (None flattens the tensor first, same as `torch.roll`).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        shifts,
        dims=None,
    ) -> torch.Tensor:
        ctx.shifts = shifts
        ctx.dims = dims
        return torch.roll(input, shifts, dims)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        shifts = ctx.shifts
        if isinstance(shifts, (list, tuple)):
            neg_shifts = tuple(-s for s in shifts)
        else:
            neg_shifts = -shifts
        return torch.roll(grad_output, neg_shifts, ctx.dims), None, None


@_allow_in_graph
def roll(input: torch.Tensor, shifts, dims=None) -> torch.Tensor:
    """Mirrors `torch.roll`. shifts: int or tuple; dims: int, tuple, or None."""
    return Roll.apply(input, shifts, dims)


class ScaledDotProductAttention(torch.autograd.Function):
    """Naive (non-Flash) scaled dot-product attention with explicit backward.

    Forward (row-vector convention, features on last dim):
        S = (Q @ K^T) * scale         # scores, shape (..., L, S)
        S += mask / -inf where masked # optional causal or attn_mask
        P = softmax(S, dim=-1)        # attention probs
        O = P @ V                     # output, shape (..., L, D_v)

    Backward (closed form, derivable from chain rule over the three ops):
        dV = P^T @ g
        dP = g @ V^T
        dS = P * (dP - sum(P * dP, dim=-1, keepdim=True))      # softmax_backward
        dQ = (dS * scale) @ K
        dK = (dS * scale)^T @ Q

    ctx saves Q, K, V (originals; pre-GQA-expand), and P. Saving P is the
    "naive" memory strategy — O(B*H*L*S). Flash Attention saves only the
    normalization stats (L = log-sum-exp, M = max), recomputes P from Q/K
    in backward at O(B*H*L) extra compute but O(B*H*L) memory. We do not.

    Args
    ----
    query, key, value : Tensor
        Shapes (..., L, D_k), (..., S, D_k), (..., S, D_v).
    attn_mask : Tensor or None
        bool tensor → True keeps the position, False masks (sets score to -inf).
        float tensor → added to scores (use -inf or large negative to mask).
        Mutually exclusive with is_causal.
    is_causal : bool
        Lower-triangular causal mask (tril with diagonal=S-L offset, so it
        works for both L==S training and L<S cached generation).
    scale : float or None
        Defaults to 1/sqrt(d_k).
    enable_gqa : bool
        Grouped-query attention. If True and Q has more heads than K/V,
        repeats K/V along the heads dim by factor G = H_q / H_kv.
        Forward expands K/V via repeat_interleave; backward sums grads
        from the G replicas back into H_kv slots (via unflatten + sum).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        if is_causal and attn_mask is not None:
            raise ValueError("is_causal and attn_mask are mutually exclusive")

        d_k = query.shape[-1]
        if scale is None:
            scale = 1.0 / math.sqrt(d_k)

        # GQA: repeat K/V heads to match Q heads.
        H_q = query.shape[-3] if query.ndim >= 3 else 1
        H_kv = key.shape[-3] if key.ndim >= 3 else 1
        if enable_gqa and H_q != H_kv:
            assert H_q % H_kv == 0, (
                f"GQA needs H_q ({H_q}) divisible by H_kv ({H_kv})"
            )
            G = H_q // H_kv
            key_e = key.repeat_interleave(G, dim=-3)
            value_e = value.repeat_interleave(G, dim=-3)
        else:
            G = 1
            key_e, value_e = key, value

        # scores = Q @ K^T * scale
        scores = (query @ key_e.transpose(-2, -1)) * scale

        if is_causal:
            L, S = scores.shape[-2], scores.shape[-1]
            # Upper-left aligned causal (matches PyTorch SDPA semantics).
            # For L==S this is standard lower-triangular. For L != S, the
            # causal triangle sits at the top-left of the L×S grid, NOT
            # right-aligned to the K edge — this is PyTorch's documented
            # convention. Cached-generation use cases (where you want
            # right-aligned causal) should pass an explicit attn_mask.
            causal_mask = torch.ones(L, S, dtype=torch.bool, device=scores.device).tril()
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask

        probs = torch.softmax(scores, dim=-1)
        output = probs @ value_e

        ctx.save_for_backward(query, key, value, probs)
        ctx.scale = scale
        ctx.G = G
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None]:
        query, key, value, probs = ctx.saved_tensors
        scale = ctx.scale
        G = ctx.G

        # Re-expand K/V to match probs's head count (cheap; same op as forward).
        if G > 1:
            key_e = key.repeat_interleave(G, dim=-3)
            value_e = value.repeat_interleave(G, dim=-3)
        else:
            key_e, value_e = key, value

        # dL/dV_e = P^T @ g
        grad_value_e = probs.transpose(-2, -1) @ grad_output

        # dL/dP = g @ V_e^T
        grad_probs = grad_output @ value_e.transpose(-2, -1)

        # softmax_backward: dL/dS = P * (dP - sum(P*dP, dim=-1, keepdim))
        # Note: masked positions have P=0, so dS=0 there — masks naturally
        # don't propagate grad. Also: this is the same formula nanoops's
        # Softmax uses (the "save y, not x" trick).
        #
        # Compact form: P*(dP - <P,dP>) = P*dP - P*<P,dP>. Compute pg = P*dP
        # once, then subtract `P * <P,dP>` in-place via addcmul_(value=-1).
        # That's a single fused kernel — no `P * <P,dP>` intermediate. Saves
        # one P-sized transient per SDPA backward (the (B,H,L,S) attention
        # matrix is the dominant memory term, so cutting one is meaningful).
        grad_scores = probs * grad_probs                                    # 1 P-size alloc
        inner = grad_scores.sum(dim=-1, keepdim=True)                       # tiny (B,H,L,1)
        grad_scores.addcmul_(probs, inner, value=-1)                        # in-place: grad_scores -= probs * inner

        # Fold scale into dS once, used by both dQ and dK. In-place since
        # grad_scores has no other references.
        grad_scores.mul_(scale)

        # dL/dQ = dS @ K_e
        grad_query = grad_scores @ key_e

        # dL/dK_e = dS^T @ Q
        grad_key_e = grad_scores.transpose(-2, -1) @ query

        # GQA backward: collapse G replicas of K/V back into H_kv heads.
        # repeat_interleave([a, b], G, dim=-3) -> [a]*G ++ [b]*G; backward
        # sums each G-block back to a single element.
        if G > 1:
            # (..., H_q, S, D) -> (..., H_kv, G, S, D) -> sum over G (dim=-3).
            grad_key = grad_key_e.unflatten(-3, (-1, G)).sum(dim=-3)
            grad_value = grad_value_e.unflatten(-3, (-1, G)).sum(dim=-3)
        else:
            grad_key = grad_key_e
            grad_value = grad_value_e

        # query, key, value, attn_mask, is_causal, scale, enable_gqa
        return grad_query, grad_key, grad_value, None, None, None, None


@_allow_in_graph
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Mirrors `F.scaled_dot_product_attention` (causal + mask + GQA)."""
    return ScaledDotProductAttention.apply(
        query, key, value, attn_mask, is_causal, scale, enable_gqa
    )
