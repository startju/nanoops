"""Functional ops, mirroring `torch.nn.functional`."""

import torch


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
    # Sum out broadcasted dimensions to match target shape
    while grad.ndim > len(target_shape):
        grad = grad.sum(dim=0)
    for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if g_dim != t_dim:
            grad = grad.sum(dim=i, keepdim=True)
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


def linear(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    # weight: (out, in); input: (..., in) -> (..., out)
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
