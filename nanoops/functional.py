"""Functional ops, mirroring `torch.nn.functional`."""
import torch


class Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        assert left.ndim == 2, f"left must be 2D, got {left.ndim}D"
        assert right.ndim == 2, f"right must be 2D, got {right.ndim}D"
        ctx.save_for_backward(left, right)
        output = left @ right
        return output

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left, right = ctx.saved_tensors
        grad_left = grad_output @ right.T
        grad_right = left.T @ grad_output
        return grad_left, grad_right


def linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    # weight: (out, in); input: (..., in) -> (..., out)
    out = input @ weight.transpose(-1, -2)
    if bias is not None:
        out = out + bias
    return out
