"""Module-style ops, mirroring `torch.nn`."""

import torch
from torch import nn

from . import functional as F


class Linear(nn.Module):
    """Drop-in replacement for `torch.nn.Linear` (API and forward-pass identical).

    Init deliberately diverges from PyTorch's `nn.Linear` default:
      - weight: `kaiming_uniform_(a=1)` -> bound = sqrt(3 / fan_in),
                i.e. gain=1 / "linear" nonlinearity / variance-preserving.
                PyTorch's default `a=sqrt(5)` is a Torch7 historical
                inheritance with no real activation justification (it gives
                bound = 1/sqrt(fan_in), sqrt(3)x smaller, variance 3x smaller).
      - bias:   zeros. PyTorch's `uniform_(-1/sqrt(fan_in), +1/sqrt(fan_in))`
                has no theoretical basis for bias (bias has no fan_in) and
                amounts to negligible perturbation anyway.
    This matches nanochat's own init for attention/MLP projections (where
    `s = sqrt(3/n_embd)` is used for c_q/c_k/c_v) — see `nanochat/gpt.py`.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Embedding(nn.Module):
    """Drop-in replacement for `torch.nn.Embedding` (core subset).

    Init matches PyTorch default: N(0, 1) per element. nanochat overrides
    this for its own `wte` with std=0.8 (see `nanochat/gpt.py`), so the
    default here only matters when nanoops is used outside nanochat.

    Not implemented (add when needed):
      - padding_idx       (zero & freeze a specific row)
      - max_norm/norm_type (renormalize over-large rows)
      - scale_grad_by_freq (divide row gradient by token frequency in batch)
      - sparse             (sparse gradient tensor + SparseAdam)
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.weight)

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
