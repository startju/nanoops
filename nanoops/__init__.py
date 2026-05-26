"""nanoops: a from-scratch reimplementation of the PyTorch operators used by nanochat.

The public API mirrors `torch.nn` and `torch.nn.functional` so that nanochat code
can switch implementations by changing imports only.
"""

from . import nn
from . import functional

__all__ = ["nn", "functional"]
