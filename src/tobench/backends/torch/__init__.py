"""PyTorch eager and TorchInductor adapters."""

from .eager import TorchEagerAdapter
from .inductor import TorchInductorAdapter

__all__ = ["TorchEagerAdapter", "TorchInductorAdapter"]
