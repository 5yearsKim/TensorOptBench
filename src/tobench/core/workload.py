"""Shared base for workloads expressed as PyTorch modules."""

from abc import ABC, abstractmethod
from typing import Literal

from torch import Tensor, nn


class Workload(nn.Module, ABC):
    """A computation with shared input-generation configuration.

    Concrete workloads define their computation in ``forward``. Construction
    does not allocate inputs; the runner supplies tensors to the module.
    """

    def __init__(
        self,
        dtype: Literal["float16", "bfloat16"] = "float16",
        seed: int = 0,
    ) -> None:
        super().__init__()
        if dtype not in ("float16", "bfloat16"):
            raise ValueError("dtype must be 'float16' or 'bfloat16'")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.dtype = dtype
        self.seed = seed

    def to_config(self) -> dict:
        """Return serializable parameters; operators extend this with their shapes."""
        return {"name": type(self).__name__, "dtype": self.dtype, "seed": self.seed}

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        """Compute the workload output from the supplied inputs."""
        ...
