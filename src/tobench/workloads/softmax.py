"""Batched last-axis softmax workload."""

from typing import Literal

import torch
from torch import Tensor

from .base_workload import BaseWorkload


class Softmax(BaseWorkload):
    """Compute a numerically stable softmax over X[B, M, N]'s last axis."""

    op = "softmax"
    layout = "row_major"

    def __init__(
        self,
        B: int = 8,
        M: int = 128,
        N: int = 1024,
        dtype: Literal["float16", "bfloat16"] = "float16",
        seed: int = 0,
    ) -> None:
        super().__init__(dtype=dtype, seed=seed)
        for name, value in (("B", B), ("M", M), ("N", N)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.B, self.M, self.N = B, M, N

    @property
    def input_shapes(self) -> tuple[tuple[int, int, int]]:
        return ((self.B, self.M, self.N),)

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return (self.B, self.M, self.N)

    @property
    def flop_count(self) -> int:
        """Approximate max/subtract/exp/sum/div work as five ops per element."""
        return 5 * self.B * self.M * self.N

    def forward(self, X: Tensor) -> Tensor:
        return torch.softmax(X.float(), dim=-1).to(X.dtype)

    def to_config(self) -> dict:
        return {
            **super().to_config(),
            "op": self.op,
            "B": self.B,
            "M": self.M,
            "N": self.N,
            "axis": -1,
            "accumulation_dtype": "float32",
            "layout": self.layout,
            "flop_count_convention": "5*B*M*N",
        }
