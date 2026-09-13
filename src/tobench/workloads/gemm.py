"""GEMM workload expressed as a PyTorch module."""

from typing import Literal

import torch
from torch import Tensor

from tobench.workloads.base_workload import BaseWorkload


class GEMM(BaseWorkload):
    """Compute C[M, N] = A[M, K] @ B[K, N].

    The benchmark supplies contiguous row-major matrices in the configured
    dtype. There is no transpose, bias, scaling, or existing output accumulation.
    Numerics follow torch.matmul and the backend's precision settings.
    """

    op = "matmul"
    layout = "row_major"

    def __init__(
        self,
        M: int,
        N: int,
        K: int,
        dtype: Literal["float16", "bfloat16"] = "float16",
        seed: int = 0,
    ) -> None:
        super().__init__(dtype=dtype, seed=seed)
        for name, value in (("M", M), ("N", N), ("K", K)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.M = M
        self.N = N
        self.K = K

    @property
    def input_shapes(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Shapes of A and B, in forward argument order."""
        return ((self.M, self.K), (self.K, self.N))

    @property
    def output_shape(self) -> tuple[int, int]:
        """Shape of output C."""
        return (self.M, self.N)

    @property
    def flop_count(self) -> int:
        """Conventional operation count: two operations per multiply-add."""
        return 2 * self.M * self.N * self.K

    def forward(self, A: Tensor, B: Tensor) -> Tensor:
        return torch.matmul(A, B)

    def to_config(self) -> dict:
        return {
            **super().to_config(),
            "op": self.op,
            "M": self.M,
            "N": self.N,
            "K": self.K,
            "layout": self.layout,
        }
