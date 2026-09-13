"""RMS normalization followed by a bias-free linear projection."""

import math
from typing import Literal

import torch
from torch import Tensor

from .base_workload import BaseWorkload


class RMSNormLinear(BaseWorkload):
    """Compute rmsnorm(X, G) @ W.T, normalizing the last dimension.

    X has shape [M, K], scale G [1, K], and weight W [N, K]. Normalization
    and scaling use FP32; the normalized value is cast to the input dtype
    before the projection. Output has shape [M, N] in the input dtype.
    """

    op = "rmsnorm_linear"
    layout = "row_major"

    def __init__(
        self, M: int = 16, N: int = 406, K: int = 4096,
        dtype: Literal["float16", "bfloat16"] = "float16",
        seed: int = 0, eps: float = 1e-6,
    ) -> None:
        super().__init__(dtype=dtype, seed=seed)
        for name, value in (("M", M), ("N", N), ("K", K)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(eps, bool) or not isinstance(eps, (int, float)):
            raise TypeError("eps must be a number")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.M, self.N, self.K = M, N, K
        self.eps = float(eps)

    @property
    def input_shapes(self) -> tuple[tuple[int, int], ...]:
        """Shapes in forward argument order: X, G, W."""
        return ((self.M, self.K), (1, self.K), (self.N, self.K))

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.M, self.N)

    @property
    def flop_count(self) -> int:
        """Approximate arithmetic count, treating rsqrt as one operation.

        Projection uses 2*M*N*K; normalization/scaling uses 4*M*K + 2*M.
        Casts and memory operations are excluded.
        """
        return 2 * self.M * self.N * self.K + 4 * self.M * self.K + 2 * self.M

    def forward(self, X: Tensor, G: Tensor, W: Tensor) -> Tensor:
        x = X.float()
        inverse_rms = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
        normalized = (x * inverse_rms * G.float()).to(X.dtype)
        return torch.nn.functional.linear(normalized, W)

    def to_config(self) -> dict:
        return {
            **super().to_config(), "op": self.op,
            "M": self.M, "N": self.N, "K": self.K,
            "eps": self.eps, "normalization_axis": -1,
            "normalization_dtype": "float32", "layout": self.layout,
            "weight_layout": "out_features_in_features",
            "flop_count_convention": "2*M*N*K + 4*M*K + 2*M; rsqrt counts as one",
        }
