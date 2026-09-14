"""Explicit batched multi-head attention workload."""

import math
from typing import Literal

import torch
from torch import Tensor

from .base_workload import BaseWorkload


class Attention(BaseWorkload):
    """Compute unmasked scaled dot-product attention for Q, K, and V."""

    op = "attention"
    layout = "BHSD"

    def __init__(
        self,
        B: int = 2,
        H: int = 8,
        S: int = 128,
        D: int = 64,
        dtype: Literal["float16", "bfloat16"] = "float16",
        seed: int = 0,
    ) -> None:
        super().__init__(dtype=dtype, seed=seed)
        for name, value in (("B", B), ("H", H), ("S", S), ("D", D)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.B, self.H, self.S, self.D = B, H, S, D
        self.scale = 1.0 / math.sqrt(D)

    @property
    def input_shapes(self) -> tuple[tuple[int, int, int, int], ...]:
        shape = (self.B, self.H, self.S, self.D)
        return (shape, shape, shape)

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        return (self.B, self.H, self.S, self.D)

    @property
    def flop_count(self) -> int:
        matmuls = 4 * self.B * self.H * self.S * self.S * self.D
        scale_and_softmax = 6 * self.B * self.H * self.S * self.S
        return matmuls + scale_and_softmax

    def forward(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        probabilities = torch.softmax(scores.float(), dim=-1).to(Q.dtype)
        return torch.matmul(probabilities, V)

    def to_config(self) -> dict:
        return {
            **super().to_config(),
            "op": self.op,
            "B": self.B,
            "H": self.H,
            "S": self.S,
            "D": self.D,
            "scale": self.scale,
            "causal": False,
            "softmax_accumulation_dtype": "float32",
            "layout": self.layout,
            "flop_count_convention": "4*B*H*S*S*D + 6*B*H*S*S",
        }
