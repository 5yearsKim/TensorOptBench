"""Batched convolution, inference BatchNorm, and ReLU workload."""

import math
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from .base_workload import BaseWorkload


class ConvBNReLU(BaseWorkload):
    """Compute Conv2D followed by inference BatchNorm and ReLU in NCHW layout."""

    op = "conv_bn_relu"
    layout = "NCHW"
    weight_layout = "OIHW"

    def __init__(
        self,
        B: int = 8,
        C_in: int = 64,
        C_out: int = 64,
        H: int = 56,
        W: int = 56,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        dtype: Literal["float16", "bfloat16"] = "float16",
        seed: int = 0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(dtype=dtype, seed=seed)
        dimensions = (
            ("B", B),
            ("C_in", C_in),
            ("C_out", C_out),
            ("H", H),
            ("W", W),
            ("kernel_size", kernel_size),
            ("stride", stride),
            ("dilation", dilation),
            ("groups", groups),
        )
        for name, value in dimensions:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(padding, bool) or not isinstance(padding, int):
            raise TypeError("padding must be an integer")
        if padding < 0:
            raise ValueError("padding must be non-negative")
        if C_in % groups or C_out % groups:
            raise ValueError("C_in and C_out must be divisible by groups")
        if isinstance(eps, bool) or not isinstance(eps, (int, float)):
            raise TypeError("eps must be a number")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")

        effective_kernel = dilation * (kernel_size - 1) + 1
        output_height = (H + 2 * padding - effective_kernel) // stride + 1
        output_width = (W + 2 * padding - effective_kernel) // stride + 1
        if output_height <= 0 or output_width <= 0:
            raise ValueError("kernel configuration produces an empty output")

        self.B, self.C_in, self.C_out = B, C_in, C_out
        self.H, self.W = H, W
        self.kernel_size, self.stride = kernel_size, stride
        self.padding, self.dilation, self.groups = padding, dilation, groups
        self.eps = float(eps)
        self.output_height, self.output_width = output_height, output_width

    @property
    def input_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Shapes of X, convolution weight, gamma, beta, mean, and variance."""
        channel_shape = (self.C_out,)
        return (
            (self.B, self.C_in, self.H, self.W),
            (self.C_out, self.C_in // self.groups, self.kernel_size, self.kernel_size),
            channel_shape,
            channel_shape,
            channel_shape,
            channel_shape,
        )

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        return (self.B, self.C_out, self.output_height, self.output_width)

    @property
    def flop_count(self) -> int:
        output_elements = math.prod(self.output_shape)
        convolution = (
            2
            * output_elements
            * (self.C_in // self.groups)
            * self.kernel_size
            * self.kernel_size
        )
        batch_norm_relu = 5 * output_elements + 2 * self.C_out
        return convolution + batch_norm_relu

    def forward(
        self,
        X: Tensor,
        weight: Tensor,
        gamma: Tensor,
        beta: Tensor,
        running_mean: Tensor,
        running_variance: Tensor,
    ) -> Tensor:
        convolved = F.conv2d(
            X,
            weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        shape = (1, self.C_out, 1, 1)
        mean = running_mean.float().reshape(shape)
        inverse_std = torch.rsqrt(running_variance.float().reshape(shape) + self.eps)
        normalized = (convolved.float() - mean) * inverse_std
        normalized = normalized * gamma.float().reshape(shape) + beta.float().reshape(
            shape
        )
        return torch.relu(normalized.to(X.dtype))

    def to_config(self) -> dict:
        return {
            **super().to_config(),
            "op": self.op,
            "B": self.B,
            "C_in": self.C_in,
            "C_out": self.C_out,
            "H": self.H,
            "W": self.W,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "groups": self.groups,
            "eps": self.eps,
            "layout": self.layout,
            "weight_layout": self.weight_layout,
            "batch_norm_training": False,
            "batch_norm_dtype": "float32",
            "convolution_bias": False,
            "flop_count_convention": "conv multiply-adds + 5 ops/output + 2 ops/channel",
        }
