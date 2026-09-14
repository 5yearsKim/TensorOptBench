"""Torch-TensorRT prepared graphs and executable handles."""

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn


@dataclass(frozen=True)
class TensorRTPreparedInput:
    exported_program: Any
    inputs: tuple[Tensor, ...]


@dataclass(frozen=True)
class TensorRTExecutable:
    module: nn.Module
    inputs: tuple[Tensor, ...]
    input_signature: tuple
