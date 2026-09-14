"""PyTorch/XLA prepared graph and executable state."""

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from tobench.workloads import BaseWorkload


@dataclass(frozen=True)
class XLAPreparedInput:
    workload: BaseWorkload
    inputs: tuple[Tensor, ...]
    host_input_signature: tuple
    device: Any
    device_type: str
    cache_policy: str


@dataclass(frozen=True)
class XLAExecutable:
    module: nn.Module
    inputs: tuple[Tensor, ...]
    host_input_signature: tuple
    device: Any
    device_type: str
