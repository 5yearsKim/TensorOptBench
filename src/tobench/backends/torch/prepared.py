"""PyTorch runtime state; no serialized configuration objects."""

from dataclasses import dataclass

from torch import Tensor, nn

from tobench.workloads import BaseWorkload


@dataclass(frozen=True)
class TorchPreparedInput:
    workload: BaseWorkload
    inputs: tuple[Tensor, ...]


@dataclass(frozen=True)
class TorchExecutable:
    module: nn.Module
    inputs: tuple[Tensor, ...]
