"""TVM-specific prepared IR and executable handles."""

from dataclasses import dataclass
from typing import Any

from torch import Tensor


@dataclass(frozen=True)
class TVMPreparedInput:
    mod: Any
    inputs: tuple[Tensor, ...]
    target: Any
    device: Any


@dataclass(frozen=True)
class TVMExecutable:
    vm: Any
    inputs: tuple[Tensor, ...]
    device: Any
    input_signature: tuple
