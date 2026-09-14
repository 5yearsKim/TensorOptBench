"""IREE-specific exported programs and runtime handles."""

from dataclasses import dataclass
from typing import Any

from torch import Tensor


@dataclass(frozen=True)
class IREEPreparedInput:
    export_output: Any
    inputs: tuple[Tensor, ...]
    target_device: str
    target_backend: str
    target_arch: str | None
    runtime_device_uri: str


@dataclass(frozen=True)
class IREEExecutable:
    module: Any
    function: Any
    device: Any
    inputs: tuple[Tensor, ...]
    input_signature: tuple
