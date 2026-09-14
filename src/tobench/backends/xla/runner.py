"""Compile PyTorch workloads with OpenXLA and execute through PJRT."""

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType
from typing import Any

import torch
from torch import Tensor

try:
    import torch_xla
except ModuleNotFoundError as error:
    if error.name != "torch_xla" and not (error.name or "").startswith("torch_xla."):
        raise
    torch_xla = None
    _IMPORT_ERROR = error
else:
    _IMPORT_ERROR = None

from tobench.backends.base_runner import BaseRunner
from tobench.benchmarking import Benchmarker
from tobench.core.budget import OptimizationBudget

from .adapter import _host_signature
from .prepared import XLAExecutable, XLAPreparedInput


def _require_xla() -> ModuleType:
    if torch_xla is None:
        raise ImportError(
            "The XLA backend is optional. Install matching torch and torch-xla "
            "versions in a separate environment; for example: "
            "pip install torch==2.9.0 torch-xla==2.9.0"
        ) from _IMPORT_ERROR
    return torch_xla


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


class XLARunner(BaseRunner[XLAPreparedInput, XLAExecutable]):
    def __init__(self, benchmarker: Benchmarker | None = None) -> None:
        super().__init__(benchmarker)

    def _build(
        self, prepared: XLAPreparedInput, budget: OptimizationBudget | None
    ) -> XLAExecutable:
        if not isinstance(prepared, XLAPreparedInput):
            raise TypeError("expected XLAPreparedInput from XLAAdapter.prepare")
        _require_xla()
        module = torch.compile(
            prepared.workload, backend="openxla", fullgraph=True, dynamic=False
        )
        executable = XLAExecutable(
            module=module,
            inputs=prepared.inputs,
            host_input_signature=prepared.host_input_signature,
            device=prepared.device,
            device_type=prepared.device_type,
        )
        self._run(executable, None)
        self.synchronize(executable)
        return executable

    @torch.inference_mode()
    def _run(
        self, executable: XLAExecutable, inputs: Sequence[Tensor] | None
    ) -> Tensor:
        if not isinstance(executable, XLAExecutable):
            raise TypeError("expected XLAExecutable from XLARunner.build")
        if inputs is None:
            values = executable.inputs
        else:
            values = tuple(inputs)
            if _host_signature(values) != executable.host_input_signature:
                raise ValueError(
                    "inputs must match the CPU shapes, strides, dtype, and device used in build"
                )
            values = tuple(value.to(executable.device) for value in values)
        output = executable.module(*values)
        if not isinstance(output, Tensor):
            raise TypeError("XLA workloads must return exactly one tensor")
        return output

    def synchronize(self, state: XLAPreparedInput | XLAExecutable) -> None:
        _require_xla().sync(wait=True)

    def collect_metadata(self, prepared: XLAPreparedInput) -> dict[str, Any]:
        if not isinstance(prepared, XLAPreparedInput):
            raise TypeError("expected XLAPreparedInput from XLAAdapter.prepare")
        xla = _require_xla()
        return {
            "backend": "xla",
            "backend_version": _package_version("torch-xla"),
            "budget_enforced": False,
            "progress_reporting": False,
            "configuration": {
                "frontend": "torch.compile openxla",
                "runtime": "PJRT",
                "device_type": prepared.device_type,
                "device": str(prepared.device),
                "fullgraph": True,
                "dynamic_shapes": False,
                "eval_mode": True,
                "inference_mode": True,
                "inputs_resident_during_measurement": True,
                "runtime_input_copy": "CPU to XLA when replacement inputs are supplied",
                "synchronization": "torch_xla.sync(wait=True)",
                "persistent_cache": prepared.cache_policy,
                "torch_xla_version": str(getattr(xla, "__version__", "unknown")),
            },
        }
