"""Export complete PyTorch workload graphs through IREE Turbine AOT."""

from collections.abc import Sequence
from types import ModuleType

import torch
from torch import Tensor

try:
    import iree.turbine.aot as aot
except ModuleNotFoundError as error:
    if error.name != "iree" and not (error.name or "").startswith("iree."):
        raise
    aot = None
    _IMPORT_ERROR = error
else:
    _IMPORT_ERROR = None

from tobench.backends.base_adapter import BaseAdapter
from tobench.workloads import BaseWorkload

from .prepared import IREEPreparedInput


def _require_turbine() -> ModuleType:
    if aot is None:
        raise ImportError(
            'The IREE backend is optional. Install it with: pip install -e ".[iree]"'
        ) from _IMPORT_ERROR
    return aot


class IREEAdapter(BaseAdapter[IREEPreparedInput]):
    @torch.no_grad()
    def prepare(
        self, workload: BaseWorkload, inputs: Sequence[Tensor]
    ) -> IREEPreparedInput:
        if not isinstance(workload, BaseWorkload):
            raise TypeError("workload must be a BaseWorkload module")
        if not isinstance(inputs, (tuple, list)) or not inputs:
            raise TypeError("inputs must be a non-empty tuple or list of tensors")
        if not all(isinstance(value, Tensor) for value in inputs):
            raise TypeError("inputs must contain only tensors")
        device = inputs[0].device
        if device.type not in ("cpu", "cuda"):
            raise ValueError("IREEAdapter supports CPU and CUDA inputs only")
        if any(value.device != device for value in inputs):
            raise ValueError("all inputs must be on the same device")
        if any(not value.is_contiguous() for value in inputs):
            raise ValueError("IREEAdapter requires contiguous inputs")

        if device.type == "cuda":
            major, minor = torch.cuda.get_device_capability(device)
            target_device = target_backend = "cuda"
            target_arch = f"sm_{major}{minor}"
            runtime_device_uri = f"cuda://{device.index or 0}"
        else:
            target_device = "local"
            target_backend = "llvm-cpu"
            target_arch = None
            runtime_device_uri = "local-task"

        turbine = _require_turbine()
        export_output = turbine.export(workload.eval(), args=tuple(inputs))
        return IREEPreparedInput(
            export_output=export_output,
            inputs=tuple(inputs),
            target_device=target_device,
            target_backend=target_backend,
            target_arch=target_arch,
            runtime_device_uri=runtime_device_uri,
        )
