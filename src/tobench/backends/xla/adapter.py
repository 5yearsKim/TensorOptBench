"""Move complete PyTorch workloads and inputs onto a PJRT XLA device."""

from collections.abc import Sequence
import copy
import os

import torch
from torch import Tensor

try:
    import torch_xla
    import torch_xla.runtime as xr
except ModuleNotFoundError as error:
    if error.name != "torch_xla" and not (error.name or "").startswith("torch_xla."):
        raise
    torch_xla = xr = None
    _IMPORT_ERROR = error
else:
    _IMPORT_ERROR = None

from tobench.backends.base_adapter import BaseAdapter
from tobench.workloads import BaseWorkload
from .prepared import XLAPreparedInput


def _host_signature(inputs: Sequence[Tensor]) -> tuple:
    return tuple(
        (tuple(value.shape), value.stride(), value.dtype, value.device)
        for value in inputs
    )


def _require_xla():
    if torch_xla is None or xr is None:
        raise ImportError(
            "The XLA backend is optional. Install matching torch and torch-xla "
            "versions in a separate environment; for example: "
            "pip install torch==2.9.0 torch-xla==2.9.0"
        ) from _IMPORT_ERROR
    return torch_xla, xr


class XLAAdapter(BaseAdapter[XLAPreparedInput]):
    def __init__(self, device_type: str = "CPU") -> None:
        if not isinstance(device_type, str) or device_type.upper() not in ("CPU", "TPU"):
            raise ValueError("device_type must be CPU or TPU")
        self.device_type = device_type.upper()

    @torch.no_grad()
    def prepare(
        self, workload: BaseWorkload, inputs: Sequence[Tensor]
    ) -> XLAPreparedInput:
        if not isinstance(workload, BaseWorkload):
            raise TypeError("workload must be a BaseWorkload module")
        if not isinstance(inputs, (tuple, list)) or not inputs:
            raise TypeError("inputs must be a non-empty tuple or list of tensors")
        if not all(isinstance(value, Tensor) for value in inputs):
            raise TypeError("inputs must contain only tensors")
        if any(value.device.type != "cpu" for value in inputs):
            raise ValueError("XLAAdapter requires CPU source inputs")
        if any(not value.is_contiguous() for value in inputs):
            raise ValueError("XLAAdapter requires contiguous inputs")

        xla, runtime = _require_xla()
        runtime.set_device_type(self.device_type)
        cache_path = os.environ.get("TOBENCH_XLA_CACHE_PATH")
        if cache_path is not None:
            runtime.initialize_cache(cache_path, readonly=False)
        device = xla.device()
        xla_workload = copy.deepcopy(workload).eval().to(device)
        xla_inputs = tuple(value.to(device) for value in inputs)
        xla.sync(wait=True)
        return XLAPreparedInput(
            workload=xla_workload,
            inputs=xla_inputs,
            host_input_signature=_host_signature(inputs),
            device=device,
            device_type=self.device_type,
            cache_policy="fresh_temporary" if cache_path is not None else "uncontrolled",
        )
