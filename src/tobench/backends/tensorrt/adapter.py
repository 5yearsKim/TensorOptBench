"""Export complete PyTorch workload graphs for Torch-TensorRT."""

from collections.abc import Sequence

import torch
from torch import Tensor

from tobench.backends.base_adapter import BaseAdapter
from tobench.workloads import BaseWorkload
from .prepared import TensorRTPreparedInput


class TensorRTAdapter(BaseAdapter[TensorRTPreparedInput]):
    @torch.no_grad()
    def prepare(
        self, workload: BaseWorkload, inputs: Sequence[Tensor]
    ) -> TensorRTPreparedInput:
        if not isinstance(workload, BaseWorkload):
            raise TypeError("workload must be a BaseWorkload module")
        if not isinstance(inputs, (tuple, list)) or not inputs:
            raise TypeError("inputs must be a non-empty tuple or list of tensors")
        if not all(isinstance(value, Tensor) for value in inputs):
            raise TypeError("inputs must contain only tensors")
        device = inputs[0].device
        if device.type != "cuda":
            raise ValueError("TensorRTAdapter requires CUDA inputs")
        if any(value.device != device for value in inputs):
            raise ValueError("all inputs must be on the same CUDA device")
        if any(not value.is_contiguous() for value in inputs):
            raise ValueError("TensorRTAdapter requires contiguous inputs")

        exported_program = torch.export.export(workload.eval(), tuple(inputs))
        return TensorRTPreparedInput(exported_program, tuple(inputs))
