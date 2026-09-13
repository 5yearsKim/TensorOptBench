"""Prepare a PyTorch workload for eager or Inductor execution."""

from collections.abc import Sequence
from torch import Tensor
from tobench.backends.base_adapter import BaseAdapter
from tobench.workloads import BaseWorkload
from .prepared import TorchPreparedInput


class TorchAdapter(BaseAdapter[TorchPreparedInput]):
    def prepare(self, workload: BaseWorkload, inputs: Sequence[Tensor]) -> TorchPreparedInput:
        if not isinstance(workload, BaseWorkload):
            raise TypeError("workload must be a BaseWorkload module")
        if not isinstance(inputs, (tuple, list)) or not inputs:
            raise TypeError("inputs must be a non-empty tuple or list of tensors")
        if not all(isinstance(value, Tensor) for value in inputs):
            raise TypeError("inputs must contain only tensors")
        return TorchPreparedInput(workload.eval(), tuple(inputs))
