"""PyTorch eager baseline adapter."""

from collections.abc import Callable, Sequence
from typing import Any

from tobench.core.budget import OptimizationBudget

import torch
from torch import Tensor, nn

from tobench.backends.base import BackendAdapter
from tobench.core import Workload


class TorchEagerAdapter(BackendAdapter):
    """Execute a workload in evaluation and inference mode.

    Inputs are positional tensors. The caller supplies their shapes, dtype, and
    layout as specified by the workload. Prepare switches the module to eval
    mode; neither the module nor its inputs are moved or cast.
    """

    def __init__(self) -> None:
        self._workload: Workload | None = None
        self._inputs: tuple[Tensor, ...] = ()

    def prepare(self, workload: Workload, inputs: Sequence[Tensor]) -> None:
        if not isinstance(workload, Workload):
            raise TypeError("workload must be a Workload module")
        if not isinstance(inputs, (tuple, list)) or not inputs:
            raise TypeError("inputs must be a non-empty tuple or list of tensors")
        if not all(isinstance(value, Tensor) for value in inputs):
            raise TypeError("inputs must contain only tensors")
        self._workload = workload.eval()
        self._inputs = tuple(inputs)

    def _prepared_workload(self) -> Workload:
        if self._workload is None:
            raise RuntimeError("prepare must be called before build")
        return self._workload

    def build(
        self,
        budget: OptimizationBudget | None,
        report_progress: Callable[[float, float], None] | None = None,
    ) -> nn.Module:
        """Return the eager module; no optimization or progress reporting."""
        return self._prepared_workload()

    @torch.inference_mode()
    def run(self, executable: nn.Module, inputs: Sequence[Tensor]) -> Tensor:
        return executable(*inputs)

    def collect_metadata(self) -> dict[str, Any]:
        return {
            "backend": "torch_eager",
            "backend_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "budget_enforced": False,
            "progress_reporting": False,
            "configuration": {
                "eval_mode": True,
                "inference_mode": True,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "allow_fp16_reduced_precision_reduction": (
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
                ),
                "allow_bf16_reduced_precision_reduction": (
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                ),
            },
        }
