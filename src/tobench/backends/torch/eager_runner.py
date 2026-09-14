"""PyTorch eager execution with optional benchmarking."""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from tobench.backends.base_runner import BaseRunner
from tobench.core.budget import OptimizationBudget

from .prepared import TorchExecutable, TorchPreparedInput


class TorchEagerRunner(BaseRunner[TorchPreparedInput, TorchExecutable]):
    def _build(
        self, prepared: TorchPreparedInput, budget: OptimizationBudget | None
    ) -> TorchExecutable:
        if not isinstance(prepared, TorchPreparedInput):
            raise TypeError("expected TorchPreparedInput from TorchAdapter.prepare")
        return TorchExecutable(prepared.workload, prepared.inputs)

    @torch.inference_mode()
    def _run(
        self, executable: TorchExecutable, inputs: Sequence[Tensor] | None
    ) -> Tensor:
        return executable.module(*(executable.inputs if inputs is None else inputs))

    def synchronize(self, state: TorchPreparedInput | TorchExecutable) -> None:
        for device in {x.device for x in state.inputs if x.is_cuda}:
            torch.cuda.synchronize(device)

    def collect_metadata(self, prepared: TorchPreparedInput) -> dict[str, Any]:
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
