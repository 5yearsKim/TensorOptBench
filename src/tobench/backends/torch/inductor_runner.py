"""TorchInductor compilation and execution."""

from typing import Any

import torch

from tobench.benchmarking import Benchmarker
from tobench.core.budget import OptimizationBudget

from .eager_runner import TorchEagerRunner
from .prepared import TorchExecutable, TorchPreparedInput


class TorchInductorRunner(TorchEagerRunner):
    def __init__(
        self, benchmarker: Benchmarker | None = None, *, mode: str = "default"
    ) -> None:
        super().__init__(benchmarker)
        self.mode = mode

    def _build(
        self, prepared: TorchPreparedInput, budget: OptimizationBudget | None
    ) -> TorchExecutable:
        if not isinstance(prepared, TorchPreparedInput):
            raise TypeError("expected TorchPreparedInput from TorchAdapter.prepare")
        module = torch.compile(
            prepared.workload,
            backend="inductor",
            mode=self.mode,
            fullgraph=True,
            dynamic=False,
        )
        executable = TorchExecutable(module, prepared.inputs)
        self._run(executable, None)
        self.synchronize(executable)
        return executable

    def collect_metadata(self, prepared: TorchPreparedInput) -> dict[str, Any]:
        metadata = super().collect_metadata(prepared)
        metadata["backend"] = "torchinductor"
        metadata["configuration"].update(
            {
                "compiler_backend": "inductor",
                "mode": self.mode,
                "fullgraph": True,
                "dynamic": False,
            }
        )
        return metadata
