"""TorchInductor adapter using torch.compile."""

from collections.abc import Callable
from typing import Any

from tobench.core.budget import OptimizationBudget

import torch
from torch import nn

from .eager import TorchEagerAdapter


class TorchInductorAdapter(TorchEagerAdapter):
    """Compile a full, static graph for the prepared inputs.

    Measurement must use the same tensor shapes, strides, dtype, device, and
    execution settings as build. Changing these can trigger recompilation.
    The runner owns timing; callers must arrange cache isolation. Build cannot enforce a hard
    wall-clock budget, and reports that limitation in metadata.
    """

    def __init__(self, mode: str = "default") -> None:
        super().__init__()
        self.mode = mode

    def build(
        self,
        budget: OptimizationBudget | None,
        report_progress: Callable[[float, float], None] | None = None,
    ) -> nn.Module:
        """Compile and execute once so lazy compilation is included in build.

        Uses the same inference context as run. No tuning progress callback is
        exposed; budget enforcement and cancellation are unsupported.
        """
        workload = self._prepared_workload()
        executable = torch.compile(
            workload, backend="inductor", mode=self.mode, fullgraph=True, dynamic=False
        )
        self.run(executable, self._inputs)
        for device in {value.device for value in self._inputs if value.is_cuda}:
            torch.cuda.synchronize(device)
        return executable

    def collect_metadata(self) -> dict[str, Any]:
        metadata = super().collect_metadata()
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
