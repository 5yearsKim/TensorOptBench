"""Backend execution contract with optional measurement instrumentation."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from torch import Tensor

from tobench.benchmarking import Benchmarker
from tobench.core.budget import OptimizationBudget
from tobench.core.config import RuntimeConfig

Prepared = TypeVar("Prepared")
Executable = TypeVar("Executable")


class BaseRunner(ABC, Generic[Prepared, Executable]):
    def __init__(self, benchmarker: Benchmarker | None = None) -> None:
        self.benchmarker = benchmarker

    def build(
        self, prepared: Prepared, budget: OptimizationBudget | None = None
    ) -> Executable:
        """Build once, optionally recording elapsed time and budget overrun."""
        if self.benchmarker is None:
            return self._build(prepared, budget)
        return self.benchmarker.measure_build(
            lambda: self._build(prepared, budget),
            lambda: self.synchronize(prepared),
            budget,
        )

    def run(
        self,
        executable: Executable,
        inputs: Sequence[Tensor] | None = None,
        *,
        measure: bool = True,
    ) -> Tensor:
        """Execute exactly once. Disable measurement for correctness checks."""
        if self.benchmarker is None or not measure:
            return self._run(executable, inputs)
        return self.benchmarker.measure_run(
            lambda: self._run(executable, inputs), lambda: self.synchronize(executable)
        )

    def benchmark_run(
        self,
        executable: Executable,
        *,
        warmup: int = 20,
        repetitions: int = 100,
    ) -> dict[str, float | None]:
        """Replace latency samples with an explicit warmup/repetition experiment."""
        if self.benchmarker is None:
            raise RuntimeError("benchmark_run requires an injected Benchmarker")
        RuntimeConfig(warmup=warmup, repetitions=repetitions)
        self.benchmarker.latency_samples_ms.clear()
        self.benchmarker.phase = "warmup"
        for _ in range(warmup):
            self.run(executable, measure=False)
        self.synchronize(executable)
        for _ in range(repetitions):
            self.run(executable)
        return self.benchmarker.summary()

    @abstractmethod
    def _build(
        self, prepared: Prepared, budget: OptimizationBudget | None
    ) -> Executable:
        """Compile and materialize an executable without recording run samples."""
        ...

    @abstractmethod
    def _run(
        self, executable: Executable, inputs: Sequence[Tensor] | None
    ) -> Tensor: ...

    @abstractmethod
    def synchronize(self, state: Prepared | Executable) -> None:
        """Wait for backend work at a measurement boundary."""
        ...

    @abstractmethod
    def collect_metadata(self, prepared: Prepared) -> dict[str, Any]: ...
