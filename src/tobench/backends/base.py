"""Shared backend contract for workloads expressed as PyTorch modules."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from tobench.core.budget import OptimizationBudget

from tobench.core import Workload


class BackendAdapter(ABC):
    """Prepare, build, and execute a workload through a common interface.

    The runner owns timing, measurement, and result storage. Concrete adapters
    own backend-specific state, input bindings, and executable representations.
    """

    @abstractmethod
    def prepare(self, workload: Workload, inputs: Any) -> None:
        """Bind shared inputs and set up state without compilation or tuning."""
        ...

    @abstractmethod
    def build(
        self,
        budget: OptimizationBudget | None,
        report_progress: Callable[[float, float], None] | None = None,
    ) -> Any:
        """Optimize and compile, returning an executable ready for measurement.

        Complete lazy compilation and autotuning here, including any necessary
        first invocation. A baseline may simply return a callable. The budget
        applies to the entire build, subject to backend enforcement support.

        When supported, report progress as (elapsed optimization seconds, best
        observed latency milliseconds), using the start of build as the time
        origin. Backends without progress reporting may ignore the callback.
        """
        ...

    @abstractmethod
    def run(self, executable: Any, inputs: Any) -> Any:
        """Execute on supplied inputs and return outputs without further tuning
        or compilation.
        """
        ...

    @abstractmethod
    def collect_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable backend version and configuration metadata."""
        ...
