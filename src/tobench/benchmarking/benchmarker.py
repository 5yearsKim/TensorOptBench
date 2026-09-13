"""Timing and statistics, independent of backend execution code."""

import math
from collections.abc import Callable
from statistics import median
from time import perf_counter
from typing import TypeVar

from tobench.core.budget import OptimizationBudget

T = TypeVar("T")


class Benchmarker:
    """Record synchronized wall-clock build time and individual executions.

    One instance belongs to one runner and is reset for each build. Cache and
    process isolation are caller responsibilities. Memory is not measured yet.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.optimization_time_seconds: float | None = None
        self.budget_overrun_seconds: float | None = None
        self.latency_samples_ms: list[float] = []
        self.phase = "build"

    def measure_build(
        self, operation: Callable[[], T], synchronize: Callable[[], None],
        budget: OptimizationBudget | None,
    ) -> T:
        self.reset()
        synchronize()
        started = perf_counter()
        try:
            output = operation()
            synchronize()
            return output
        finally:
            self.optimization_time_seconds = perf_counter() - started
            if budget is not None:
                self.budget_overrun_seconds = max(
                    0.0, self.optimization_time_seconds - budget.max_time_seconds
                )

    def measure_run(self, operation: Callable[[], T], synchronize: Callable[[], None]) -> T:
        self.phase = "runtime"
        synchronize()
        started = perf_counter()
        output = operation()
        synchronize()
        elapsed_ms = (perf_counter() - started) * 1000
        if elapsed_ms <= 0:
            raise RuntimeError("timer returned a non-positive latency")
        self.latency_samples_ms.append(elapsed_ms)
        return output

    def summary(self) -> dict[str, float | None]:
        samples = self.latency_samples_ms
        return {
            "median_latency_ms": median(samples) if samples else None,
            "p95_latency_ms": sorted(samples)[math.ceil(0.95 * len(samples)) - 1] if samples else None,
        }
