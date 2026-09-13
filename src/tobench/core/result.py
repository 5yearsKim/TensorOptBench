"""Benchmark summaries and raw measurements stored as JSON."""

import json
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


NonNegativeFloat = Annotated[float, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class BenchmarkResult(BaseModel):
    """Validate metric types and ranges during creation and field assignment."""

    model_config = ConfigDict(
        strict=True, extra="forbid", validate_assignment=True, allow_inf_nan=False
    )

    backend: str
    workload: dict[str, JsonValue]
    configuration: dict[str, JsonValue]
    environment: dict[str, JsonValue]
    timestamp: str
    status: Literal["success", "error", "correctness_failed"] = "error"
    backend_version: str | None = None
    backend_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    preparation_time_seconds: NonNegativeFloat | None = None
    optimization_time_seconds: NonNegativeFloat | None = None
    budget_enforced: bool = False
    budget_overrun_seconds: NonNegativeFloat | None = None
    median_latency_ms: PositiveFloat | None = None
    p95_latency_ms: PositiveFloat | None = None
    latency_samples_ms: list[PositiveFloat] = Field(default_factory=list)
    throughput: NonNegativeFloat | None = None
    throughput_unit: str | None = None
    peak_device_memory_bytes: NonNegativeInt | None = None
    peak_host_memory_bytes: NonNegativeInt | None = None
    error: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        # In-place list/dict changes bypass Pydantic's assignment validation.
        # Revalidate at the export boundary before persisting measurements.
        validated = type(self).model_validate(self.model_dump())
        return validated.model_dump()

    def save_json(self, path: str | Path) -> None:
        """Atomically write a result, creating its parent directory if needed."""
        payload = json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as file:
                temporary_path = Path(file.name)
                file.write(payload)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
