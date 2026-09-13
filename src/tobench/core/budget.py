"""Optimization budget shared by runners and adapters."""

from pydantic import BaseModel, ConfigDict, Field


class OptimizationBudget(BaseModel):
    """Immutable, strictly validated wall-clock budget in seconds."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", validate_default=True)

    max_time_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
