"""Validated experiment configuration, independent of runtime objects."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .budget import OptimizationBudget


class _ConfigModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", validate_default=True)


class GEMMConfig(_ConfigModel):
    name: Literal["gemm"] = "gemm"
    M: int = Field(default=128, gt=0)
    N: int = Field(default=128, gt=0)
    K: int = Field(default=128, gt=0)
    dtype: Literal["float16", "bfloat16"] = "float16"
    seed: int = Field(default=0, ge=0)


class RuntimeConfig(_ConfigModel):
    warmup: int = Field(default=20, ge=0)
    repetitions: int = Field(default=100, gt=0)


class RMSNormLinearConfig(_ConfigModel):
    name: Literal["rmsnorm_linear"] = "rmsnorm_linear"
    M: int = Field(default=16, gt=0)
    N: int = Field(default=406, gt=0)
    K: int = Field(default=4096, gt=0)
    dtype: Literal["float16", "bfloat16"] = "float16"
    seed: int = Field(default=0, ge=0)
    eps: float = Field(default=1e-6, gt=0, allow_inf_nan=False)


class BenchmarkConfig(_ConfigModel):
    """One backend/workload experiment; paths are relative to the working directory."""

    workload: GEMMConfig | RMSNormLinearConfig = Field(
        default_factory=GEMMConfig, discriminator="name"
    )
    backend: Literal["eager", "inductor", "tvm"] = "eager"
    device: Literal["cpu", "cuda"] = "cpu"
    budget: OptimizationBudget = Field(default_factory=OptimizationBudget)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    output: str | None = Field(default=None, min_length=1)

    @field_validator("workload", mode="before")
    @classmethod
    def default_workload_name(cls, value):
        # Preserve existing GEMM JSON files which omit the operator name.
        if isinstance(value, dict) and "name" not in value:
            return {"name": "gemm", **value}
        return value

    @classmethod
    def from_json(cls, path: str | Path) -> "BenchmarkConfig":
        """Read a JSON file, rejecting unknown fields and invalid values."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
