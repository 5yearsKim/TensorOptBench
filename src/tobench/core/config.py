"""Validated experiment configuration, independent of runtime objects."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .budget import OptimizationBudget


class _ConfigModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", validate_default=True)


class GEMMConfig(_ConfigModel):
    name: Literal["gemm"] = "gemm"
    B: int = Field(default=8, gt=0)
    M: int = Field(default=128, gt=0)
    N: int = Field(default=128, gt=0)
    K: int = Field(default=128, gt=0)
    dtype: Literal["float16", "bfloat16"] = "float16"
    seed: int = Field(default=0, ge=0)


class RuntimeConfig(_ConfigModel):
    warmup: int = Field(default=20, ge=0)
    repetitions: int = Field(default=100, gt=0)


class CorrectnessConfig(_ConfigModel):
    enabled: bool = True
    reference: Literal["torch_eager"] = "torch_eager"
    rtol: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    atol: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    def resolved(self, dtype: str) -> "CorrectnessConfig":
        """Preserve existing FP16/BF16 defaults when thresholds are omitted."""
        default = 1e-2 if dtype == "bfloat16" else 1e-3
        return self.model_copy(update={
            "rtol": default if self.rtol is None else self.rtol,
            "atol": default if self.atol is None else self.atol,
        })


class RMSNormLinearConfig(_ConfigModel):
    name: Literal["rmsnorm_linear"] = "rmsnorm_linear"
    B: int = Field(default=4, gt=0)
    M: int = Field(default=16, gt=0)
    N: int = Field(default=406, gt=0)
    K: int = Field(default=4096, gt=0)
    dtype: Literal["float16", "bfloat16"] = "float16"
    seed: int = Field(default=0, ge=0)
    eps: float = Field(default=1e-6, gt=0, allow_inf_nan=False)


class SoftmaxConfig(_ConfigModel):
    name: Literal["softmax"] = "softmax"
    B: int = Field(default=8, gt=0)
    M: int = Field(default=128, gt=0)
    N: int = Field(default=1024, gt=0)
    dtype: Literal["float16", "bfloat16"] = "float16"
    seed: int = Field(default=0, ge=0)


class AttentionConfig(_ConfigModel):
    name: Literal["attention"] = "attention"
    B: int = Field(default=2, gt=0)
    H: int = Field(default=8, gt=0)
    S: int = Field(default=128, gt=0)
    D: int = Field(default=64, gt=0)
    dtype: Literal["float16", "bfloat16"] = "float16"
    seed: int = Field(default=0, ge=0)


class ConvBNReLUConfig(_ConfigModel):
    name: Literal["conv_bn_relu"] = "conv_bn_relu"
    B: int = Field(default=8, gt=0)
    C_in: int = Field(default=64, gt=0)
    C_out: int = Field(default=64, gt=0)
    H: int = Field(default=56, gt=0)
    W: int = Field(default=56, gt=0)
    kernel_size: int = Field(default=3, gt=0)
    stride: int = Field(default=1, gt=0)
    padding: int = Field(default=1, ge=0)
    dilation: int = Field(default=1, gt=0)
    groups: int = Field(default=1, gt=0)
    dtype: Literal["float16", "bfloat16"] = "float16"
    seed: int = Field(default=0, ge=0)
    eps: float = Field(default=1e-5, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_convolution(self) -> Self:
        if self.C_in % self.groups or self.C_out % self.groups:
            raise ValueError("C_in and C_out must be divisible by groups")
        effective_kernel = self.dilation * (self.kernel_size - 1) + 1
        if self.H + 2 * self.padding < effective_kernel:
            raise ValueError("kernel configuration produces an empty output height")
        if self.W + 2 * self.padding < effective_kernel:
            raise ValueError("kernel configuration produces an empty output width")
        return self


class MetaScheduleConfig(_ConfigModel):
    max_trials_global: int = Field(default=256, gt=0)
    max_trials_per_task: int | None = Field(default=None, gt=0)
    num_trials_per_iter: int = Field(default=64, gt=0)
    seed: int = Field(default=0, ge=0, le=2147483647)
    cost_model: Literal["xgb", "random"] = "xgb"
    work_dir: str | None = Field(default=None, min_length=1)
    target: str | None = Field(default=None, min_length=1)


class IREEConfig(_ConfigModel):
    optimization_level: Literal["O0", "O1", "O2", "O3"] = "O3"


class BenchmarkConfig(_ConfigModel):
    """One backend/workload experiment; paths are relative to the working directory."""

    workload: (
        GEMMConfig | RMSNormLinearConfig | SoftmaxConfig | AttentionConfig
        | ConvBNReLUConfig
    ) = Field(
        default_factory=GEMMConfig, discriminator="name"
    )
    backend: Literal[
        "eager", "inductor", "iree", "tensorrt", "tvm", "tvm_metaschedule"
    ] = "eager"
    iree: IREEConfig = Field(default_factory=IREEConfig)
    metaschedule: MetaScheduleConfig = Field(default_factory=MetaScheduleConfig)
    device: Literal["cpu", "cuda"] = "cpu"
    budget: OptimizationBudget = Field(default_factory=OptimizationBudget)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    correctness: CorrectnessConfig = Field(default_factory=CorrectnessConfig)
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
