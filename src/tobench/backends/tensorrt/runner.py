"""Compile exported PyTorch graphs into TensorRT engines and execute them."""

import os
from collections.abc import Sequence
from types import ModuleType
from typing import Any

import torch
from torch import Tensor

try:
    import torch_tensorrt
except ModuleNotFoundError as error:
    if error.name != "torch_tensorrt" and not (error.name or "").startswith(
        ("torch_tensorrt.", "tensorrt")
    ):
        raise
    torch_tensorrt = None
    _IMPORT_ERROR = error
else:
    _IMPORT_ERROR = None

from tobench.backends.base_runner import BaseRunner
from tobench.benchmarking import Benchmarker
from tobench.core.budget import OptimizationBudget

from .prepared import TensorRTExecutable, TensorRTPreparedInput


def _signature(inputs: Sequence[Tensor]) -> tuple:
    return tuple(
        (tuple(value.shape), value.stride(), value.dtype, value.device)
        for value in inputs
    )


def _require_torch_tensorrt() -> ModuleType:
    if torch_tensorrt is None:
        raise ImportError(
            "The TensorRT backend is optional. Install it in a CUDA environment "
            "with compatible versions of: torch-tensorrt and tensorrt"
        ) from _IMPORT_ERROR
    return torch_tensorrt


class TensorRTRunner(BaseRunner[TensorRTPreparedInput, TensorRTExecutable]):
    def __init__(
        self,
        benchmarker: Benchmarker | None = None,
        *,
        min_block_size: int = 1,
        optimization_level: int | None = None,
    ) -> None:
        super().__init__(benchmarker)
        if isinstance(min_block_size, bool) or not isinstance(min_block_size, int):
            raise TypeError("min_block_size must be an integer")
        if min_block_size < 1:
            raise ValueError("min_block_size must be at least 1")
        if optimization_level is not None and (
            isinstance(optimization_level, bool)
            or not isinstance(optimization_level, int)
            or not 0 <= optimization_level <= 5
        ):
            raise ValueError(
                "optimization_level must be an integer from 0 to 5 or None"
            )
        self.min_block_size = min_block_size
        self.optimization_level = optimization_level

    def _build(
        self, prepared: TensorRTPreparedInput, budget: OptimizationBudget | None
    ) -> TensorRTExecutable:
        if not isinstance(prepared, TensorRTPreparedInput):
            raise TypeError(
                "expected TensorRTPreparedInput from TensorRTAdapter.prepare"
            )
        trt = _require_torch_tensorrt()
        options: dict[str, Any] = {
            "arg_inputs": prepared.inputs,
            "require_full_compilation": True,
            "pass_through_build_failures": True,
            "min_block_size": self.min_block_size,
            "cache_built_engines": False,
            "reuse_cached_engines": False,
        }
        if self.optimization_level is not None:
            options["optimization_level"] = self.optimization_level
        timing_cache_path = os.environ.get("TOBENCH_TENSORRT_TIMING_CACHE_PATH")
        if timing_cache_path is not None:
            options["timing_cache_path"] = timing_cache_path

        module = trt.dynamo.compile(prepared.exported_program, **options)
        executable = TensorRTExecutable(
            module=module,
            inputs=prepared.inputs,
            input_signature=_signature(prepared.inputs),
        )
        self._run(executable, None)
        self.synchronize(executable)
        return executable

    @torch.inference_mode()
    def _run(
        self, executable: TensorRTExecutable, inputs: Sequence[Tensor] | None
    ) -> Tensor:
        values = executable.inputs if inputs is None else tuple(inputs)
        if _signature(values) != executable.input_signature:
            raise ValueError(
                "inputs must match the shapes, strides, dtype, and device used in build"
            )
        return executable.module(*values)

    def synchronize(self, state: TensorRTPreparedInput | TensorRTExecutable) -> None:
        for device in {value.device for value in state.inputs if value.is_cuda}:
            torch.cuda.synchronize(device)

    def collect_metadata(self, prepared: TensorRTPreparedInput) -> dict[str, Any]:
        version = (
            None
            if torch_tensorrt is None
            else str(getattr(torch_tensorrt, "__version__", "unknown"))
        )
        return {
            "backend": "tensorrt",
            "backend_version": version,
            "budget_enforced": False,
            "progress_reporting": False,
            "configuration": {
                "frontend": "torch.export -> torch_tensorrt.dynamo",
                "require_full_compilation": True,
                "pass_through_build_failures": True,
                "min_block_size": self.min_block_size,
                "optimization_level": self.optimization_level,
                "dynamic_shapes": False,
                "eval_mode": True,
                "inference_mode": True,
                "engine_cache_enabled": False,
                "timing_cache": (
                    "fresh_temporary"
                    if os.environ.get("TOBENCH_TENSORRT_TIMING_CACHE_PATH")
                    else "uncontrolled"
                ),
            },
        }
