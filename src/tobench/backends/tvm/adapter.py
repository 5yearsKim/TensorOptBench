"""Compile PyTorch workloads through TVM's Relax frontend."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from tobench.core.budget import OptimizationBudget

import torch
from torch import Tensor

from tobench.backends.base import BackendAdapter
from tobench.core import Workload


def _signature(inputs: Sequence[Tensor]) -> tuple:
    return tuple((tuple(x.shape), x.stride(), x.dtype, x.device) for x in inputs)


def _convert_matmul(node: Any, importer: Any) -> Any:
    """Keep PyTorch's output dtype after TVM's FP32 matmul accumulation."""
    from tvm import relax

    lhs, rhs = importer.retrieve_args(node)
    dtype = lhs.ty.dtype
    accumulation_dtype = "float32" if dtype in ("float16", "bfloat16") else dtype
    result = importer.block_builder.emit(
        relax.op.matmul(lhs, rhs, out_dtype=accumulation_dtype)
    )
    if dtype != accumulation_dtype:
        result = importer.block_builder.emit(relax.op.astype(result, dtype))
    return result


@dataclass
class TVMExecutable:
    """VM and input contract retained for execution without recompilation."""

    vm: Any
    device: Any
    input_signature: tuple


class TVMAdapter(BackendAdapter):
    """Export an nn.Module to Relax and compile for LLVM or CUDA.

    TVM is loaded during prepare. This initial adapter uses TVM's default
    compilation pipeline without MetaSchedule search. Hard wall-clock budgets
    and optimization progress reporting are unsupported. The runner owns build
    timing and budget overrun reporting. Callers must arrange cache/process isolation.
    """

    def __init__(self, target: str | None = None) -> None:
        self.target = target
        self._tvm: Any = None
        self._target: Any = None
        self._device: Any = None
        self._workload: Workload | None = None
        self._inputs: tuple[Tensor, ...] = ()

    def prepare(self, workload: Workload, inputs: Sequence[Tensor]) -> None:
        if not isinstance(workload, Workload):
            raise TypeError("workload must be a Workload module")
        if not isinstance(inputs, (tuple, list)) or not inputs:
            raise TypeError("inputs must be a non-empty tuple or list of tensors")
        if not all(isinstance(x, Tensor) for x in inputs):
            raise TypeError("inputs must contain only tensors")
        device = inputs[0].device
        if device.type not in ("cpu", "cuda"):
            raise ValueError("TVMAdapter supports CPU and CUDA inputs only")
        if any(x.device != device for x in inputs):
            raise ValueError("all inputs must be on the same device")
        if any(not x.is_contiguous() for x in inputs):
            raise ValueError("TVMAdapter requires contiguous inputs")

        try:
            tvm = import_module("tvm")
        except ModuleNotFoundError as error:
            if error.name != "tvm":
                raise
            raise ImportError(
                'TVM is optional. Install it with: pip install -e ".[tvm]"'
            ) from error

        target_name = self.target
        if target_name is None:
            if device.type == "cuda":
                major, minor = torch.cuda.get_device_capability(device)
                target_name = f"cuda -arch=sm_{major}{minor}"
            else:
                target_name = "llvm"
        target = tvm.target.Target(target_name)
        expected_kind = "cuda" if device.type == "cuda" else "llvm"
        if target.kind.name != expected_kind:
            raise ValueError(f"{device.type} inputs require a {expected_kind} target")
        if not tvm.runtime.enabled(expected_kind):
            raise RuntimeError(f"Installed TVM does not support {expected_kind}")

        self._tvm = tvm
        self._target = target
        self._device = tvm.device(device.type, device.index or 0)
        self._workload = workload.eval()
        self._inputs = tuple(inputs)

    @torch.no_grad()
    def build(
        self,
        budget: OptimizationBudget | None,
        report_progress: Callable[[float, float], None] | None = None,
    ) -> TVMExecutable:
        """Export, lower, compile, and perform the first invocation in build."""
        if self._workload is None:
            raise RuntimeError("prepare must be called before build")
        from tvm import relax
        from tvm.relax.frontend.torch import from_exported_program

        exported = torch.export.export(self._workload, self._inputs)
        mod = from_exported_program(
            exported,
            unwrap_unit_return_tuple=True,
            custom_convert_map={
                "mm.default": _convert_matmul,
                "matmul.default": _convert_matmul,
            },
        )
        compiled = relax.build(mod, target=self._target)
        executable = TVMExecutable(
            vm=relax.VirtualMachine(compiled, self._device),
            device=self._device,
            input_signature=_signature(self._inputs),
        )
        self.run(executable, self._inputs)
        executable.device.sync()
        return executable

    @torch.no_grad()
    def run(self, executable: TVMExecutable, inputs: Sequence[Tensor]) -> Tensor:
        """Share tensors through DLPack and execute the already built VM.

        CUDA work is synchronized at the framework boundaries for correctness
        across PyTorch and TVM streams. These synchronizations are part of run.
        """
        if _signature(inputs) != executable.input_signature:
            raise ValueError("inputs must match the shapes, strides, dtype, and device used in build")
        if inputs[0].is_cuda:
            torch.cuda.synchronize(inputs[0].device)
        tvm_inputs = tuple(self._tvm.runtime.from_dlpack(x.detach()) for x in inputs)
        output = executable.vm["main"](*tvm_inputs)
        if inputs[0].is_cuda:
            executable.device.sync()
        return torch.from_dlpack(output)

    def collect_metadata(self) -> dict[str, Any]:
        return {
            "backend": "tvm",
            "backend_version": self._tvm.__version__ if self._tvm else None,
            "budget_enforced": False,
            "progress_reporting": False,
            "configuration": {
                "target": str(self._target) if self._target is not None else self.target,
                "frontend": "torch.export -> relax",
                "pipeline": "default",
                "autotuning": False,
                "matmul_accumulation_dtype": "float32 for float16/bfloat16",
                "eval_mode": True,
                "grad_enabled": False,
                "tensor_interop": "dlpack",
                "cuda_boundary_synchronization": True,
            },
        }
