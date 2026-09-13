"""Build and execute already prepared Relax programs."""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from tobench.backends.base_runner import BaseRunner
from tobench.core.budget import OptimizationBudget
from .prepared import TVMExecutable, TVMPreparedInput


def _signature(inputs: Sequence[Tensor]) -> tuple:
    return tuple((tuple(x.shape), x.stride(), x.dtype, x.device) for x in inputs)


class TVMRunner(BaseRunner[TVMPreparedInput, TVMExecutable]):
    def _build(self, prepared: TVMPreparedInput, budget: OptimizationBudget | None) -> TVMExecutable:
        if not isinstance(prepared, TVMPreparedInput):
            raise TypeError("expected TVMPreparedInput from TVMAdapter.prepare")
        from tvm import relax

        compiled = relax.build(prepared.mod, target=prepared.target)
        executable = TVMExecutable(
            vm=relax.VirtualMachine(compiled, prepared.device),
            device=prepared.device, inputs=prepared.inputs, tvm=prepared.tvm,
            input_signature=_signature(prepared.inputs),
        )
        self._run(executable, None)
        self.synchronize(executable)
        return executable

    @torch.no_grad()
    def _run(self, executable: TVMExecutable, inputs: Sequence[Tensor] | None) -> Tensor:
        inputs = executable.inputs if inputs is None else tuple(inputs)
        if _signature(inputs) != executable.input_signature:
            raise ValueError("inputs must match the shapes, strides, dtype, and device used in build")
        if inputs[0].is_cuda:
            torch.cuda.synchronize(inputs[0].device)
        tvm_inputs = tuple(executable.tvm.runtime.from_dlpack(x.detach()) for x in inputs)
        output = executable.vm["main"](*tvm_inputs)
        if inputs[0].is_cuda:
            executable.device.sync()
        return torch.from_dlpack(output)

    def synchronize(self, state: TVMPreparedInput | TVMExecutable) -> None:
        for device in {x.device for x in state.inputs if x.is_cuda}:
            torch.cuda.synchronize(device)
        state.device.sync()

    def collect_metadata(self, prepared: TVMPreparedInput) -> dict[str, Any]:
        return {
            "backend": "tvm",
            "backend_version": prepared.tvm.__version__,
            "budget_enforced": False,
            "progress_reporting": False,
            "configuration": {
                "target": str(prepared.target),
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
