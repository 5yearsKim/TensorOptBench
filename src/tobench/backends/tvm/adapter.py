"""Prepare Relax IR and target state from PyTorch workload code."""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

try:
    import tvm
    from tvm import relax
    from tvm.relax.frontend.torch import from_exported_program
except ModuleNotFoundError as error:
    if error.name != "tvm" and not (error.name or "").startswith("tvm."):
        raise
    raise ImportError(
        'The TVM backend is optional. Install it with: pip install -e ".[tvm]"'
    ) from error

from tobench.backends.base_adapter import BaseAdapter
from tobench.workloads import BaseWorkload
from .prepared import TVMPreparedInput


def _convert_matmul(node: Any, importer: Any) -> Any:
    """Keep PyTorch's output dtype after TVM's FP32 matmul accumulation."""
    lhs, rhs = importer.retrieve_args(node)
    dtype = lhs.ty.dtype
    accumulation_dtype = "float32" if dtype in ("float16", "bfloat16") else dtype
    result = importer.block_builder.emit(
        relax.op.matmul(lhs, rhs, out_dtype=accumulation_dtype)
    )
    if dtype != accumulation_dtype:
        result = importer.block_builder.emit(relax.op.astype(result, dtype))
    return result


class TVMAdapter(BaseAdapter[TVMPreparedInput]):
    def __init__(self, target: str | None = None) -> None:
        self.target = target

    @torch.no_grad()
    def prepare(self, workload: BaseWorkload, inputs: Sequence[Tensor]) -> TVMPreparedInput:
        """Export/import the graph without compiling a VM executable."""
        if not isinstance(workload, BaseWorkload):
            raise TypeError("workload must be a BaseWorkload module")
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

        target_name = self.target
        if target_name is None:
            if device.type == "cuda":
                major, minor = torch.cuda.get_device_capability(device)
                target_name = {"kind": "cuda", "arch": f"sm_{major}{minor}"}
            else:
                target_name = "llvm"
        target = tvm.target.Target(target_name)
        expected_kind = "cuda" if device.type == "cuda" else "llvm"
        if target.kind.name != expected_kind:
            raise ValueError(f"{device.type} inputs require a {expected_kind} target")
        if not tvm.runtime.enabled(expected_kind):
            raise RuntimeError(f"Installed TVM does not support {expected_kind}")

        exported = torch.export.export(workload.eval(), tuple(inputs))
        mod = from_exported_program(
            exported,
            unwrap_unit_return_tuple=True,
            custom_convert_map={
                "bmm.default": _convert_matmul,
                "mm.default": _convert_matmul,
                "matmul.default": _convert_matmul,
            },
        )
        return TVMPreparedInput(
            mod=mod, inputs=tuple(inputs), target=target,
            device=tvm.device(device.type, device.index or 0),
        )
