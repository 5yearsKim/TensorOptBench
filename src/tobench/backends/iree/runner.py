"""Compile Turbine exports and execute them with the IREE runtime."""

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType
from typing import Any

import torch
from torch import Tensor

try:
    import iree.compiler as ireec
    import iree.runtime as ireert
except ModuleNotFoundError as error:
    if error.name != "iree" and not (error.name or "").startswith("iree."):
        raise
    ireec = ireert = None
    _IMPORT_ERROR = error
else:
    _IMPORT_ERROR = None

from tobench.backends.base_runner import BaseRunner
from tobench.benchmarking import Benchmarker
from tobench.core.budget import OptimizationBudget

from .prepared import IREEExecutable, IREEPreparedInput


def _signature(inputs: Sequence[Tensor]) -> tuple:
    return tuple(
        (tuple(value.shape), value.stride(), value.dtype, value.device)
        for value in inputs
    )


def _require_iree() -> tuple[ModuleType, ModuleType]:
    if ireec is None or ireert is None:
        raise ImportError(
            'The IREE backend is optional. Install it with: pip install -e ".[iree]"'
        ) from _IMPORT_ERROR
    return ireec, ireert


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


class IREERunner(BaseRunner[IREEPreparedInput, IREEExecutable]):
    def __init__(
        self,
        benchmarker: Benchmarker | None = None,
        *,
        optimization_level: str = "O3",
    ) -> None:
        super().__init__(benchmarker)
        if optimization_level not in ("O0", "O1", "O2", "O3"):
            raise ValueError("optimization_level must be O0, O1, O2, or O3")
        self.optimization_level = optimization_level

    def _build(
        self, prepared: IREEPreparedInput, budget: OptimizationBudget | None
    ) -> IREEExecutable:
        if not isinstance(prepared, IREEPreparedInput):
            raise TypeError("expected IREEPreparedInput from IREEAdapter.prepare")
        _, runtime = _require_iree()

        flags = [
            f"--iree-hal-target-device={prepared.target_device}",
            f"--iree-opt-level={self.optimization_level}",
        ]
        if prepared.target_device == "local":
            flags.append(
                f"--iree-hal-local-target-device-backends={prepared.target_backend}"
            )
        elif prepared.target_arch is not None:
            flags.append(f"--iree-cuda-target={prepared.target_arch}")
        prepared.export_output.session.set_flags(*flags)
        binary = prepared.export_output.compile(save_to=None, target_backends=None)

        device = runtime.get_device(prepared.runtime_device_uri)
        config = runtime.Config(device=device)
        vm_module = runtime.VmModule.copy_buffer(
            config.vm_instance, binary.map_memory()
        )
        module = runtime.load_vm_module(vm_module, config)
        executable = IREEExecutable(
            module=module,
            function=module.main,
            device=device,
            inputs=prepared.inputs,
            input_signature=_signature(prepared.inputs),
        )
        self._run(executable, None)
        self.synchronize(executable)
        return executable

    @torch.inference_mode()
    def _run(
        self, executable: IREEExecutable, inputs: Sequence[Tensor] | None
    ) -> Tensor:
        values = executable.inputs if inputs is None else tuple(inputs)
        if _signature(values) != executable.input_signature:
            raise ValueError(
                "inputs must match the shapes, strides, dtype, and device used in build"
            )
        if values[0].is_cuda:
            # The capsule-level IREE API has no stream argument, so complete
            # PyTorch producers before the IREE runtime consumes their buffers.
            torch.cuda.synchronize(values[0].device)
        iree_inputs = tuple(
            executable.device.from_dlpack_capsule(torch.utils.dlpack.to_dlpack(value))
            for value in values
        )
        output = executable.function(*iree_inputs)
        if isinstance(output, tuple):
            if len(output) != 1:
                raise TypeError("IREE workloads must return exactly one tensor")
            output = output[0]
        buffer_view = getattr(output, "_buffer_view", None)
        if buffer_view is None:
            raise TypeError("IREE workload output is not a device array")
        device_type, device_id = values[0].__dlpack_device__()
        capsule = executable.device.create_dlpack_capsule(
            buffer_view, device_type, device_id
        )
        return torch.from_dlpack(capsule)

    def synchronize(self, state: IREEPreparedInput | IREEExecutable) -> None:
        for device in {value.device for value in state.inputs if value.is_cuda}:
            torch.cuda.synchronize(device)

    def collect_metadata(self, prepared: IREEPreparedInput) -> dict[str, Any]:
        compiler, runtime = _require_iree()
        return {
            "backend": "iree",
            "backend_version": _package_version("iree-base-compiler"),
            "budget_enforced": False,
            "progress_reporting": False,
            "configuration": {
                "frontend": "iree-turbine AOT",
                "compiler_target_device": prepared.target_device,
                "compiler_target_backend": prepared.target_backend,
                "target_arch": prepared.target_arch,
                "runtime_device_uri": prepared.runtime_device_uri,
                "optimization_level": self.optimization_level,
                "dynamic_shapes": False,
                "eval_mode": True,
                "grad_enabled": False,
                "tensor_interop": "dlpack",
                "host_tensor_copies": False,
                "cuda_boundary_synchronization": True,
                "iree_turbine_version": _package_version("iree-turbine"),
                "iree_compiler_version": _package_version("iree-base-compiler"),
                "iree_runtime_version": _package_version("iree-base-runtime"),
                "target_available": prepared.target_backend
                in compiler.query_available_targets(),
                "runtime_driver_available": prepared.runtime_device_uri.split(":", 1)[0]
                in runtime.query_available_drivers(),
            },
        }
