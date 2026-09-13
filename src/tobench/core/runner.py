"""Minimal benchmark runner using synchronized wall-clock measurements."""

import math
import platform
from datetime import datetime, timezone
from statistics import median
from time import perf_counter

import torch
from torch import Tensor

from tobench.backends.base import BackendAdapter

from .budget import OptimizationBudget
from .config import BenchmarkConfig
from .result import BenchmarkResult
from .workload import Workload


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_from_config(config: BenchmarkConfig) -> BenchmarkResult:
    """Create shared inputs and an adapter, then run the configured experiment.

    Returns the result without writing files. Input generation uses a dedicated
    generator and is outside all measured stages.
    """
    from tobench.backends.torch import TorchEagerAdapter, TorchInductorAdapter
    from tobench.workloads import GEMM

    workload = GEMM(**config.workload.model_dump(exclude={"name"}))
    device = torch.device(config.device)
    generator = torch.Generator(device=device).manual_seed(workload.seed)
    inputs = tuple(
        torch.randn(
            shape, dtype=getattr(torch, workload.dtype), device=device, generator=generator
        )
        for shape in workload.input_shapes
    )
    if config.backend == "tvm":
        from tobench.backends.tvm import TVMAdapter

        adapter = TVMAdapter()
    else:
        adapter = TorchEagerAdapter() if config.backend == "eager" else TorchInductorAdapter()
    result = benchmark(
        adapter, workload, inputs, budget=config.budget,
        warmup=config.runtime.warmup, repetitions=config.runtime.repetitions,
    )
    result.configuration = {
        **result.configuration,
        "experiment": config.model_dump(mode="json"),
        "input_generation": "torch.randn with dedicated device generator",
    }
    return result


def benchmark(
    adapter: BackendAdapter,
    workload: Workload,
    inputs: tuple[Tensor, ...],
    *,
    budget: OptimizationBudget | None = None,
    warmup: int = 20,
    repetitions: int = 100,
) -> BenchmarkResult:
    """Measure one backend on caller-supplied inputs, validating against eager.

    Prepare, correctness checking, and warmup are outside the build timer.
    Latencies include adapter dispatch, interop, and synchronization overhead;
    they are not kernel-only timings. No caches are cleared and no subprocess
    is spawned. Cache/process policy and unmeasured memory are explicit in JSON.

    Budgets are passed to the adapter. The runner records overrun but accepts a
    late executable, because current adapters cannot enforce cancellation.
    Operational failures return an error result; invalid arguments raise.
    """
    for name, value, minimum in (("warmup", warmup, 0), ("repetitions", repetitions, 1)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    if not isinstance(inputs, (tuple, list)) or not inputs:
        raise ValueError("inputs must be a non-empty tuple or list of tensors")
    if not all(isinstance(x, Tensor) for x in inputs):
        raise TypeError("inputs must contain only tensors")
    inputs = tuple(inputs)
    device = inputs[0].device
    if device.type not in ("cpu", "cuda") or any(x.device != device for x in inputs):
        raise ValueError("inputs must share one CPU or CUDA device")
    if budget is not None and not isinstance(budget, OptimizationBudget):
        raise TypeError("budget must be an OptimizationBudget or None")
    shapes = getattr(workload, "input_shapes", None)
    if shapes is not None and tuple(tuple(x.shape) for x in inputs) != shapes:
        raise ValueError("input shapes do not match the workload")
    if any(x.dtype != getattr(torch, workload.dtype) for x in inputs):
        raise ValueError("input dtype does not match the workload")
    if any(not x.is_contiguous() for x in inputs):
        raise ValueError("benchmark inputs must be contiguous")

    rtol, atol = (1e-3, 1e-3) if workload.dtype == "float16" else (1e-2, 1e-2)
    result = BenchmarkResult(
        backend=type(adapter).__name__,
        workload=workload.to_config(),
        configuration={
            "warmup": warmup,
            "repetitions": repetitions,
            "budget_seconds": budget.max_time_seconds if budget is not None else None,
            "budget_policy": "report_overrun_and_continue",
            "timing_method": "synchronized_wall_clock",
            "latency_scope": "adapter.run including dispatch, interop, and synchronization",
            "p95_method": "nearest_rank",
            "cache_policy": "uncontrolled",
            "process_isolation": False,
            "memory_measurement": "not_implemented",
            "correctness": {"reference": "pytorch_eager", "rtol": rtol, "atol": atol},
            "inputs": [
                {"shape": list(x.shape), "stride": list(x.stride()), "dtype": str(x.dtype)}
                for x in inputs
            ],
        },
        environment={
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    stage = "prepare"
    try:
        adapter.prepare(workload, inputs)
        metadata = adapter.collect_metadata()
        result.backend_metadata = metadata
        result.backend = metadata.get("backend", result.backend)
        result.backend_version = metadata.get("backend_version")
        result.budget_enforced = metadata.get("budget_enforced", False)

        stage = "build"
        _synchronize(device)
        started = perf_counter()
        try:
            executable = adapter.build(budget=budget)
            _synchronize(device)
        finally:
            result.optimization_time_seconds = perf_counter() - started
            if budget is not None:
                result.budget_overrun_seconds = max(
                    0.0, result.optimization_time_seconds - budget.max_time_seconds
                )

        stage = "correctness"
        with torch.inference_mode():
            reference = workload(*inputs)
        output = adapter.run(executable, inputs)
        _synchronize(device)
        torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)
        del output, reference

        stage = "warmup"
        for _ in range(warmup):
            adapter.run(executable, inputs)
        _synchronize(device)

        stage = "runtime"
        for _ in range(repetitions):
            _synchronize(device)
            started = perf_counter()
            output = adapter.run(executable, inputs)
            _synchronize(device)
            elapsed_ms = (perf_counter() - started) * 1000.0
            del output
            if elapsed_ms <= 0:
                raise RuntimeError("timer returned a non-positive latency")
            result.latency_samples_ms.append(elapsed_ms)

        result.median_latency_ms = median(result.latency_samples_ms)
        ordered = sorted(result.latency_samples_ms)
        result.p95_latency_ms = ordered[math.ceil(0.95 * repetitions) - 1]
        flop_count = getattr(workload, "flop_count", None)
        if flop_count is not None:
            result.throughput = flop_count / (result.median_latency_ms * 1e-3) / 1e12
            result.throughput_unit = "TFLOP/s"
        result.status = "success"
    except Exception as error:
        result.status = "correctness_failed" if stage == "correctness" else "error"
        result.error = {"stage": stage, "type": type(error).__name__, "message": str(error)}
    return result
