"""Coordinate preparation, backend runners, correctness, and benchmark results."""

import platform
from datetime import datetime, timezone
from time import perf_counter

import torch
from torch import Tensor

from tobench.backends.base_adapter import BaseAdapter
from tobench.backends.base_runner import BaseRunner

from .budget import OptimizationBudget
from .result import BenchmarkResult
from tobench.workloads.base_workload import BaseWorkload


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    adapter: BaseAdapter,
    runner: BaseRunner,
    workload: BaseWorkload,
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

    Budgets are passed to the backend runner. The runner records overrun but accepts a
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
        backend=type(runner).__name__,
        workload=workload.to_config(),
        configuration={
            "warmup": warmup,
            "repetitions": repetitions,
            "budget_seconds": budget.max_time_seconds if budget is not None else None,
            "budget_policy": "report_overrun_and_continue",
            "timing_method": "synchronized_wall_clock",
            "latency_scope": "runner.run including dispatch, interop, and synchronization",
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
    if runner.benchmarker is None:
        raise ValueError("benchmark requires a runner with an injected Benchmarker")
    benchmarker = runner.benchmarker
    benchmarker.reset()
    stage = "prepare"
    try:
        _synchronize(device)
        started = perf_counter()
        try:
            prepared = adapter.prepare(workload, inputs)
            _synchronize(device)
        finally:
            result.preparation_time_seconds = perf_counter() - started
        metadata = runner.collect_metadata(prepared)
        result.backend_metadata = metadata
        result.backend = metadata.get("backend", result.backend)
        result.backend_version = metadata.get("backend_version")
        result.budget_enforced = metadata.get("budget_enforced", False)

        stage = "build"
        executable = runner.build(prepared, budget=budget)

        stage = "correctness"
        with torch.inference_mode():
            reference = workload(*inputs)
        output = runner.run(executable, measure=False)
        runner.synchronize(executable)
        torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)
        del output, reference

        stage = "runtime"
        summary = runner.benchmark_run(executable, warmup=warmup, repetitions=repetitions)
        result.median_latency_ms = summary["median_latency_ms"]
        result.p95_latency_ms = summary["p95_latency_ms"]
        flop_count = getattr(workload, "flop_count", None)
        if flop_count is not None:
            result.throughput = flop_count / (result.median_latency_ms * 1e-3) / 1e12
            result.throughput_unit = "TFLOP/s"
        result.status = "success"
    except Exception as error:
        if stage == "runtime":
            stage = benchmarker.phase
        result.status = "correctness_failed" if stage == "correctness" else "error"
        result.error = {"stage": stage, "type": type(error).__name__, "message": str(error)}
    finally:
        result.optimization_time_seconds = benchmarker.optimization_time_seconds
        result.budget_overrun_seconds = benchmarker.budget_overrun_seconds
        result.latency_samples_ms = list(benchmarker.latency_samples_ms)
    return result
