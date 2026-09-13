"""Shared workload examples, CLI configuration, and result reporting."""

import argparse
from pathlib import Path

import torch
from pydantic import ValidationError

from tobench.core.config import BenchmarkConfig
from tobench.core.result import BenchmarkResult
from tobench.workloads import BaseWorkload, GEMM, RMSNormLinear


def parse_config(backend: str, argv: list[str] | None = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON configuration file")
    parser.add_argument("--work-type", "--work_type", choices=("gemm", "rmsnorm_linear"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--dtype", choices=("float16", "bfloat16"))
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--budget-seconds", type=float)
    parser.add_argument("--output")
    parser.add_argument("--correctness", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--atol", type=float)
    if backend == "tvm_metaschedule":
        parser.add_argument("--max-trials-global", type=int)
        parser.add_argument("--max-trials-per-task", type=int)
        parser.add_argument("--num-trials-per-iter", type=int)
        parser.add_argument("--tuning-seed", type=int)
        parser.add_argument("--cost-model", choices=("xgb", "random"))
        parser.add_argument("--work-dir")
        parser.add_argument("--target")
    args = parser.parse_args(argv)

    try:
        config = BenchmarkConfig.from_json(args.config) if args.config else BenchmarkConfig()
        data = config.model_dump()
        data["backend"] = backend
        if args.work_type is not None and args.work_type != config.workload.name:
            data["workload"] = {"name": args.work_type}
        # Only explicitly supplied flags override the file's settings.
        for field in ("device", "output"):
            value = getattr(args, field)
            if value is not None:
                data[field] = value
        for flag, field in (("m", "M"), ("n", "N"), ("k", "K"), ("dtype", "dtype"), ("seed", "seed"), ("eps", "eps")):
            value = getattr(args, flag)
            if value is not None:
                data["workload"][field] = value
        for field in ("warmup", "repetitions"):
            value = getattr(args, field)
            if value is not None:
                data["runtime"][field] = value
        if args.budget_seconds is not None:
            data["budget"]["max_time_seconds"] = args.budget_seconds
        if args.correctness is not None:
            data["correctness"]["enabled"] = args.correctness
        for field in ("rtol", "atol"):
            if getattr(args, field) is not None:
                data["correctness"][field] = getattr(args, field)
        if backend == "tvm_metaschedule":
            for flag, field in (
                ("max_trials_global", "max_trials_global"),
                ("max_trials_per_task", "max_trials_per_task"),
                ("num_trials_per_iter", "num_trials_per_iter"),
                ("tuning_seed", "seed"), ("cost_model", "cost_model"),
                ("work_dir", "work_dir"), ("target", "target"),
            ):
                value = getattr(args, flag)
                if value is not None:
                    data["metaschedule"][field] = value
        return BenchmarkConfig.model_validate(data)
    except (OSError, ValidationError) as error:
        parser.error(str(error))

def create_workload(
    work_type: str, *, device: str = "cpu", parameters: dict | None = None,
) -> tuple[BaseWorkload, tuple[torch.Tensor, ...]]:
    """Map a workload name to a small example and reproducible input tensors.

    Extend this function when adding workload examples. Parameters optionally
    override the hardcoded values for the selected operator.
    """
    if work_type == "gemm":
        factory = GEMM
        values = {"M": 128, "N": 128, "K": 128, "dtype": "float16", "seed": 0}
    elif work_type == "rmsnorm_linear":
        factory = RMSNormLinear
        values = {"M": 16, "N": 406, "K": 4096, "dtype": "float16", "seed": 0, "eps": 1e-6}
    else:
        raise ValueError(f"Unknown workload: {work_type}")
    values.update(parameters or {})
    workload = factory(**values)
    generator = torch.Generator(device=device).manual_seed(workload.seed)
    inputs = tuple(
        torch.randn(
            shape, dtype=getattr(torch, workload.dtype), device=device, generator=generator
        )
        for shape in workload.input_shapes
    )
    return workload, inputs



def save_and_report(result: BenchmarkResult, config: BenchmarkConfig) -> None:
    """Store the resolved configuration and measurements, then print a summary."""
    result.configuration = {
        **result.configuration,
        "experiment": config.model_dump(mode="json"),
        "input_generation": "torch.randn with dedicated device generator",
    }
    output = Path(config.output or f"results/{config.workload.name}_{config.backend}.json")
    result.save_json(output)
    print(f"Status: {result.status}")
    check = result.correctness
    print(f"Correctness: {check.status} (reference={check.reference}, rtol={check.rtol}, atol={check.atol})")
    if check.failed_elements is not None:
        print(f"Failed elements: {check.failed_elements}/{check.total_elements}")
        print(f"Maximum absolute error: {check.max_absolute_error}")
    print(f"Results: {output}")
    if result.status != "success":
        print(f"Error: {result.error}")
        raise SystemExit(1)
    print(f"Preparation: {result.preparation_time_seconds:.6f} s")
    print(f"Build: {result.optimization_time_seconds:.6f} s")
    print(f"Median latency: {result.median_latency_ms:.6f} ms")
    print(f"P95 latency: {result.p95_latency_ms:.6f} ms")
    print(f"Throughput: {result.throughput:.6f} {result.throughput_unit}")
    print(f"Budget overrun: {result.budget_overrun_seconds:.6f} s (not enforced)")
