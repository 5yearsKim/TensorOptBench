"""Benchmark GEMM using a JSON configuration or command-line settings."""

import argparse
from pathlib import Path

from pydantic import ValidationError

from tobench.core.config import BenchmarkConfig
from tobench.core.runner import benchmark_from_config


def parse_config(argv: list[str] | None = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON configuration file")
    parser.add_argument("--backend", choices=("eager", "inductor", "tvm"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--dtype", choices=("float16", "bfloat16"))
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--budget-seconds", type=float)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    try:
        config = BenchmarkConfig.from_json(args.config) if args.config else BenchmarkConfig()
        data = config.model_dump()
        # Only explicitly supplied flags override the file's settings.
        for field in ("backend", "device", "output"):
            value = getattr(args, field)
            if value is not None:
                data[field] = value
        for flag, field in (("m", "M"), ("n", "N"), ("k", "K"), ("dtype", "dtype"), ("seed", "seed")):
            value = getattr(args, flag)
            if value is not None:
                data["workload"][field] = value
        for field in ("warmup", "repetitions"):
            value = getattr(args, field)
            if value is not None:
                data["runtime"][field] = value
        if args.budget_seconds is not None:
            data["budget"]["max_time_seconds"] = args.budget_seconds
        return BenchmarkConfig.model_validate(data)
    except (OSError, ValidationError) as error:
        parser.error(str(error))


def main() -> None:
    config = parse_config()
    print(f"Benchmarking {config.backend} GEMM on {config.device}...", flush=True)
    result = benchmark_from_config(config)
    output = Path(config.output or f"results/gemm_{config.backend}.json")
    result.save_json(output)
    print(f"Status: {result.status}")
    print(f"Results: {output}")
    if result.status != "success":
        print(f"Error: {result.error}")
        raise SystemExit(1)
    print(f"Build: {result.optimization_time_seconds:.6f} s")
    print(f"Median latency: {result.median_latency_ms:.6f} ms")
    print(f"P95 latency: {result.p95_latency_ms:.6f} ms")
    print(f"Throughput: {result.throughput:.6f} {result.throughput_unit}")
    print(f"Budget overrun: {result.budget_overrun_seconds:.6f} s (not enforced)")


if __name__ == "__main__":
    main()
