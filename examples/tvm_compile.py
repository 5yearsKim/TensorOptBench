"""Benchmark a selected workload with tvm."""

from tobench.backends.tvm import TVMAdapter, TVMRunner
from tobench.benchmarking import Benchmarker
from tobench.core.experiment import benchmark

if __package__:
    from .utils import create_workload, parse_config, save_and_report
else:
    from utils import create_workload, parse_config, save_and_report


def main(argv: list[str] | None = None) -> None:
    config = parse_config("tvm", argv)
    workload, inputs = create_workload(
        config.workload.name,
        device=config.device,
        parameters=config.workload.model_dump(exclude={"name"}),
    )
    adapter = TVMAdapter()
    runner = TVMRunner(benchmarker=Benchmarker())
    print(f"Benchmarking {config.workload.name} with tvm on {config.device}...", flush=True)
    result = benchmark(
        adapter, runner, workload, inputs,
        budget=config.budget,
        warmup=config.runtime.warmup,
        repetitions=config.runtime.repetitions,
    )
    save_and_report(result, config)


if __name__ == "__main__":
    main()
