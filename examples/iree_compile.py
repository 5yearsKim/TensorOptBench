"""Benchmark a selected workload with IREE Turbine AOT."""

from tobench.backends.iree import IREEAdapter, IREERunner
from tobench.benchmarking import Benchmarker
from tobench.core.experiment import benchmark

if __package__:
    from .utils import create_workload, parse_config, save_and_report
else:
    from utils import create_workload, parse_config, save_and_report


def main(argv: list[str] | None = None) -> None:
    config = parse_config("iree", argv)
    workload, inputs = create_workload(
        config.workload.name,
        device=config.device,
        parameters=config.workload.model_dump(exclude={"name"}),
    )
    adapter = IREEAdapter()
    runner = IREERunner(
        benchmarker=Benchmarker(),
        optimization_level=config.iree.optimization_level,
    )
    print(
        f"Benchmarking {config.workload.name} with IREE on {config.device}...",
        flush=True,
    )
    result = benchmark(
        adapter,
        runner,
        workload,
        inputs,
        budget=config.budget,
        correctness=config.correctness,
        warmup=config.runtime.warmup,
        repetitions=config.runtime.repetitions,
    )
    save_and_report(result, config)


if __name__ == "__main__":
    main()
