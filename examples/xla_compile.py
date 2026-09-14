"""Benchmark a selected workload with PyTorch/XLA and PJRT."""

from tobench.backends.xla import XLAAdapter, XLARunner
from tobench.benchmarking import Benchmarker
from tobench.core.experiment import benchmark

if __package__:
    from .utils import create_workload, parse_config, save_and_report
else:
    from utils import create_workload, parse_config, save_and_report


def main(argv: list[str] | None = None) -> None:
    config = parse_config("xla", argv)
    workload, inputs = create_workload(
        config.workload.name,
        device="cpu",
        parameters=config.workload.model_dump(exclude={"name"}),
    )
    adapter = XLAAdapter(device_type=config.device.upper())
    runner = XLARunner(benchmarker=Benchmarker())
    print(
        f"Benchmarking {config.workload.name} with XLA on {config.device}...",
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
