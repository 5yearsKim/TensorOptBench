"""Validate measurement boundaries, failure handling, and result storage."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tobench.backends.torch import TorchAdapter, TorchEagerRunner
from tobench.benchmarking import Benchmarker
from tobench.core.budget import OptimizationBudget
from tobench.core.experiment import benchmark
from tobench.workloads import GEMM


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.workload = GEMM(2, 2, 3)
        self.inputs = (torch.ones(2, 3, dtype=torch.float16), torch.ones(3, 2, dtype=torch.float16))

    def test_measurement_boundaries_and_summaries(self):
        clock = [0.0]
        durations = iter([100, 200, 0.001, 0.002, 0.003, 0.004])

        class TimedAdapter(TorchAdapter):
            def prepare(self, workload, inputs):
                clock[0] += 1000
                return super().prepare(workload, inputs)

        class TimedRunner(TorchEagerRunner):
            def _build(self, prepared, budget):
                self.received_budget = budget
                clock[0] += 2
                return super()._build(prepared, budget)

            def _run(self, executable, inputs):
                clock[0] += next(durations)
                return super()._run(executable, inputs)

        def sync(device):
            clock[0] += 0.01

        runner = TimedRunner(benchmarker=Benchmarker())
        budget = OptimizationBudget(max_time_seconds=1)
        with (
            patch("tobench.benchmarking.benchmarker.perf_counter", side_effect=lambda: clock[0]),
            patch("tobench.core.experiment.perf_counter", side_effect=lambda: clock[0]),
            patch.object(runner, "synchronize", side_effect=sync),
        ):
            result = benchmark(
                TimedAdapter(), runner, self.workload, self.inputs,
                budget=budget, warmup=1, repetitions=4,
            )
        self.assertEqual(result.status, "success", result.error)
        self.assertIs(runner.received_budget, budget)
        self.assertAlmostEqual(result.preparation_time_seconds, 1000)
        self.assertAlmostEqual(result.optimization_time_seconds, 2.01)
        self.assertAlmostEqual(result.budget_overrun_seconds, 1.01)
        for actual, expected in zip(result.latency_samples_ms, [11, 12, 13, 14]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(result.median_latency_ms, 12.5)
        self.assertAlmostEqual(result.p95_latency_ms, 14)
        self.assertAlmostEqual(result.throughput, 24 / 0.0125 / 1e12, places=15)
        self.assertEqual(result.throughput_unit, "TFLOP/s")

    def test_wrong_output_has_no_runtime_metrics(self):
        class WrongRunner(TorchEagerRunner):
            def _run(self, executable, inputs):
                return super()._run(executable, inputs) + 1

        result = benchmark(TorchAdapter(), WrongRunner(benchmarker=Benchmarker()), self.workload, self.inputs, repetitions=2)
        self.assertEqual(result.status, "correctness_failed")
        self.assertEqual(result.latency_samples_ms, [])
        self.assertIsNone(result.median_latency_ms)
        self.assertIsNone(result.throughput)

    def test_failed_build_preserves_error_and_elapsed_time(self):
        class FailingRunner(TorchEagerRunner):
            def _build(self, prepared, budget):
                raise RuntimeError("compiler failed")

        result = benchmark(TorchAdapter(), FailingRunner(benchmarker=Benchmarker()), self.workload, self.inputs)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error["stage"], "build")
        self.assertIn("compiler failed", result.error["message"])
        self.assertIsNotNone(result.optimization_time_seconds)
        self.assertEqual(result.latency_samples_ms, [])

    def test_failed_prepare_does_not_reuse_previous_measurements(self):
        class FailingAdapter(TorchAdapter):
            def prepare(self, workload, inputs):
                raise RuntimeError("export failed")

        runner = TorchEagerRunner(benchmarker=Benchmarker())
        benchmark(TorchAdapter(), runner, self.workload, self.inputs, warmup=0, repetitions=1)
        result = benchmark(FailingAdapter(), runner, self.workload, self.inputs)
        self.assertEqual(result.error["stage"], "prepare")
        self.assertIsNotNone(result.preparation_time_seconds)
        self.assertIsNone(result.optimization_time_seconds)
        self.assertEqual(result.latency_samples_ms, [])

    def test_warmup_failure_is_not_a_runtime_sample(self):
        class WarmupFailure(TorchEagerRunner):
            calls = 0

            def _run(self, executable, inputs):
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("warmup failed")
                return super()._run(executable, inputs)

        result = benchmark(
            TorchAdapter(), WarmupFailure(benchmarker=Benchmarker()),
            self.workload, self.inputs, warmup=1, repetitions=1,
        )
        self.assertEqual(result.error["stage"], "warmup")
        self.assertEqual(result.latency_samples_ms, [])

    def test_eager_result_json(self):
        result = benchmark(
            TorchAdapter(), TorchEagerRunner(benchmarker=Benchmarker()), self.workload, self.inputs, warmup=0, repetitions=1
        )
        self.assertEqual(result.status, "success", result.error)
        self.assertGreater(result.median_latency_ms, 0)
        self.assertEqual(result.median_latency_ms, result.p95_latency_ms)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results" / "gemm.json"
            result.save_json(path)
            data = json.loads(path.read_text())
        self.assertEqual(data["workload"]["M"], 2)
        self.assertEqual(len(data["latency_samples_ms"]), 1)
        self.assertIsNone(data["peak_device_memory_bytes"])
        self.assertIsNone(data["peak_host_memory_bytes"])
        self.assertEqual(data["configuration"]["cache_policy"], "uncontrolled")

    def test_invalid_arguments(self):
        for kwargs in ({"warmup": -1}, {"repetitions": 0}, {"repetitions": True}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((ValueError, TypeError)):
                    benchmark(TorchAdapter(), TorchEagerRunner(benchmarker=Benchmarker()), self.workload, self.inputs, **kwargs)
        with self.assertRaisesRegex(ValueError, "shapes"):
            benchmark(TorchAdapter(), TorchEagerRunner(benchmarker=Benchmarker()), self.workload, (self.inputs[0][:1], self.inputs[1]))

    def test_invalid_budget(self):
        for value in (0, -1, float("inf"), float("nan"), True, "60"):
            with self.subTest(value=value):
                with self.assertRaises((ValueError, TypeError)):
                    OptimizationBudget(max_time_seconds=value)


if __name__ == "__main__":
    unittest.main()
