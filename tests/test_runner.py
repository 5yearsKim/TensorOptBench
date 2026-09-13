"""Validate measurement boundaries, failure handling, and result storage."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tobench.backends.torch import TorchEagerAdapter
from tobench.core.budget import OptimizationBudget
from tobench.core.runner import benchmark
from tobench.workloads import GEMM


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.workload = GEMM(2, 2, 3)
        self.inputs = (torch.ones(2, 3, dtype=torch.float16), torch.ones(3, 2, dtype=torch.float16))

    def test_measurement_boundaries_and_summaries(self):
        clock = [0.0]
        durations = iter([100, 200, 0.001, 0.002, 0.003, 0.004])

        class TimedAdapter(TorchEagerAdapter):
            def prepare(self, workload, inputs):
                clock[0] += 1000  # Setup must not count toward build.
                super().prepare(workload, inputs)

            def build(self, budget, report_progress=None):
                self.received_budget = budget
                clock[0] += 2
                return super().build(budget)

            def run(self, executable, inputs):
                clock[0] += next(durations)
                return super().run(executable, inputs)

        def sync(device):
            clock[0] += 0.01

        adapter = TimedAdapter()
        budget = OptimizationBudget(max_time_seconds=1)
        with patch("tobench.core.runner.perf_counter", side_effect=lambda: clock[0]), patch(
            "tobench.core.runner._synchronize", side_effect=sync
        ):
            result = benchmark(
                adapter, self.workload, self.inputs,
                budget=budget, warmup=1, repetitions=4,
            )
        self.assertEqual(result.status, "success", result.error)
        self.assertIs(adapter.received_budget, budget)
        self.assertAlmostEqual(result.optimization_time_seconds, 2.01)
        self.assertAlmostEqual(result.budget_overrun_seconds, 1.01)
        for actual, expected in zip(result.latency_samples_ms, [11, 12, 13, 14]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(result.median_latency_ms, 12.5)
        self.assertAlmostEqual(result.p95_latency_ms, 14)
        self.assertAlmostEqual(result.throughput, 24 / 0.0125 / 1e12, places=15)
        self.assertEqual(result.throughput_unit, "TFLOP/s")

    def test_wrong_output_has_no_runtime_metrics(self):
        class WrongAdapter(TorchEagerAdapter):
            def run(self, executable, inputs):
                return super().run(executable, inputs) + 1

        result = benchmark(WrongAdapter(), self.workload, self.inputs, repetitions=2)
        self.assertEqual(result.status, "correctness_failed")
        self.assertEqual(result.latency_samples_ms, [])
        self.assertIsNone(result.median_latency_ms)
        self.assertIsNone(result.throughput)

    def test_failed_build_preserves_error_and_elapsed_time(self):
        class FailingAdapter(TorchEagerAdapter):
            def build(self, budget, report_progress=None):
                raise RuntimeError("compiler failed")

        result = benchmark(FailingAdapter(), self.workload, self.inputs)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error["stage"], "build")
        self.assertIn("compiler failed", result.error["message"])
        self.assertIsNotNone(result.optimization_time_seconds)
        self.assertEqual(result.latency_samples_ms, [])

    def test_eager_result_json(self):
        result = benchmark(
            TorchEagerAdapter(), self.workload, self.inputs, warmup=0, repetitions=1
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
                    benchmark(TorchEagerAdapter(), self.workload, self.inputs, **kwargs)
        with self.assertRaisesRegex(ValueError, "shapes"):
            benchmark(TorchEagerAdapter(), self.workload, (self.inputs[0][:1], self.inputs[1]))

    def test_invalid_budget(self):
        for value in (0, -1, float("inf"), float("nan"), True, "60"):
            with self.subTest(value=value):
                with self.assertRaises((ValueError, TypeError)):
                    OptimizationBudget(max_time_seconds=value)


if __name__ == "__main__":
    unittest.main()
