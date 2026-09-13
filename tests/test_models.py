"""Validation at configuration, assignment, and JSON persistence boundaries."""

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from tobench.core.budget import OptimizationBudget
from tobench.core.result import BenchmarkResult


class ModelTests(unittest.TestCase):
    def result(self, **overrides):
        fields = {
            "backend": "torch_eager",
            "workload": {"name": "GEMM", "M": 2},
            "configuration": {},
            "environment": {},
            "timestamp": "2026-09-12T00:00:00+00:00",
        }
        return BenchmarkResult(**(fields | overrides))

    def test_budget_is_strict_positive_finite_and_frozen(self):
        self.assertEqual(OptimizationBudget().max_time_seconds, 60)
        budget = OptimizationBudget(max_time_seconds=1)
        with self.assertRaises(ValidationError):
            budget.max_time_seconds = 2
        for value in ("60", True, 0, -1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                OptimizationBudget(max_time_seconds=value)
        with self.assertRaises(ValidationError):
            OptimizationBudget(max_time_second=1)

    def test_invalid_metric_types_ranges_and_status(self):
        for fields in (
            {"median_latency_ms": "1"},
            {"median_latency_ms": True},
            {"median_latency_ms": 0},
            {"p95_latency_ms": float("inf")},
            {"optimization_time_seconds": -1},
            {"throughput": float("nan")},
            {"latency_samples_ms": [1.0, -1.0]},
            {"peak_host_memory_bytes": 1.5},
            {"status": "sucess"},
            {"configuration": {"object": object()}},
            {"unknown_field": 1},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                self.result(**fields)

    def test_assignment_validation_preserves_previous_value(self):
        result = self.result(median_latency_ms=1)
        with self.assertRaises(ValidationError):
            result.median_latency_ms = -1
        self.assertEqual(result.median_latency_ms, 1)
        result.status = "success"
        self.assertEqual(result.status, "success")

    def test_json_round_trip(self):
        result = self.result(latency_samples_ms=[1.0, 2.0], median_latency_ms=1.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            result.save_json(path)
            restored = BenchmarkResult.model_validate_json(path.read_text())
            data = json.loads(path.read_text())
        self.assertEqual(restored, result)
        self.assertIsNone(data["peak_device_memory_bytes"])
        self.assertEqual(data["latency_samples_ms"], [1.0, 2.0])

    def test_save_revalidates_mutated_containers(self):
        result = self.result()
        result.latency_samples_ms.append(-1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            with self.assertRaises(ValidationError):
                result.save_json(path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
