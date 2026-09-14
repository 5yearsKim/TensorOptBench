"""Structured comparison, configuration, and unmeasured correctness execution."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from pydantic import ValidationError

from examples.utils import parse_config
from tobench.backends.torch import TorchAdapter, TorchEagerRunner
from tobench.benchmarking import Benchmarker, CorrectnessChecker, TorchEagerReference
from tobench.core.config import CorrectnessConfig
from tobench.core.experiment import benchmark
from tobench.workloads import GEMM


class CorrectnessTests(unittest.TestCase):
    def test_statistics_and_threshold(self):
        checker = CorrectnessChecker(CorrectnessConfig(rtol=0.0, atol=0.125))
        check = checker.compare(torch.tensor([1.125, 2.5]), torch.tensor([1., 2.]))
        self.assertEqual(check.status, "failed")
        self.assertEqual(check.failed_elements, 1)
        self.assertEqual(check.total_elements, 2)
        self.assertEqual(check.max_absolute_error, 0.5)
        self.assertEqual(check.mean_absolute_error, 0.3125)

    def test_shape_and_dtype_mismatch(self):
        checker = CorrectnessChecker()
        shape = checker.compare(torch.ones(2, 1), torch.ones(2))
        self.assertFalse(shape.shape_match)
        self.assertEqual(shape.status, "failed")
        self.assertIsNone(shape.failed_elements)
        dtype = checker.compare(torch.ones(2).half(), torch.ones(2))
        self.assertFalse(dtype.dtype_match)
        self.assertEqual(dtype.status, "failed")

    def test_nonfinite_values_are_rejected_and_json_safe(self):
        check = CorrectnessChecker().compare(
            torch.tensor([float("nan"), float("inf"), 2.]),
            torch.tensor([float("nan"), float("inf"), 2.]),
        )
        self.assertEqual(check.status, "failed")
        self.assertEqual(check.actual_nonfinite_count, 2)
        self.assertEqual(check.reference_nonfinite_count, 2)
        self.assertEqual(check.failed_elements, 2)
        self.assertEqual(check.finite_elements, 1)
        self.assertEqual(check.max_absolute_error, 0)
        json.dumps(check.model_dump(), allow_nan=False)
        check = CorrectnessChecker().compare(torch.tensor([float("inf")]), torch.ones(1))
        self.assertIsNone(check.max_absolute_error)
        self.assertEqual(check.failed_elements, 1)

    def test_empty_outputs(self):
        check = CorrectnessChecker().compare(torch.empty(0), torch.empty(0))
        self.assertEqual(check.status, "passed")
        self.assertEqual(check.total_elements, 0)
        self.assertIsNone(check.mean_absolute_error)

    def test_default_tolerances_and_validation(self):
        for dtype, threshold in ((torch.float16, 1e-3), (torch.bfloat16, 1e-2)):
            check = CorrectnessChecker().compare(torch.ones(1, dtype=dtype), torch.ones(1, dtype=dtype))
            self.assertEqual(check.rtol, threshold)
            self.assertEqual(check.atol, threshold)
        for params in ({"rtol": -1}, {"atol": float("inf")}, {"rtol": True}, {"reference": "unknown"}):
            with self.subTest(params=params), self.assertRaises(ValidationError):
                CorrectnessConfig(**params)

    def test_cli_overrides_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"correctness": {"rtol": 0.1, "atol": 0.2}}))
            config = parse_config("inductor", ["--config", str(path), "--rtol", "0", "--no-correctness"])
        self.assertEqual(config.correctness.rtol, 0)
        self.assertEqual(config.correctness.atol, 0.2)
        self.assertFalse(config.correctness.enabled)

    def test_reference_snapshot_does_not_alias(self):
        class Identity(GEMM):
            def forward(self, A, B):
                return A

        inputs = (torch.ones(1, 1), torch.ones(1, 1))
        output = TorchEagerReference().run(Identity(1, 1, 1), inputs)
        inputs[0].zero_()
        self.assertEqual(output.item(), 1)


class CorrectnessExperimentTests(unittest.TestCase):
    def setUp(self):
        self.workload = GEMM(2, 2, 3, B=1)
        self.inputs = (
            torch.ones(1, 2, 3, dtype=torch.float16),
            torch.ones(1, 3, 2, dtype=torch.float16),
        )
        self.monitor = Benchmarker()

    def run_benchmark(self, runner=None, **kwargs):
        return benchmark(
            TorchAdapter(), runner or TorchEagerRunner(self.monitor),
            self.workload, self.inputs, warmup=0, repetitions=2, **kwargs,
        )

    def test_success_records_only_measured_runs(self):
        result = self.run_benchmark()
        self.assertEqual(result.correctness.status, "passed")
        self.assertEqual(result.correctness.failed_elements, 0)
        self.assertEqual(len(self.monitor.latency_samples_ms), 2)
        self.assertEqual(result.to_dict()["correctness"]["reference"], "torch_eager")

    def test_failure_records_stats_and_no_runtime(self):
        class Wrong(TorchEagerRunner):
            def _run(self, executable, inputs):
                return super()._run(executable, inputs) + 1

        result = self.run_benchmark(Wrong(self.monitor))
        self.assertEqual(result.status, "correctness_failed")
        self.assertEqual(result.correctness.failed_elements, 4)
        self.assertEqual(result.latency_samples_ms, [])
        self.assertIsNone(result.median_latency_ms)

    def test_reference_error_is_distinct(self):
        with patch.object(TorchEagerReference, "run", side_effect=RuntimeError("reference unavailable")):
            result = self.run_benchmark()
        self.assertEqual(result.status, "error")
        self.assertEqual(result.correctness.status, "error")
        self.assertEqual(result.correctness.error_stage, "correctness_reference")
        self.assertEqual(result.latency_samples_ms, [])

    def test_candidate_error_is_distinct(self):
        class Broken(TorchEagerRunner):
            def _run(self, executable, inputs):
                raise RuntimeError("execution failed")

        result = self.run_benchmark(Broken(self.monitor))
        self.assertEqual(result.status, "error")
        self.assertEqual(result.correctness.error_stage, "correctness_candidate")

    def test_disabled_check_does_not_execute_reference(self):
        with patch.object(TorchEagerReference, "run") as reference:
            result = self.run_benchmark(correctness=CorrectnessConfig(enabled=False))
        reference.assert_not_called()
        self.assertEqual(result.status, "success")
        self.assertEqual(result.correctness.status, "skipped")
        self.assertEqual(len(result.latency_samples_ms), 2)


if __name__ == "__main__":
    unittest.main()
