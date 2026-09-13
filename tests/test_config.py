"""JSON configuration validation and configured benchmark execution."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from pydantic import ValidationError

from examples.benchmark_gemm import parse_config
from tobench.core.config import BenchmarkConfig
from tobench.core.runner import benchmark_from_config


class ConfigTests(unittest.TestCase):
    def test_json_load_and_cli_overrides(self):
        data = {
            "workload": {"M": 3, "N": 5, "K": 7, "dtype": "bfloat16", "seed": 42},
            "backend": "inductor",
            "runtime": {"warmup": 0, "repetitions": 3},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(data))
            config = BenchmarkConfig.from_json(path)
            overridden = parse_config(["--config", str(path), "--backend", "eager", "--m", "9"])
        self.assertEqual(config.workload.M, 3)
        self.assertEqual(config.runtime.warmup, 0)
        self.assertEqual(config.budget.max_time_seconds, 60)
        self.assertEqual(overridden.backend, "eager")
        self.assertEqual(overridden.workload.M, 9)
        self.assertEqual(overridden.workload.N, 5)
        self.assertEqual(overridden.workload.dtype, "bfloat16")
        self.assertEqual(overridden.runtime.repetitions, 3)

    def test_invalid_configuration(self):
        for data in (
            {"backend": "unknown"}, {"device": "mps"},
            {"workload": {"name": "unknown"}}, {"workload": {"M": 0}},
            {"workload": {"M": "128"}}, {"workload": {"M": True}},
            {"runtime": {"warmup": -1}}, {"runtime": {"repetitions": 0}},
            {"runtime": {"repetitons": 10}}, {"budget": {"max_time_seconds": -1}},
        ):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                BenchmarkConfig.model_validate_json(json.dumps(data))
        with self.assertRaises(ValidationError):
            BenchmarkConfig.model_validate_json('{"backend":')

    def test_default_cli_and_legacy_flags(self):
        self.assertEqual(parse_config([]), BenchmarkConfig())
        config = parse_config(["--m", "8", "--repetitions", "2", "--budget-seconds", "1"])
        self.assertEqual(config.workload.M, 8)
        self.assertEqual(config.runtime.repetitions, 2)
        self.assertEqual(config.budget.max_time_seconds, 1)

    def test_configured_run_preserves_rng_and_records_resolved_config(self):
        config = BenchmarkConfig.model_validate({
            "workload": {"M": 2, "N": 3, "K": 4},
            "runtime": {"warmup": 0, "repetitions": 1},
        })
        before = torch.random.get_rng_state().clone()
        result = benchmark_from_config(config)
        self.assertEqual(result.status, "success", result.error)
        self.assertEqual(result.workload["N"], 3)
        self.assertEqual(result.configuration["experiment"], config.model_dump(mode="json"))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
