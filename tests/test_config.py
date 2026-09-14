"""JSON configuration validation and configured benchmark execution."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from pydantic import ValidationError

from examples.utils import create_workload, parse_config
from tobench.core.config import BenchmarkConfig


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
            overridden = parse_config("tvm", ["--config", str(path), "--m", "9"])
        self.assertEqual(config.workload.M, 3)
        self.assertEqual(config.runtime.warmup, 0)
        self.assertEqual(config.budget.max_time_seconds, 60)
        self.assertEqual(overridden.backend, "tvm")
        self.assertEqual(overridden.workload.M, 9)
        self.assertEqual(overridden.workload.N, 5)
        self.assertEqual(overridden.workload.dtype, "bfloat16")
        self.assertEqual(overridden.runtime.repetitions, 3)

    def test_invalid_configuration(self):
        for data in (
            {"backend": "unknown"}, {"device": "mps"},
            {"workload": {"name": "unknown"}}, {"workload": {"M": 0}},
            {"workload": {"name": "softmax", "B": 0}},
            {"workload": {"name": "attention", "H": 0}},
            {"workload": {"name": "conv_bn_relu", "C_in": 3, "groups": 2}},
            {"workload": {"M": "128"}}, {"workload": {"M": True}},
            {"runtime": {"warmup": -1}}, {"runtime": {"repetitions": 0}},
            {"runtime": {"repetitons": 10}}, {"budget": {"max_time_seconds": -1}},
            {"iree": {"optimization_level": "O4"}},
        ):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                BenchmarkConfig.model_validate_json(json.dumps(data))
        with self.assertRaises(ValidationError):
            BenchmarkConfig.model_validate_json('{"backend":')

    def test_default_cli_and_legacy_flags(self):
        self.assertEqual(parse_config("inductor", []), BenchmarkConfig(backend="inductor"))
        config = parse_config("inductor", ["--m", "8", "--repetitions", "2", "--budget-seconds", "1"])
        self.assertEqual(config.workload.M, 8)
        self.assertEqual(config.runtime.repetitions, 2)
        self.assertEqual(config.budget.max_time_seconds, 1)
        self.assertEqual(parse_config("tensorrt", []).backend, "tensorrt")
        self.assertEqual(parse_config("iree", []).backend, "iree")
        self.assertEqual(
            parse_config("iree", ["--iree-opt-level", "O2"]).iree.optimization_level,
            "O2",
        )

    def test_workload_factory_preserves_rng_and_uses_parameters(self):
        before = torch.random.get_rng_state().clone()
        workload, inputs = create_workload(
            "gemm", parameters={"B": 2, "M": 2, "N": 3, "K": 4}
        )
        self.assertEqual(workload.N, 3)
        self.assertEqual(tuple(inputs[0].shape), (2, 2, 4))
        self.assertEqual(tuple(inputs[1].shape), (2, 4, 3))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        _, repeated = create_workload(
            "gemm", parameters={"B": 2, "M": 2, "N": 3, "K": 4}
        )
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(inputs, repeated)))
        with self.assertRaisesRegex(ValueError, "Unknown workload"):
            create_workload("unknown")

    def test_work_type_argument_aliases(self):
        for flag in ("--work-type", "--work_type"):
            for workload in (
                "gemm", "rmsnorm_linear", "softmax", "attention", "conv_bn_relu",
            ):
                with self.subTest(flag=flag, workload=workload):
                    config = parse_config("inductor", [flag, workload])
                    self.assertEqual(config.workload.name, workload)

    def test_batched_workload_cli_dimensions(self):
        gemm = parse_config("inductor", ["--batch-size", "3", "--m", "4"])
        self.assertEqual((gemm.workload.B, gemm.workload.M), (3, 4))
        attention = parse_config(
            "inductor",
            [
                "--work-type", "attention", "--b", "2", "--heads", "3",
                "--seq-len", "5", "--head-dim", "7",
            ],
        )
        self.assertEqual(
            (attention.workload.B, attention.workload.H, attention.workload.S, attention.workload.D),
            (2, 3, 5, 7),
        )
        convolution = parse_config(
            "inductor",
            [
                "--work-type", "conv_bn_relu", "--batch-size", "2",
                "--in-channels", "4", "--out-channels", "6",
                "--height", "8", "--width", "10", "--kernel-size", "3",
                "--stride", "2", "--padding", "1", "--groups", "2",
            ],
        )
        self.assertEqual(
            (
                convolution.workload.B, convolution.workload.C_in,
                convolution.workload.C_out, convolution.workload.H,
                convolution.workload.W, convolution.workload.groups,
            ),
            (2, 4, 6, 8, 10, 2),
        )

    def test_new_workload_factories(self):
        softmax, (X,) = create_workload(
            "softmax", parameters={"B": 2, "M": 3, "N": 4}
        )
        self.assertEqual(softmax.output_shape, (2, 3, 4))
        self.assertEqual(tuple(X.shape), softmax.output_shape)
        attention, inputs = create_workload(
            "attention", parameters={"B": 1, "H": 2, "S": 3, "D": 4}
        )
        self.assertEqual(attention.output_shape, (1, 2, 3, 4))
        self.assertTrue(all(tuple(value.shape) == (1, 2, 3, 4) for value in inputs))


if __name__ == "__main__":
    unittest.main()
