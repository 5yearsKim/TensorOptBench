"""RMSNorm -> linear semantics and backend execution."""

import importlib.util
import unittest

import torch
from pydantic import ValidationError

from examples.utils import create_workload, parse_config
from tobench.backends.torch import TorchAdapter, TorchInductorRunner
from tobench.core.config import BenchmarkConfig
from tobench.workloads import RMSNormLinear


class RMSNormLinearTests(unittest.TestCase):
    def test_requested_shapes_and_cli_defaults(self):
        workload = RMSNormLinear()
        self.assertEqual(workload.input_shapes, ((16, 4096), (1, 4096), (406, 4096)))
        self.assertEqual(workload.output_shape, (16, 406))
        config = parse_config("inductor", ["--work-type", "rmsnorm_linear"])
        self.assertEqual((config.workload.M, config.workload.N, config.workload.K), (16, 406, 4096))
        self.assertEqual(config.workload.eps, 1e-6)

    def test_fp64_reference_and_weight_orientation(self):
        for dtype in ("float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                workload, (X, G, W) = create_workload(
                    "rmsnorm_linear", parameters={"M": 2, "N": 3, "K": 8, "dtype": dtype}
                )
                x = X.double()
                normalized = x / torch.sqrt(x.square().mean(-1, keepdim=True) + workload.eps)
                normalized = (normalized * G.double()).to(X.dtype)
                expected = (normalized.double() @ W.double().T).to(X.dtype)
                output = workload(X, G, W)
                torch.testing.assert_close(output, expected)
                self.assertEqual(output.shape, (2, 3))
                self.assertEqual(output.dtype, X.dtype)

    def test_zero_input_is_finite(self):
        workload = RMSNormLinear(M=2, N=3, K=4)
        X = torch.zeros(2, 4, dtype=torch.float16)
        G = torch.ones(1, 4, dtype=torch.float16)
        W = torch.ones(3, 4, dtype=torch.float16)
        torch.testing.assert_close(workload(X, G, W), torch.zeros(2, 3, dtype=X.dtype))

    def test_invalid_parameters(self):
        for params in ({"M": 0}, {"N": True}, {"K": 1.5}, {"eps": 0},
                       {"eps": float("nan")}, {"eps": True}):
            with self.subTest(params=params), self.assertRaises((ValueError, TypeError)):
                RMSNormLinear(**params)
        with self.assertRaises(ValidationError):
            BenchmarkConfig.model_validate({"workload": {"name": "rmsnorm_linear", "eps": -1}})

    def test_inductor_matches_eager(self):
        for dtype in ("float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                workload, inputs = create_workload(
                    "rmsnorm_linear", parameters={"M": 2, "N": 3, "K": 8, "dtype": dtype}
                )
                prepared = TorchAdapter().prepare(workload, inputs)
                runner = TorchInductorRunner()
                actual = runner.run(runner.build(prepared))
                torch.testing.assert_close(actual, workload(*inputs))

    @unittest.skipUnless(importlib.util.find_spec("tvm"), "optional TVM not installed")
    def test_tvm_matches_eager(self):
        import tvm
        from tobench.backends.tvm import TVMAdapter, TVMRunner

        if not tvm.runtime.enabled("llvm"):
            self.skipTest("TVM lacks LLVM")
        for dtype in ("float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                workload, inputs = create_workload(
                    "rmsnorm_linear", parameters={"M": 2, "N": 3, "K": 8, "dtype": dtype}
                )
                prepared = TVMAdapter().prepare(workload, inputs)
                runner = TVMRunner()
                torch.testing.assert_close(runner.run(runner.build(prepared)), workload(*inputs))


if __name__ == "__main__":
    unittest.main()
