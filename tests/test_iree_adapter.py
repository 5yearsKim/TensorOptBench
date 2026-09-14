"""IREE adapter contracts and optional real compiler integration tests."""

import importlib.util
import json
import unittest

import torch

from tobench.workloads import Attention, GEMM, RMSNormLinear, Softmax

try:
    IREE_AVAILABLE = all(
        importlib.util.find_spec(name) is not None
        for name in ("iree.turbine.aot", "iree.compiler", "iree.runtime")
    )
except ModuleNotFoundError:
    IREE_AVAILABLE = False
if IREE_AVAILABLE:
    from tobench.backends.iree import IREEAdapter, IREERunner


@unittest.skipUnless(IREE_AVAILABLE, "optional IREE dependencies not installed")
class IREEAdapterContractTests(unittest.TestCase):
    def test_build_requires_prepared_result(self):
        with self.assertRaisesRegex(TypeError, "IREEPreparedInput"):
            IREERunner().build(None)

    def test_invalid_inputs(self):
        workload = GEMM(2, 2, 3, B=1)
        for inputs in ([], [1], [torch.ones(3, 2).T]):
            with self.subTest(inputs=inputs), self.assertRaises((TypeError, ValueError)):
                IREEAdapter().prepare(workload, inputs)

    def test_options_are_validated(self):
        with self.assertRaisesRegex(ValueError, "optimization_level"):
            IREERunner(optimization_level="fast")


@unittest.skipUnless(IREE_AVAILABLE, "optional IREE dependencies not installed")
class IREEIntegrationTests(unittest.TestCase):
    def test_gemm_matches_reference_for_both_dtypes(self):
        for dtype in ("float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                workload = GEMM(M=3, N=5, K=7, dtype=dtype)
                generator = torch.Generator().manual_seed(42)
                inputs = tuple(
                    torch.randn(shape, dtype=getattr(torch, dtype), generator=generator)
                    for shape in workload.input_shapes
                )
                prepared = IREEAdapter().prepare(workload, inputs)
                runner = IREERunner()
                executable = runner.build(prepared)
                for scale in (1, 2):
                    run_inputs = (inputs[0] * scale, inputs[1])
                    output = runner.run(executable, run_inputs)
                    expected = workload(*run_inputs)
                    torch.testing.assert_close(output, expected)
                    self.assertEqual(output.dtype, getattr(torch, dtype))
                with self.assertRaisesRegex(ValueError, "match"):
                    runner.run(executable, (inputs[0][:, :1], inputs[1]))

                metadata = json.loads(json.dumps(runner.collect_metadata(prepared)))
                self.assertEqual(metadata["backend"], "iree")
                self.assertEqual(
                    metadata["configuration"]["compiler_target_backend"], "llvm-cpu"
                )
                self.assertEqual(metadata["configuration"]["tensor_interop"], "dlpack")
                self.assertTrue(metadata["configuration"]["target_available"])
                self.assertTrue(metadata["configuration"]["runtime_driver_available"])

    def test_small_rmsnorm_linear_matches_eager(self):
        workload = RMSNormLinear(M=2, N=3, K=4)
        generator = torch.Generator().manual_seed(7)
        inputs = tuple(
            torch.randn(shape, dtype=torch.float16, generator=generator)
            for shape in workload.input_shapes
        )
        prepared = IREEAdapter().prepare(workload, inputs)
        runner = IREERunner()
        output = runner.run(runner.build(prepared))
        torch.testing.assert_close(output, workload(*inputs), rtol=1e-3, atol=1e-3)

    def test_softmax_and_attention_match_eager(self):
        workloads = (
            Softmax(B=2, M=3, N=7),
            Attention(B=1, H=2, S=4, D=8),
        )
        for workload in workloads:
            with self.subTest(workload=workload.__class__.__name__):
                generator = torch.Generator().manual_seed(7)
                inputs = tuple(
                    torch.randn(shape, dtype=torch.float16, generator=generator)
                    for shape in workload.input_shapes
                )
                prepared = IREEAdapter().prepare(workload, inputs)
                runner = IREERunner()
                output = runner.run(runner.build(prepared))
                torch.testing.assert_close(
                    output, workload(*inputs), rtol=1e-3, atol=1e-3
                )


if __name__ == "__main__":
    unittest.main()
