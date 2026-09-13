"""Optional TVM integration tests, using LLVM on CPU."""

import importlib.util
import json
import unittest
from unittest.mock import patch

import torch

from tobench.backends.tvm import TVMAdapter, TVMRunner
from tobench.workloads import GEMM


class TVMAdapterContractTests(unittest.TestCase):
    def test_build_requires_prepared_result(self):
        with self.assertRaisesRegex(TypeError, "TVMPreparedInput"):
            TVMRunner().build(None)

    def test_missing_dependency_has_install_hint(self):
        error = ModuleNotFoundError("No module named 'tvm'", name="tvm")
        inputs = (torch.ones(2, 3), torch.ones(3, 2))
        with patch("tobench.backends.tvm.adapter.import_module", side_effect=error):
            with self.assertRaisesRegex(ImportError, r"\[tvm\]"):
                TVMAdapter().prepare(GEMM(2, 2, 3), inputs)

    def test_invalid_inputs(self):
        for inputs in ([], [1], [torch.ones(3, 2).T]):
            with self.subTest(inputs=inputs):
                with self.assertRaises((TypeError, ValueError)):
                    TVMAdapter().prepare(GEMM(2, 2, 3), inputs)


@unittest.skipUnless(importlib.util.find_spec("tvm"), "optional TVM dependency not installed")
class TVMAdapterIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tvm

        if not tvm.runtime.enabled("llvm"):
            raise unittest.SkipTest("TVM was built without LLVM")

    def test_gemm_matches_reference_for_both_dtypes(self):
        for dtype in ("float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                workload = GEMM(M=3, N=5, K=7, dtype=dtype)
                generator = torch.Generator().manual_seed(42)
                inputs = tuple(
                    torch.randn(s, dtype=getattr(torch, dtype), generator=generator)
                    for s in workload.input_shapes
                )
                adapter = TVMAdapter(target="llvm")
                prepared = adapter.prepare(workload, inputs)
                self.assertIsNotNone(prepared.mod)
                runner = TVMRunner()
                executable = runner.build(prepared)
                for scale in (1, 2):
                    run_inputs = (inputs[0] * scale, inputs[1])
                    output = runner.run(executable, run_inputs)
                    # Independent, high-precision reference on the quantized inputs.
                    expected = (run_inputs[0].double() @ run_inputs[1].double()).to(
                        getattr(torch, dtype)
                    )
                    torch.testing.assert_close(output, expected)
                    self.assertEqual(output.dtype, getattr(torch, dtype))
                with self.assertRaisesRegex(ValueError, "match"):
                    runner.run(executable, (inputs[0][:1], inputs[1]))
                metadata = json.loads(json.dumps(runner.collect_metadata(prepared)))
                self.assertEqual(metadata["backend"], "tvm")
                self.assertFalse(metadata["budget_enforced"])
                self.assertFalse(metadata["configuration"]["autotuning"])

    def test_target_must_match_input_device(self):
        with self.assertRaisesRegex(ValueError, "llvm target"):
            TVMAdapter(target="cuda").prepare(
                GEMM(2, 2, 3), (torch.ones(2, 3), torch.ones(3, 2))
            )


if __name__ == "__main__":
    unittest.main()
