"""Adapter lifecycle and actual TorchInductor execution checks on CPU."""

import json
import unittest
from unittest.mock import patch

import torch

from tobench.backends.torch import TorchEagerAdapter, TorchInductorAdapter
from tobench.workloads import GEMM


class TorchAdapterTests(unittest.TestCase):
    def setUp(self):
        self.workload = GEMM(M=2, N=2, K=3)
        self.inputs = (
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float16),
            torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.float16),
        )

    def test_build_requires_prepare(self):
        for adapter in (TorchEagerAdapter(), TorchInductorAdapter()):
            with self.subTest(adapter=type(adapter).__name__):
                with self.assertRaisesRegex(RuntimeError, "prepare"):
                    adapter.build(budget=None)

    def test_prepare_validates_inputs(self):
        adapter = TorchEagerAdapter()
        for inputs in ([], self.inputs[0], [1]):
            with self.subTest(inputs=inputs):
                with self.assertRaises(TypeError):
                    adapter.prepare(self.workload, inputs)

    def test_eager_output_and_inference_mode(self):
        adapter = TorchEagerAdapter()
        inputs = tuple(value.clone().requires_grad_() for value in self.inputs)
        adapter.prepare(self.workload, inputs)
        self.assertFalse(self.workload.training)
        with patch("torch.compile") as compile_mock:
            executable = adapter.build(budget=None)
            output = adapter.run(executable, inputs)
        compile_mock.assert_not_called()
        torch.testing.assert_close(
            output, torch.tensor([[58, 64], [139, 154]], dtype=torch.float16),
            rtol=0, atol=0,
        )
        self.assertFalse(output.requires_grad)

    def test_inductor_completes_first_invocation_in_build(self):
        adapter = TorchInductorAdapter()
        adapter.prepare(self.workload, self.inputs)
        calls = []

        def compiled(A, B):
            calls.append(torch.is_inference_mode_enabled())
            return A @ B

        with patch("torch.compile", return_value=compiled) as compile_mock:
            executable = adapter.build(budget=None)
            self.assertEqual(calls, [True])
            adapter.run(executable, self.inputs)
        self.assertEqual(calls, [True, True])
        compile_mock.assert_called_once_with(
            self.workload, backend="inductor", mode="default",
            fullgraph=True, dynamic=False,
        )

    def test_metadata_is_serializable_and_reports_budget_limitation(self):
        for adapter, name in (
            (TorchEagerAdapter(), "torch_eager"),
            (TorchInductorAdapter(), "torchinductor"),
        ):
            with self.subTest(backend=name):
                metadata = json.loads(json.dumps(adapter.collect_metadata()))
                self.assertEqual(metadata["backend"], name)
                self.assertEqual(metadata["backend_version"], str(torch.__version__))
                self.assertFalse(metadata["budget_enforced"])

    def test_real_inductor_matches_eager(self):
        for dtype in ("float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                workload = GEMM(M=2, N=2, K=3, dtype=dtype)
                inputs = tuple(value.to(getattr(torch, dtype)) for value in self.inputs)
                eager = TorchEagerAdapter()
                compiled = TorchInductorAdapter()
                eager.prepare(workload, inputs)
                compiled.prepare(workload, inputs)
                reference = eager.build(budget=None)
                executable = compiled.build(budget=None)
                for scale in (1, 2):
                    # New values with the same tensor specifications.
                    run_inputs = (inputs[0] * scale, inputs[1])
                    torch.testing.assert_close(
                        compiled.run(executable, run_inputs),
                        eager.run(reference, run_inputs), rtol=0, atol=0,
                    )


if __name__ == "__main__":
    unittest.main()
