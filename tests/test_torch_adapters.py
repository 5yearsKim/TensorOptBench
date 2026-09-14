"""Preparation, runner lifecycle, and actual Inductor compilation on CPU."""

import json
import unittest
from unittest.mock import patch

import torch

from tobench.backends.torch import TorchAdapter, TorchEagerRunner, TorchInductorRunner
from tobench.benchmarking import Benchmarker
from tobench.workloads import GEMM


class TorchBackendTests(unittest.TestCase):
    def setUp(self):
        self.workload = GEMM(2, 2, 3, B=1)
        self.inputs = (
            torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float16),
            torch.tensor([[[7, 8], [9, 10], [11, 12]]], dtype=torch.float16),
        )

    def test_prepare_validates_inputs(self):
        for inputs in ([], self.inputs[0], [1]):
            with self.subTest(inputs=inputs), self.assertRaises(TypeError):
                TorchAdapter().prepare(self.workload, inputs)

    def test_prepared_results_are_independent(self):
        adapter = TorchAdapter()
        first = adapter.prepare(self.workload, self.inputs)
        second = adapter.prepare(
            GEMM(1, 1, 1, B=1), (torch.ones(1, 1, 1), torch.ones(1, 1, 1))
        )
        self.assertIs(first.inputs[0], self.inputs[0])
        self.assertIsNot(first, second)
        self.assertEqual(first.workload.M, 2)
        self.assertFalse(hasattr(adapter, "build"))
        self.assertFalse(hasattr(adapter, "run"))

    def test_eager_without_benchmarker(self):
        inputs = tuple(x.clone().requires_grad_() for x in self.inputs)
        prepared = TorchAdapter().prepare(self.workload, inputs)
        runner = TorchEagerRunner()
        with patch("torch.compile") as compile_mock:
            executable = runner.build(prepared)
            output = runner.run(executable)
        compile_mock.assert_not_called()
        self.assertFalse(output.requires_grad)
        self.assertIsNone(runner.benchmarker)
        torch.testing.assert_close(
            output, torch.tensor([[[58, 64], [139, 154]]], dtype=torch.float16)
        )
        with self.assertRaisesRegex(RuntimeError, "Benchmarker"):
            runner.benchmark_run(executable)

    def test_lazy_compile_is_not_a_runtime_sample(self):
        monitor = Benchmarker()
        runner = TorchInductorRunner(benchmarker=monitor)
        prepared = TorchAdapter().prepare(self.workload, self.inputs)
        calls = []

        def compiled(A, B):
            calls.append(torch.is_inference_mode_enabled())
            return A @ B

        with patch("torch.compile", return_value=compiled) as compile_mock:
            executable = runner.build(prepared)
            self.assertEqual(calls, [True])
            self.assertEqual(monitor.latency_samples_ms, [])
            self.assertIsNotNone(monitor.optimization_time_seconds)
            runner.run(executable)
            self.assertEqual(len(monitor.latency_samples_ms), 1)
            self.assertEqual(len(calls), 2)
            runner.run(executable, measure=False)
            self.assertEqual(len(monitor.latency_samples_ms), 1)
            runner.benchmark_run(executable, warmup=2, repetitions=3)
            self.assertEqual(len(monitor.latency_samples_ms), 3)
            self.assertEqual(len(calls), 8)
            runner.build(prepared)
            self.assertEqual(monitor.latency_samples_ms, [])
        self.assertEqual(compile_mock.call_count, 2)

    def test_metadata(self):
        prepared = TorchAdapter().prepare(self.workload, self.inputs)
        for runner, name in ((TorchEagerRunner(), "torch_eager"), (TorchInductorRunner(), "torchinductor")):
            metadata = json.loads(json.dumps(runner.collect_metadata(prepared)))
            self.assertEqual(metadata["backend"], name)
            self.assertFalse(metadata["budget_enforced"])

    def test_real_inductor_matches_eager(self):
        for dtype in ("float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                workload = GEMM(2, 2, 3, B=1, dtype=dtype)
                inputs = tuple(x.to(getattr(torch, dtype)) for x in self.inputs)
                prepared = TorchAdapter().prepare(workload, inputs)
                eager, compiled = TorchEagerRunner(), TorchInductorRunner()
                reference, executable = eager.build(prepared), compiled.build(prepared)
                for scale in (1, 2):
                    values = (inputs[0] * scale, inputs[1])
                    torch.testing.assert_close(
                        compiled.run(executable, values), eager.run(reference, values), rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
