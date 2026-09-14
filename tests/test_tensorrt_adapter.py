"""TensorRT backend contracts that can be tested without CUDA or TensorRT."""

import json
import unittest
from unittest.mock import patch

import torch

from tobench.backends.tensorrt import TensorRTAdapter, TensorRTPreparedInput, TensorRTRunner
from tobench.backends.tensorrt import runner as runner_module
from tobench.benchmarking import Benchmarker
from tobench.workloads import GEMM


class _FakeTorchTensorRT:
    __version__ = "test-version"

    class dynamo:
        calls = []

        @classmethod
        def compile(cls, exported_program, **options):
            cls.calls.append((exported_program, options))
            return lambda A, B: A @ B


class TensorRTBackendTests(unittest.TestCase):
    def setUp(self):
        self.workload = GEMM(2, 2, 3)
        self.inputs = (
            torch.ones(2, 3, dtype=torch.float16),
            torch.ones(3, 2, dtype=torch.float16),
        )
        _FakeTorchTensorRT.dynamo.calls.clear()

    def test_adapter_requires_cuda_inputs(self):
        for inputs in ([], [1], list(self.inputs)):
            with self.subTest(inputs=inputs), self.assertRaises((TypeError, ValueError)):
                TensorRTAdapter().prepare(self.workload, inputs)

    def test_build_compiles_the_exported_graph_without_fallback(self):
        prepared = TensorRTPreparedInput(object(), self.inputs)
        runner = TensorRTRunner(benchmarker=Benchmarker())
        with (
            patch.object(runner_module, "torch_tensorrt", _FakeTorchTensorRT),
            patch.dict("os.environ", {"TOBENCH_TENSORRT_TIMING_CACHE_PATH": "/tmp/test-timing.cache"}),
        ):
            executable = runner.build(prepared)
            output = runner.run(executable, measure=False)
            metadata = runner.collect_metadata(prepared)

        torch.testing.assert_close(output, self.inputs[0] @ self.inputs[1])
        self.assertEqual(len(_FakeTorchTensorRT.dynamo.calls), 1)
        exported, options = _FakeTorchTensorRT.dynamo.calls[0]
        self.assertIs(exported, prepared.exported_program)
        self.assertIs(options["arg_inputs"], prepared.inputs)
        self.assertTrue(options["require_full_compilation"])
        self.assertTrue(options["pass_through_build_failures"])
        self.assertEqual(options["min_block_size"], 1)
        self.assertFalse(options["cache_built_engines"])
        self.assertFalse(options["reuse_cached_engines"])
        self.assertEqual(options["timing_cache_path"], "/tmp/test-timing.cache")
        self.assertEqual(metadata["backend"], "tensorrt")
        self.assertEqual(metadata["backend_version"], "test-version")
        self.assertEqual(metadata["configuration"]["timing_cache"], "fresh_temporary")

    def test_runtime_inputs_must_match_build_signature(self):
        prepared = TensorRTPreparedInput(object(), self.inputs)
        with patch.object(runner_module, "torch_tensorrt", _FakeTorchTensorRT):
            runner = TensorRTRunner()
            executable = runner.build(prepared)
        with self.assertRaisesRegex(ValueError, "match"):
            runner.run(executable, (self.inputs[0][:1], self.inputs[1]))

    def test_missing_optional_dependency_has_install_hint(self):
        prepared = TensorRTPreparedInput(object(), self.inputs)
        with patch.object(runner_module, "torch_tensorrt", None):
            with self.assertRaisesRegex(ImportError, "torch-tensorrt and tensorrt"):
                TensorRTRunner().build(prepared)

    def test_options_are_validated_and_metadata_is_json_safe(self):
        for kwargs in (
            {"min_block_size": 0},
            {"min_block_size": True},
            {"optimization_level": -1},
            {"optimization_level": 6},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                TensorRTRunner(**kwargs)
        prepared = TensorRTPreparedInput(object(), self.inputs)
        json.dumps(TensorRTRunner().collect_metadata(prepared))


if __name__ == "__main__":
    unittest.main()
