"""PyTorch/XLA adapter contracts using a dependency-isolated PJRT stand-in."""

import json
import os
import unittest
from unittest.mock import patch

import torch

from tobench.backends.xla import XLAAdapter, XLARunner
from tobench.backends.xla import adapter as adapter_module
from tobench.backends.xla import runner as runner_module
from tobench.workloads import GEMM


class _FakeXLA:
    __version__ = "test"

    def __init__(self):
        self.sync_calls = []

    def device(self):
        return torch.device("cpu")

    def sync(self, *, wait=False):
        self.sync_calls.append(wait)


class _FakeRuntime:
    def __init__(self):
        self.device_type = None
        self.cache = None

    def set_device_type(self, device_type):
        self.device_type = device_type

    def initialize_cache(self, path, *, readonly):
        self.cache = (path, readonly)


class XLABackendTests(unittest.TestCase):
    def setUp(self):
        self.workload = GEMM(M=2, N=4, K=3, B=1)
        self.inputs = (
            torch.randn(1, 2, 3, dtype=torch.float16),
            torch.randn(1, 3, 4, dtype=torch.float16),
        )

    def test_options_and_inputs_are_validated(self):
        with self.assertRaisesRegex(ValueError, "device_type"):
            XLAAdapter("gpu")
        for inputs in ([], [1], [torch.ones(3, 2).T]):
            with self.subTest(inputs=inputs), self.assertRaises((TypeError, ValueError)):
                XLAAdapter().prepare(self.workload, inputs)

    def test_missing_optional_dependency_has_install_hint(self):
        missing = ModuleNotFoundError("No module named 'torch_xla'", name="torch_xla")
        with (
            patch.object(adapter_module, "torch_xla", None),
            patch.object(adapter_module, "xr", None),
            patch.object(adapter_module, "_IMPORT_ERROR", missing),
            self.assertRaisesRegex(ImportError, "torch==2.9.0 torch-xla==2.9.0"),
        ):
            XLAAdapter().prepare(self.workload, self.inputs)

    def test_prepare_build_run_and_metadata(self):
        fake_xla = _FakeXLA()
        fake_runtime = _FakeRuntime()
        with (
            patch.object(adapter_module, "torch_xla", fake_xla),
            patch.object(adapter_module, "xr", fake_runtime),
            patch.dict(os.environ, {"TOBENCH_XLA_CACHE_PATH": "/tmp/xla-test-cache"}),
        ):
            prepared = XLAAdapter("cpu").prepare(self.workload, self.inputs)
        self.assertEqual(fake_runtime.device_type, "CPU")
        self.assertEqual(fake_runtime.cache, ("/tmp/xla-test-cache", False))
        self.assertEqual(prepared.cache_policy, "fresh_temporary")

        compile_calls = []

        def compile_module(module, **options):
            compile_calls.append(options)
            return module

        runner = XLARunner()
        with (
            patch.object(runner_module, "torch_xla", fake_xla),
            patch.object(runner_module.torch, "compile", side_effect=compile_module),
        ):
            executable = runner.build(prepared)
            output = runner.run(executable, (self.inputs[0] * 2, self.inputs[1]))
            expected = self.workload(self.inputs[0] * 2, self.inputs[1])
            torch.testing.assert_close(output, expected)
            with self.assertRaisesRegex(ValueError, "match"):
                runner.run(executable, (self.inputs[0][:, :1], self.inputs[1]))
            metadata = json.loads(json.dumps(runner.collect_metadata(prepared)))

        self.assertEqual(
            compile_calls,
            [{"backend": "openxla", "fullgraph": True, "dynamic": False}],
        )
        self.assertEqual(metadata["backend"], "xla")
        self.assertEqual(metadata["configuration"]["device_type"], "CPU")
        self.assertTrue(metadata["configuration"]["inputs_resident_during_measurement"])
        self.assertTrue(all(fake_xla.sync_calls))

    def test_build_requires_prepared_result(self):
        with self.assertRaisesRegex(TypeError, "XLAPreparedInput"):
            XLARunner().build(None)


if __name__ == "__main__":
    unittest.main()
