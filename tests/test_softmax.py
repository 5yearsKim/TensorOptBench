"""Batched softmax semantics, validation, and compiled execution."""

import unittest

import torch

from examples.utils import create_workload
from tobench.backends.torch import TorchAdapter, TorchInductorRunner
from tobench.workloads import BaseWorkload, Softmax


class SoftmaxTests(unittest.TestCase):
    def test_shapes_metadata_and_validation(self):
        workload = Softmax(B=2, M=3, N=4)
        self.assertIsInstance(workload, BaseWorkload)
        self.assertEqual(workload.input_shapes, ((2, 3, 4),))
        self.assertEqual(workload.output_shape, (2, 3, 4))
        self.assertEqual(workload.flop_count, 120)
        self.assertEqual(workload.to_config()["accumulation_dtype"], "float32")
        for name in ("B", "M", "N"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                Softmax(**{"B": 2, "M": 3, "N": 4, name: 0})

    def test_numerical_contract_for_both_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                workload = Softmax(B=2, M=1, N=2, dtype=str(dtype).removeprefix("torch."))
                X = torch.tensor([[[0.0, 0.0]], [[0.0, 2.0]]], dtype=dtype)
                output = workload(X)
                expected = torch.softmax(X.float(), dim=-1).to(dtype)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    output.float().sum(-1), torch.ones(2, 1), rtol=1e-3, atol=1e-3
                )

    def test_inductor_matches_eager(self):
        workload, inputs = create_workload(
            "softmax", parameters={"B": 2, "M": 3, "N": 8}
        )
        prepared = TorchAdapter().prepare(workload, inputs)
        runner = TorchInductorRunner()
        torch.testing.assert_close(runner.run(runner.build(prepared)), workload(*inputs))


if __name__ == "__main__":
    unittest.main()
