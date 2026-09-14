"""Explicit batched multi-head attention semantics and execution."""

import unittest

import torch

from examples.utils import create_workload
from tobench.backends.torch import TorchAdapter, TorchInductorRunner
from tobench.workloads import Attention, BaseWorkload


class AttentionTests(unittest.TestCase):
    def test_shapes_metadata_and_validation(self):
        workload = Attention(B=2, H=3, S=4, D=5)
        self.assertIsInstance(workload, BaseWorkload)
        self.assertEqual(workload.input_shapes, ((2, 3, 4, 5),) * 3)
        self.assertEqual(workload.output_shape, (2, 3, 4, 5))
        self.assertEqual(workload.flop_count, 2 * 3 * 4 * 4 * (4 * 5 + 6))
        self.assertFalse(workload.to_config()["causal"])
        for name in ("B", "H", "S", "D"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                Attention(**{"B": 2, "H": 3, "S": 4, "D": 5, name: 0})

    def test_uniform_scores_average_values_for_both_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                name = str(dtype).removeprefix("torch.")
                workload = Attention(B=1, H=1, S=2, D=2, dtype=name)
                Q = torch.zeros(1, 1, 2, 2, dtype=dtype)
                K = torch.ones_like(Q)
                V = torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]], dtype=dtype)
                expected = torch.tensor([[[[4.0, 6.0], [4.0, 6.0]]]], dtype=dtype)
                torch.testing.assert_close(workload(Q, K, V), expected, rtol=0, atol=0)

    def test_inductor_matches_eager(self):
        workload, inputs = create_workload(
            "attention", parameters={"B": 1, "H": 2, "S": 4, "D": 8}
        )
        prepared = TorchAdapter().prepare(workload, inputs)
        runner = TorchInductorRunner()
        torch.testing.assert_close(
            runner.run(runner.build(prepared)), workload(*inputs), rtol=1e-3, atol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
