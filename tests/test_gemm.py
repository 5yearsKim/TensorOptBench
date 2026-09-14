"""GEMM configuration and execution checks on CPU."""

import unittest
import torch
from torch import nn

from tobench.workloads.base_workload import BaseWorkload
from tobench.workloads import GEMM


class GEMMTests(unittest.TestCase):
    def test_defaults(self):
        workload = GEMM(M=32, N=64, K=16)
        self.assertEqual((workload.B, workload.M, workload.N, workload.K), (8, 32, 64, 16))
        self.assertEqual(workload.dtype, "float16")
        self.assertEqual(workload.seed, 0)

    def test_bfloat16_and_explicit_seed(self):
        workload = GEMM(M=1, N=1, K=1, dtype="bfloat16", seed=42)
        self.assertEqual(workload.dtype, "bfloat16")
        self.assertEqual(workload.seed, 42)

    def test_rectangular_matmul_specification(self):
        workload = GEMM(B=2, M=3, N=5, K=7)
        self.assertEqual(workload.op, "matmul")
        self.assertEqual(workload.input_shapes, ((2, 3, 7), (2, 7, 5)))
        self.assertEqual(workload.output_shape, (2, 3, 5))
        self.assertEqual(workload.flop_count, 420)

    def test_numerical_contract_for_both_dtypes(self):
        for dtype in ("float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                workload = GEMM(B=2, M=1, N=1, K=1, dtype=dtype)
                self.assertEqual(workload.layout, "row_major")
                self.assertEqual(workload.output_shape, (2, 1, 1))

    def test_invalid_dimensions(self):
        for name in ("B", "M", "N", "K"):
            for value in (0, -1, 1.5, "32", True, None):
                with self.subTest(name=name, value=value):
                    dimensions = {"B": 8, "M": 32, "N": 64, "K": 16, name: value}
                    error = ValueError if type(value) is int else TypeError
                    with self.assertRaisesRegex(error, name):
                        GEMM(**dimensions)

    def test_unsupported_dtype(self):
        for dtype in ("float32", "fp16", "", None):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(ValueError, "dtype"):
                    GEMM(M=32, N=64, K=16, dtype=dtype)

    def test_invalid_seed(self):
        for seed in (-1, 1.5, "42", True, None):
            with self.subTest(seed=seed):
                error = ValueError if type(seed) is int else TypeError
                with self.assertRaisesRegex(error, "seed"):
                    GEMM(M=32, N=64, K=16, seed=seed)

    def test_is_pytorch_workload(self):
        workload = GEMM(M=2, N=2, K=3)
        self.assertIsInstance(workload, BaseWorkload)
        self.assertIsInstance(workload, nn.Module)

    def test_forward_known_rectangular_product(self):
        for name, dtype in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
            with self.subTest(dtype=dtype):
                workload = GEMM(B=2, M=2, N=2, K=3, dtype=name)
                A = torch.tensor(
                    [[[1, 2, 3], [4, 5, 6]], [[2, 4, 6], [1, 3, 5]]], dtype=dtype
                )
                B = torch.tensor(
                    [[[7, 8], [9, 10], [11, 12]], [[1, 2], [3, 4], [5, 6]]],
                    dtype=dtype,
                )
                before_A, before_B = A.clone(), B.clone()
                output = workload(A, B)
                expected = torch.tensor(
                    [[[58, 64], [139, 154]], [[44, 56], [35, 44]]], dtype=dtype
                )
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                self.assertEqual(tuple(output.shape), workload.output_shape)
                self.assertEqual(output.dtype, dtype)
                self.assertTrue(output.is_contiguous())
                torch.testing.assert_close(A, before_A, rtol=0, atol=0)
                torch.testing.assert_close(B, before_B, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
