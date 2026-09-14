"""Focused ConvBNReLU shape, numerical, and example checks."""

import unittest

import torch

from examples.utils import create_workload
from tobench.workloads import BaseWorkload, ConvBNReLU


class ConvBNReLUTests(unittest.TestCase):
    def test_shapes_metadata_and_validation(self):
        workload = ConvBNReLU(
            B=2, C_in=4, C_out=6, H=7, W=9,
            kernel_size=3, stride=2, padding=1, groups=2,
        )
        self.assertIsInstance(workload, BaseWorkload)
        self.assertEqual(
            workload.input_shapes,
            ((2, 4, 7, 9), (6, 2, 3, 3), (6,), (6,), (6,), (6,)),
        )
        self.assertEqual(workload.output_shape, (2, 6, 4, 5))
        self.assertFalse(workload.to_config()["batch_norm_training"])
        with self.assertRaisesRegex(ValueError, "divisible"):
            ConvBNReLU(C_in=3, C_out=4, groups=2)
        with self.assertRaisesRegex(ValueError, "empty output"):
            ConvBNReLU(H=2, W=2, kernel_size=5, padding=0)

    def test_inference_batch_norm_and_relu(self):
        workload = ConvBNReLU(
            B=1, C_in=1, C_out=1, H=2, W=2,
            kernel_size=1, padding=0, eps=1e-5,
        )
        X = torch.tensor([[[[-1.0, 1.0], [2.0, 3.0]]]], dtype=torch.float16)
        weight = torch.tensor([[[[2.0]]]], dtype=torch.float16)
        gamma = torch.tensor([3.0], dtype=torch.float16)
        beta = torch.tensor([1.0], dtype=torch.float16)
        mean = torch.tensor([2.0], dtype=torch.float16)
        variance = torch.tensor([4.0], dtype=torch.float16)
        output = workload(X, weight, gamma, beta, mean, variance)
        convolved = torch.nn.functional.conv2d(X, weight)
        expected = torch.relu(
            ((convolved.float() - 2.0) * torch.rsqrt(torch.tensor(4.0 + 1e-5)) * 3.0 + 1.0)
            .to(torch.float16)
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

        example, inputs = create_workload(
            "conv_bn_relu",
            parameters={"B": 1, "C_in": 2, "C_out": 3, "H": 4, "W": 5},
        )
        self.assertEqual(example.output_shape, (1, 3, 4, 5))
        self.assertTrue(torch.all(inputs[-1] > 0))


if __name__ == "__main__":
    unittest.main()
