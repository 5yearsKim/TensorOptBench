"""Compile GEMM with TVM and compare its output with eager PyTorch.

Install: pip install -e '.[tvm]'
Run: python examples/gemm_tvm.py --device cpu
"""

import argparse

import torch

from tobench.backends.torch import TorchEagerAdapter
from tobench.backends.tvm import TVMAdapter
from tobench.workloads import GEMM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    args = parser.parse_args()
    device = torch.device(args.device)
    workload = GEMM(M=32, N=64, K=16, dtype=args.dtype, seed=0)
    generator = torch.Generator(device=device).manual_seed(workload.seed)
    inputs = tuple(
        torch.randn(
            shape, dtype=getattr(torch, workload.dtype), device=device, generator=generator
        )
        for shape in workload.input_shapes
    )

    adapter = TVMAdapter()
    adapter.prepare(workload, inputs)
    print(f"Compiling GEMM with TVM on {device}...", flush=True)
    executable = adapter.build(budget=None)
    output = adapter.run(executable, inputs)

    eager = TorchEagerAdapter()
    eager.prepare(workload, inputs)
    reference = eager.run(eager.build(budget=None), inputs)
    torch.testing.assert_close(output, reference)
    print(f"Output: {tuple(output.shape)}, {output.dtype}")
    print("Correctness check against eager PyTorch passed.")
    print(adapter.collect_metadata())


if __name__ == "__main__":
    main()
