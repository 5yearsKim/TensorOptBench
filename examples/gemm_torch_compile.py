"""Compile and execute GEMM, then compare with eager PyTorch.

Run from the repository root after installing tobench:
    .venv/bin/python examples/gemm_torch_compile.py
"""

import torch

from tobench.backends.torch import TorchEagerAdapter, TorchInductorAdapter
from tobench.workloads import GEMM


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workload = GEMM(M=128, N=128, K=128, dtype="float16", seed=0)
    generator = torch.Generator(device=device).manual_seed(workload.seed)
    inputs = tuple(
        torch.randn(
            shape,
            dtype=getattr(torch, workload.dtype),
            device=device,
            generator=generator,
        )
        for shape in workload.input_shapes
    )

    adapter = TorchInductorAdapter()
    adapter.prepare(workload, inputs)
    print(f"Compiling GEMM on {device}...", flush=True)
    executable = adapter.build(budget=None)
    output = adapter.run(executable, inputs)

    eager = TorchEagerAdapter()
    eager.prepare(workload, inputs)
    reference = eager.run(eager.build(budget=None), inputs)
    # Dtype-aware PyTorch tolerances allow small floating-point differences.
    torch.testing.assert_close(output, reference)

    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output dtype: {output.dtype}")
    print("Correctness check against eager PyTorch passed.")


if __name__ == "__main__":
    main()
