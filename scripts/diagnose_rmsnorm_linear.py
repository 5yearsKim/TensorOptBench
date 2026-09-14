"""Locate numerical differences between PyTorch and TVM on RMSNormLinear.

Run: uv run --extra tvm python scripts/diagnose_rmsnorm_linear.py
"""

import json
from pathlib import Path

import torch

from tobench.backends.tvm import TVMAdapter, TVMRunner
from tobench.workloads import BaseWorkload, RMSNormLinear


class Stage(BaseWorkload):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage

    def forward(self, X, G):
        x = X.float()
        mean = (x * x).mean(-1, keepdim=True)
        if self.stage == "mean":
            return mean
        inv = torch.rsqrt(mean + 1e-6)
        if self.stage == "inverse_rms":
            return inv
        normalized = x * inv * G.float()
        if self.stage == "normalized_fp32":
            return normalized
        return normalized.to(X.dtype)


class Projection(BaseWorkload):
    def forward(self, X, W):
        return torch.nn.functional.linear(X, W)


def difference(actual, reference):
    actual, reference = actual.double(), reference.double()
    error = (actual - reference).abs()
    return {
        "max_absolute_error": error.max().item(),
        "mean_absolute_error": error.mean().item(),
        "different_elements": (error != 0).sum().item(),
        "failed_elements": (error > 1e-3 + 1e-3 * reference.abs()).sum().item(),
        "elements": actual.numel(),
    }


def main():
    output_dir = Path("results/rmsnorm_diagnosis")
    output_dir.mkdir(parents=True, exist_ok=True)
    workload = RMSNormLinear()
    generator = torch.Generator().manual_seed(0)
    X, G, W = tuple(torch.randn(shape, dtype=torch.float16, generator=generator)
                    for shape in workload.input_shapes)

    def compile_module(module, inputs, name):
        prepared = TVMAdapter().prepare(module, inputs)
        (output_dir / f"{name}.py").write_text(prepared.mod.script())
        if name == "mean":
            from tvm import relax

            lowered = relax.transform.LegalizeOps()(prepared.mod)
            (output_dir / "mean_lowered.py").write_text(lowered.script())
        runner = TVMRunner()
        executable = runner.build(prepared)
        return lambda *values: runner.run(executable, values).clone()

    stats = {}
    stage_values = {}
    for name in ("mean", "inverse_rms", "normalized_fp32", "normalized_fp16"):
        stage = Stage(name)
        eager = stage(X, G)
        tvm_value = compile_module(stage, (X, G), name)(X, G)
        stage_values[name] = (eager, tvm_value)
        stats[name] = difference(tvm_value, eager)
        print(name, stats[name], flush=True)

    eager_norm, tvm_norm = stage_values["normalized_fp16"]
    projection = compile_module(Projection(), (eager_norm, W), "projection")
    eager_out = workload(X, G, W)
    full_out = compile_module(workload, (X, G, W), "full")(X, G, W)
    variants = {
        "full_tvm_vs_eager": (full_out, eager_out),
        "tvm_projection_on_eager_norm_vs_eager": (projection(eager_norm, W), eager_out),
        "torch_projection_on_tvm_norm_vs_eager": (tvm_norm @ W.T, eager_out),
        "split_tvm_vs_full_tvm": (projection(tvm_norm, W), full_out),
    }
    x64 = X.double()
    # Reproduce a serial FP32 sum to identify TVM's reduction behavior.
    squared = X.float().square()
    sequential_sum = torch.zeros(*X.shape[:-1], 1, dtype=torch.float32)
    for k in range(X.shape[-1]):
        sequential_sum = sequential_sum + squared[..., k:k + 1]
    sequential_mean = sequential_sum / X.shape[-1]
    norm64 = x64 * torch.rsqrt(x64.square().mean(-1, keepdim=True) + 1e-6) * G.double()
    ref_out = (norm64.half().double() @ W.double().T).half()
    variants.update({
        "sequential_fp32_mean_vs_tvm": (sequential_mean, stage_values["mean"][1]),
        "eager_norm_vs_fp64_rounded": (eager_norm, norm64.half()),
        "tvm_norm_vs_fp64_rounded": (tvm_norm, norm64.half()),
        "eager_output_vs_fp64_reference": (eager_out, ref_out),
        "tvm_output_vs_fp64_reference": (full_out, ref_out),
        "tvm_mean_vs_fp64": (stage_values["mean"][1], x64.square().mean(-1, keepdim=True)),
        "torch_mean_vs_fp64": (stage_values["mean"][0], x64.square().mean(-1, keepdim=True)),
    })
    for name, values in variants.items():
        stats[name] = difference(*values)
        print(name, stats[name], flush=True)
    import tvm
    report = {"torch_version": str(torch.__version__), "tvm_version": tvm.__version__,
              "workload": workload.to_config(), "comparisons": stats}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
