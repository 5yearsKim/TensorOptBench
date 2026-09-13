# RMSNormLinear TVM Numerical Discrepancy Report

TVM's sequential FP32 mean-square reduction is the main source of the observed
CPU correctness failure. Its accumulated rounding error changes normalized
values after conversion to FP16, and the following projection propagates those
differences into the output. Controlled intermediate-input comparisons support
this diagnosis.

## Experiment

- Workload: `RMSNormLinear`, computing `rmsnorm(X, G) @ W.T`.
- Shapes: `X[16,4096]`, `G[1,4096]`, `W[406,4096]`; output `[16,406]`.
- Inputs: FP16, generated with `torch.randn` using a dedicated generator and seed 0.
- Normalization: last dimension, FP32 arithmetic and scaling, epsilon `1e-6`.
- The normalized tensor is cast to FP16 before projection.
- Environment: CPU execution, PyTorch `2.14.0+cu130`, TVM `0.26.0`, LLVM target.
- Correctness criterion: `abs(actual - reference) <= 1e-3 + 1e-3 * abs(reference)`.

The full TVM output failed this criterion on **40 of 6,496 elements** against
eager PyTorch. The benchmark recorded `correctness_failed` and omitted runtime
metrics.

## Where the difference begins

| Stage: TVM versus eager PyTorch | Maximum absolute difference | Unequal elements |
| --- | ---: | ---: |
| Mean of squared inputs | `3.21865e-6` | 16 / 16 |
| Inverse RMS | `1.54972e-6` | 16 / 16 |
| Normalized and scaled FP32 tensor | `1.09673e-5` | 65,536 / 65,536 |
| Normalized tensor after FP16 cast | `0.00390625` | 88 / 65,536 |
| Final projected output | `0.125` | 559 / 6,496 |

Unequal elements are not necessarily tolerance failures. All intermediate
comparisons passed the stated tolerance; the final output had 40 failures.
The maximum absolute difference is over all elements, including those that
pass because the relative tolerance permits larger differences at larger values.

The lowered TVM IR uses a reduction accumulator initialized to FP32 zero,
adds the 4,096 squared values, and divides by 4,096. A manually implemented
sequential FP32 sum reproduced TVM's mean values **exactly for all 16 rows**.

## Controlled comparisons

| Computation compared with eager PyTorch output | Failing elements |
| --- | ---: |
| Full TVM graph | 40 / 6,496 |
| TVM projection using PyTorch-normalized inputs | 0 / 6,496 |
| PyTorch projection using TVM-normalized inputs | 39 / 6,496 |

Separately compiled TVM normalization and projection produced exactly the same
output as the full TVM graph. These experiments isolate normalization as the
dominant source. Projection arithmetic also differs, but projection alone on
the same PyTorch-normalized inputs stays within tolerance. The counts do not
imply that one particular remaining element has a single independent cause.

## Higher-precision reference

The reference normalizes and scales the already quantized inputs in FP64,
casts that intermediate to FP16 to preserve the workload's rounding boundary,
then projects in FP64 and casts the output to FP16.

- Mean-square maximum error against FP64: TVM `3.26719e-6`, PyTorch `1.02477e-7`.
- Normalized FP16 values differing from the rounded FP64 reference: TVM 86,
  PyTorch 2.
- Final output tolerance failures against this reference: TVM 39, PyTorch 0.

This supports accumulated reduction error as the primary explanation, beyond
merely observing disagreement between two implementations.

## Reproduction and artifacts

Run from the repository root:

```bash
uv run --extra tvm python scripts/diagnose_rmsnorm_linear.py
```

- [Diagnostic script](scripts/diagnose_rmsnorm_linear.py)
- [Detailed measurements](results/rmsnorm_diagnosis/report.json)
- [Lowered mean-reduction IR](results/rmsnorm_diagnosis/mean_lowered.py)
- [Workload implementation](src/tobench/workloads/rmsnorm_linear.py)

Files under `results/` are generated artifacts, ignored by Git; rerun the script
to recreate them.

## Scope and next step

These findings apply to the tested CPU build, shape, dtype, and seed. They do
not establish CUDA behavior or behavior under other schedules. The investigation
did not change workload semantics, compiler settings, or correctness tolerances.

The next experiment should evaluate a more accurate reduction strategy, such
as a balanced FP32 reduction or higher-precision accumulation, and measure both
correctness and performance. No such fix has been implemented or verified here.
