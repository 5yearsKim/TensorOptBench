# TensorOptBench

Correctness validation lives in `benchmarking/correctness.py`; eager PyTorch
reference execution lives in `benchmarking/reference.py`. The experiment checks
outputs after build, before warmup and timed runs. Both examples print the check
status and save a structured `correctness` report in result JSON.

Configure validation in an experiment JSON:

```json
"correctness": {
  "enabled": true,
  "reference": "torch_eager",
  "rtol": 0.001,
  "atol": 0.001
}
```

Omitted or `null` tolerances preserve dtype defaults: FP16 uses `1e-3`, BF16 uses
`1e-2` for each threshold. Explicit thresholds must be finite and nonnegative.
The actual resolved values are stored in the correctness report. CLI overrides
are available in both examples:

```bash
uv run python examples/torch_compile.py --work-type gemm --rtol 0.001 --atol 0.001
uv run --extra tvm python examples/tvm_compile.py --config configs/rmsnorm_linear.json
```

Use `--no-correctness` to skip validation explicitly; the report then says
`skipped`, even if runtime measurement succeeds. `--correctness` enables it.

The checker requires matching shapes and dtypes and rejects NaNs/infinities,
including matching nonfinite values. Comparisons use
`abs(actual-reference) <= atol + rtol*abs(reference)` in FP64 on CPU, outside
the measured stages. Reports include shape/dtype agreement, nonfinite counts,
failed/total element counts, and maximum/mean absolute error over finite pairs.
Shape mismatches have no elementwise statistics; when no finite pairs exist,
absolute-error statistics are `null`.

Numerical mismatches produce `correctness_failed` and no runtime metrics.
Reference execution, candidate execution, and comparison exceptions instead
produce `error`, with the failing stage recorded separately. Eager PyTorch is
a consistency reference, not a mathematical accuracy guarantee. These checks
do not change the existing RMSNormLinear tolerance or repair its TVM discrepancy.

Tensor compiler benchmarks with PyTorch workloads, backend-specific preparation
and execution, and optional benchmark instrumentation.

Run an experiment from [configs/gemm.json](configs/gemm.json):

```bash
uv run python examples/torch_compile.py --work-type gemm
uv run python examples/torch_compile.py --config configs/gemm.json
uv run --extra tvm python examples/tvm_compile.py --work-type gemm
```

The JSON separates `workload`, `backend`, `device`, `budget`, and `runtime`
settings, with an optional `output` path. Pydantic validates the configuration
before execution. Explicit CLI flags override file settings; omitted settings
use defaults. Relative paths resolve from the working directory. Results include
the fully resolved configuration.

Defaults are CPU, FP16, `M=N=K=128`, 20 warmup iterations, and 100 measured runs.
Flags include `--device cuda`, `--dtype bfloat16`, `--m`, `--n`, `--k`, `--seed`,
`--warmup`, `--repetitions`, `--budget-seconds`, and `--output`. The budget is
reported, not enforced; late builds remain eligible for runtime measurement.

The code is organized as follows:

```text
src/tobench/
├── backends/
│   ├── base_adapter.py
│   ├── base_runner.py
│   ├── torch/
│   │   ├── adapter.py
│   │   ├── prepared.py
│   │   ├── eager_runner.py
│   │   └── inductor_runner.py
│   └── tvm/
│       ├── adapter.py
│       ├── prepared.py
│       └── runner.py
├── benchmarking/benchmarker.py
├── core/
│   ├── experiment.py
│   ├── config.py
│   ├── budget.py
│   └── result.py
└── workloads/
    ├── base_workload.py
    └── gemm.py
```

`BaseAdapter[Prepared]` only prepares a workload. Its result is a backend-specific
runtime dataclass. `BaseRunner[Prepared, Executable]` builds and executes that
state, supplies synchronization, and reports backend metadata. Runners accept
an optional `Benchmarker`, which owns clocks, raw samples, and statistics.
`core/experiment.py` coordinates configuration, preparation, correctness checks,
and benchmark results; the caller controls saving JSON.

PyTorch eager and Inductor share `TorchAdapter` and `TorchPreparedInput`:

```python
import torch
from tobench.workloads import GEMM
from tobench.backends.torch import TorchAdapter, TorchInductorRunner
from tobench.benchmarking import Benchmarker

workload = GEMM(M=32, N=64, K=16, dtype="float16", seed=0)
generator = torch.Generator().manual_seed(workload.seed)
inputs = tuple(torch.randn(shape, dtype=torch.float16, generator=generator)
               for shape in workload.input_shapes)
prepared = TorchAdapter().prepare(workload, inputs)

monitor = Benchmarker()
runner = TorchInductorRunner(benchmarker=monitor)
executable = runner.build(prepared)
output = runner.run(executable)  # exactly one execution and one latency sample
runner.benchmark_run(executable, warmup=20, repetitions=100)
print(monitor.optimization_time_seconds)
print(monitor.summary())
```

Omit `benchmarker` to execute without measurement. `run(..., measure=False)`
executes once without recording a sample. `benchmark_run` explicitly replaces
runtime samples, performs unmeasured warmup, then measures the requested number
of executions. It requires an injected benchmarker. Each build resets the
benchmarker; use one benchmarker per runner and experiment.

Inductor finishes its first invocation in `build`, under the same inference
context as execution. It does not record that invocation as a runtime sample.
Subsequent inputs must preserve shapes, strides, dtype, device, and execution
settings to avoid recompilation. `TorchEagerRunner` executes without compilation.

TVM is optional and uses the tested 0.26 API series. `TVMAdapter.prepare` exports
the PyTorch module and imports it into Relax IR, returning `TVMPreparedInput`.
`TVMRunner.build` compiles that IR and completes a first invocation. Use the same
runner API with `TVMRunner`, or run the example:

```bash
uv run --extra tvm python examples/tvm_compile.py --device cpu --dtype bfloat16
uv run python examples/torch_compile.py
```

TVM GEMM preserves FP16/BF16 output dtype after FP32 accumulation. Tensor interop
uses DLPack without NumPy copies; CUDA framework boundaries synchronize for
correctness. MetaSchedule search and hard budget cancellation are not implemented.

The examples own workload/input creation and backend selection. Shared helpers
in `examples/utils.py` provide `create_workload(work_type)`, argument parsing,
and result reporting. The workload factory contains a hardcoded GEMM example
(`M=N=K=128`, FP16, seed 0) and the RMSNormLinear example described below;
JSON and CLI parameters can override these values.
`gemm` and `rmsnorm_linear` are supported. Add new workload mappings in that helper
and extend the configuration/CLI choices when adding operators.

Both scripts accept `--work-type gemm` (alias `--work_type gemm`). Their backend
is fixed by the script, overriding the JSON `backend` field. Both validate
correctness, record preparation/build/runtime measurements, and save JSON to
`results/gemm_inductor.json` or `results/gemm_tvm.json` by default. Use `--output`
to select another path. They also work with `python -m examples.torch_compile`
and `python -m examples.tvm_compile`.

The reusable experiment API is `benchmark(adapter, runner, workload, inputs, ...)`
from `core.experiment`. Supply a runner with an injected benchmarker. There is no
configuration-to-backend factory in the core; that setup is explicit in each
example.

Results record preparation time separately from build time. **TVM export/import
now counts as preparation**, so its build times are not directly comparable to
results from the earlier combined adapter implementation. PyTorch graph capture
remains inside Inductor build because it is part of `torch.compile`.

Runtime measurements use synchronized wall-clock timing, including runner
instrumentation dispatch, interop, and synchronization overhead. They are not
kernel-only latency. Results contain raw samples, median, nearest-rank p95,
TFLOP/s derived from median latency, metadata, and failures. Input generation,
correctness checking, and warmup are outside measured build/runtime samples.
Correctness uses eager PyTorch with FP16 `rtol=atol=1e-3` and BF16 `rtol=atol=1e-2`.
Incorrect outputs receive no runtime metrics.

Caches and process isolation are uncontrolled and labeled in JSON; build times
can reflect cache reuse. Host/device memory metrics remain `null`. CPU execution
is tested; CUDA execution and timing have not been tested on hardware.

Configuration, budget, and result schemas use strict Pydantic validation.
`OptimizationBudget(max_time_seconds=60)` is immutable. Results validate field
assignments and revalidate nested containers before JSON export. Workloads
remain PyTorch modules; prepared state and executables remain runtime dataclasses.

GEMM computes `A[M,K] @ B[K,N] -> C[M,N]` with contiguous row-major inputs in
FP16/BF16, without bias, transpose, scaling, or existing output accumulation.
Construction does not allocate inputs or alter the global random seed. Each
workload has its own file and inherits `BaseWorkload`.

Run tests (including TVM when installed with LLVM support):

```bash
uv run --extra tvm python -m unittest discover -s tests -v
```

`RMSNormLinear` in `workloads/rmsnorm_linear.py` implements normalization followed
by a bias-free linear projection:

```python
normalized = X.float() * torch.rsqrt(X.float().square().mean(-1, keepdim=True) + eps)
normalized = (normalized * G.float()).to(X.dtype)
output = normalized @ W.T
```

Defaults match the requested graph: `X[16,4096]`, `G[1,4096]`, `W[406,4096]`,
FP16, and output `[16,406]`. Because W stores `[out_features,in_features]`, the
projection uses its transpose. Epsilon defaults to `1e-6` and can be set with
`--eps` or in JSON. Normalization and scale multiplication use FP32 with one
cast before projection. This defines the numerical contract explicitly; no
external graph library's unspecified epsilon or rounding rules are assumed.

```bash
uv run python examples/torch_compile.py --work-type rmsnorm_linear
uv run --extra tvm python examples/tvm_compile.py --config configs/rmsnorm_linear.json
```

Dimensions remain configurable through `--m`, `--n`, and `--k`. Changing
`--work-type` selects that operator's defaults before applying CLI overrides.
For this workload, reported TFLOP/s uses the approximate arithmetic count
`2*M*N*K + 4*M*K + 2*M`, counting reciprocal square root as one operation and
excluding casts and memory operations. It is not a hardware instruction count.

The requested full-size FP16 example passes correctness checks with TorchInductor
on CPU. With the tested TVM 0.26 CPU build, it fails the current FP16 tolerance
on some outputs, although small FP16/BF16 cases pass. The benchmark records
`correctness_failed` and omits runtime metrics in that case. The tolerance has
not been relaxed. CUDA has not been tested.

The CPU discrepancy was traced to the FP32 mean-square reduction. TVM's lowered
reduction and a manually accumulated sequential FP32 sum agree exactly. At the
default shape and seed, the mean differs from PyTorch by up to `3.22e-6`, changing
88 normalized values after FP16 rounding. Feeding those values to PyTorch's
projection produces 39 tolerance failures; the full TVM graph produces 40.
TVM projection on PyTorch-normalized inputs produces no tolerance failures.
This identifies reduction error followed by FP16 rounding as the main source.

Reproduce the stage comparisons and FP64 reference checks:

```bash
uv run --extra tvm python scripts/diagnose_rmsnorm_linear.py
```

The script writes comparison statistics and Relax/lowered IR under
`results/rmsnorm_diagnosis/`. It does not alter workload semantics or tolerances.
