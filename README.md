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
uv run python scripts/run_benchmark.py --backend inductor --work-type gemm --rtol 0.001 --atol 0.001
uv run --extra tvm python scripts/run_benchmark.py --backend tvm --config configs/rmsnorm_linear.json
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
a consistency reference, not a mathematical accuracy guarantee.

Tensor compiler benchmarks with PyTorch workloads, backend-specific preparation
and execution, and optional benchmark instrumentation.

Importing `tobench` or the Torch backend does not require IREE, TVM, or
Torch-TensorRT.
Optional backends report dependency guidance when a package is needed.

Use `scripts/run_benchmark.py` as the measurement entry point. It starts a fresh
Python process for every experiment, imports the selected backend before any
metric begins, and gives TorchInductor and Triton new temporary cache directories.
The cache tree is deleted after the worker exits. Results record
`process_isolation=true` and `cache_policy=fresh_temporary`. Directly running an
example remains useful while developing, but records an uncontrolled cache and
no process isolation.

Results distinguish `preparation_time_seconds` and
`optimization_time_seconds`. For TVM these mean PyTorch export plus Relax import,
and Relax compilation. Python and backend import time is intentionally excluded.

Run an experiment from [configs/gemm.json](configs/gemm.json):

```bash
uv run python scripts/run_benchmark.py --backend inductor --work-type gemm
uv run --extra iree python scripts/run_benchmark.py --backend iree --device cpu --work-type gemm
.venv/bin/python scripts/run_benchmark.py --backend tensorrt --device cuda --work-type gemm
uv run python scripts/run_benchmark.py --backend inductor --config configs/gemm.json
uv run --extra tvm python scripts/run_benchmark.py --backend tvm --work-type gemm
```

The JSON separates `workload`, `backend`, `device`, `budget`, and `runtime`
settings, with an optional `output` path. Pydantic validates the configuration
before execution. Explicit CLI flags override file settings; omitted settings
use defaults. Relative paths resolve from the working directory. Results include
the fully resolved configuration.

GEMM defaults are CPU, FP16, `B=8`, `M=N=K=128`, 20 warmup iterations, and 100
measured runs. Dimension flags include `--batch-size`, `--m`, `--n`, `--k`,
`--heads`, `--seq-len`, and `--head-dim`; common flags include `--device cuda`,
`--dtype bfloat16`, `--seed`, `--warmup`, `--repetitions`, `--budget-seconds`, and
`--output`. ConvBNReLU also accepts `--in-channels`, `--out-channels`, `--height`,
`--width`, `--kernel-size`, `--stride`, `--padding`, `--dilation`, and `--groups`.
The budget is reported, not enforced; late builds remain eligible for runtime
measurement.

The code is organized as follows:

```text
src/tobench/
├── backends/
│   ├── base_adapter.py
│   ├── base_runner.py
│   ├── iree/
│   │   ├── adapter.py
│   │   ├── prepared.py
│   │   └── runner.py
│   ├── torch/
│   │   ├── adapter.py
│   │   ├── prepared.py
│   │   ├── eager_runner.py
│   │   └── inductor_runner.py
│   ├── tensorrt/
│   │   ├── adapter.py
│   │   ├── prepared.py
│   │   └── runner.py
│   └── tvm/
│       ├── adapter.py
│       ├── metaschedule_runner.py
│       ├── prepared.py
│       └── runner.py
├── benchmarking/benchmarker.py
├── core/
│   ├── experiment.py
│   ├── config.py
│   ├── budget.py
│   └── result.py
└── workloads/
    ├── attention.py
    ├── base_workload.py
    ├── conv_bn_relu.py
    ├── gemm.py
    ├── rmsnorm_linear.py
    └── softmax.py
scripts/
└── run_benchmark.py
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

workload = GEMM(B=2, M=32, N=64, K=16, dtype="float16", seed=0)
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

IREE is optional and uses Turbine's ahead-of-time PyTorch frontend. `IREEAdapter`
exports the complete static workload to MLIR. `IREERunner` compiles an in-memory
VM bytecode module, loads it through the IREE runtime, and completes a first
invocation during build. CPU inputs select the `llvm-cpu` target with the
`local-task` runtime. CUDA inputs select the `cuda` compiler and runtime and use
the input GPU's `sm_XX` architecture. Inputs and outputs cross the PyTorch/IREE
boundary through DLPack without host tensor copies.

```bash
uv run --extra iree python scripts/run_benchmark.py --backend iree --device cpu --work-type gemm
uv run --extra iree python scripts/run_benchmark.py --backend iree --device cuda --work-type gemm
```

IREE defaults to optimization level `O3`; select another level with
`--iree-opt-level O0`, `O1`, `O2`, or `O3`. Runtime inputs must retain the build
shapes, strides, dtypes, and device. FP16 and BF16 batched GEMM plus small FP16
RMSNormLinear, Softmax, and Attention graphs are tested through the real LLVM
CPU compiler. CUDA VM bytecode generation is tested offline; CUDA runtime
execution and cross-runtime synchronization still require validation on the
target GPU.

Torch-TensorRT is an optional CUDA-only graph backend. `TensorRTAdapter` exports
the complete workload with `torch.export`; `TensorRTRunner` compiles it through
the Dynamo frontend with `require_full_compilation=True`,
`pass_through_build_failures=True`, and `min_block_size=1`. Unsupported graphs
and engine builds therefore fail, and single-operation workloads remain eligible
for compilation. Build includes TensorRT tactic selection and a first invocation.
Runtime inputs must retain the build shapes, strides, dtypes, and device.

```bash
uv sync
uv pip install --python .venv/bin/python torch-tensorrt tensorrt
.venv/bin/python scripts/run_benchmark.py --backend tensorrt --device cuda --work-type gemm
```

Use matching PyTorch, Torch-TensorRT, TensorRT, CUDA, driver, and GPU versions.
Torch-TensorRT is deliberately installed outside the shared CPU-generated lock;
pin compatible versions when creating the target GPU environment.
The isolated launcher assigns a fresh TensorRT timing cache and disables engine
caching. Direct example runs use an uncontrolled timing-cache policy. TensorRT
cannot run in the CPU-only development environment.

TVM is optional and uses the tested 0.26 API series. `TVMAdapter.prepare` exports
the PyTorch module and imports it into Relax IR, returning `TVMPreparedInput`.
`TVMRunner.build` compiles that IR and completes a first invocation. Use the same
runner API with `TVMRunner`, or run the example:

```bash
uv run --extra tvm python scripts/run_benchmark.py --backend tvm --device cpu --dtype bfloat16
uv run python scripts/run_benchmark.py --backend inductor
```

TVM GEMM preserves FP16/BF16 output dtype after FP32 accumulation. Tensor interop
uses DLPack without NumPy copies; CUDA framework boundaries synchronize for
correctness. MetaSchedule search is available through the separate runner below. Hard budget cancellation is not implemented.

The examples own workload/input creation and backend selection. Shared helpers
in `examples/utils.py` provide `create_workload(work_type)`, argument parsing,
and result reporting. The workload factory contains default examples for
batched GEMM, RMSNormLinear, Softmax, Attention, and ConvBNReLU; JSON and CLI
parameters can override their values. The supported names are `gemm`,
`rmsnorm_linear`, `softmax`, `attention`, and `conv_bn_relu`.

The backend example scripts accept `--work-type gemm` (alias `--work_type gemm`).
Each fixes its backend, overriding the JSON `backend` field. They validate
correctness, record preparation/build/runtime measurements, and save JSON under
`results/` by default. Use `--output` to select another path. They also work as
modules, including `python -m examples.iree_compile`.

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

The subprocess launcher isolates process state and creates fresh TorchInductor
and Triton caches. Direct example runs are labeled uncontrolled and can reflect
cache reuse. Host/device memory metrics remain `null`. CPU execution is tested;
CUDA execution and timing have not been tested on hardware.

Configuration, budget, and result schemas use strict Pydantic validation.
`OptimizationBudget(max_time_seconds=60)` is immutable. Results validate field
assignments and revalidate nested containers before JSON export. Workloads
remain PyTorch modules; prepared state and executables remain runtime dataclasses.

GEMM computes `A[B,M,K] @ B[B,K,N] -> C[B,M,N]` with contiguous row-major inputs
in FP16/BF16, without bias, transpose, scaling, or existing output accumulation.
Construction does not allocate inputs or alter the global random seed. Each
workload has its own file and inherits `BaseWorkload`.

Run tests (including TVM when installed with LLVM support):

```bash
uv run --extra iree --extra tvm python -m unittest discover -s tests -v
```

`RMSNormLinear` in `workloads/rmsnorm_linear.py` implements normalization followed
by a bias-free linear projection:

```python
normalized = X.float() * torch.rsqrt(X.float().square().mean(-1, keepdim=True) + eps)
normalized = (normalized * G.float()).to(X.dtype)
output = normalized @ W.T
```

Defaults are `X[4,16,4096]`, `G[1,1,4096]`, `W[406,4096]`, FP16, and output
`[4,16,406]`. The weight is shared across the batch and stores
`[out_features,in_features]`, so the projection uses its transpose. Epsilon
defaults to `1e-6` and can be set with `--eps` or in JSON. Normalization and
scale multiplication use FP32 with one cast before projection. This defines
the numerical contract explicitly; no external graph library's unspecified
epsilon or rounding rules are assumed.

```bash
uv run python scripts/run_benchmark.py --backend inductor --work-type rmsnorm_linear
uv run --extra tvm python scripts/run_benchmark.py --backend tvm --config configs/rmsnorm_linear.json
```

Dimensions remain configurable through `--batch-size`, `--m`, `--n`, and `--k`. Changing
`--work-type` selects that operator's defaults before applying CLI overrides.
For this workload, reported TFLOP/s uses the approximate arithmetic count
`B*(2*M*N*K + 4*M*K + 2*M)`, counting reciprocal square root as one operation and
excluding casts and memory operations. It is not a hardware instruction count.

`Softmax` applies last-axis softmax to `X[B,M,N]`, using FP32 for the reduction
and returning the configured input dtype. `Attention` implements unmasked,
scaled dot-product attention over BHSD tensors as explicit QK transpose
matmul, FP32 softmax, and probability-value matmul. Its defaults are
`B=2`, `H=8`, `S=128`, and `D=64`.

```bash
uv run python scripts/run_benchmark.py --backend inductor --work-type softmax
uv run --extra iree python scripts/run_benchmark.py --backend iree --work-type attention
```

`ConvBNReLU` applies a bias-free NCHW convolution, inference BatchNorm, and ReLU.
Its explicit inputs are the activation, OIHW convolution weight, BatchNorm
gamma, beta, running mean, and positive running variance. BatchNorm arithmetic
uses FP32 before casting back to the configured dtype. Defaults represent a
`3x3` ResNet-style feature convolution with `B=8`, 64 input/output channels,
and a `56x56` feature map.

```bash
uv run python scripts/run_benchmark.py --backend inductor --work-type conv_bn_relu
```

Small batched FP16 RMSNormLinear cases pass correctness checks with
TorchInductor, IREE LLVM CPU, and TVM LLVM CPU. CUDA execution has not been
tested. Larger reductions can amplify backend-specific FP32 ordering
differences before the normalized values are rounded to FP16.

Reproduce the stage comparisons and FP64 reference checks:

```bash
uv run --extra tvm python scripts/diagnose_rmsnorm_linear.py
```

The script writes batched stage comparisons and Relax/lowered IR under
`results/rmsnorm_diagnosis/`. It does not alter workload semantics or tolerances.

TVM MetaSchedule tuning
-----------------------

Install the tuning dependencies and run the dedicated example:

```bash
uv run --extra tvm-tuning python scripts/run_benchmark.py --backend tvm_metaschedule --config configs/gemm_metaschedule.json
uv run --extra tvm-tuning python scripts/run_benchmark.py --backend tvm_metaschedule --device cuda --max-trials-global 256
```

`TVMMetaScheduleRunner` shares `TVMAdapter` and the existing VM execution path.
Its measured build lowers/fuses Relax into TIR, runs MetaSchedule search,
compiles with the resulting database, and completes the first invocation.
The original `examples/tvm_compile.py` remains the untuned comparison.
The integration follows TVM's
[Relax MetaSchedule APIs](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html).

Configure the `metaschedule` JSON object or use `--max-trials-global`,
`--max-trials-per-task`, `--num-trials-per-iter`, `--tuning-seed`,
`--cost-model xgb|random`, `--work-dir`, and `--target`.
XGBoost is the default cost model; `random` retains measured schedule search
but replaces the learned cost model. A target with attributes must use a JSON
string in TVM 0.26, e.g. `--target '{"kind":"llvm","num-cores":1}'`.
LLVM targets without `num-cores` use PyTorch's configured thread count.

By default, each build creates and retains a fresh database under
`results/metaschedule/run-*`. An explicit `--work-dir` reuses existing records
and warm-starts further tuning; it does not skip search. Backend metadata records
the effective target, tuning settings, database path, record count, and reuse
policy. A build with no valid records fails instead of reporting an untuned
executable as tuned. Individual functions without matching records may still
use TVM's fallback compilation; a nonempty database does not imply full coverage.

Trial counts limit search. `--budget-seconds` still reports total build overrun
without enforcing a wall-clock deadline; a small trial count can take longer
than the requested budget. Optimization trajectories are not yet collected.
The subprocess launcher also isolates MetaSchedule process state. Its tuning
database is fresh by default; an explicit `--work-dir` intentionally reuses that
database. Existing timing and memory limitations described above still apply.

Run the optional real-search integration test with:

```bash
OMP_NUM_THREADS=1 TOBENCH_TEST_TUNING=1 uv run --extra tvm-tuning python -m unittest discover -s tests -p test_tvm_metaschedule.py -v
```
