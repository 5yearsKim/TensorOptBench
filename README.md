# TensorOptBench

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
(`M=N=K=128`, FP16, seed 0); JSON and CLI parameters can override these values.
Only `gemm` is currently supported. Add new workload mappings in that helper
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
