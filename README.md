# TensorOptBench

Benchmark settings can be loaded from [configs/gemm.json](configs/gemm.json):

```bash
uv run python examples/benchmark_gemm.py --config configs/gemm.json
uv run python examples/benchmark_gemm.py --config configs/gemm.json --backend inductor
```

The JSON separates `workload`, `backend`, `device`, `budget`, and `runtime`
settings, with an optional `output` path. Pydantic rejects unknown fields and
invalid values before execution. Omitted settings use model defaults, and
explicit command-line flags override file settings. Relative output paths are
resolved from the working directory. The fully resolved experiment configuration
is included in the result JSON.

```python
from tobench.core.config import BenchmarkConfig
from tobench.core.runner import benchmark_from_config

config = BenchmarkConfig.from_json("configs/gemm.json")
result = benchmark_from_config(config)
result.save_json(config.output or "results/gemm.json")
```

`benchmark_from_config` returns a result; the caller controls saving it.

Run a GEMM benchmark and save its measurements:

```bash
uv run python examples/benchmark_gemm.py --backend eager --output results/eager.json
uv run python examples/benchmark_gemm.py --backend inductor --output results/inductor.json
uv run --extra tvm python examples/benchmark_gemm.py --backend tvm --output results/tvm.json
```

Each command runs one backend. Defaults are CPU, FP16, `M=N=K=128`, 20 warmup
iterations, and 100 measured repetitions. Use `--device cuda`, `--dtype bfloat16`,
`--m`, `--n`, `--k`, `--warmup`, `--repetitions`, or `--budget-seconds` to change
the experiment. The requested budget is reported, not enforced; a late build
is still measured and its overrun is recorded.

JSON results include build wall time, raw latency samples, median and p95
latency, TFLOP/s derived from median latency, workload configuration, backend
metadata, environment information, and any failure. P95 uses the nearest-rank
definition. Correctness failures produce no runtime measurements. Missing TVM
or compiler failures are saved as error results and the example exits nonzero.

The runner times `build` after `prepare`, validates against eager PyTorch, then
warms up and measures repeated `run` calls. CUDA is synchronized at timing
boundaries. Latency includes Python dispatch, adapter interop, and synchronization
overhead; it is **not kernel-only latency**. Input generation, correctness checks,
and warmup are outside measured build/runtime samples. Correctness tolerances
are explicit in JSON: FP16 uses `rtol=atol=1e-3`; BF16 uses `rtol=atol=1e-2`.

This is an initial measurement pipeline: it does not clear compiler caches or
isolate runs inside the runner, and labels those policies in the output. Build
times can therefore reflect cache reuse. Host/device memory metrics are `null`
until a consistent measurement method is implemented.

The same runner is available from Python:

```python
from tobench.core.budget import OptimizationBudget
from tobench.core.runner import benchmark

# adapter, workload, and inputs are supplied by the caller.
result = benchmark(adapter, workload, inputs,
                   budget=OptimizationBudget(max_time_seconds=60), warmup=20, repetitions=100)
result.save_json("results/gemm.json")
```

`OptimizationBudget` and `BenchmarkResult` are Pydantic models with strict
runtime validation. Construct budgets with keyword arguments, as above. Budgets
are immutable; result fields validate assignments, metric ranges, and finite
values. JSON export revalidates nested containers before writing. Runtime
handles such as `TVMExecutable` remain plain dataclasses, and workloads remain
PyTorch modules.

Workloads are PyTorch `nn.Module` subclasses. Each operator lives in a separate
file under `src/tobench/workloads/`; the shared `Workload` base lives in
`src/tobench/core/workload.py`. Tests live in `tests/`.

GEMM is the first workload, defined in `src/tobench/workloads/gemm.py`:

```python
import torch
from tobench.workloads import GEMM

workload = GEMM(M=32, N=64, K=16, dtype="float16", seed=0)
generator = torch.Generator().manual_seed(workload.seed)
A, B = [
    torch.randn(shape, dtype=getattr(torch, workload.dtype), generator=generator)
    for shape in workload.input_shapes
]
C = workload(A, B)  # shape: (32, 64)
```

Its computation is ordinary PyTorch module code:

```python
def forward(self, A: Tensor, B: Tensor) -> Tensor:
    return torch.matmul(A, B)
```

The GEMM benchmark contract is:

- Equation: `C[i, j] = sum(A[i, k] * B[k, j] for k in range(K))`.
- Inputs: contiguous row-major `A[M, K]` and `B[K, N]`, in that order.
- Output: a new contiguous `C[M, N]`, described by `workload.output_shape`.
- Dimensions must be positive integers; dtypes are `float16` or `bfloat16`.
- Seed is a non-negative integer for input generation. Module construction
  neither allocates inputs nor changes the global random seed.
- Numerical behavior follows `torch.matmul` and backend precision settings.
  Backend metadata records precision settings; comparisons should use consistent settings.
- No transpose, bias, scaling, batching, or existing output accumulation.
- `workload.flop_count` is `2*M*N*K`, counting two operations per multiply-add.

The caller supplies tensors matching the declared shapes, dtype, and layout.
`forward` contains only the computation so backends can capture it for compilation.
The configuration does not cast or reshape supplied tensors automatically.

Install the package in a virtual environment, then run CPU tests:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover configuration validation and known matrix products in FP16 and BF16.
The benchmark runner measures build/runtime and checks correctness. CPU execution
is tested; CUDA measurements have not been tested on hardware.

PyTorch adapters live under `src/tobench/backends/torch/`:

```python
from tobench.backends.torch import TorchEagerAdapter, TorchInductorAdapter

adapter = TorchInductorAdapter()  # or TorchEagerAdapter()
adapter.prepare(workload, (A, B))
executable = adapter.build(budget=None)
C = adapter.run(executable, (A, B))
metadata = adapter.collect_metadata()
```

Both adapters use evaluation and inference mode. Inductor builds with
`torch.compile(backend="inductor", fullgraph=True, dynamic=False)` and performs
the first invocation during `build`, including CUDA synchronization when needed.
Subsequent runs must preserve input shapes, strides, dtype, device, and execution
settings to avoid recompilation. `mode` can be passed to `TorchInductorAdapter`.

Neither adapter enforces a wall-clock budget or reports optimization progress;
metadata records these limitations and the active matrix multiplication precision
settings. The runner measures timing and reports budget overruns; cache/process
isolation remains a future step. The current adapter tests include actual Inductor compilation on CPU.

Run the complete GEMM example with automatic CUDA/CPU selection and an eager
correctness comparison:

```bash
.venv/bin/python examples/gemm_torch_compile.py
```

The optional `TVMAdapter` lives in `src/tobench/backends/tvm/adapter.py`.
Install the tested TVM 0.26 API series and run GEMM on CPU:

```bash
.venv/bin/python -m pip install -e '.[tvm]'
.venv/bin/python examples/gemm_tvm.py --device cpu
.venv/bin/python examples/gemm_tvm.py --device cpu --dtype bfloat16
```

With `uv`, select the optional extra when running so environment synchronization
keeps TVM installed:

```bash
uv run --extra tvm python examples/gemm_tvm.py --device cpu
```

With CUDA-enabled PyTorch and TVM, use `--device cuda`. CPU/LLVM execution is
tested locally; the CUDA path has not been tested on hardware.

```python
from tobench.backends.tvm import TVMAdapter

adapter = TVMAdapter()  # infer LLVM or CUDA from inputs; optional target string
adapter.prepare(workload, (A, B))
executable = adapter.build(budget=None)
C = adapter.run(executable, (A, B))
```

Build exports the existing PyTorch module, imports it through the
[TVM Relax PyTorch frontend](https://tvm.apache.org/docs/reference/api/python/relax/frontend.html),
compiles it, and completes a first invocation. GEMM uses FP32 accumulation with
an explicit cast back to FP16/BF16 inside the compiled graph to preserve the
PyTorch output dtype. Inputs and outputs are shared through DLPack, without a
NumPy round trip. Runs reject changes to input shapes, strides, dtype, or device.
CUDA runs synchronize both frameworks at their boundaries; future latency
measurements must account for these synchronizations and interop overhead.

This initial TVM adapter uses the default compilation pipeline. It does **not**
perform MetaSchedule search or enforce a hard wall-clock budget; metadata records
both limitations. MetaSchedule tuning remains a separate implementation step.
TVM stays optional: its integration tests skip when it is absent or lacks LLVM,
and importing the PyTorch adapters does not require it.
