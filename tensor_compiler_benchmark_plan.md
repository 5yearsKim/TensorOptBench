# Tensor Compiler Benchmark Framework — Basic Implementation Plan

## 1. Project Goal

Build a unified benchmark framework for comparing tensor compilers and auto-optimization systems under the same workloads and hardware environment.

Evaluate two aspects:

- **Optimization cost:** total optimization time and peak host memory usage.
- **Generated program quality:** execution latency, throughput, and peak device memory usage.

The central question is:

> How much optimization effort is required to obtain a given level of runtime performance?

Keep the first implementation small, modular, and focused on operators and small fused subgraphs.

## 2. Initial Scope

- NVIDIA GPU, with single-GPU execution.
- Static shapes and FP16 / BF16 inputs.
- Reproducible environments and a common API across backends.
- Operators and small fused subgraphs, before full models.

Defer distributed execution, dynamic shapes, other accelerator vendors, training, and full-model import/export.

## 3. Backends

| Backend | Role |
| --- | --- |
| PyTorch eager / cuBLAS / cuDNN | Baseline runtime performance |
| Torch-TensorRT | PyTorch graph compilation and TensorRT tactic selection |
| IREE | PyTorch graph compilation to portable VM executables |
| PyTorch/XLA | OpenXLA graph compilation through the PJRT CPU/TPU runtime |
| TVM MetaSchedule | Search-based scheduling and compilation |
| TorchInductor | Graph compilation and fusion |

Start with PyTorch and Torch-TensorRT, then add IREE, PyTorch/XLA, TVM MetaSchedule, and TorchInductor. Possible later integrations include TileLang, CUTLASS-based implementations, and custom optimizers.

The common API must support backends with or without autotuning.

## 4. Workloads

### Level 1 — Primitive operators

- Elementwise Add
- ReLU
- Reduction
- Batched GEMM
- GEMM + RMSNorm (matrix multiplication followed by RMSNorm)
- Conv2D

### Level 2 — Common ML operators

- Softmax
- LayerNorm
- RMSNorm
- Attention
- MLP block

### Level 3 — Small fused subgraphs

- MatMul → Bias → GELU
- QK MatMul → Softmax → PV MatMul
- Conv2D → inference BatchNorm → ReLU

These workloads expose fusion, intermediate materialization, layout choices, and kernel boundaries. Each workload specifies computation semantics, shapes, dtype, and operator parameters independently of backend implementation.

```python
Workload(
    op="matmul",
    shapes=[(8, 4096, 4096), (8, 4096, 4096)],
    dtype="float16",
)
```

For GEMM + RMSNorm, also specify the normalization axis, epsilon, and scale tensor so every backend implements the same computation.

## 5. Architecture and Adapter Interface

```text
Benchmark specification → Workload → Orchestrator
                                        ↓
                               BackendAdapter base
                                        ↓
                      Backend-specific implementation
                                        ↓
                       Runtime measurement → Results
                                        ↓
                               Analysis / plots
```

Compiler-specific logic stays inside adapters. The orchestrator owns timing, runtime measurement, and result storage.

All adapters inherit the same abstract base class in `backends/base.py`:

```python
from abc import ABC, abstractmethod


class BackendAdapter(ABC):
    @abstractmethod
    def prepare(self, workload, inputs):
        """Bind common inputs and prepare backend-specific workload state."""
        ...

    @abstractmethod
    def build(self, budget, report_progress=None):
        """Optimize and compile; return an executable ready for measurement."""
        ...

    @abstractmethod
    def run(self, executable, inputs):
        """Execute the prepared program on the supplied inputs."""
        ...

    @abstractmethod
    def collect_metadata(self):
        """Return backend version and configuration metadata."""
        ...
```

Concrete implementations are `TVMAdapter(BackendAdapter)`, `TensorRTAdapter(BackendAdapter)`, `IREEAdapter(BackendAdapter)`, `TorchInductorAdapter(BackendAdapter)`, and `BaselineAdapter(BackendAdapter)`. Each implements these methods; the runner uses only the base interface.

`prepare` performs workload setup without tuning or compilation. `build` encapsulates each backend's internal search and compilation order, including any first invocation needed to finish lazy compilation or autotuning. A baseline's `build` can simply return a callable. `run` must not trigger further tuning or compilation during runtime measurement.

The optional progress callback reports elapsed optimization time and best observed latency when a backend exposes that information.

## 6. Metrics and Optimization Budget

### Optimization cost

Use one **total optimization time** metric. Search / tuning selects implementations or configurations, while compilation translates a selected implementation into executable code. In practice, tuning often compiles and benchmarks alternatives, and some backends expose these operations through a single call. Combining them avoids ambiguous boundaries and double counting.

Measure wall-clock elapsed time around `build`, after `prepare` and before runtime warmup. Synchronize device work at the timing boundaries. This includes search, tuning, compilation, and internal trial execution, but excludes common input creation, backend setup, and final runtime measurement.

Also record peak host memory during `build`, using the same process-tree measurement procedure for each backend.

### Generated program quality

- Median and p95 execution latency after warmup, with device synchronization or device events as appropriate.
- Throughput derived from workload size and measured latency, with workload-specific units stated explicitly.
- Peak device memory during runtime measurement, using a consistent measurement scope across backends.

Keep metrics separate rather than combining them into one score.

### Budget

Use a wall-clock optimization budget:

```python
OptimizationBudget(max_time_seconds=60)
```

The budget applies to the entire `build` stage. Adapters should stop optimization within it where supported. Record whether the backend can enforce the limit and any overrun from non-interruptible work; use actual elapsed time in comparisons. A run that produces no executable within the allowed policy is recorded with a status instead of runtime metrics.

### Optional optimization trajectory

When a backend exposes progress, record `(elapsed optimization time, best observed latency)` pairs. Use the same time origin as total optimization time. Compare best latency at fixed elapsed times and plot latency against optimization time. Backends without progress reporting provide a final point only.

## 7. Results and Configuration

Each experiment produces a common result containing:

```text
backend
backend_version
workload
hardware
status
optimization_time_seconds
budget_enforced
budget_overrun_seconds
median_latency_ms
p95_latency_ms
throughput
throughput_unit
peak_device_memory_bytes
peak_host_memory_bytes
optimization_trajectory (when available)
timestamp
configuration
environment
```

Retain raw runtime timing samples and backend configuration metadata alongside summaries. Start with JSON / JSONL results and CSV summaries; add SQLite or Parquet only when needed.

Example configuration:

```yaml
workload:
  name: matmul
  B: 8
  M: 4096
  N: 4096
  K: 4096
  dtype: float16

backends:
  - tensorrt
  - iree
  - xla
  - tvm
  - torchinductor

budget:
  type: wall_clock
  seconds: 60

runtime:
  warmup: 20
  repetitions: 100
```

## 8. Benchmark Execution and Fairness

1. Load the workload and create shared input data.
2. Initialize the adapter and call `prepare`.
3. Time `build`, collect peak host memory, and record available progress.
4. Warm up the returned executable.
5. Measure runtime latency, throughput, and peak device memory.
6. Save results, timing samples, configuration, and environment metadata.

Standardize workload semantics, hardware, inputs, dtype, optimization budget, and runtime measurement procedure. Run backends in isolated processes to keep memory measurements and compiler state separate.

Use a consistent cache policy: begin measured builds with empty compilation and tuning caches, and record the policy. Any warm-cache experiments should be reported separately.

Record GPU model, GPU driver, CUDA and Python versions, compiler/framework versions or commits, relevant environment variables, configuration, and random seed. Record budget enforcement limitations so unsupported time limits are visible in comparisons.

## 9. Repository Structure

```text
tensor-compiler-bench/
├── benchmarks/
│   ├── matmul/
│   ├── gemm_rmsnorm/
│   ├── softmax/
│   ├── rmsnorm/
│   └── fused/
├── backends/
│   ├── base.py
│   ├── tvm/
│   ├── iree/
│   ├── tensorrt/
│   ├── torchinductor/
│   └── baseline/
├── core/
│   ├── workload.py
│   ├── runner.py
│   ├── budget.py
│   ├── result.py
│   └── environment.py
├── metrics/
│   ├── latency.py
│   ├── memory.py
│   └── optimization_time.py
├── analysis/
│   ├── plots.py
│   └── tables.py
├── configs/
├── results/
├── scripts/
├── tests/
└── README.md
```

Keep backend-specific dependencies isolated from the benchmark core.

## 10. Development Milestones

| Phase | Deliverable |
| --- | --- |
| 0 — Skeleton | Workload definition, shared adapter base class, runner, budget, result schema, and CLI; use mock adapters initially. |
| 1 — Baseline + TensorRT | Implement both adapters; support GEMM, GEMM + RMSNorm, and an elementwise operation; collect optimization time, latency, throughput, and memory usage. |
| 2 — IREE | Add Turbine AOT import and LLVM CPU/CUDA compilation through the IREE runtime. |
| 3 — PyTorch/XLA | Add OpenXLA compilation and PJRT CPU/TPU execution. |
| 4 — TVM MetaSchedule | Add the TVM adapter, wall-clock budget handling, and optional optimization trajectories. |
| 5 — TorchInductor | Add the TorchInductor adapter and exercise operators and fused subgraphs through the same API. |
| 6 — Analysis | Produce latency comparisons, best latency versus optimization time, and memory versus final latency plots. |
| 7 — Extensions | Add backends and workloads once the common interface and measurement procedure are stable. |

## 11. Minimum Viable Version

- **Workloads:** Batched GEMM, RMSNormLinear, Softmax, Attention, and ConvBNReLU.
- **Backends:** PyTorch baseline, Torch-TensorRT, IREE, PyTorch/XLA, and TVM MetaSchedule.
- **Metrics:** total optimization time, runtime latency, throughput, peak host memory, and peak device memory.
- **Outputs:** JSON results, CSV summaries, latency comparisons, and optimization trajectory plots where available.

The first concrete target is GEMM with PyTorch and Torch-TensorRT, a fixed wall-clock optimization budget, and JSON output. Extend that pipeline with GEMM + RMSNorm and the remaining workloads, then TVM MetaSchedule.

## 12. Design Principles

- Keep workload semantics independent of compiler implementation.
- Require every backend to inherit the shared adapter base class.
- Preserve timing samples, available optimization history, and configuration metadata.
- Compare runtime performance against optimization time and memory usage using separate metrics and plots.
- Make every result reproducible from its configuration and environment metadata.

The same interface can later support research into scheduling, cost models, layout optimization, fusion, and memory-aware optimization.
