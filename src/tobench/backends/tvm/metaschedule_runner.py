"""MetaSchedule search and compilation for prepared Relax workloads."""

from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import torch
import tvm
from tvm import relax
from tvm.s_tir.meta_schedule import relax_integration

from tobench.benchmarking import Benchmarker
from tobench.core.budget import OptimizationBudget
from tobench.core.config import MetaScheduleConfig

from .prepared import TVMPreparedInput
from .runner import TVMRunner


class TVMMetaScheduleRunner(TVMRunner):
    def __init__(
        self,
        benchmarker: Benchmarker | None = None,
        *,
        tuning: MetaScheduleConfig | None = None,
    ) -> None:
        super().__init__(benchmarker)
        self.tuning = tuning or MetaScheduleConfig()
        self.work_dir: str | None = None
        self.tuning_records: int | None = None

    def _target(self, prepared: TVMPreparedInput) -> Any:
        target = prepared.target
        # LLVM's search rules require the execution thread count explicitly.
        if target.kind.name == "llvm" and "num-cores" not in target.attrs:
            target = tvm.target.Target(
                {**dict(target.export()), "num-cores": torch.get_num_threads()}
            )
        return target

    def _compile(
        self, prepared: TVMPreparedInput, budget: OptimizationBudget | None
    ) -> Any:
        self.work_dir = None
        self.tuning_records = None
        if self.tuning.cost_model == "xgb":
            try:
                import xgboost  # noqa: F401
            except ModuleNotFoundError as error:
                if error.name != "xgboost":
                    raise
                raise ImportError(
                    'MetaSchedule XGBoost tuning requires pip install -e ".[tvm-tuning]"'
                ) from error
        if self.tuning.work_dir is None:
            root = Path("results/metaschedule")
            root.mkdir(parents=True, exist_ok=True)
            self.work_dir = mkdtemp(prefix="run-", dir=str(root.resolve()))
        else:
            directory = Path(self.tuning.work_dir).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            self.work_dir = str(directory)
        target = self._target(prepared)
        # Task extraction expects legalized, fused TIR in TVM 0.26.
        with target:
            mod = relax.get_pipeline("zero")(prepared.mod)
        database = relax_integration.tune_relax(
            mod=mod,
            params={},
            target=target,
            work_dir=self.work_dir,
            max_trials_global=self.tuning.max_trials_global,
            max_trials_per_task=self.tuning.max_trials_per_task,
            num_trials_per_iter=self.tuning.num_trials_per_iter,
            seed=self.tuning.seed,
            cost_model=self.tuning.cost_model,
        )
        self.tuning_records = len(database)
        if not self.tuning_records:
            raise RuntimeError(
                "MetaSchedule produced no valid tuning records; inspect "
                + self.work_dir
            )
        return relax_integration.compile_relax(
            # compile_relax performs legalization/fusion itself. Passing the
            # already-fused module adds op_pattern attrs and breaks record lookup.
            database=database,
            mod=prepared.mod,
            target=target,
            params={},
            enable_warning=True,
        )

    def collect_metadata(self, prepared: TVMPreparedInput) -> dict[str, Any]:
        metadata = super().collect_metadata(prepared)
        metadata["backend"] = "tvm_metaschedule"
        metadata["configuration"].update(
            {
                "pipeline": "meta_schedule.relax_integration.tune_relax -> compile_relax",
                "target": str(self._target(prepared)),
                "autotuning": True,
                "tuning": self.tuning.model_dump(mode="json"),
                "work_dir": self.work_dir,
                "tuning_records": self.tuning_records,
                "tuning_database_policy": "fresh"
                if self.tuning.work_dir is None
                else "reuse",
                "budget_limitation": "trial limits only; wall-clock budget is not enforced",
            }
        )
        return metadata
