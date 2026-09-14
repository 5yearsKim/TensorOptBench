"""Run one benchmark in an isolated process with fresh compiler caches."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKERS = {
    "inductor": PROJECT_ROOT / "examples" / "torch_compile.py",
    "iree": PROJECT_ROOT / "examples" / "iree_compile.py",
    "tensorrt": PROJECT_ROOT / "examples" / "tensorrt_compile.py",
    "tvm": PROJECT_ROOT / "examples" / "tvm_compile.py",
    "tvm_metaschedule": PROJECT_ROOT / "examples" / "tvm_metaschedule.py",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=tuple(WORKERS))
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="Parent directory for the temporary cache tree (deleted after the run)",
    )
    args, worker_args = parser.parse_known_args(argv)
    if worker_args[:1] == ["--"]:
        worker_args = worker_args[1:]

    cache_root = args.cache_root
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="tobench-cache-", dir=cache_root
    ) as temporary_directory:
        cache_directory = Path(temporary_directory)
        inductor_cache = cache_directory / "torchinductor"
        triton_cache = cache_directory / "triton"
        tensorrt_timing_cache = cache_directory / "tensorrt-timing.cache"
        inductor_cache.mkdir()
        triton_cache.mkdir()

        environment = os.environ.copy()
        environment.update(
            {
                "TORCHINDUCTOR_CACHE_DIR": str(inductor_cache),
                "TRITON_CACHE_DIR": str(triton_cache),
                "TOBENCH_TENSORRT_TIMING_CACHE_PATH": str(tensorrt_timing_cache),
                "TOBENCH_PROCESS_ISOLATED": "1",
                "TOBENCH_CACHE_POLICY": "fresh_temporary",
            }
        )
        command = [sys.executable, str(WORKERS[args.backend]), *worker_args]
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, env=environment, check=False
        )
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
