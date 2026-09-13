"""Tests for the process and cache isolation launcher."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_benchmark import main


class SubprocessLauncherTests(unittest.TestCase):
    def test_forwards_arguments_and_marks_isolated_environment(self):
        observed = {}

        def run(command, *, cwd, env, check):
            observed.update(command=command, cwd=cwd, env=env, check=check)
            self.assertTrue(Path(env["TORCHINDUCTOR_CACHE_DIR"]).is_dir())
            self.assertTrue(Path(env["TRITON_CACHE_DIR"]).is_dir())
            return subprocess.CompletedProcess(command, 7)

        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.run_benchmark.subprocess.run", side_effect=run
        ):
            status = main(
                [
                    "--backend", "tvm", "--cache-root", directory,
                    "--work-type", "gemm", "--no-correctness",
                ]
            )

        self.assertEqual(status, 7)
        self.assertTrue(observed["command"][1].endswith("examples/tvm_compile.py"))
        self.assertEqual(observed["command"][2:], ["--work-type", "gemm", "--no-correctness"])
        self.assertEqual(observed["env"]["TOBENCH_PROCESS_ISOLATED"], "1")
        self.assertEqual(observed["env"]["TOBENCH_CACHE_POLICY"], "fresh_temporary")
        self.assertFalse(observed["check"])


if __name__ == "__main__":
    unittest.main()
