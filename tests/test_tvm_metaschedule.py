"""MetaSchedule configuration, record application, and optional real tuning."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from pydantic import ValidationError

from examples.utils import create_workload, parse_config
from tobench.benchmarking import Benchmarker
from tobench.core.config import MetaScheduleConfig
from tobench.core.experiment import benchmark

TVM_AVAILABLE = importlib.util.find_spec('tvm') is not None
if TVM_AVAILABLE:
    from tobench.backends.tvm import TVMAdapter, TVMMetaScheduleRunner


class MetaScheduleConfigTests(unittest.TestCase):
    def test_invalid_limits(self):
        for field in ('max_trials_global', 'max_trials_per_task', 'num_trials_per_iter'):
            for value in (0, -1, True, '2'):
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    MetaScheduleConfig(**{field: value})

    def test_cli_overrides_tuning_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'metaschedule': {'max_trials_global': 10, 'seed': 9}}))
            config = parse_config('tvm_metaschedule', [
                '--config', str(path), '--max-trials-global', '2', '--cost-model', 'random',
            ])
        self.assertEqual(config.metaschedule.max_trials_global, 2)
        self.assertEqual(config.metaschedule.seed, 9)
        self.assertEqual(config.metaschedule.cost_model, 'random')

    @unittest.skipUnless(TVM_AVAILABLE, 'optional TVM dependency not installed')
    def test_requires_prepared_input(self):
        with self.assertRaisesRegex(TypeError, 'TVMPreparedInput'):
            TVMMetaScheduleRunner().build(None)


@unittest.skipUnless(TVM_AVAILABLE, 'optional TVM dependency not installed')
class MetaScheduleIntegrationTests(unittest.TestCase):
    def setUp(self):
        import tvm
        if not tvm.runtime.enabled('llvm'):
            self.skipTest('TVM was built without LLVM')
        self.workload, self.inputs = create_workload('gemm', parameters={'M': 8, 'N': 8, 'K': 8})
        self.prepared = TVMAdapter().prepare(self.workload, self.inputs)

    def test_empty_database_is_not_reported_as_tuned(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = TVMMetaScheduleRunner(tuning=MetaScheduleConfig(
                work_dir=directory, cost_model='random', max_trials_global=1,
            ))
            with patch('tvm.s_tir.meta_schedule.relax_integration.tune_relax', return_value=[]):
                with self.assertRaisesRegex(RuntimeError, 'no valid tuning records'):
                    runner.build(self.prepared)

    @unittest.skipUnless(os.environ.get('TOBENCH_TEST_TUNING') == '1', 'opt-in real compiler search')
    def test_real_tuning_compilation_correctness_and_metadata(self):
        import tvm
        from tvm import relax

        applied = []
        apply_database = relax.transform.MetaScheduleApplyDatabase

        def inspect_application(*args, **kwargs):
            transform = apply_database(*args, **kwargs)

            def apply(mod):
                mod = transform(mod)
                applied.extend(
                    gv.name_hint for gv, func in mod.functions.items()
                    if isinstance(func, tvm.tirx.PrimFunc)
                    and func.attrs.get('tirx.is_scheduled', False)
                )
                return mod

            return apply

        with tempfile.TemporaryDirectory() as directory:
            runner = TVMMetaScheduleRunner(benchmarker=Benchmarker(), tuning=MetaScheduleConfig(
                work_dir=directory, max_trials_global=2, num_trials_per_iter=2,
                cost_model='random',
            ))
            with patch.object(relax.transform, 'MetaScheduleApplyDatabase',
                              side_effect=inspect_application):
                result = benchmark(TVMAdapter(), runner, self.workload, self.inputs,
                                   warmup=1, repetitions=2)
            self.assertIn('fused_matmul_cast', applied)
            self.assertEqual(result.status, 'success', result.error)
            self.assertEqual(result.correctness.status, 'passed')
            self.assertEqual(len(result.latency_samples_ms), 2)
            metadata = result.backend_metadata['configuration']
            self.assertGreater(metadata['tuning_records'], 0)
            self.assertEqual(metadata['work_dir'], directory)
            self.assertFalse(result.budget_enforced)
            self.assertTrue((Path(directory) / 'database_tuning_record.json').exists())
            executable = runner.build(self.prepared)
            output = runner.run(executable, (self.inputs[0] * 2, self.inputs[1]))
            torch.testing.assert_close(output, (self.inputs[0] * 2) @ self.inputs[1])


if __name__ == '__main__':
    unittest.main()
