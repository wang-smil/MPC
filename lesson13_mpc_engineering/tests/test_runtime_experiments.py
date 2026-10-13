import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .test_managed_timing import fg_config

if importlib.util.find_spec('lesson13_mpc_engineering.src.run_runtime'):
    from lesson13_mpc_engineering.src.run_runtime import (
        benchmark_solver, make_replay_sequence, run_runtime_experiments,
    )
    from lesson13_mpc_engineering.src.runtime_metrics import calculate_managed_metrics, calculate_runtime_metrics
    from lesson13_mpc_engineering.src.managed_closed_loop import build_managed_controller
else:
    benchmark_solver = make_replay_sequence = run_runtime_experiments = calculate_runtime_metrics = None


class RuntimeExperimentsTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, '_temp'):
            cls._temp.cleanup()

    def setUp(self):
        self.assertIsNotNone(run_runtime_experiments, 'F/G experiment runner not implemented')

    def short_run(self):
        cls = type(self)
        if not hasattr(cls, '_summary'):
            config = fg_config()
            config['experiment_fg']['fault_scenarios'] = {
                'g_nominal': {},
                'g_iteration_limit': {i: {'kind': 'max_iter', 'max_iter': 1} for i in range(3)},
                'g_deadline': {0: {'kind': 'delay', 'completion_s': .007},
                               1: {'kind': 'delay', 'completion_s': .012}},
                'g_error_recovery': {0: {'kind': 'error'}, 2: {'kind': 'nonfinite'}},
            }
            before = copy.deepcopy(config)
            cls._temp = tempfile.TemporaryDirectory()
            cls._root = Path(cls._temp.name)
            cls._summary = run_runtime_experiments(config, cls._root)
            self.assertEqual(config, before)
        return cls._summary, cls._root

    def test_replay_is_identical_and_problem_is_reused(self):
        config = fg_config()
        config['experiment_fg']['repetitions'] = 2
        sequence = make_replay_sequence(config, 3)
        controllers, calls = [], []

        def factory(config, settings=None):
            controller, design = build_managed_controller(config, settings)
            controllers.append(controller)
            solve = controller.solve

            def record(x_hat, x_ref, u_prev):
                calls.append((id(controller.problem), x_hat.copy(), x_ref.copy(), u_prev))
                return solve(x_hat, x_ref, u_prev)

            controller.solve = record
            return controller, design

        with patch('lesson13_mpc_engineering.src.run_runtime.build_managed_controller', side_effect=factory):
            result = benchmark_solver(config, sequence)
        self.assertEqual(len(controllers), 4)
        self.assertEqual(len(result['rows']), 12)
        self.assertEqual(sum(row['cold'] for row in result['rows']), 4)
        self.assertTrue(all(row['candidate_valid'] for row in result['rows']))
        for start in range(0, 12, 3):
            self.assertEqual(len({item[0] for item in calls[start:start + 3]}), 1)
            for i, (_, estimate, reference, previous) in enumerate(calls[start:start + 3]):
                np.testing.assert_array_equal(estimate, sequence['x_hat'][i])
                np.testing.assert_array_equal(reference, sequence['x_ref'][i])
                self.assertEqual(previous, sequence['u_prev'][i])
        self.assertEqual(len({item[0] for item in calls}), 4)
        self.assertEqual([r['mode'] for r in result['rows'][::3]], ['warm_on', 'warm_off', 'warm_off', 'warm_on'])

    def test_runtime_statistics_match_hand_fixture(self):
        rows = [dict(mode='warm_on', repetition=0, sample_index=i, cold=i == 0,
                     candidate_valid=True, wall_time_s=wall, iterations=10 + i,
                     backend_solve_time_s=None, run_time_s=None)
                for i, wall in enumerate((.004, .005, .007, .012))]
        m = calculate_runtime_metrics(rows, .005, .010)['warm_on']
        self.assertEqual(m['all']['count'], 4)
        self.assertAlmostEqual(m['all']['wall_mean_ms'], 7.)
        self.assertAlmostEqual(m['all']['wall_p95_ms'], 11.25)
        self.assertAlmostEqual(m['all']['wall_max_ms'], 12.)
        self.assertEqual(m['all']['budget_miss_count'], 2)
        self.assertEqual(m['all']['deadline_miss_count'], 1)
        self.assertAlmostEqual(m['steady']['wall_mean_ms'], 8.)
        self.assertAlmostEqual(m['steady']['wall_p95_ms'], 11.5)
        self.assertIsNone(m['all']['backend_solve_mean_ms'])
        self.assertEqual(m['cold']['count'], 1)
        self.assertEqual(m['repetitions']['0']['all']['count'], 4)

    def test_fg_runner_covers_scenarios_and_reconciles(self):
        summary, root = self.short_run()
        self.assertEqual(set(summary['cases']), {'f_measured_sync', 'g_nominal', 'g_iteration_limit', 'g_deadline', 'g_error_recovery'})
        saved = json.loads((root / 'logs/experiment_fg_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(saved, summary)
        with (root / 'logs/runtime_benchmark.csv').open(encoding='utf-8') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 8)
        with np.load(root / 'logs/runtime_parameter_sequence.npz') as sequence:
            self.assertEqual(sequence['x_hat'].shape, (4, 2))
        noise = []
        for name, case in summary['cases'].items():
            with (root / case['csv']).open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            jobs = json.loads((root / case['jobs']).read_text(encoding='utf-8'))
            with (root / case['events']).open(encoding='utf-8') as stream:
                events = list(csv.DictReader(stream))
            noise.append([float(row['measurement_noise_rad']) for row in rows])
            metrics = case['metrics']
            self.assertEqual(metrics['budget_miss_count'], sum(job['budget_missed'] for job in jobs))
            self.assertEqual(metrics['deadline_miss_count'], sum(job['deadline_missed'] for job in jobs))
            self.assertEqual(metrics['busy_skip_count'], sum(e['event'] == 'busy_skip' for e in events))
            self.assertEqual(metrics['late_discard_count'], sum(e['event'] == 'late_result_discarded' for e in events))
            self.assertEqual(metrics['recovery_count'], sum(e['recovered'] == 'True' for e in events))
            self.assertEqual(metrics['control_cycle_count'], 8)
            self.assertEqual(metrics['submitted_job_count'] + metrics['busy_skip_count'], 8)
            self.assertEqual(case['status'], 'complete')
            self.assertEqual(case['unapplied_fault_cycles'], [])
            self.assertLessEqual(max(abs(float(row['torque_actual_nm'])) for row in rows), 2 + 1e-9)
            self.assertLessEqual(max(abs(float(row['delta_actual_nm'])) for row in rows), .05 + 1e-9)
            for event in events:
                if event['command_elapsed_s']:
                    self.assertLessEqual(abs(float(event['command_delta_nm'])), 50 * float(event['command_elapsed_s']) + 1e-9)
            with np.load(root / case['plans']) as plans:
                self.assertEqual(plans['EPS'].shape[1:], (1, 20))
        for other in noise[1:]:
            np.testing.assert_array_equal(noise[0], other)
        for relative in ('figures/solver_runtime.png', 'figures/fallback_recovery.png', 'reports/runtime_fallback_report.md'):
            self.assertGreater((root / relative).stat().st_size, 100)

    def test_old_candidate_never_applied_after_fault(self):
        summary, root = self.short_run()
        for name in ('g_deadline', 'g_iteration_limit', 'g_error_recovery'):
            case = summary['cases'][name]
            jobs = json.loads((root / case['jobs']).read_text(encoding='utf-8'))
            with (root / case['events']).open(encoding='utf-8') as stream:
                events = list(csv.DictReader(stream))
            streak = 0
            for event in events:
                if event['event'] in ('budget_timeout', 'busy_skip'):
                    streak = 0
                elif event['event'] == 'result_decision':
                    job = next(j for j in jobs if str(j['job_id']) == event['job_id'])
                    streak = streak + 1 if job['accepted'] else 0
                if event['recovered'] == 'True':
                    self.assertGreaterEqual(streak, 3)
            for job in jobs:
                if job['discarded'] or not job['candidate_valid']:
                    self.assertFalse(job['executed'])
                    self.assertFalse(any(e['job_id'] == str(job['job_id']) and e['selected_source'] == 'mpc' for e in events))
            self.assertGreaterEqual(case['metrics']['recovery_count'], 1)

    def test_single_sample_keeps_missing_steady_statistics(self):
        config = fg_config(.01)
        config['experiment_fg']['replay_samples'] = 1
        config['experiment_fg']['fault_scenarios'] = {name: {} for name in
            ('g_nominal', 'g_iteration_limit', 'g_deadline', 'g_error_recovery')}
        with tempfile.TemporaryDirectory() as temporary:
            summary = run_runtime_experiments(config, Path(temporary))
            self.assertEqual(summary['benchmark']['metrics']['warm_on']['steady']['count'], 0)
            self.assertIsNone(summary['benchmark']['metrics']['warm_on']['steady']['wall_mean_ms'])
            self.assertIn('无稳态样本', (Path(temporary) / 'reports/runtime_fallback_report.md').read_text(encoding='utf-8'))

    def test_partial_stop_integrates_only_applied_physical_intervals(self):
        result = {'status': 'safety_stop', 'expected_samples': 80, 'final_state_time_s': .001,
                  'final_state': np.array([2., 10.]), 'jobs': [], 'events': [],
                  'log': {'time_s': np.array([0., .001]), 'q_true_rad': np.array([0., 2.]),
                          'q_ref_rad': np.array([0., 0.]), 'dq_true_rad_s': np.array([2., 10.]),
                          'torque_actual_nm': np.array([.05, np.nan]),
                          'delta_actual_nm': np.array([.05, np.nan]), 'control_update': np.array([True, False])}}
        metrics = calculate_managed_metrics(result, fg_config())
        self.assertAlmostEqual(metrics['actual_overspeed_integral_rad'], .0005)
        self.assertAlmostEqual(metrics['actual_overspeed_duration_s'], .001)
        self.assertAlmostEqual(metrics['actual_overspeed_peak_rad_s'], 8.5)
        self.assertEqual(metrics['position_violation_observation_count'], 1)


if __name__ == '__main__':
    unittest.main()
