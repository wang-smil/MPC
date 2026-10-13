import copy
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .test_feasibility_closed_loop import feasibility_config

if importlib.util.find_spec('lesson13_mpc_engineering.src.run_feasibility'):
    from lesson13_mpc_engineering.src.run_feasibility import run_feasibility_experiments
else:
    run_feasibility_experiments = None


def cde_config():
    config = feasibility_config()
    config['constraints']['velocity_limit_rad_s'] = 1.5
    config['experiment_cde'] = {'duration_s': .04, 'target_deg': 30., 'initial_velocity_rad_s': 2.,
                                'seed': 2026, 'slack_weights': [100., 10000., 1000000.]}
    return config


def read_log(root, case):
    with (root / case['csv']).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


class ExperimentCDETest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(run_feasibility_experiments, 'C/D/E runner not implemented')

    def test_diagnostic_hard_infeasible_soft_feasible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_feasibility_experiments(cde_config(), root)
            diag = json.loads((root / 'logs/feasibility_diagnostic.json').read_text(encoding='utf-8'))
            self.assertIn('infeasible', diag['hard']['status'])
            self.assertIsNone(diag['hard']['U'])
            self.assertIn(diag['soft']['status'], ('optimal', 'optimal_inaccurate'))
            self.assertGreaterEqual(diag['soft']['EPS'][0][0], .5 - 1e-5)
            self.assertEqual(diag['velocity_indices'], list(range(21)))

    def test_runner_covers_cde_and_reuses_noise(self):
        config = cde_config()
        original = copy.deepcopy(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = run_feasibility_experiments(config, root)
            self.assertEqual(config, original)
            self.assertEqual(set(summary['cases']), {'d_soft', 'e_rho100', 'e_rho10000', 'e_rho1e6'})
            logs = {}
            for name, case in summary['cases'].items():
                self.assertEqual(case['status'], 'complete', case['failure_reason'])
                logs[name] = read_log(root, case)
                np.testing.assert_array_equal([r['measurement_noise_rad'] for r in logs[name]],
                                               [r['measurement_noise_rad'] for r in logs['d_soft']])
                self.assertEqual(case['effective_config'], summary['cases']['d_soft']['effective_config'])
            for field in ('q_true_rad', 'dq_true_rad_s', 'torque_request_nm', 'torque_actual_nm'):
                np.testing.assert_allclose([float(r[field]) for r in logs['d_soft']],
                                           [float(r[field]) for r in logs['e_rho10000']], atol=1e-12)

    def test_exported_slack_and_metrics_reconcile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = run_feasibility_experiments(cde_config(), root)
            self.assertTrue(summary['versions']['cvxpy'])
            for case in summary['cases'].values():
                rows = read_log(root, case)
                speed = np.array([float(r['dq_true_rad_s']) for r in rows])
                excess = np.maximum(np.abs(speed) - 1.5, 0.)
                self.assertAlmostEqual(case['metrics']['actual_overspeed_integral_rad'], np.sum(excess) * .001)
                with np.load(root / case['plans']) as plans:
                    self.assertEqual(plans['EPS'].shape, (4, 1, 21))
                    self.assertLessEqual(np.max(np.abs(plans['X'][:, 1, :]) - 1.5 - plans['EPS'][:, 0, :]), 1e-5)
                    self.assertLessEqual(np.max(np.abs(plans['U'])), 2 + 1e-5)
                    self.assertLessEqual(np.max(np.abs(plans['DU'])), .5 + 1e-5)
                    self.assertLessEqual(np.max(np.abs(plans['X'][:, 0, 1:])), np.deg2rad(60) + 1e-5)
                events = [r for r in rows if r['control_update'] == 'True']
                self.assertAlmostEqual(case['metrics']['initial_slack_hold_integral_rad'],
                                       sum(max(0., float(r['qp_initial_slack_rad_s'])) for r in events) * .01)
                self.assertEqual(case['final_state']['time_s'], .04)
            for path in ('figures/velocity_softening.png', 'figures/slack_penalty_comparison.png',
                         'reports/feasibility_report.md', 'logs/experiment_cde_summary.json'):
                self.assertTrue((root / path).is_file())

    def test_failure_remains_partial_and_visible(self):
        config = cde_config()
        config['constraints']['position_limit_deg'] = .0001
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = run_feasibility_experiments(config, root)
            for case in summary['cases'].values():
                self.assertEqual(case['status'], 'solver_failure')
                self.assertTrue(case['failure_reason'])
                self.assertEqual(case['final_state']['time_s'], 0.)
                self.assertEqual(len(read_log(root, case)), 1)
                with np.load(root / case['plans']) as plans:
                    self.assertEqual(plans['EPS'].shape, (0, 1, 21))
                self.assertIsNone(case['metrics']['max_initial_slack_rad_s'])

    def test_nearby_rhos_have_distinct_cases_and_files(self):
        config = cde_config()
        config['experiment_cde'].update(duration_s=.01, slack_weights=[100., 100.00001, 100.00002])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = run_feasibility_experiments(config, root)
            self.assertEqual(len(summary['cases']), 4)
            self.assertEqual(len({case['csv'] for case in summary['cases'].values()}), 4)
            self.assertEqual(len({case['plans'] for case in summary['cases'].values()}), 4)
            self.assertEqual({case['rho'] for case in summary['cases'].values()}, {100., 100.00001, 100.00002, 10000.})

    def test_report_uses_effective_configuration(self):
        config = cde_config()
        config['plant']['dt_s'] = .002
        config['mpc'].update(dt_s=.02, horizon_steps=6)
        config['actuator']['torque_limit_nm'] = 1.25
        config['constraints'].update(position_limit_deg=45., velocity_limit_rad_s=1.8)
        config['experiment_cde']['duration_s'] = .02
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_feasibility_experiments(config, root)
            report = (root / 'reports/feasibility_report.md').read_text(encoding='utf-8')
            for expected in ('|物理/KF周期|2 ms|', '|MPC周期|20 ms|', '|预测步数|6|',
                             '|扭矩幅值|±1.25 N·m|', '|位置边界|±45°|', '|速度边界|±1.8 rad/s|'):
                self.assertIn(expected, report)


if __name__ == '__main__':
    unittest.main()
