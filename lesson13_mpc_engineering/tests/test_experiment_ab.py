import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from .test_move_closed_loop import ab_config

if importlib.util.find_spec('lesson13_mpc_engineering.src.run_experiments'):
    from lesson13_mpc_engineering.src.run_experiments import run_experiments
else:
    run_experiments = None


class ExperimentABTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(run_experiments, 'A/B runner not implemented')

    def test_runner_covers_all_ab_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run_experiments(ab_config(.04), Path(directory))
            self.assertEqual(set(summary['cases']), {'a_s0', 'a_s0p5', 'a_s5', 'b_no_rate', 'b_hard_rate'})
            noises = []
            for case in summary['cases'].values():
                self.assertEqual(case['status'], 'complete')
                self.assertEqual(case['metrics']['solver_success_count'], 4)
                with (Path(directory) / case['csv']).open(newline='', encoding='utf-8') as stream:
                    rows = list(csv.DictReader(stream))
                noises.append([float(row['measurement_noise_rad']) for row in rows])
            for noise in noises[1:]:
                np.testing.assert_array_equal(noise, noises[0])
            self.assertEqual(summary['config']['actuator']['torque_limit_nm'], 2)
            self.assertTrue(summary['versions']['cvxpy'])

    def test_summary_and_csv_reconcile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_experiments(ab_config(.04), root)
            summary = json.loads((root / 'logs/experiment_ab_summary.json').read_text(encoding='utf-8'))
            for name, case in summary['cases'].items():
                with (root / case['csv']).open(newline='', encoding='utf-8') as stream:
                    rows = list(csv.DictReader(stream))
                actual = np.array([float(row['torque_actual_nm']) for row in rows])
                requests = np.array([float(row['torque_request_nm']) for row in rows
                                     if row['control_update'] == 'True'])
                self.assertAlmostEqual(case['metrics']['peak_actual_torque_nm'], np.max(np.abs(actual)))
                self.assertAlmostEqual(case['metrics']['max_request_delta_nm'], np.max(np.abs(np.diff(np.r_[0., requests]))))
                if name == 'b_hard_rate':
                    with np.load(root / case['plans']) as plans:
                        self.assertLessEqual(np.max(np.abs(plans['DU'])), .5 + 1e-5)
            self.assertTrue((root / 'figures/input_move_penalty.png').is_file())
            self.assertTrue((root / 'figures/input_rate_constraint.png').is_file())


if __name__ == '__main__':
    unittest.main()
