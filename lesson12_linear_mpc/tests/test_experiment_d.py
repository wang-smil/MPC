"""Experiment D compares torque-only feedback with speed-aware MPC."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from lesson12_linear_mpc.src.run_experiment_d import run_experiment_d


class ExperimentDRunnerTest(TestCase):
    def test_runner_produces_fair_comparison_and_separate_velocity_metrics(self):
        config_path = Path(__file__).resolve().parents[1] / 'config' / 'mpc.yaml'
        with TemporaryDirectory() as directory:
            result = run_experiment_d(config_path, directory, duration_s=0.2)
            self.assertEqual(set(result['traces']), {'lqr_clip', 'mpc_velocity_limited'})
            for path in result['paths'].values():
                self.assertTrue(path.is_file(), path)
            baseline = result['traces']['lqr_clip']
            constrained = result['traces']['mpc_velocity_limited']
            np.testing.assert_array_equal(
                baseline['measurement_noise_rad'], constrained['measurement_noise_rad']
            )
            self.assertGreater(result['metrics']['lqr_clip']['peak_true_velocity_rad_s'], 1.5)
            self.assertIsNone(result['metrics']['lqr_clip']['predicted_velocity_violation_events'])
            self.assertIsNone(result['metrics']['lqr_clip']['solver_deadline_miss_count'])
            self.assertGreater(result['metrics']['lqr_clip']['max_true_velocity_excess_rad_s'], 0.1)
            self.assertLessEqual(
                result['metrics']['mpc_velocity_limited']['peak_predicted_velocity_rad_s'],
                1.5 + 1e-5,
            )
            for values in result['metrics'].values():
                self.assertIn('true_velocity_violation_samples', values)
                self.assertIn('max_true_velocity_excess_rad_s', values)
                self.assertIn('solver_deadline_miss_count', values)
                self.assertIn('solver_status_counts', values)
