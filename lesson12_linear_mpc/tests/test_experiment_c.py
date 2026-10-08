"""Experiment C: torque clipping versus planning within the torque limit."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from lesson12_linear_mpc.src.model_loader import load_config
from lesson12_linear_mpc.src.mpc_controller import LinearMPCController
from lesson12_linear_mpc.src.closed_loop import simulate_closed_loop


class TorqueConstraintTest(TestCase):
    def test_mpc_plans_every_input_inside_bounds(self):
        controller = LinearMPCController(
            np.array([[1.0]]), np.array([[1.0]]),
            np.array([[1.0]]), np.array([[0.01]]),
            np.array([[1.0]]), 3, torque_bounds=(-2.0, 2.0),
        )
        result = controller.solve(x_hat=10.0, x_ref=0.0)
        self.assertEqual(result['status'], 'optimal')
        self.assertTrue(np.all(result['U'] >= -2.0 - 1e-6))
        self.assertTrue(np.all(result['U'] <= 2.0 + 1e-6))
        self.assertAlmostEqual(result['u0'][0], -2.0, places=5)

    def test_lqr_clips_request_but_mpc_plan_is_bounded(self):
        config = load_config(Path(__file__).resolve().parents[1] / 'config' / 'mpc.yaml')
        config['experiment_b']['duration_s'] = 0.02
        noise = np.zeros(20)
        lqr = simulate_closed_loop(config, 'lqr', measurement_noise=noise, torque_limit_nm=2.0)
        mpc = simulate_closed_loop(
            config, 'mpc', horizon=20, measurement_noise=noise, torque_limit_nm=2.0,
        )
        self.assertGreater(lqr['torque_request_nm'][0], 2.0)
        self.assertEqual(lqr['torque_applied_nm'][0], 2.0)
        self.assertLessEqual(np.max(np.abs(mpc['torque_applied_nm'])), 2.0 + 1e-6)
        self.assertTrue(np.all(mpc['safety_clip_active'] == 0))
        np.testing.assert_array_equal(lqr['measurement_noise_rad'], mpc['measurement_noise_rad'])

    def test_runner_writes_comparison_artifacts(self):
        from lesson12_linear_mpc.src.run_experiment_c import run_experiment_c

        config_path = Path(__file__).resolve().parents[1] / 'config' / 'mpc.yaml'
        with TemporaryDirectory() as directory:
            output = run_experiment_c(config_path, directory, duration_s=0.04)
            self.assertEqual(set(output['traces']), {'lqr_clip', 'mpc_constrained'})
            for path in output['paths'].values():
                self.assertTrue(path.is_file(), path)
            for values in output['metrics'].values():
                self.assertIn('tracking_rmse_rad', values)
                self.assertIn('settling_time_s', values)
                self.assertIn('peak_requested_torque_nm', values)
                self.assertIn('constraint_violation_count', values)
