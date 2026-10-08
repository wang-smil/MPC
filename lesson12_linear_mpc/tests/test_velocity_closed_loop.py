"""Output-feedback simulation of the hard velocity-constrained MPC."""

from pathlib import Path
from unittest import TestCase

import numpy as np

from lesson12_linear_mpc.src.closed_loop import simulate_closed_loop
from lesson12_linear_mpc.src.model_loader import load_config


class VelocityClosedLoopTest(TestCase):
    def test_velocity_constrained_mpc_logs_legal_predictions(self):
        config = load_config(Path(__file__).resolve().parents[1] / 'config' / 'mpc.yaml')
        config['experiment_b']['duration_s'] = 0.2
        noise = np.zeros(200)
        baseline = simulate_closed_loop(
            config, 'mpc', horizon=20, measurement_noise=noise, torque_limit_nm=2.0,
        )
        constrained = simulate_closed_loop(
            config, 'mpc', horizon=20, measurement_noise=noise,
            torque_limit_nm=2.0, velocity_limit_rad_s=1.5,
        )
        updates = constrained['control_update']
        self.assertGreater(np.max(np.abs(baseline['dq_true_rad_s'])), 1.5)
        self.assertLessEqual(
            np.max(constrained['predicted_velocity_max_abs_rad_s'][updates]), 1.5 + 1e-5
        )
        self.assertTrue(np.any(constrained['velocity_constraint_active'][updates]))
        self.assertTrue(np.all(constrained['safety_clip_active'][updates] == 0))
        np.testing.assert_array_equal(
            baseline['measurement_noise_rad'], constrained['measurement_noise_rad']
        )
