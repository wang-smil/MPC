"""Tests for the causal Kalman/LQR closed-loop simulation."""

from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from lesson10_kalman_filter.src.closed_loop import simulate_closed_loop
from lesson10_kalman_filter.src.model_loader import load_config


class KalmanClosedLoopTest(unittest.TestCase):
    """The simulation is the only component allowed to own true state."""

    def setUp(self) -> None:
        path = Path("lesson10_kalman_filter/config/kalman.yaml")
        self.config = load_config(path)

    def test_kalman_loop_logs_applied_torque_covariance_and_nis(self) -> None:
        log = simulate_closed_loop(self.config, estimator_kind="kalman")

        self.assertIn("P_qq", log)
        self.assertIn("P_dqdq", log)
        self.assertIn("kalman_gain_q", log)
        self.assertIn("nis", log)
        self.assertTrue(
            np.all(np.abs(log["torque_applied_nm"]) <= self.config["actuator"]["torque_limit_nm"])
        )

    def test_noise_free_kalman_loop_reduces_initial_position_error(self) -> None:
        config = deepcopy(self.config)
        config["measurement"]["position_noise_std_deg"] = 1e-6
        config["process_noise"]["disturbance_torque_std_nm"] = 1e-6
        log = simulate_closed_loop(config, estimator_kind="kalman")

        self.assertLess(
            abs(log["position_error_rad"][-1]),
            abs(log["position_error_rad"][0]),
        )


if __name__ == "__main__":
    unittest.main()
