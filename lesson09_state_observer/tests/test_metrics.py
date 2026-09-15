"""Tests for observer and control engineering metrics."""

import unittest

import numpy as np

from lesson09_state_observer.src.metrics import calculate_metrics


class ObserverMetricsTest(unittest.TestCase):
    """Metrics must be reproducible from the recorded physical signals."""

    def test_metrics_match_known_logged_values(self) -> None:
        """RMSE, torque, innovation, and saturation use their named log fields."""

        log = {
            "time_s": np.array([0.1, 0.2]),
            "q_true_rad": np.array([1.0, 2.0]),
            "q_ref_rad": np.array([0.0, 0.0]),
            "q_estimation_error_rad": np.array([1.0, 0.0]),
            "dq_estimation_error_rad_s": np.array([2.0, 0.0]),
            "innovation_rad": np.array([3.0, 4.0]),
            "torque_cmd_nm": np.array([3.0, 4.0]),
            "saturated": np.array([False, True]),
        }

        metrics = calculate_metrics(log)

        self.assertAlmostEqual(metrics["q_est_rmse_rad"], np.sqrt(0.5))
        self.assertAlmostEqual(metrics["dq_est_rmse_rad_s"], np.sqrt(2.0))
        self.assertAlmostEqual(metrics["innovation_rms_rad"], np.sqrt(12.5))
        self.assertAlmostEqual(metrics["tracking_rmse_rad"], np.sqrt(2.5))
        self.assertAlmostEqual(metrics["rms_torque_nm"], np.sqrt(12.5))
        self.assertEqual(metrics["peak_torque_nm"], 4.0)
        self.assertEqual(metrics["saturation_ratio_percent"], 50.0)

    def test_control_velocity_metrics_use_selected_estimate_when_logged(self) -> None:
        """Experiment C must score the velocity source actually sent to LQR."""

        log = {
            "time_s": np.array([0.1, 0.2]),
            "q_true_rad": np.array([0.0, 0.0]),
            "q_ref_rad": np.array([0.0, 0.0]),
            "q_estimation_error_rad": np.array([0.0, 0.0]),
            "dq_estimation_error_rad_s": np.array([0.0, 0.0]),
            "dq_control_estimation_error_rad_s": np.array([3.0, 4.0]),
            "innovation_rad": np.array([0.0, 0.0]),
            "torque_cmd_nm": np.array([0.0, 0.0]),
            "saturated": np.array([False, False]),
        }

        metrics = calculate_metrics(log)

        self.assertAlmostEqual(metrics["control_velocity_rmse_rad_s"], 5.0 / np.sqrt(2.0))
        self.assertAlmostEqual(metrics["control_velocity_error_std_rad_s"], 0.5)


if __name__ == "__main__":
    unittest.main()
