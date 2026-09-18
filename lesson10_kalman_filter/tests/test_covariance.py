"""Tests for physical noise covariance construction."""

import unittest

import numpy as np

from lesson10_kalman_filter.src.noise_model import build_noise_covariances


class NoiseCovarianceTest(unittest.TestCase):
    """Covariances must preserve their physical scalar meanings."""

    def setUp(self) -> None:
        self.config = {
            "measurement": {"position_noise_std_deg": 0.05},
            "process_noise": {"disturbance_torque_std_nm": 0.15},
            "initial_uncertainty": {
                "position_std_deg": 5.0,
                "velocity_std_rad_s": 1.0,
            },
        }
        self.Bd = np.array([[0.0001], [0.01]])

    def test_physical_standard_deviations_become_scalar_variances(self) -> None:
        covariance = build_noise_covariances(self.config, self.Bd)

        self.assertEqual(covariance["Gd"].shape, (2, 1))
        self.assertEqual(covariance["Q_process"].shape, (1, 1))
        self.assertEqual(covariance["R_measurement"].shape, (1, 1))
        self.assertEqual(covariance["P0"].shape, (2, 2))
        self.assertAlmostEqual(
            covariance["R_measurement"].item(),
            np.deg2rad(0.05) ** 2,
        )
        self.assertGreater(covariance["Q_process"].item(), 0.0)

    def test_nonpositive_noise_scales_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_noise_covariances(self.config, self.Bd, r_scale=0.0)
        with self.assertRaises(ValueError):
            build_noise_covariances(self.config, self.Bd, q_scale=-1.0)


if __name__ == "__main__":
    unittest.main()
