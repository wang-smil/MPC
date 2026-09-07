"""Tests for the smooth-Coulomb-friction identification model."""

import unittest

import numpy as np

from lesson05_system_identification.src.estimator import identify_j_b_tau_c


class FrictionEstimatorTest(unittest.TestCase):
    def test_recovers_known_parameters_from_noise_free_data(self) -> None:
        velocity = np.linspace(-2.0, 2.0, 101)
        acceleration = np.cos(velocity)
        torque = 0.02 * acceleration + 0.08 * velocity + 0.10 * np.tanh(velocity / 0.02)

        estimate = identify_j_b_tau_c(
            velocity,
            acceleration,
            torque,
            friction_smoothing_rad_s=0.02,
        )

        self.assertAlmostEqual(estimate["inertia_hat"], 0.02, places=10)
        self.assertAlmostEqual(estimate["damping_hat"], 0.08, places=10)
        self.assertAlmostEqual(estimate["coulomb_friction_hat"], 0.10, places=10)


if __name__ == "__main__":
    unittest.main()
