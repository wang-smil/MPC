"""Tests for Experiment C: model A versus smooth-friction model B."""

import unittest

from lesson05_system_identification.src.quality_experiments import (
    compare_friction_models,
)


class FrictionModelComparisonTest(unittest.TestCase):
    def test_friction_model_reduces_noise_free_torque_residual(self) -> None:
        results = compare_friction_models(
            duration_s=4.0,
            dt_s=0.01,
            position_noise_std_deg=0.0,
            window_length=31,
        )

        self.assertIn("model_a", results)
        self.assertIn("model_b", results)
        self.assertLess(results["model_b"]["torque_rmse"], results["model_a"]["torque_rmse"])
        self.assertGreater(results["model_b"]["coulomb_friction_hat"], 0.0)


if __name__ == "__main__":
    unittest.main()
