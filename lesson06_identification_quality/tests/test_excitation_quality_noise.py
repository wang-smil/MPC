"""Tests for noisy excitation-quality identification."""

import unittest

from lesson06_identification_quality.src.quality_experiments import (
    compare_noisy_identification,
)


class NoisyExcitationQualityTest(unittest.TestCase):
    def test_noisy_comparison_reports_parameter_errors(self) -> None:
        results = compare_noisy_identification(
            duration_s=2.0,
            dt_s=0.01,
            position_noise_std_deg=0.02,
        )

        self.assertEqual(set(results), {"constant", "single_sine", "multisine", "prbs"})
        for estimate in results.values():
            self.assertGreaterEqual(estimate["inertia_abs_error"], 0.0)
            self.assertGreaterEqual(estimate["damping_abs_error"], 0.0)
            self.assertGreaterEqual(estimate["torque_rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
