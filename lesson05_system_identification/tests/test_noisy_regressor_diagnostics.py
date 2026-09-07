"""Ensure noisy identification reports diagnostics for the fitted regressor."""

import unittest

from lesson05_system_identification.src.quality_experiments import (
    compare_noisy_identification,
)


class NoisyRegressorDiagnosticsTest(unittest.TestCase):
    def test_noisy_estimates_include_their_own_regressor_diagnostics(self) -> None:
        results = compare_noisy_identification(
            duration_s=2.0,
            dt_s=0.01,
            position_noise_std_deg=0.02,
        )

        for estimate in results.values():
            self.assertIn("rank", estimate)
            self.assertIn("sigma_min", estimate)
            self.assertGreater(estimate["condition_number"], 0.0)


if __name__ == "__main__":
    unittest.main()
