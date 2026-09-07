"""Tests for Experiment B: Savitzky-Golay window sensitivity."""

import unittest

from lesson05_system_identification.src.quality_experiments import (
    compare_savgol_windows,
)


class SavgolSensitivityTest(unittest.TestCase):
    def test_each_requested_window_returns_a_parameter_estimate(self) -> None:
        results = compare_savgol_windows(
            window_lengths=[11, 31],
            duration_s=2.0,
            dt_s=0.01,
            position_noise_std_deg=0.02,
        )

        self.assertEqual(set(results), {11, 31})
        for estimate in results.values():
            self.assertGreaterEqual(estimate["inertia_abs_error"], 0.0)
            self.assertGreaterEqual(estimate["damping_abs_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
