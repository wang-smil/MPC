from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from signal_processing import savgol_derivatives


class SavgolDerivativeTest(unittest.TestCase):
    def test_quadratic_signal_has_expected_derivatives(self):
        time_s = np.linspace(0.0, 1.0, 101)
        position_rad = 2.0 * time_s**2

        velocity, acceleration = savgol_derivatives(
            position_rad,
            dt_s=0.01,
            window_length=31,
        )

        self.assertEqual(velocity.shape, position_rad.shape)
        self.assertEqual(acceleration.shape, position_rad.shape)
        self.assertAlmostEqual(velocity[50], 2.0, places=2)
        self.assertAlmostEqual(acceleration[50], 4.0, places=2)

    def test_even_window_is_rejected(self):
        with self.assertRaises(ValueError):
            savgol_derivatives(np.zeros(20), dt_s=0.01, window_length=10)


if __name__ == "__main__":
    unittest.main()
