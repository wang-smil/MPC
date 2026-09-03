from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from excitation import ConstantTorque, PRBSExcitation, SingleSine


class ExcitationStrategyTest(unittest.TestCase):
    def test_constant_and_single_sine_evaluate_known_values(self):
        self.assertEqual(ConstantTorque(0.5).evaluate(2.0), 0.5)
        self.assertAlmostEqual(SingleSine(0.4, 1.0).evaluate(0.25), 0.4)

    def test_prbs_is_bounded_and_holds_value_within_interval(self):
        signal = PRBSExcitation(0.5, hold_time_s=0.1, seed=2026)
        values = signal.evaluate(np.array([0.00, 0.03, 0.09, 0.10, 0.19]))

        self.assertTrue(np.all(np.abs(values) <= 0.5))
        self.assertEqual(values[0], values[1])
        self.assertEqual(values[1], values[2])

    def test_invalid_prbs_hold_time_is_rejected(self):
        with self.assertRaises(ValueError):
            PRBSExcitation(0.5, hold_time_s=0.0, seed=1)


if __name__ == "__main__":
    unittest.main()
