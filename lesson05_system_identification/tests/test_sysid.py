from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from excitation import MultiSineExcitation


class ExcitationTest(unittest.TestCase):
    def test_multisine_is_zero_at_time_zero_and_has_expected_value(self):
        signal = MultiSineExcitation([0.5, 1.0], [0.4, 0.2])

        self.assertAlmostEqual(signal.evaluate(0.0), 0.0)
        expected = 0.4 * np.sin(2.0 * np.pi * 0.5 * 0.25) + 0.2
        self.assertAlmostEqual(signal.evaluate(0.25), expected)

    def test_mismatched_lists_are_rejected(self):
        with self.assertRaises(ValueError):
            MultiSineExcitation([0.5], [0.4, 0.2])


if __name__ == "__main__":
    unittest.main()
