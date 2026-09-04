from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from identifiability import analyse_regressor


class IdentifiabilityTest(unittest.TestCase):
    def test_independent_columns_have_rank_two(self):
        phi = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])

        result = analyse_regressor(phi)

        self.assertEqual(result["rank"], 2)
        self.assertGreater(result["sigma_max"], result["sigma_min"])
        self.assertGreater(result["condition_number"], 1.0)

    def test_rank_deficient_columns_are_reported(self):
        phi = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])

        result = analyse_regressor(phi)

        self.assertEqual(result["rank"], 1)


if __name__ == "__main__":
    unittest.main()
