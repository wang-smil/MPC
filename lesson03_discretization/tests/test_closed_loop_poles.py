from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import closed_loop_poles
import discretize_demo


class ClosedLoopAlgebraTest(unittest.TestCase):
    def test_closed_loop_matrix_matches_state_feedback_formula(self) -> None:
        ad, bd, _, _, _ = discretize_demo.discretize_zoh(0.01)

        actual = closed_loop_poles.closed_loop_matrix(0.01, 15.0, 2.0)
        expected = ad - bd @ np.array([[15.0, 2.0]])

        np.testing.assert_allclose(actual, expected)

    def test_closed_loop_poles_report_discrete_stability(self) -> None:
        poles = closed_loop_poles.closed_loop_poles(0.01, 15.0, 2.0)

        self.assertEqual(poles.shape, (2,))
        self.assertTrue(closed_loop_poles.is_stable(poles))


if __name__ == "__main__":
    unittest.main()
