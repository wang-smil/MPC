from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

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


class ClosedLoopArtifactTest(unittest.TestCase):
    def test_kp_sweep_has_one_pair_of_poles_per_gain(self) -> None:
        kp_values = np.array([0.0, 10.0, 20.0])

        history = closed_loop_poles.sweep_kp(
            kp_values,
            kd=2.0,
            sample_time_s=0.01,
        )

        self.assertEqual(history.shape, (3, 2))

    def test_plot_functions_write_png_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(closed_loop_poles, "ROOT", Path(temp_dir)):
                case_path = closed_loop_poles.plot_closed_loop_poles(
                    closed_loop_poles.DEFAULT_CASES
                )
                sweep_path = closed_loop_poles.plot_gain_sweep(
                    np.linspace(0.0, 200.0, 20),
                    kd=2.0,
                    sample_time_s=0.01,
                )

            self.assertTrue(case_path.is_file())
            self.assertTrue(sweep_path.is_file())
            self.assertGreater(case_path.stat().st_size, 0)
            self.assertGreater(sweep_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
