"""Integration tests for the prediction validation experiment."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from lesson12_linear_mpc.src.run_prediction_validation import run_validation


LESSON_DIR = Path(__file__).resolve().parents[1]


class PredictionValidationTest(unittest.TestCase):
    def test_runner_checks_predictions_and_writes_figure(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            result = run_validation(
                LESSON_DIR / "config" / "mpc.yaml",
                output_dir=temporary_dir,
            )

            self.assertLessEqual(result["max_abs_error"], 1e-10)
            self.assertEqual(result["X_matrix"].shape, (20, 2))
            self.assertEqual(result["X_rollout"].shape, (20, 2))
            np.testing.assert_allclose(
                result["X_matrix"], result["X_rollout"], rtol=0, atol=1e-10
            )
            self.assertEqual(
                result["figure_path"],
                Path(temporary_dir) / "prediction_validation.png",
            )
            self.assertTrue(result["figure_path"].is_file())
            self.assertGreater(result["figure_path"].stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
