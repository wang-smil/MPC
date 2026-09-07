"""Test the executable Experiment C runner."""

import tempfile
import unittest
from pathlib import Path

from lesson06_identification_quality.src.run_friction_model_comparison import run_experiment


class FrictionModelRunnerTest(unittest.TestCase):
    def test_runner_writes_a_model_comparison_figure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_experiment(
                duration_s=2.0,
                dt_s=0.01,
                output_root=Path(temporary_directory),
            )

            self.assertTrue(result["figure_path"].is_file())
            self.assertLess(
                result["results"]["model_b"]["torque_rmse"],
                result["results"]["model_a"]["torque_rmse"],
            )


if __name__ == "__main__":
    unittest.main()
