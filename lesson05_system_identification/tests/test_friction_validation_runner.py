"""Test the executable Experiment D runner."""

import tempfile
import unittest
from pathlib import Path

from lesson05_system_identification.src.run_friction_validation import run_experiment


class FrictionValidationRunnerTest(unittest.TestCase):
    def test_runner_writes_a_validation_figure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_experiment(
                duration_s=2.0,
                dt_s=0.01,
                output_root=Path(temporary_directory),
            )

            self.assertTrue(result["figure_path"].is_file())
            self.assertIn("nominal", result["results"])


if __name__ == "__main__":
    unittest.main()
