"""Test the executable Experiment B runner."""

import tempfile
import unittest
from pathlib import Path

from lesson06_identification_quality.src.run_savgol_sensitivity import run_experiment


class SavgolSensitivityRunnerTest(unittest.TestCase):
    def test_runner_writes_a_window_comparison_figure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_experiment(
                duration_s=2.0,
                dt_s=0.01,
                output_root=Path(temporary_directory),
            )

            self.assertEqual(set(result["results"]), {11, 31, 61, 101})
            self.assertTrue(result["figure_path"].is_file())


if __name__ == "__main__":
    unittest.main()
