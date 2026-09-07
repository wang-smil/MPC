"""Tests for the executable Experiment A runner."""

import tempfile
import unittest
from pathlib import Path

from lesson05_system_identification.src.run_excitation_quality import run_experiment


class ExcitationQualityRunnerTest(unittest.TestCase):
    def test_runner_writes_a_comparison_figure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_experiment(
                duration_s=1.0,
                dt_s=0.01,
                output_root=Path(temporary_directory),
            )

            self.assertEqual(set(result["diagnostics"]), {"constant", "single_sine", "multisine", "prbs"})
            self.assertTrue(result["figure_path"].is_file())


if __name__ == "__main__":
    unittest.main()
