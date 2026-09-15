"""Integration test for Lesson 09 experiment artifacts."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from lesson09_state_observer.src.run_experiments import run_all


CONFIG_PATH = Path(__file__).parents[1] / "config" / "observer.yaml"


class ExperimentRunnerTest(unittest.TestCase):
    """The runner must make the observer lesson reproducible and reviewable."""

    def test_runner_writes_experiments_figures_and_report(self) -> None:
        """All required scenarios and teaching artifacts are created."""

        with TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            result = run_all(CONFIG_PATH, output_root=output_root)

            self.assertTrue(
                {
                    "convergence",
                    "speed_tradeoff",
                    "velocity_sources",
                    "payload",
                    "bias",
                    "normal",
                    "disturbance",
                    "stress",
                }.issubset(result)
            )
            self.assertTrue(
                (output_root / "figures" / "observer_convergence.png").is_file()
            )
            self.assertTrue(
                (output_root / "figures" / "observer_speed_tradeoff.png").is_file()
            )
            self.assertTrue(
                (output_root / "figures" / "velocity_source_comparison.png").is_file()
            )
            self.assertTrue(
                (output_root / "figures" / "observer_robustness.png").is_file()
            )
            self.assertTrue(
                (output_root / "reports" / "observer_engineering_report.md").is_file()
            )


if __name__ == "__main__":
    unittest.main()
