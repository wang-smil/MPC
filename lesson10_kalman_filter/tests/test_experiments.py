"""Integration test for reproducible Lesson 10 experiments."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import yaml

from lesson10_kalman_filter.src.model_loader import load_config
from lesson10_kalman_filter.src.run_experiments import run_all


class ExperimentRunnerTest(unittest.TestCase):
    """The runner must produce the lesson's teaching artifacts in one call."""

    def test_runner_writes_required_figures_logs_and_report(self) -> None:
        source_config = Path("lesson10_kalman_filter/config/kalman.yaml")
        config = load_config(source_config)
        config["simulation"]["duration_s"] = 0.05
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            config_path = output / "fast_test_kalman.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            result = run_all(config_path, output_root=output)

            for name in (
                "baseline_confidence.png",
                "r_q_tuning.png",
                "estimator_comparison.png",
                "robustness.png",
            ):
                self.assertTrue((output / "figures" / name).exists())
            self.assertTrue((output / "logs" / "normal.csv").exists())
            self.assertTrue((output / "reports" / "kalman_engineering_report.md").exists())
            self.assertIn("normal", result["metrics"])
            self.assertIn("kalman", result["comparison_metrics"])


if __name__ == "__main__":
    unittest.main()