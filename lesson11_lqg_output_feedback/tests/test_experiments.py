from pathlib import Path
import tempfile
import unittest

import yaml

from lesson11_lqg_output_feedback.src.run_experiments import run_all


class ExperimentRunnerTest(unittest.TestCase):
    def test_runner_writes_all_lqg_artifacts(self):
        source_config = Path(__file__).parents[1] / "config" / "lqg.yaml"
        config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
        config["simulation"]["duration_s"] = 0.05

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "short_lqg.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            result = run_all(config_path, output_root=root)

            for name in (
                "separation_poles.png",
                "full_state_vs_lqg.png",
                "recursive_vs_steady_state.png",
                "q_tuning_coupling.png",
                "saturation_boundary.png",
            ):
                self.assertTrue((root / "figures" / name).exists())
            self.assertTrue((root / "reports" / "lqg_engineering_report.md").exists())
            self.assertIn("separation", result)
            self.assertGreaterEqual(
                result["boundary_metrics"]["reduced_limit"]["saturation_ratio_percent"],
                result["boundary_metrics"]["normal"]["saturation_ratio_percent"],
            )


if __name__ == "__main__":
    unittest.main()
