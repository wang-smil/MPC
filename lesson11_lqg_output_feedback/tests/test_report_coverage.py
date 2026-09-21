from pathlib import Path
import tempfile
import unittest

import yaml

from lesson11_lqg_output_feedback.src.run_experiments import run_all


class ReportCoverageTest(unittest.TestCase):
    def test_report_separates_recursive_and_steady_startup_metrics(self):
        source_config = Path(__file__).parents[1] / "config" / "lqg.yaml"
        config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
        config["simulation"]["duration_s"] = 0.05

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "short_lqg.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            result = run_all(config_path, output_root=root)
            report = result["report"].read_text(encoding="utf-8")

        self.assertIn("| recursive startup |", report)
        self.assertIn("| steady_state steady |", report)


if __name__ == "__main__":
    unittest.main()
