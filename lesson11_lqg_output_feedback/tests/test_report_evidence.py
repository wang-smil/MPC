from pathlib import Path
import tempfile
import unittest

import yaml

from lesson11_lqg_output_feedback.src.run_experiments import run_all


class ReportEvidenceTest(unittest.TestCase):
    def test_report_and_logs_expose_peak_innovation_and_numeric_poles(self):
        source_config = Path(__file__).parents[1] / "config" / "lqg.yaml"
        config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
        config["simulation"]["duration_s"] = 0.01

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "short_lqg.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            result = run_all(config_path, output_root=root)
            report = result["report"].read_text(encoding="utf-8")

            self.assertTrue((root / "logs" / "separation_poles.csv").exists())
        self.assertIn("peak torque / N m", report)
        self.assertIn("innovation RMS / rad", report)
        self.assertIn("Controller poles", report)


if __name__ == "__main__":
    unittest.main()
