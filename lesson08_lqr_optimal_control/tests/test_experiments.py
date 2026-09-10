"""Integration test for the Lesson 08 A-D LQR experiment runner."""

from pathlib import Path
import tempfile
import unittest


class LqrExperimentRunnerTest(unittest.TestCase):
    def test_run_all_writes_four_figures_logs_summary_and_report(self) -> None:
        from lesson08_lqr_optimal_control.src.run_experiments import run_all

        config_path = Path("lesson08_lqr_optimal_control/config/lqr.yaml")
        with tempfile.TemporaryDirectory() as directory:
            result = run_all(config_path, output_root=Path(directory))

            self.assertEqual(
                set(result["figures"]),
                {
                    "qr_tradeoff",
                    "lqr_vs_pole_placement",
                    "payload_mismatch",
                    "load_saturation_stress",
                },
            )
            self.assertTrue(all(path.is_file() for path in result["figures"].values()))
            self.assertTrue(all(path.is_file() for path in result["logs"].values()))
            self.assertTrue(result["summary_path"].is_file())
            self.assertTrue(result["report_path"].is_file())


if __name__ == "__main__":
    unittest.main()
