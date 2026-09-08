"""Integration test for the Lesson 07 A-D experiment runner."""

from pathlib import Path
import tempfile
import unittest


class ExperimentRunnerTest(unittest.TestCase):
    def test_run_all_writes_required_figures_logs_and_report(self) -> None:
        from lesson07_state_feedback_controllability.src.run_experiments import run_all

        config_path = Path("lesson07_state_feedback_controllability/config/controller.yaml")
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_all(config_path, output_root=Path(temporary_directory))

            self.assertEqual(
                set(result["figures"]),
                {
                    "pole_placement",
                    "speed_tradeoff",
                    "uncontrollable_case",
                    "payload_robustness",
                },
            )
            self.assertTrue(all(path.is_file() for path in result["figures"].values()))
            self.assertTrue(all(path.is_file() for path in result["logs"].values()))
            self.assertTrue(result["report_path"].is_file())
            self.assertEqual(result["controllability"]["rank"], 2)


if __name__ == "__main__":
    unittest.main()
