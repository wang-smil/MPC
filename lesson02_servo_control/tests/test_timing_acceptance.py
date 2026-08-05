from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_servo_test
import run_timing_acceptance


class AcceptanceArtifactContractTest(unittest.TestCase):
    def test_required_config_and_runner_exist(self) -> None:
        config_path = (
            PROJECT_ROOT / "config" / "timing_acceptance.yaml"
        )
        runner_path = (
            PROJECT_ROOT / "src" / "run_timing_acceptance.py"
        )

        self.assertTrue(
            config_path.is_file(),
            "timing_acceptance.yaml 尚未创建",
        )
        self.assertTrue(
            runner_path.is_file(),
            "run_timing_acceptance.py 尚未创建",
        )

    def test_config_defines_exact_experiment_set(self) -> None:
        config_path = (
            PROJECT_ROOT / "config" / "timing_acceptance.yaml"
        )

        with config_path.open("r", encoding="utf-8") as file:
            acceptance_config = yaml.safe_load(file)

        self.assertEqual(
            set(acceptance_config["experiments"]),
            {
                "acceptance_normal",
                "delay_1step",
                "delay_5steps",
                "sample_5ms",
                "jitter",
            },
        )


class TimingAcceptanceBehaviorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.run_acceptance = getattr(
            run_timing_acceptance,
            "run_acceptance",
            None,
        )
        self.normal_passes = getattr(
            run_timing_acceptance,
            "normal_passes",
            None,
        )
        self.generate_artifacts = getattr(
            run_timing_acceptance,
            "generate_artifacts",
            None,
        )

        self.assertIsNotNone(
            self.run_acceptance,
            "run_acceptance() 尚未实现",
        )
        self.assertIsNotNone(
            self.normal_passes,
            "normal_passes() 尚未实现",
        )
        self.assertIsNotNone(
            self.generate_artifacts,
            "generate_artifacts() 尚未实现",
        )

        self.base_config = run_servo_test.load_config()

        with (
            PROJECT_ROOT
            / "config"
            / "timing_acceptance.yaml"
        ).open("r", encoding="utf-8") as file:
            self.acceptance_config = yaml.safe_load(file)

    def test_cases_report_period_delay_and_normal_pass(self) -> None:
        results = self.run_acceptance(
            self.base_config,
            self.acceptance_config,
        )

        self.assertEqual(
            set(results),
            set(self.acceptance_config["experiments"]),
        )
        self.assertAlmostEqual(
            results["delay_5steps"]["nominal_delay_ms"],
            5.0,
        )
        self.assertAlmostEqual(
            results["sample_5ms"]["nominal_delay_ms"],
            5.0,
        )
        self.assertAlmostEqual(
            results["delay_5steps"]["nominal_dt_ms"],
            1.0,
        )
        self.assertAlmostEqual(
            results["sample_5ms"]["nominal_dt_ms"],
            5.0,
        )
        self.assertGreater(
            results["jitter"]["metrics"]["jitter_std_ms"],
            0.0,
        )

        passed, checks = self.normal_passes(
            results["acceptance_normal"]["metrics"],
            self.acceptance_config["normal_limits"],
        )

        self.assertTrue(passed)
        self.assertTrue(all(checks.values()))

        unsafe_metrics = deepcopy(
            results["acceptance_normal"]["metrics"]
        )
        unsafe_metrics["safety_fault_count"] = 1

        unsafe_passed, unsafe_checks = self.normal_passes(
            unsafe_metrics,
            self.acceptance_config["normal_limits"],
        )

        self.assertFalse(unsafe_passed)
        self.assertFalse(unsafe_checks["safety_fault_count"])

    def test_artifacts_are_generated_from_real_results(self) -> None:
        results = self.run_acceptance(
            self.base_config,
            self.acceptance_config,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)

            with (
                patch.object(run_timing_acceptance, "ROOT", output_root),
                patch.object(run_servo_test, "ROOT", output_root),
            ):
                artifacts = self.generate_artifacts(
                    results=results,
                    base_config=self.base_config,
                    acceptance_config=self.acceptance_config,
                )

            self.assertTrue(artifacts["step_vs_cubic"].is_file())
            self.assertTrue(artifacts["timing_comparison"].is_file())
            self.assertTrue(artifacts["report"].is_file())
            self.assertEqual(
                set(artifacts["logs"]),
                set(self.acceptance_config["experiments"]),
            )

            for log_path in artifacts["logs"].values():
                self.assertTrue(log_path.is_file())

            header = artifacts["logs"]["acceptance_normal"].read_text(
                encoding="utf-8",
            ).splitlines()[0]

            self.assertIn("actual_dt", header)
            self.assertIn("target", header)
            self.assertIn("target_velocity", header)

            report = artifacts["report"].read_text(encoding="utf-8")
            self.assertIn("acceptance_normal", report)
            self.assertIn("硬实时", report)


if __name__ == "__main__":
    unittest.main()
