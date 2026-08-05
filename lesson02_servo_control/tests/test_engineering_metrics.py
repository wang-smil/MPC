from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_servo_test


class TimingMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.timing_metrics = getattr(
            run_servo_test,
            "timing_metrics",
            None,
        )
        self.assertIsNotNone(
            self.timing_metrics,
            "timing_metrics() 尚未实现",
        )

    def test_known_periods_produce_expected_statistics(self) -> None:
        dt_log = np.array([0.0010, 0.0011, 0.0009])

        metrics = self.timing_metrics(dt_log)

        self.assertAlmostEqual(metrics["mean_dt_ms"], 1.0)
        self.assertAlmostEqual(metrics["max_dt_ms"], 1.1)
        self.assertAlmostEqual(
            metrics["jitter_std_ms"],
            np.std(dt_log) * 1000.0,
        )

    def test_empty_period_log_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.timing_metrics(np.array([]))


class EngineeringMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engineering_metrics = getattr(
            run_servo_test,
            "engineering_metrics",
            None,
        )
        self.assertIsNotNone(
            self.engineering_metrics,
            "engineering_metrics() 尚未实现",
        )

    def test_known_motion_produces_expected_peaks(self) -> None:
        data = {
            "velocity": np.deg2rad(
                np.array([0.0, -20.0, 10.0])
            ),
            "target": np.deg2rad(
                np.array([0.0, 10.0, 20.0])
            ),
            "position": np.deg2rad(
                np.array([0.0, 8.0, 17.0])
            ),
        }

        metrics = self.engineering_metrics(data)

        self.assertAlmostEqual(metrics["max_velocity_deg_s"], 20.0)
        self.assertAlmostEqual(
            metrics["max_position_error_deg"],
            3.0,
        )

    def test_mismatched_position_arrays_are_rejected(self) -> None:
        data = {
            "velocity": np.deg2rad(np.array([0.0, 1.0])),
            "target": np.deg2rad(np.array([0.0, 1.0, 2.0])),
            "position": np.deg2rad(np.array([0.0, 1.0])),
        }

        with self.assertRaises(ValueError):
            self.engineering_metrics(data)


if __name__ == "__main__":
    unittest.main()
