from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_servo_test


class CalculateMetricsTest(unittest.TestCase):
    def test_known_response_produces_expected_metrics(self) -> None:
        calculate_metrics = getattr(
            run_servo_test,
            "calculate_metrics",
            None,
        )

        self.assertIsNotNone(
            calculate_metrics,
            "calculate_metrics() 尚未实现",
        )

        time = np.array([0.0, 1.0, 2.0, 3.0])
        position = np.deg2rad(
            np.array([0.0, 8.0, 10.0, 10.0])
        )
        target = np.deg2rad(10.0)
        torque = np.array([0.0, 2.0, 1.0, 0.0])
        saturated = np.array([False, True, False, False])

        metrics = calculate_metrics(
            time=time,
            position=position,
            target=target,
            torque=torque,
            saturated=saturated,
        )

        self.assertAlmostEqual(metrics["final_error_deg"], 0.0)
        self.assertAlmostEqual(metrics["overshoot_percent"], 0.0)
        self.assertAlmostEqual(metrics["settling_time_s"], 2.0)
        self.assertAlmostEqual(metrics["max_torque_nm"], 2.0)
        self.assertAlmostEqual(
            metrics["rms_torque_nm"],
            np.sqrt(1.25),
        )
        self.assertAlmostEqual(
            metrics["saturation_ratio_percent"],
            25.0,
        )


if __name__ == "__main__":
    unittest.main()