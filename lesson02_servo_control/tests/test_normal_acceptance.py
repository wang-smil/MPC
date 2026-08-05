from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_servo_test


class NormalAcceptanceTest(unittest.TestCase):
    def test_cubic_normal_meets_reference_and_safety_contract(self) -> None:
        config = run_servo_test.load_config()
        data = run_servo_test.simulate(config, scenario="normal")

        self.assertIn("target_velocity", data)
        self.assertAlmostEqual(data["target"][0], 0.0)
        self.assertAlmostEqual(data["target_velocity"][0], 0.0)
        self.assertAlmostEqual(
            data["target"][-1],
            np.deg2rad(30.0),
        )
        self.assertAlmostEqual(data["target_velocity"][-1], 0.0)

        metrics = run_servo_test.calculate_metrics(
            time=data["time"],
            position=data["position"],
            target=data["target"][-1],
            torque=data["torque_applied"],
            saturated=data["saturated"],
        )

        max_velocity_deg_s = float(
            np.max(np.abs(np.rad2deg(data["velocity"])))
        )
        safety_fault_count = int(np.sum(data["safety_fault"]))

        self.assertEqual(safety_fault_count, 0)
        self.assertLess(metrics["final_error_deg"], 0.5)
        self.assertLess(metrics["overshoot_percent"], 5.0)
        self.assertLess(max_velocity_deg_s, 300.0)
        self.assertLess(metrics["saturation_ratio_percent"], 10.0)
        self.assertLess(metrics["settling_time_s"], 1.5)

    def test_step_reference_remains_available_for_comparison(self) -> None:
        config = deepcopy(run_servo_test.load_config())
        reference = config.get("reference")
        self.assertIsNotNone(
            reference,
            "reference 配置尚未实现",
        )
        reference["type"] = "step"

        data = run_servo_test.simulate(config, scenario="normal")

        np.testing.assert_allclose(
            data["target"],
            np.deg2rad(30.0),
        )
        np.testing.assert_allclose(data["target_velocity"], 0.0)


if __name__ == "__main__":
    unittest.main()
