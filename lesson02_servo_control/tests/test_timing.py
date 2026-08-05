from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_servo_test


class TimeAxisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.build_time_axis = getattr(
            run_servo_test,
            "build_time_axis",
            None,
        )
        self.assertIsNotNone(
            self.build_time_axis,
            "build_time_axis() 尚未实现",
        )

    def test_fixed_period_keeps_exact_endpoint(self) -> None:
        time, dt_log = self.build_time_axis(
            nominal_dt=0.001,
            duration=0.01,
            rng=np.random.default_rng(42),
            jitter_std_ratio=0.0,
        )

        self.assertEqual(len(time), 11)
        self.assertEqual(len(dt_log), len(time))
        self.assertAlmostEqual(time[-1], 0.01)
        np.testing.assert_allclose(dt_log[:-1], 0.001)
        self.assertAlmostEqual(dt_log[-1], dt_log[-2])

    def test_jitter_is_bounded_and_changes_period(self) -> None:
        time, dt_log = self.build_time_axis(
            nominal_dt=0.001,
            duration=0.02,
            rng=np.random.default_rng(42),
            jitter_std_ratio=0.05,
        )

        applied_dt = dt_log[:-1]

        self.assertEqual(len(dt_log), len(time))
        self.assertGreaterEqual(time[-1], 0.02)
        self.assertTrue(np.all(applied_dt >= 0.0008))
        self.assertTrue(np.all(applied_dt <= 0.0012))
        self.assertGreater(float(np.std(applied_dt)), 0.0)
        self.assertAlmostEqual(dt_log[-1], dt_log[-2])

    def test_invalid_timing_parameters_are_rejected(self) -> None:
        invalid_cases = [
            (0.0, 1.0, 0.0),
            (0.001, 0.0, 0.0),
            (0.001, 1.0, -0.01),
        ]

        for nominal_dt, duration, jitter_std_ratio in invalid_cases:
            with self.subTest(
                nominal_dt=nominal_dt,
                duration=duration,
                jitter_std_ratio=jitter_std_ratio,
            ):
                with self.assertRaises(ValueError):
                    self.build_time_axis(
                        nominal_dt=nominal_dt,
                        duration=duration,
                        rng=np.random.default_rng(42),
                        jitter_std_ratio=jitter_std_ratio,
                    )


class SimulationTimingLogTest(unittest.TestCase):
    def test_simulation_exposes_applied_periods(self) -> None:
        config = deepcopy(run_servo_test.load_config())
        config["simulation"]["jitter_std_ratio"] = 0.0

        data = run_servo_test.simulate(config, scenario="normal")

        self.assertIn("actual_dt", data)
        self.assertEqual(len(data["actual_dt"]), len(data["time"]))
        np.testing.assert_allclose(
            data["actual_dt"][:-1],
            float(config["simulation"]["dt"]),
        )


if __name__ == "__main__":
    unittest.main()
