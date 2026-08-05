from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_servo_test


class CubicReferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cubic_reference = getattr(
            run_servo_test,
            "cubic_reference",
            None,
        )

        self.assertIsNotNone(
            self.cubic_reference,
            "cubic_reference() 尚未实现",
        )

        self.start = np.deg2rad(0.0)
        self.target = np.deg2rad(30.0)
        self.duration = 0.8

    def test_start_midpoint_end_and_hold(self) -> None:
        q_start, dq_start = self.cubic_reference(
            0.0,
            self.start,
            self.target,
            self.duration,
        )
        self.assertAlmostEqual(q_start, self.start)
        self.assertAlmostEqual(dq_start, 0.0)

        q_mid, dq_mid = self.cubic_reference(
            0.4,
            self.start,
            self.target,
            self.duration,
        )
        self.assertAlmostEqual(q_mid, np.deg2rad(15.0))
        self.assertAlmostEqual(dq_mid, np.deg2rad(56.25))

        q_end, dq_end = self.cubic_reference(
            0.8,
            self.start,
            self.target,
            self.duration,
        )
        self.assertAlmostEqual(q_end, self.target)
        self.assertAlmostEqual(dq_end, 0.0)

        q_after, dq_after = self.cubic_reference(
            1.2,
            self.start,
            self.target,
            self.duration,
        )
        self.assertAlmostEqual(q_after, self.target)
        self.assertAlmostEqual(dq_after, 0.0)

    def test_nonpositive_duration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.cubic_reference(
                0.0,
                self.start,
                self.target,
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
