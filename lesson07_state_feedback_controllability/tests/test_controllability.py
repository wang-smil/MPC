"""Tests for controllability diagnostics and the teaching failure model."""

import unittest

from lesson07_state_feedback_controllability.src.controllability import (
    build_uncontrollable_model,
    controllability_report,
)
from lesson07_state_feedback_controllability.src.model import build_continuous_model


class ControllabilityTest(unittest.TestCase):
    def test_nominal_joint_is_controllable(self) -> None:
        A, B = build_continuous_model(inertia=0.02, damping=0.08)

        report = controllability_report(A, B)

        self.assertEqual(report["rank"], 2)
        self.assertEqual(report["state_dimension"], 2)
        self.assertGreater(report["sigma_min"], 0.0)

    def test_uncontrolled_unstable_mode_is_not_controllable(self) -> None:
        A_bad, B_bad = build_uncontrollable_model(inertia=0.02, damping=0.08)

        report = controllability_report(A_bad, B_bad)

        self.assertEqual(report["rank"], 2)
        self.assertEqual(report["state_dimension"], 3)
        self.assertGreater(A_bad[2, 2], 0.0)


if __name__ == "__main__":
    unittest.main()
