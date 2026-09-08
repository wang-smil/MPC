"""Tests for the runnable controllability diagnostic experiment."""

import unittest

import numpy as np


class ControllabilityExperimentTest(unittest.TestCase):
    def test_experiment_reports_nominal_and_uncontrollable_cases(self) -> None:
        from lesson07_state_feedback_controllability.src.run_controllability import (
            calculate_controllability_cases,
        )

        cases = calculate_controllability_cases(inertia=0.02, damping=0.08)

        self.assertEqual(cases["nominal"]["rank"], 2)
        self.assertEqual(cases["uncontrollable"]["rank"], 2)
        self.assertEqual(cases["uncontrollable"]["state_dimension"], 3)
        self.assertTrue(np.isinf(cases["uncontrollable"]["condition_number"]))


if __name__ == "__main__":
    unittest.main()
