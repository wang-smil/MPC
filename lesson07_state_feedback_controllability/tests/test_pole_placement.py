"""Tests for discrete pole placement and saturated state feedback."""

import unittest

import numpy as np

from lesson07_state_feedback_controllability.src.model import (
    build_continuous_model,
    discretize_zoh,
)


class PolePlacementTest(unittest.TestCase):
    def test_achieved_discrete_poles_match_requested_poles(self) -> None:
        from lesson07_state_feedback_controllability.src.pole_design import (
            design_discrete_feedback,
            second_order_poles,
        )

        A, B = build_continuous_model(inertia=0.02, damping=0.08)
        Ad, Bd = discretize_zoh(A, B, dt_s=0.01)
        continuous_poles = second_order_poles(0.8, 0.8)

        design = design_discrete_feedback(Ad, Bd, continuous_poles, dt_s=0.01)

        np.testing.assert_allclose(
            np.sort_complex(design["computed_poles"]),
            np.sort_complex(design["requested_poles"]),
        )
        self.assertTrue(np.all(np.abs(design["requested_poles"]) < 1.0))

    def test_feedback_logs_unsaturated_and_applied_torque(self) -> None:
        from lesson07_state_feedback_controllability.src.state_feedback import (
            feedback_torque,
        )

        result = feedback_torque(
            K=np.array([[10.0, 2.0]]),
            state_estimate=np.array([1.0, 0.0]),
            reference_state=np.zeros(2),
            torque_limit_nm=3.0,
        )

        self.assertEqual(result["torque_unsat_nm"], -10.0)
        self.assertEqual(result["torque_applied_nm"], -3.0)
        self.assertTrue(result["saturated"])


if __name__ == "__main__":
    unittest.main()
