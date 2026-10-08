"""MPC must plan the future velocity, not clip it after motion."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np

from lesson12_linear_mpc.src.mpc_controller import LinearMPCController


class VelocityConstraintTest(TestCase):
    def controller(self, *, torque_bounds=(-2.0, 2.0), velocity_bounds=(-1.5, 1.5)):
        return LinearMPCController(
            Ad=np.array([[1.0, 1.0], [0.0, 1.0]]),
            Bd=np.array([[0.0], [1.0]]),
            Q=np.diag([1.0, 0.1]),
            R=np.array([[0.01]]),
            P_terminal=np.eye(2),
            horizon=3,
            torque_bounds=torque_bounds,
            velocity_bounds=velocity_bounds,
        )

    def test_every_future_velocity_including_terminal_stays_inside_limit(self):
        result = self.controller().solve(np.array([0.0, 0.0]), np.array([10.0, 0.0]))
        self.assertEqual(result['status'], 'optimal')
        self.assertTrue(np.all(result['X'][1, 1:] <= 1.5 + 1e-6))
        self.assertTrue(np.all(result['X'][1, 1:] >= -1.5 - 1e-6))
        self.assertGreater(np.max(result['X'][1, 1:]), 1.4)

    def test_current_over_limit_may_recover_but_unbrakeable_future_is_infeasible(self):
        recoverable = self.controller().solve(np.array([0.0, 1.8]), np.array([0.0, 0.0]))
        self.assertIn(recoverable['status'], ('optimal', 'optimal_inaccurate'))
        self.assertLessEqual(recoverable['X'][1, 1], 1.5 + 1e-6)

        impossible = self.controller(torque_bounds=(-0.1, 0.1)).solve(
            np.array([0.0, 5.0]), np.array([0.0, 0.0])
        )
        self.assertIn(impossible['status'], ('infeasible', 'infeasible_inaccurate'))
        self.assertIsNone(impossible['u0'])

    def test_inaccurate_status_cannot_execute_a_plan_outside_hard_bounds(self):
        controller = self.controller()
        good = controller.solve(np.array([0.0, 0.0]), np.array([10.0, 0.0]))
        self.assertEqual(good['status'], 'optimal')

        def inject_invalid_plan(*args, **kwargs):
            controller.problem._status = 'optimal_inaccurate'
            controller.U.value = good['U'].copy()
            controller.X.value = good['X'].copy()
            controller.U.value[0, 0] = 2.1

        with patch.object(controller.problem, 'solve', side_effect=inject_invalid_plan):
            result = controller.solve(np.array([0.0, 0.0]), np.array([10.0, 0.0]))
        self.assertEqual(result['status'], 'invalid_solution')
        self.assertIsNone(result['u0'])
        self.assertIsNone(result['U'])
