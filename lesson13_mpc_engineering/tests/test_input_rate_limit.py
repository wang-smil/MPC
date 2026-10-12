import importlib.util
import unittest

import numpy as np

from lesson12_linear_mpc.src.mpc_controller import LinearMPCController

if importlib.util.find_spec('lesson13_mpc_engineering.src.mpc_controller'):
    from lesson13_mpc_engineering.src.mpc_controller import EngineeringMPCController
else:
    EngineeringMPCController = None


def simple_controller(**kwargs):
    return EngineeringMPCController(np.eye(2), np.array([[0.], [1.]]),
                                   np.zeros((2, 2)), np.ones((1, 1)),
                                   np.diag([0., 1.]), kwargs.pop('horizon', 1), **kwargs)


class InputMoveTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(EngineeringMPCController, 'EngineeringMPCController not implemented')

    def test_zero_s_matches_lesson12(self):
        args = (np.eye(2), np.array([[0.], [1.]]), np.eye(2), np.ones((1, 1)), np.eye(2), 5)
        old = LinearMPCController(*args, torque_bounds=(-2, 2)).solve([0, 0], [0, 1])
        new = EngineeringMPCController(*args).solve([0, 0], [0, 1], .3)
        np.testing.assert_allclose(new['u0'], old['u0'], atol=1e-5)
        self.assertAlmostEqual(new['objective'], old['objective'], places=5)

    def test_one_step_penalty_matches_hand_solution(self):
        for weight, expected in ((0, 1), (.5, .8), (5, 2 / 7)):
            result = simple_controller(input_rate_weight=weight, torque_bounds=None).solve([0, 0], [0, 2], 0.)
            self.assertAlmostEqual(result['u0'][0], expected, places=5)

    def test_first_delta_uses_feedback(self):
        controller = simple_controller(delta_bounds=(-.5, .5))
        problem = controller.problem
        for previous, expected in ((.4, .9), (-.2, .3)):
            result = controller.solve([0, 0], [0, 10], previous)
            self.assertAlmostEqual(result['u0'][0], expected, places=5)
            self.assertAlmostEqual(result['DU'][0, 0], .5, places=5)
            self.assertIs(problem, controller.problem)

    def test_all_planned_deltas_respect_bounds(self):
        result = simple_controller(horizon=5, delta_bounds=(-.5, .5)).solve([0, 0], [0, 10], .2)
        expected = np.diff(np.concatenate(([.2], result['U'][0])))
        np.testing.assert_allclose(result['DU'][0], expected, atol=1e-10)
        self.assertLessEqual(np.max(np.abs(expected)), .5 + 1e-5)
        self.assertLessEqual(result['max_hard_violation'], 1e-5)

    def test_infeasible_does_not_return_stale_plan(self):
        controller = simple_controller(delta_bounds=(-.5, .5))
        self.assertIsNotNone(controller.solve([0, 0], [0, 2], 0)['u0'])
        result = controller.solve([0, 0], [0, 2], 10.)
        self.assertIn('infeasible', result['status'])
        for key in ('u0', 'X', 'U', 'DU'):
            self.assertIsNone(result[key])

    def test_invalid_move_parameters(self):
        for value in (-1, True, np.nan, np.inf):
            with self.assertRaises(ValueError):
                simple_controller(input_rate_weight=value)
        for bounds in ((1, -1), (.1, .5), (True, 1), (0, np.inf)):
            with self.assertRaises(ValueError):
                simple_controller(delta_bounds=bounds)
        controller = simple_controller()
        for previous in (True, np.nan, np.inf, [0]):
            with self.assertRaises(ValueError):
                controller.solve([0, 0], [0, 1], previous)
        for state in ([0], [True, 0], [0, np.inf]):
            with self.assertRaises(ValueError):
                controller.solve(state, [0, 1], 0.)
        with self.assertRaises(ValueError):
            EngineeringMPCController(np.eye(1), np.ones((1, 1)), np.eye(1), np.eye(1), np.eye(1), 1)
        result = simple_controller(position_bounds=(-.1, .1)).solve([1, 0], [0, 0], 0)
        self.assertIsNone(result['u0'])


if __name__ == '__main__':
    unittest.main()
