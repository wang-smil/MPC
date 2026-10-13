import unittest
from unittest.mock import patch

import numpy as np

from .test_input_rate_limit import simple_controller


class VelocityConstraintsTest(unittest.TestCase):
    def controller(self, **kwargs):
        return simple_controller(horizon=3, delta_bounds=(-.5, .5), **kwargs)

    def test_velocity_off_preserves_existing_plan(self):
        old = self.controller().solve([0, 0], [0, 1], .2)
        new = self.controller(velocity_mode='off').solve([0, 0], [0, 1], .2)
        np.testing.assert_allclose(new['U'], old['U'], atol=1e-8)
        np.testing.assert_allclose(new['X'], old['X'], atol=1e-8)
        self.assertAlmostEqual(new['objective'], old['objective'])
        self.assertIsNone(new['EPS'])

    def test_current_hard_constraint_is_infeasible(self):
        controller = self.controller(velocity_mode='hard', velocity_bounds=(-1.5, 1.5),
                                     include_current_velocity=True)
        result = controller.solve([0, 2], [0, 0], 0.)
        self.assertIn('infeasible', result['status'])
        self.assertEqual(controller.velocity_indices, (0, 1, 2, 3))
        for key in ('u0', 'X', 'U', 'DU', 'EPS'):
            self.assertIsNone(result[key])

    def test_future_only_does_not_constrain_x0(self):
        controller = self.controller(velocity_mode='hard', velocity_bounds=(-1.5, 1.5))
        problem = controller.problem
        result = controller.solve([0, 2], [0, 0], 0.)
        self.assertIsNotNone(result['u0'])
        self.assertAlmostEqual(result['u0'][0], -.5, places=5)
        self.assertEqual(controller.velocity_indices, (1, 2, 3))
        self.assertLessEqual(np.max(np.abs(result['X'][1, 1:])), 1.5 + 1e-5)
        controller.solve([0, 1], [0, 0], 0.)
        self.assertIs(problem, controller.problem)

    def test_soft_current_has_real_epsilon_zero(self):
        controller = self.controller(velocity_mode='soft', velocity_bounds=(-1.5, 1.5),
                                     include_current_velocity=True, slack_weight=1e4)
        result = controller.solve([0, 2], [0, 0], 0.)
        self.assertIsNotNone(result['u0'])
        self.assertEqual(result['EPS'].shape, (1, 4))
        self.assertAlmostEqual(result['EPS'][0, 0], .5, places=5)
        # Three -0.5 inputs + terminal speed0.5 + unavoidable rho*(0.5)^2.
        self.assertAlmostEqual(result['objective'], 2501., places=4)
        self.assertLessEqual(np.max(np.abs(result['X'][1]) - 1.5 - result['EPS'][0]), 1e-5)
        self.assertGreaterEqual(np.min(result['EPS']), -1e-5)
        self.assertLessEqual(result['max_soft_violation'], 1e-5)
        self.assertLessEqual(np.max(np.abs(result['U'])), 2 + 1e-5)
        self.assertLessEqual(np.max(np.abs(result['DU'])), .5 + 1e-5)

    def test_softening_does_not_remove_hard_conflicts(self):
        controller = self.controller(velocity_mode='soft', velocity_bounds=(-1.5, 1.5),
                                     include_current_velocity=True)
        self.assertIsNotNone(controller.solve([0, 2], [0, 0], 0.)['EPS'])
        failed = controller.solve([0, 2], [0, 0], 10.)
        for key in ('u0', 'X', 'U', 'DU', 'EPS'):
            self.assertIsNone(failed[key])
        position_conflict = self.controller(velocity_mode='soft', velocity_bounds=(-1.5, 1.5),
                                            position_bounds=(-.1, .1))
        self.assertIsNone(position_conflict.solve([1, 2], [0, 0], 0.)['u0'])

    def test_invalid_velocity_options_are_rejected(self):
        for kwargs in ({'velocity_mode': 'invalid'}, {'velocity_mode': True},
                       {'velocity_mode': 'soft'}, {'velocity_bounds': (-1.5, 1.5)},
                       {'include_current_velocity': True}, {'include_current_velocity': 1},
                       {'velocity_mode': 'hard', 'velocity_bounds': (1, -1)},
                       {'velocity_mode': 'soft', 'velocity_bounds': (True, 2)},
                       {'velocity_mode': 'soft', 'velocity_bounds': (-1, np.inf)},
                       *({'slack_weight': value} for value in (0, -1, True, np.nan, np.inf))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.controller(**kwargs)

    def test_corrupted_slack_solution_is_not_executable(self):
        for value in (np.nan, -.1, 0.):
            controller = self.controller(velocity_mode='soft', velocity_bounds=(-1.5, 1.5),
                                         include_current_velocity=True)
            solve = controller.problem.solve

            def corrupt(**kwargs):
                solve(**kwargs)
                # Inject corruption at the solver boundary; keep validation real.
                eps = next(variable for variable in controller.problem.variables() if variable.name() == 'EPS')
                eps.save_value(np.full(eps.shape, value))

            with patch.object(controller.problem, 'solve', side_effect=corrupt):
                result = controller.solve([0, 2], [0, 0], 0.)
            self.assertEqual(result['status'], 'invalid_solution')
            for key in ('u0', 'X', 'U', 'DU', 'EPS'):
                self.assertIsNone(result[key])


if __name__ == '__main__':
    unittest.main()
