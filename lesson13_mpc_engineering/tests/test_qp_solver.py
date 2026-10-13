import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from lesson13_mpc_engineering.src.mpc_controller import EngineeringMPCController
from lesson13_mpc_engineering.src.model_loader import build_design
from .test_input_rate_limit import simple_controller
from .test_move_closed_loop import ab_config

if importlib.util.find_spec('lesson13_mpc_engineering.src.qp_solver'):
    from lesson13_mpc_engineering.src.qp_solver import SolverSettings
else:
    SolverSettings = None


class QpSolverTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(SolverSettings, 'configurable solver not implemented')

    def test_settings_and_backend_options(self):
        for warm in (True, False):
            controller = simple_controller(torque_bounds=None, solver_settings=SolverSettings(warm, 4000))
            problem = controller.problem
            with patch.object(problem, 'solve', wraps=problem.solve) as called:
                result = controller.solve([0, 0], [0, 2], 0.)
            self.assertAlmostEqual(result['u0'][0], 1., places=6)
            self.assertEqual(called.call_args.kwargs['warm_start'], warm)
            self.assertEqual(called.call_args.kwargs['max_iter'], 4000)
            self.assertEqual(called.call_args.kwargs['eps_abs'], 1e-8)
            controller.solve([0, 0], [0, 1], 0.)
            self.assertIs(controller.problem, problem)
        for value in (0, -1, True, 2.5, np.nan):
            with self.assertRaises(ValueError):
                SolverSettings(max_iter=value)
        with self.assertRaises(ValueError):
            SolverSettings(warm_start=1)

    def test_opt_in_matches_legacy_plan(self):
        legacy = simple_controller(horizon=5).solve([0, 0], [0, 2], .1)
        controller = simple_controller(horizon=5, solver_settings=SolverSettings(True, 10000))
        result = controller.solve([0, 0], [0, 2], .1)
        np.testing.assert_allclose(result['U'], legacy['U'], atol=1e-5)
        np.testing.assert_allclose(result['X'], legacy['X'], atol=1e-5)
        self.assertAlmostEqual(result['objective'], legacy['objective'], places=5)

    def test_iteration_limit_rejects_candidate(self):
        design = build_design(ab_config())
        controller = EngineeringMPCController(design['Ad'], design['Bd'], design['Q'], design['R'],
                                              design['P'], 20, delta_bounds=(-.5, .5),
                                              solver_settings=SolverSettings(True, 1))
        result = controller.solve([0, 0], [np.deg2rad(30), 0], 0.)
        self.assertNotIn(result['status'], ('optimal', 'optimal_inaccurate'))
        self.assertTrue(result['error'])
        for field in ('u0', 'X', 'U', 'DU', 'EPS'):
            self.assertIsNone(result[field])

    def test_error_and_corrupt_candidate_do_not_reuse_previous(self):
        controller = simple_controller(solver_settings=SolverSettings())
        self.assertIsNotNone(controller.solve([0, 0], [0, 2], 0.)['u0'])
        with patch.object(controller.problem, 'solve', side_effect=RuntimeError('injected API failure')):
            result = controller.solve([0, 0], [0, 2], 0.)
        self.assertEqual(result['status'], 'solver_error')
        for field in ('u0', 'X', 'U', 'DU', 'EPS', 'solve_time_s', 'setup_time_s'):
            self.assertIsNone(result[field])
        for corruption in ('missing', 'nonfinite', 'shape', 'objective'):
            controller = simple_controller(solver_settings=SolverSettings())
            solve = controller.problem.solve
            def corrupt(**kwargs):
                solve(**kwargs)
                if corruption == 'objective':
                    controller.problem._value = np.nan
                else:
                    value = {'missing': None, 'nonfinite': np.full(controller.X.shape, np.nan),
                             'shape': np.zeros((1, 1))}[corruption]
                    controller.X.save_value(value)
            with patch.object(controller.problem, 'solve', side_effect=corrupt):
                result = controller.solve([0, 0], [0, 2], 0.)
            self.assertEqual(result['status'], 'invalid_solution')
            self.assertIsNone(result['u0'])

    def test_diagnostics_are_honest(self):
        controller = simple_controller(solver_settings=SolverSettings())
        result = controller.solve([0, 0], [0, 2], 0.)
        info = controller.problem.solver_stats.extra_stats.info
        self.assertEqual(result['backend_status'], info.status)
        for field, info_field in (('setup_time_s', 'setup_time'), ('update_time_s', 'update_time'),
                                  ('backend_solve_time_s', 'solve_time'), ('run_time_s', 'run_time')):
            self.assertEqual(result[field], getattr(info, info_field))
        self.assertEqual(result['solve_time_s'], info.run_time)
        solve = controller.problem.solve
        def no_info(**kwargs):
            solve(**kwargs)
            controller.problem.solver_stats.extra_stats = None
        with patch.object(controller.problem, 'solve', side_effect=no_info):
            missing = controller.solve([0, 0], [0, 1], 0.)
        self.assertIsNone(missing['setup_time_s'])
        self.assertIsNone(missing['backend_solve_time_s'])


if __name__ == '__main__':
    unittest.main()
