import importlib.util
import unittest

import numpy as np

if importlib.util.find_spec('lesson13_mpc_engineering.src.solver_monitor'):
    from lesson13_mpc_engineering.src.solver_monitor import assess_result, SolverMonitor
else:
    assess_result = SolverMonitor = None


def candidate(**kwargs):
    value = {'status': 'optimal', 'backend_status': 'solved', 'u0': np.array([.2]),
             'X': np.zeros((2, 2)), 'U': np.array([[.2]]), 'DU': np.array([[.2]]),
             'EPS': None, 'objective': 1., 'max_hard_violation': 0., 'max_soft_violation': None,
             'error': None}
    value.update(kwargs)
    return value


class SolverMonitorTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(assess_result, 'solver monitor not implemented')

    def test_budget_and_deadline_are_distinct(self):
        for elapsed, accepted, deadline in ((.004, True, False), (.005, True, False),
                                           (.007, False, False), (.010, False, False), (.012, False, True)):
            result = assess_result(candidate(), elapsed, .005, .01)
            self.assertEqual(result['accepted'], accepted)
            self.assertEqual(result['deadline_overrun'], deadline)
            self.assertEqual(result['budget_overrun'], elapsed > .005)
        for args in ((-1, .005, .01), (np.nan, .005, .01), (.001, .01, .005), (.001, True, .01)):
            with self.assertRaises(ValueError):
                assess_result(candidate(), *args)

    def test_bad_status_and_bad_plan_are_rejected(self):
        for status, backend, expected in (('infeasible', 'primal infeasible', 'infeasible'),
                                           ('user_limit', 'maximum iterations reached', 'iteration_limit'),
                                           ('user_limit', 'run time limit reached', 'backend_time_limit'),
                                           ('solver_error', None, 'solver_error'), ('strange', None, 'unknown_status')):
            result = assess_result(candidate(status=status, backend_status=backend), .001, .005, .01)
            self.assertFalse(result['accepted'])
            self.assertEqual(result['category'], expected)
            self.assertEqual(result['raw_status'], status)
        for change in ({'u0': np.array([np.nan])}, {'X': None}, {'objective': np.inf},
                       {'max_hard_violation': 1e-3}, {'EPS': np.array([[np.nan]]), 'max_soft_violation': 0.},
                       {'max_hard_violation': None}):
            self.assertFalse(assess_result(candidate(**change), .001, .005, .01)['accepted'])
        self.assertTrue(assess_result(candidate(status='optimal_inaccurate'), .001, .005, .01)['accepted'])

    def test_three_consecutive_good_results_recover(self):
        monitor = SolverMonitor()
        good = assess_result(candidate(), .002, .005, .01)
        monitor.reject('solver_error', 'injected')
        self.assertEqual(monitor.observe(good)['mode_after'], 'fallback')
        self.assertEqual(monitor.observe(good)['mode_after'], 'fallback')
        result = monitor.observe(good)
        self.assertEqual(result['mode_after'], 'mpc')
        self.assertTrue(result['recovered'])
        monitor.reject('solver_error', 'again')
        monitor.observe(good)
        monitor.reject('busy', 'single worker occupied')
        self.assertEqual(monitor.good_count, 0)
        self.assertEqual(monitor.observe(good)['mode_after'], 'fallback')
        for invalid in (0, True, 2.5):
            with self.assertRaises(ValueError):
                SolverMonitor(invalid)


if __name__ == '__main__':
    unittest.main()
