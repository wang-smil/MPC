import importlib.util
import unittest

import numpy as np

from .test_move_metrics import metric_log

if importlib.util.find_spec('lesson13_mpc_engineering.src.feasibility_metrics'):
    from lesson13_mpc_engineering.src.feasibility_metrics import calculate_feasibility_metrics
else:
    calculate_feasibility_metrics = None


def hand_result():
    log = {key: value[:15] for key, value in metric_log().items()}
    log['dq_true_rad_s'] = np.r_[np.full(10, 2.), np.ones(5)]
    log['qp_initial_slack_rad_s'] = np.full(15, np.nan)
    log['qp_initial_slack_rad_s'][[0, 10]] = [.5, .25]
    log['max_predicted_slack_rad_s'] = log['qp_initial_slack_rad_s'].copy()
    return {'log': log, 'final_state': np.array([0., 1.]), 'final_state_time_s': .015}


class FeasibilityMetricsTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(calculate_feasibility_metrics, 'feasibility metrics not implemented')

    def test_integrals_use_correct_intervals(self):
        metrics = calculate_feasibility_metrics(hand_result(), .01, .001, 1.5)
        self.assertEqual(metrics['max_actual_velocity_violation_rad_s'], .5)
        self.assertAlmostEqual(metrics['actual_overspeed_duration_s'], .01)
        self.assertAlmostEqual(metrics['actual_overspeed_integral_rad'], .005)
        self.assertAlmostEqual(metrics['initial_slack_hold_integral_rad'], .00625)
        self.assertEqual(metrics['max_predicted_slack_rad_s'], .5)
        self.assertEqual(metrics['solver_call_count'], 2)

    def test_missing_and_terminal_evidence(self):
        result = hand_result()
        result['log']['qp_initial_slack_rad_s'][:] = np.nan
        result['log']['max_predicted_slack_rad_s'][:] = np.nan
        result['final_state'][1] = 3.
        metrics = calculate_feasibility_metrics(result, .01, .001, 1.5)
        self.assertEqual(metrics['max_actual_velocity_violation_rad_s'], 1.5)
        self.assertAlmostEqual(metrics['actual_overspeed_integral_rad'], .005)
        self.assertIsNone(metrics['max_initial_slack_rad_s'])
        self.assertIsNone(metrics['initial_slack_hold_integral_rad'])
        for args in ((0, .001, 1.5), (.01, np.nan, 1.5), (.01, .001, True)):
            with self.assertRaises(ValueError):
                calculate_feasibility_metrics(result, *args)


if __name__ == '__main__':
    unittest.main()
