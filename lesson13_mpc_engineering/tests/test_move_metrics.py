import importlib.util
import unittest
import numpy as np

if importlib.util.find_spec('lesson13_mpc_engineering.src.metrics'):
    from lesson13_mpc_engineering.src.metrics import calculate_metrics
else:
    calculate_metrics = None


def metric_log():
    return {'time_s': np.arange(30) * .001, 'control_update': np.arange(30) % 10 == 0,
            'q_true_rad': np.zeros(30), 'q_ref_rad': np.ones(30),
            'torque_request_nm': np.repeat([1., .5, -.5], 10),
            'torque_command_nm': np.repeat([.5, 0., -.5], 10),
            'torque_actual_nm': np.zeros(30), 'delta_actual_nm': np.full(30, .05),
            'feedback_prev_nm': np.zeros(30), 'qp_first_delta_nm': np.full(30, np.nan),
            'command_delta_nm': np.where(np.arange(30) % 10 == 0, .5, np.nan),
            'guard_modified': np.where(np.arange(30) % 10 == 0, 1, np.nan),
            'solver_status': np.array(['optimal' if i % 10 == 0 else '' for i in range(30)]),
            'solver_wall_time_s': np.full(30, np.nan), 'solver_solve_time_s': np.full(30, np.nan),
            'solver_iterations': np.full(30, np.nan), 'max_planned_delta_nm': np.full(30, np.nan)}


class MoveMetricsTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(calculate_metrics, 'calculate_metrics not implemented')

    def test_metrics_use_control_events_only(self):
        metrics = calculate_metrics(metric_log(), .01)
        self.assertAlmostEqual(metrics['rms_request_delta_nm'], np.sqrt(.75))
        self.assertEqual(metrics['max_request_delta_nm'], 1)
        self.assertEqual(metrics['tracking_rmse_rad'], 1)
        self.assertEqual(metrics['solver_success_count'], 3)

    def test_metrics_separate_rate_and_move(self):
        metrics = calculate_metrics(metric_log(), .01)
        self.assertAlmostEqual(metrics['max_actual_slew_nm_s'], 50)
        self.assertEqual(metrics['max_command_delta_nm'], .5)
        self.assertEqual(metrics['mean_solver_wall_ms'], None)
        with self.assertRaises(ValueError):
            calculate_metrics(metric_log(), 0)


if __name__ == '__main__':
    unittest.main()
