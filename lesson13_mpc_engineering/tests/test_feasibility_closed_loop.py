import unittest

import numpy as np

from lesson10_kalman_filter.src.kalman_filter import DiscreteKalmanFilter
from lesson13_mpc_engineering.src.closed_loop import simulate_closed_loop
from lesson13_mpc_engineering.src.model_loader import build_design
from .test_move_closed_loop import ab_config


def feasibility_config(duration=.04):
    config = ab_config(duration)
    config['initial_condition'] = {'q_true_deg': 0., 'dq_true_rad_s': 2.}
    config['initial_estimate'] = {'position_deg': 0., 'velocity_rad_s': 2.}
    return config


def run_soft(config, **kwargs):
    return simulate_closed_loop(config, input_rate_weight=.5, hard_rate=True,
                                velocity_mode='soft', velocity_bounds=(-1.5, 1.5), **kwargs)


class FeasibilityClosedLoopTest(unittest.TestCase):
    def test_hard_current_stops_without_applying_stale_input(self):
        result = simulate_closed_loop(feasibility_config(), input_rate_weight=.5, hard_rate=True,
                                      velocity_mode='hard', velocity_bounds=(-1.5, 1.5),
                                      include_current_velocity=True, measurement_noise=np.zeros(40))
        self.assertEqual(result['status'], 'solver_failure')
        self.assertEqual(len(result['log']['time_s']), 1)
        self.assertTrue(np.isnan(result['log']['torque_actual_nm'][0]))
        self.assertEqual(result['plans'], [])
        self.assertEqual(result['final_state_time_s'], 0.)

    def test_soft_current_runs_and_logs_epsilon(self):
        result = run_soft(feasibility_config(), include_current_velocity=True, measurement_noise=np.zeros(40))
        self.assertEqual(result['status'], 'complete', result['failure_reason'])
        log = result['log']
        self.assertGreaterEqual(log['qp_initial_slack_rad_s'][0], .5 - 1e-5)
        mask = log['control_update'].astype(bool)
        for row, plan in zip(np.flatnonzero(mask), result['plans']):
            self.assertEqual(plan['velocity_indices'][0], 0)
            self.assertAlmostEqual(log['qp_initial_slack_rad_s'][row], plan['EPS'][0, 0])
        self.assertLessEqual(np.max(np.abs(log['torque_actual_nm'])), 2 + 1e-5)
        self.assertLessEqual(np.max(np.abs(log['delta_actual_nm'])), .05 + 1e-10)
        self.assertLessEqual(np.max(np.abs(log['command_delta_nm'][mask])), .5 + 1e-10)

    def test_future_slack_is_not_initial_slack(self):
        log = run_soft(feasibility_config(), measurement_noise=np.zeros(40))['log']
        self.assertTrue(np.all(np.isnan(log['qp_initial_slack_rad_s'])))
        self.assertTrue(np.isfinite(log['max_predicted_slack_rad_s'][0]))

    def test_noise_and_past_input_remain_causal(self):
        config = feasibility_config()
        noise = np.linspace(-.001, .001, 40)
        for rho in (100., 10000.):
            result = run_soft(config, include_current_velocity=True, slack_weight=rho, measurement_noise=noise)
            self.assertEqual(result['status'], 'complete')
            log, design = result['log'], build_design(config)
            np.testing.assert_array_equal(log['measurement_noise_rad'], noise)
            cov = design['covariance']
            kf = DiscreteKalmanFilter(design['Ad_fast'], design['Bd_fast'], design['C'],
                                      cov['Gd'], cov['Q_process'], cov['R_measurement'], [0., 2.], cov['P0'])
            for index in range(40):
                if index:
                    kf.predict(log['torque_actual_nm'][index - 1])
                expected = kf.update(log['q_measured_rad'][index])['x_hat']
                np.testing.assert_allclose(expected, [log['q_hat_rad'][index], log['dq_hat_rad_s'][index]], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
