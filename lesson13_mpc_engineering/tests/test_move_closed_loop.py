import importlib.util
import unittest
import numpy as np
from .test_plant_actuator_demo import demo_config

if importlib.util.find_spec('lesson13_mpc_engineering.src.closed_loop'):
    from lesson13_mpc_engineering.src.closed_loop import simulate_closed_loop
    from lesson13_mpc_engineering.src.constraint_manager import guard_command
    from lesson13_mpc_engineering.src.model_loader import build_design
else:
    simulate_closed_loop = guard_command = build_design = None


def ab_config(duration=.1):
    config = demo_config()
    config.update(mpc={'dt_s': .01, 'horizon_steps': 20},
                  experiment_ab={'duration_s': duration, 'target_deg': 30., 'seed': 2026,
                                 'input_rate_weights': [0., .5, 5.]},
                  constraints={'position_limit_deg': 60.},
                  measurement={'position_noise_std_deg': .05},
                  process_noise={'disturbance_torque_std_nm': .15},
                  initial_uncertainty={'position_std_deg': 5., 'velocity_std_rad_s': 1.},
                  design_limits={'max_position_error_deg': 5., 'max_velocity_error_rad_s': 1.5,
                                 'max_torque_nm': 3.})
    return config


class MoveClosedLoopTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(simulate_closed_loop, 'move-aware closed loop not implemented')

    def test_command_guard_amplitude_and_rate(self):
        self.assertEqual(guard_command(2., 0., 2., 50., .01), .5)
        self.assertEqual(guard_command(-2., .5, 2., 50., .01), 0.)
        self.assertEqual(guard_command(2., 0., 2., 50., .02), 1.)
        for args in ((2, 3, 2, 50, .01), (2, 0, 2, 50, 0), (np.nan, 0, 2, 50, .01)):
            with self.assertRaises(ValueError):
                guard_command(*args)

    def test_kf_uses_past_actual_torque(self):
        from lesson10_kalman_filter.src.kalman_filter import DiscreteKalmanFilter
        config = ab_config()
        result = simulate_closed_loop(config, input_rate_weight=.5, hard_rate=True,
                                      measurement_noise=np.zeros(100))
        self.assertEqual(result['status'], 'complete')
        log, design = result['log'], build_design(config)
        cov = design['covariance']
        filter_ = DiscreteKalmanFilter(design['Ad_fast'], design['Bd_fast'], design['C'],
                                      cov['Gd'], cov['Q_process'], cov['R_measurement'],
                                      np.zeros(2), cov['P0'])
        for index in range(len(log['time_s'])):
            if index:
                filter_.predict(log['torque_actual_nm'][index - 1])
            x = filter_.update(log['q_measured_rad'][index])['x_hat']
            np.testing.assert_allclose(x, [log['q_hat_rad'][index], log['dq_hat_rad_s'][index]], atol=1e-12)

    def test_feedback_and_request_deltas_are_distinct(self):
        log = simulate_closed_loop(ab_config(), input_rate_weight=0., hard_rate=False)['log']
        mask = log['control_update'].astype(bool)
        requests, commands = log['torque_request_nm'][mask], log['torque_command_nm'][mask]
        np.testing.assert_allclose(log['request_delta_nm'][mask], np.diff(np.r_[0., requests]))
        np.testing.assert_allclose(log['command_delta_nm'][mask], np.diff(np.r_[0., commands]))
        np.testing.assert_allclose(log['qp_first_delta_nm'][mask], requests - log['feedback_prev_nm'][mask])
        np.testing.assert_allclose(log['delta_actual_nm'], np.diff(np.r_[0., log['torque_actual_nm']]))
        self.assertGreater(np.max(np.abs(requests - log['feedback_prev_nm'][mask])), .5)
        self.assertLessEqual(np.max(np.abs(np.diff(np.r_[0., commands]))), .5 + 1e-12)

    def test_common_noise_is_reused(self):
        noise = np.linspace(-.001, .001, 100)
        for weight in (0., 5.):
            result = simulate_closed_loop(ab_config(), input_rate_weight=weight, hard_rate=False,
                                          measurement_noise=noise)
            np.testing.assert_array_equal(result['log']['measurement_noise_rad'], noise)

    def test_failed_solve_stops_and_preserves_log(self):
        config = ab_config()
        config['initial_condition'] = {'q_true_deg': 59., 'dq_true_rad_s': 100.}
        config['initial_estimate'] = {'position_deg': 59., 'velocity_rad_s': 100.}
        result = simulate_closed_loop(config, input_rate_weight=.5, hard_rate=True,
                                      measurement_noise=np.zeros(100))
        self.assertEqual(result['status'], 'solver_failure')
        self.assertTrue(result['failure_reason'])
        self.assertEqual(len(result['log']['time_s']), 1)
        self.assertEqual(len(result['plans']), 0)
        self.assertTrue(np.isnan(result['log']['torque_actual_nm'][0]))

    def test_post_step_safety_stop_preserves_endpoint(self):
        config = ab_config()
        config['initial_condition'] = {'q_true_deg': 59., 'dq_true_rad_s': 100.}
        config['initial_estimate'] = {'position_deg': 59., 'velocity_rad_s': 0.}
        result = simulate_closed_loop(config, input_rate_weight=.5, hard_rate=True,
                                      measurement_noise=np.zeros(100))
        self.assertEqual(result['status'], 'safety_stop')
        self.assertEqual(len(result['log']['time_s']), 1)
        self.assertLess(result['log']['q_true_rad'][-1], np.deg2rad(60.))
        self.assertGreater(result['final_state'][0], np.deg2rad(60.))
        self.assertAlmostEqual(result['final_state_time_s'], .001)

    def test_invalid_periods_and_noise(self):
        for section, name, value in (('mpc', 'dt_s', .0105), ('experiment_ab', 'duration_s', .1005),
                                     ('mpc', 'dt_s', True), ('measurement', 'position_noise_std_deg', np.nan)):
            config = ab_config()
            config[section][name] = value
            with self.assertRaises(ValueError):
                simulate_closed_loop(config, input_rate_weight=.5, hard_rate=True)
        for noise in (np.zeros(99), np.full(100, np.nan)):
            with self.assertRaises(ValueError):
                simulate_closed_loop(ab_config(), input_rate_weight=.5, hard_rate=True, measurement_noise=noise)


if __name__ == '__main__':
    unittest.main()
