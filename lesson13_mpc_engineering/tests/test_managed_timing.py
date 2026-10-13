import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from lesson10_kalman_filter.src.kalman_filter import DiscreteKalmanFilter
from lesson13_mpc_engineering.src.model_loader import build_design
from .test_move_closed_loop import ab_config

if importlib.util.find_spec('lesson13_mpc_engineering.src.managed_closed_loop'):
    from lesson13_mpc_engineering.src.managed_closed_loop import simulate_managed_closed_loop
else:
    simulate_managed_closed_loop = None


def fg_config(duration=.08):
    config = ab_config(duration)
    config['constraints']['velocity_limit_rad_s'] = 1.5
    config['solver'] = {'max_iter': 4000, 'warm_start': True}
    config['runtime'] = {'budget_s': .005, 'recovery_successes': 3, 'nominal_completion_s': .002}
    config['experiment_fg'] = {'duration_s': duration, 'target_deg': 30., 'seed': 2026,
                               'replay_samples': 4, 'repetitions': 1}
    return config


class ManagedTimingTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(simulate_managed_closed_loop, 'managed loop not implemented')

    def test_late_plan_is_dropped_at_budget(self):
        result = simulate_managed_closed_loop(fg_config(), faults={0: {'kind': 'delay', 'completion_s': .007}})
        self.assertEqual(result['status'], 'complete')
        job = result['jobs'][0]
        self.assertTrue(job['budget_missed'])
        self.assertFalse(job['deadline_missed'])
        self.assertTrue(job['discarded'])
        self.assertFalse(job['executed'])
        self.assertAlmostEqual(job['budget_event_time_s'], .005)
        self.assertAlmostEqual(job['completion_time_s'], .007)
        log = result['log']
        np.testing.assert_array_equal(log['torque_command_nm'][:5], np.zeros(5))
        self.assertEqual(log['selected_source'][5], 'fallback')
        self.assertLessEqual(abs(log['torque_command_nm'][5]), .25 + 1e-10)
        self.assertFalse(any(e['job_id'] == 0 and e.get('selected_source') == 'mpc' for e in result['events']))

    def test_deadline_and_busy_skip(self):
        result = simulate_managed_closed_loop(fg_config(), faults={0: {'kind': 'delay', 'completion_s': .012}})
        self.assertTrue(result['jobs'][0]['deadline_missed'])
        self.assertFalse(any(job['cycle_index'] == 1 for job in result['jobs']))
        self.assertTrue(any(e['event'] == 'busy_skip' and abs(e['time_s'] - .01) < 1e-12 for e in result['events']))
        self.assertTrue(any(job['cycle_index'] == 2 for job in result['jobs']))
        for old, new in zip(result['jobs'], result['jobs'][1:]):
            self.assertLessEqual(old['completion_time_s'], new['start_time_s'])

    def test_completion_ties_and_zero_duration(self):
        for latency, accepted in ((.005, True), (.010, False), (0., True)):
            result = simulate_managed_closed_loop(fg_config(), faults={0: {'kind': 'delay', 'completion_s': latency}})
            self.assertEqual(result['jobs'][0]['accepted'], accepted)
            self.assertTrue(any(job['cycle_index'] == 1 for job in result['jobs']))
            if latency == 0.:
                self.assertEqual(result['log']['torque_command_nm'][0], 0.)

    def test_recovery_and_fresh_fallback(self):
        result = simulate_managed_closed_loop(fg_config(), faults={0: {'kind': 'error'}, 2: {'kind': 'nonfinite'}})
        recoveries = [e for e in result['events'] if e.get('recovered')]
        self.assertEqual(len(recoveries), 1)
        self.assertAlmostEqual(recoveries[0]['time_s'], .052)
        log, design = result['log'], build_design(fg_config())
        reference = np.array([np.deg2rad(30), 0.])
        for event in result['events']:
            if event.get('selected_source') == 'fallback':
                row = int(round(event['time_s'] / .001))
                estimate = [log['q_hat_rad'][row], log['dq_hat_rad_s'][row]]
                expected = -(design['K'] @ (np.array(estimate) - reference)).item()
                self.assertAlmostEqual(event['selected_request_nm'], expected, places=10)
        self.assertFalse(result['jobs'][0]['executed'])
        self.assertFalse(result['jobs'][2]['executed'])

    def test_irregular_guard_and_physical_slew(self):
        config = fg_config()
        noise = np.linspace(-.001, .001, 80)
        result = simulate_managed_closed_loop(config, faults={0: {'kind': 'delay', 'completion_s': .007}}, measurement_noise=noise)
        log = result['log']
        for event in result['events']:
            if event.get('command_elapsed_s') is not None:
                self.assertLessEqual(abs(event['command_delta_nm']), 50 * event['command_elapsed_s'] + 1e-10)
        self.assertLessEqual(np.max(np.abs(log['torque_actual_nm'])), 2 + 1e-10)
        self.assertLessEqual(np.max(np.abs(log['delta_actual_nm'])), .05 + 1e-10)
        design = build_design(config)
        cov = design['covariance']
        kf = DiscreteKalmanFilter(design['Ad_fast'], design['Bd_fast'], design['C'],
                                  cov['Gd'], cov['Q_process'], cov['R_measurement'], [0., 0.], cov['P0'])
        for i in range(80):
            if i:
                kf.predict(log['torque_actual_nm'][i - 1])
            expected = kf.update(log['q_measured_rad'][i])['x_hat']
            np.testing.assert_allclose(expected, [log['q_hat_rad'][i], log['dq_hat_rad_s'][i]], atol=1e-12)

    def test_measured_sync_does_not_claim_event_timing(self):
        # Inject only the monotonic clock, not the controller or validation logic.
        with patch('lesson13_mpc_engineering.src.managed_closed_loop.perf_counter', side_effect=[0., .007, 1., 1.007]):
            result = simulate_managed_closed_loop(fg_config(.02), timing_mode='measured_sync')
        self.assertTrue(all(job['budget_missed'] for job in result['jobs']))
        self.assertTrue(all(job['injected_completion_s'] is None for job in result['jobs']))
        self.assertEqual(result['timing_mode'], 'measured_sync')

    def test_invalid_faults_are_rejected(self):
        for faults in ({True: {'kind': 'error'}}, {100: {'kind': 'error'}}, {0: {'kind': 'other'}},
                       {0: {'kind': 'delay', 'completion_s': .0015}}, {0: {'kind': 'max_iter', 'max_iter': True}}):
            with self.assertRaises(ValueError):
                simulate_managed_closed_loop(fg_config(), faults=faults)


if __name__ == '__main__':
    unittest.main()
