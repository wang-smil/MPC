import importlib.util
import unittest

import numpy as np

if importlib.util.find_spec('lesson13_mpc_engineering.src.run_plant_actuator_demo') is not None:
    from lesson13_mpc_engineering.src.run_plant_actuator_demo import simulate_demo
else:
    simulate_demo = None


def demo_config():
    return {'nominal_model': {'inertia': .019762, 'damping': .080257},
            'plant': {'dt_s': .001, 'coulomb_friction': .12, 'friction_smoothing': .02,
                      'integration': 'semi_implicit'},
            'actuator': {'torque_limit_nm': 2., 'slew_rate_nm_s': 50.,
                         'delay_s': .002, 'time_constant_s': .005},
            'demo': {'duration_s': .5, 'control_dt_s': .01}}


class PlantActuatorDemoTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(simulate_demo, 'simulate_demo is not implemented yet')

    def test_demo_alignment_and_limits(self):
        log = simulate_demo(demo_config())
        np.testing.assert_allclose(log['time_s'], np.arange(500) * .001, atol=1e-14)
        self.assertEqual(log['torque_request_nm'][49], 0.)
        self.assertEqual(log['torque_request_nm'][50], 1.)
        changes = np.flatnonzero(np.diff(log['torque_command_nm'])) + 1
        self.assertTrue(np.all(changes % 10 == 0))
        self.assertLessEqual(np.max(np.abs(np.diff(log['torque_command_nm']))), .5 + 1e-12)
        self.assertLessEqual(np.max(np.abs(log['torque_actual_nm'])), 2)
        self.assertLessEqual(np.max(np.abs(log['delta_actual_nm'])) / .001, 50 + 1e-9)
        self.assertTrue(all(np.all(np.isfinite(array)) for array in log.values()))

    def test_demo_separates_request_command_and_actual(self):
        log = simulate_demo(demo_config())
        self.assertEqual(log['torque_request_nm'][50], 1.)
        self.assertEqual(log['torque_command_nm'][50], .5)
        self.assertEqual(log['torque_actual_nm'][50], 0.)
        self.assertEqual(log['torque_actual_nm'][51], 0.)
        self.assertAlmostEqual(log['torque_actual_nm'][52], .05)
        self.assertEqual(log['dq_rad_s'][52], 0.)
        self.assertAlmostEqual(log['dq_rad_s'][53], .05 / .019762 * .001)
        self.assertAlmostEqual(log['q_rad'][53], .05 / .019762 * .001**2)

    def test_bad_duration_and_period_are_rejected(self):
        for section, key, value in [('demo', 'duration_s', .5005),
                                    ('demo', 'control_dt_s', .0105),
                                    ('demo', 'duration_s', 0),
                                    ('demo', 'control_dt_s', -1),
                                    ('plant', 'dt_s', 0)]:
            config = demo_config()
            config[section][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                simulate_demo(config)


if __name__ == '__main__':
    unittest.main()
