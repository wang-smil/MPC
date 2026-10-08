import importlib.util
import unittest

import numpy as np

if importlib.util.find_spec('lesson13_mpc_engineering.src.actuator') is not None:
    from lesson13_mpc_engineering.src.actuator import TorqueActuator
else:
    TorqueActuator = None


class TorqueActuatorTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(TorqueActuator, 'TorqueActuator is not implemented yet')

    def test_two_millisecond_delay_has_exact_interval_alignment(self):
        actuator = TorqueActuator(time_constant_s=0, slew_rate_nm_s=1e6)
        self.assertEqual([actuator.step(1)['torque_actual_nm'] for _ in range(3)], [0, 0, 1])

    def test_lag_matches_exponential_response(self):
        actuator = TorqueActuator(delay_s=0, slew_rate_nm_s=1e6)
        self.assertAlmostEqual(actuator.step(1)['torque_actual_nm'], 1 - np.exp(-.2))

    def test_positive_negative_and_reversal_respect_limits(self):
        actuator = TorqueActuator(delay_s=0, time_constant_s=0)
        previous = 0.
        for command in [100.] * 60 + [-100.] * 100:
            result = actuator.step(command)
            current = result['torque_actual_nm']
            self.assertLessEqual(abs(current), 2)
            self.assertLessEqual(abs(current - previous), .05 + 1e-12)
            self.assertAlmostEqual(result['delta_actual_nm'], current - previous)
            self.assertLessEqual(abs(result['command_nm']), 2)
            previous = current
        self.assertAlmostEqual(previous, -2)

    def test_zero_delay_and_zero_lag(self):
        actuator = TorqueActuator(delay_s=0, time_constant_s=0, slew_rate_nm_s=1e6)
        self.assertEqual(actuator.step(1)['torque_actual_nm'], 1)
        actuator = TorqueActuator(initial_torque_nm=.4, time_constant_s=0, slew_rate_nm_s=1e6)
        self.assertEqual(actuator.step(1)['torque_actual_nm'], .4)

    def test_invalid_configuration(self):
        cases = {'dt_s': [0, -1, True, np.nan, np.inf],
                 'torque_limit_nm': [0, -1, True, np.nan, np.inf],
                 'slew_rate_nm_s': [0, -1, True, np.nan, np.inf],
                 'delay_s': [-1, .0015, True, np.nan, np.inf],
                 'time_constant_s': [-1, True, np.nan, np.inf],
                 'initial_torque_nm': [2.1, -2.1, True, np.nan, np.inf]}
        for key, values in cases.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    TorqueActuator(**{key: value})

    def test_invalid_command_does_not_advance_queue(self):
        actuator, reference = TorqueActuator(), TorqueActuator()
        for instance in (actuator, reference):
            instance.step(.8)
        for bad in (True, np.nan, np.inf, [1, 2]):
            with self.assertRaises(ValueError):
                actuator.step(bad)
        for command in (-1, .5, 0, 0):
            self.assertEqual(actuator.step(command), reference.step(command))


if __name__ == '__main__':
    unittest.main()
