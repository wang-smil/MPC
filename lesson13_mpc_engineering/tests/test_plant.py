import importlib.util
import unittest

import numpy as np

if importlib.util.find_spec('lesson13_mpc_engineering.src.plant') is not None:
    from lesson13_mpc_engineering.src.plant import JointPlant
else:
    JointPlant = None


class JointPlantTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(JointPlant, 'JointPlant is not implemented yet')

    def test_zero_state_remains_at_rest(self):
        np.testing.assert_array_equal(JointPlant().step(0), [0, 0])

    def test_semi_implicit_matches_one_step(self):
        plant = JointPlant(inertia=.02, damping=.08)
        np.testing.assert_allclose(plant.step(1), [.00005, .05], atol=1e-14)

    def test_load_subtracts_from_torque(self):
        plant = JointPlant(inertia=.02, damping=.08)
        np.testing.assert_allclose(plant.step(1, .3), [.000035, .035], atol=1e-14)

    def test_friction_opposes_both_velocity_signs(self):
        for velocity in (-1., 1.):
            plant = JointPlant(damping=0, coulomb_friction=.12, initial_state=[0, velocity])
            self.assertLess(abs(plant.step(0)[1]), 1)
        np.testing.assert_array_equal(JointPlant(coulomb_friction=.12).step(0), [0, 0])

    def test_exact_zoh_matches_lesson08(self):
        from lesson08_lqr_optimal_control.src.model import build_continuous_model, discretize_zoh
        A, B = build_continuous_model(.019762, .080257)
        Ad, Bd = discretize_zoh(A, B, .001)
        plant = JointPlant(initial_state=[.1, -.2], integration='exact_zoh')
        np.testing.assert_allclose(plant.step(.8, .3), Ad @ [.1, -.2] + Bd[:, 0] * .5, atol=1e-12)
        with self.assertRaises(ValueError):
            JointPlant(integration='exact_zoh', coulomb_friction=.12)

    def test_invalid_parameters_and_inputs_leave_state_unchanged(self):
        for key, values in {'inertia': [0, -1, np.nan, np.inf, True],
                            'damping': [-1, np.nan, np.inf, True],
                            'dt_s': [0, -1, np.nan, np.inf, True],
                            'coulomb_friction': [-1, np.nan, np.inf, True],
                            'friction_smoothing': [0, -1, np.nan, np.inf, True]}.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    JointPlant(**{key: value})
        for state in ([0], [[0], [0]], [np.nan, 0], [0, np.inf], [True, 0]):
            with self.assertRaises(ValueError):
                JointPlant(initial_state=state)
        with self.assertRaises(ValueError):
            JointPlant(integration='unknown')
        plant = JointPlant(initial_state=[.1, .2])
        for bad in (np.nan, np.inf, True, [1, 2]):
            for args in ((bad, 0), (0, bad)):
                before = plant.state
                with self.assertRaises(ValueError):
                    plant.step(*args)
                np.testing.assert_array_equal(plant.state, before)

    def test_state_returns_copy(self):
        plant = JointPlant()
        state = plant.state
        state[:] = 42
        np.testing.assert_array_equal(plant.state, [0, 0])
        result = plant.step(1)
        saved = result.copy()
        result[:] = 99
        np.testing.assert_array_equal(plant.state, saved)


if __name__ == '__main__':
    unittest.main()
