from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from excitation import MultiSineExcitation
from estimator import identify_j_b
from plant import SingleAxisPlant
from signal_processing import lowpass


class ExcitationTest(unittest.TestCase):
    def test_multisine_is_zero_at_time_zero_and_has_expected_value(self):
        signal = MultiSineExcitation([0.5, 1.0], [0.4, 0.2])

        self.assertAlmostEqual(signal.evaluate(0.0), 0.0)
        expected = 0.4 * np.sin(2.0 * np.pi * 0.5 * 0.25) + 0.2
        self.assertAlmostEqual(signal.evaluate(0.25), expected)

    def test_mismatched_lists_are_rejected(self):
        with self.assertRaises(ValueError):
            MultiSineExcitation([0.5], [0.4, 0.2])


class PlantTest(unittest.TestCase):
    def test_torque_is_limited_and_acceleration_uses_applied_torque(self):
        plant = SingleAxisPlant(0.02, 0.08, 0.0, torque_limit_nm=2.0)

        sample = plant.step(5.0, 0.001)

        self.assertEqual(sample["torque_applied_nm"], 2.0)
        self.assertAlmostEqual(sample["acceleration_true_rad_s2"], 100.0)

    def test_nonpositive_dt_is_rejected(self):
        plant = SingleAxisPlant(0.02, 0.08, 0.0, torque_limit_nm=2.0)

        with self.assertRaises(ValueError):
            plant.step(0.0, 0.0)


class EstimatorTest(unittest.TestCase):
    def test_exact_data_recovers_inertia_and_damping(self):
        velocity = np.array([-2.0, -1.0, 0.5, 1.5])
        acceleration = np.array([3.0, -2.0, 1.0, 4.0])
        torque = 0.02 * acceleration + 0.08 * velocity

        result = identify_j_b(velocity, acceleration, torque)

        self.assertAlmostEqual(result["inertia_hat"], 0.02)
        self.assertAlmostEqual(result["damping_hat"], 0.08)
        self.assertAlmostEqual(result["torque_rmse"], 0.0)

    def test_rank_deficient_data_is_rejected(self):
        with self.assertRaises(ValueError):
            identify_j_b(np.ones(5), 2.0 * np.ones(5), np.ones(5))


class SignalProcessingTest(unittest.TestCase):
    def test_nonpositive_sample_period_is_rejected(self):
        with self.assertRaises(ValueError):
            lowpass(np.ones(20), 0.0, 10.0)


if __name__ == "__main__":
    unittest.main()
