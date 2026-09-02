from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from excitation import MultiSineExcitation
from estimator import identify_j_b
from metrics import calculate_metrics, rmse
from plant import SingleAxisPlant
from signal_processing import lowpass
from run_sysid import run_case, run_validation


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


class MetricsTest(unittest.TestCase):
    def test_metrics_report_parameter_and_validation_errors(self):
        metrics = calculate_metrics(
            {
                "inertia_hat": 0.021,
                "damping_hat": 0.076,
                "torque_rmse": 0.03,
                "condition_number": 8.0,
            },
            {"inertia": 0.020, "damping": 0.080},
            np.array([0.0, 1.0]),
            np.array([0.0, 1.2]),
        )

        self.assertAlmostEqual(metrics["inertia_error_percent"], 5.0)
        self.assertAlmostEqual(metrics["damping_error_percent"], 5.0)
        self.assertAlmostEqual(
            metrics["validation_position_rmse_rad"], np.sqrt(0.02)
        )


class RunnerTest(unittest.TestCase):
    def test_normal_case_writes_applied_torque_log_and_parameters(self):
        config = {
            "simulation": {"dt": 0.002, "duration_s": 2.0, "seed": 7},
            "plant_true": {
                "inertia": 0.02,
                "damping": 0.08,
                "coulomb_friction": 0.0,
            },
            "sensor": {"position_noise_std_deg": 0.0},
            "actuator": {"torque_limit_nm": 2.0},
            "excitation": {
                "frequencies_hz": [0.5, 1.3],
                "amplitudes_nm": [0.45, 0.30],
            },
            "identification": {"discard_start_s": 0.2, "lowpass_cutoff_hz": 15.0},
        }
        with tempfile.TemporaryDirectory() as output_dir:
            result = run_case("normal", config, Path(output_dir))

            self.assertIn("torque_applied_nm", result["log_columns"])
            self.assertTrue(result["parameters_path"].is_file())
            self.assertTrue(result["figure_path"].is_file())

    def test_validation_keeps_normal_case_linear(self):
        config = {
            "simulation": {"dt": 0.002, "duration_s": 4.0, "seed": 7},
            "plant_true": {
                "inertia": 0.02,
                "damping": 0.08,
                "coulomb_friction": 0.10,
            },
            "sensor": {"position_noise_std_deg": 0.0},
            "actuator": {"torque_limit_nm": 2.0},
            "excitation": {
                "frequencies_hz": [0.5, 1.3],
                "amplitudes_nm": [0.45, 0.30],
            },
            "identification": {"discard_start_s": 0.2, "lowpass_cutoff_hz": 15.0},
        }
        with tempfile.TemporaryDirectory() as output_dir:
            result = run_validation(config, 0.02, 0.08, Path(output_dir))

        self.assertLess(
            rmse(result["true_position_rad"], result["model_position_rad"]), 0.01
        )

    def test_model_mismatch_writes_residual_diagnostic(self):
        config = {
            "simulation": {"dt": 0.002, "duration_s": 2.0, "seed": 7},
            "plant_true": {
                "inertia": 0.02,
                "damping": 0.08,
                "coulomb_friction": 0.10,
            },
            "sensor": {"position_noise_std_deg": 0.02},
            "actuator": {"torque_limit_nm": 2.0},
            "excitation": {
                "frequencies_hz": [0.5, 1.3],
                "amplitudes_nm": [0.45, 0.30],
            },
            "identification": {"discard_start_s": 0.2, "lowpass_cutoff_hz": 15.0},
        }
        with tempfile.TemporaryDirectory() as output_dir:
            result = run_case("model_mismatch", config, Path(output_dir))

            self.assertTrue(result["residual_figure_path"].is_file())


if __name__ == "__main__":
    unittest.main()
