import unittest
from pathlib import Path

import numpy as np

from lesson11_lqg_output_feedback.src.closed_loop import simulate_lqg
from lesson11_lqg_output_feedback.src.lqg_controller import LQGController
from lesson11_lqg_output_feedback.src.model_loader import load_config


class SpyEstimator:
    def __init__(self):
        self.seen = None

    def step(self, measurement, applied_input):
        self.seen = (measurement, applied_input)
        return {"x_hat": np.array([0.0, 0.0]), "nis": 0.0}


class LQGControllerTest(unittest.TestCase):
    def test_controller_passes_prior_applied_torque_and_clips_request(self):
        spy = SpyEstimator()
        controller = LQGController(spy, np.array([[20.0, 0.0]]), torque_limit=1.0)

        result = controller.step(
            0.2,
            previous_applied_torque=0.4,
            x_ref=np.array([1.0, 0.0]),
        )

        self.assertEqual(spy.seen, (0.2, 0.4))
        self.assertGreater(result["torque_request"], 1.0)
        self.assertEqual(result["torque_applied"], 1.0)

    def test_controller_rejects_vector_measurement_and_bad_reference_shape(self):
        spy = SpyEstimator()
        controller = LQGController(spy, np.array([[1.0, 0.0]]), torque_limit=1.0)

        with self.assertRaises(ValueError):
            controller.step([0.2], 0.0, np.array([0.0, 0.0]))
        with self.assertRaises(ValueError):
            controller.step(0.2, 0.0, np.array([0.0]))
        with self.assertRaises(ValueError):
            LQGController(spy, np.array([[1.0, 0.0]]), torque_limit=0.0)


class SimulationTimingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_path = Path(__file__).parents[1] / "config" / "lqg.yaml"
        cls.config = load_config(config_path)
        cls.config["simulation"]["duration_s"] = 0.02

    def test_lqg_log_uses_estimate_and_records_actual_limited_torque(self):
        log = simulate_lqg(self.config, mode="recursive_lqg")

        self.assertIn("q_hat_rad", log)
        self.assertIn("nis", log)
        self.assertTrue(np.all(np.abs(log["torque_applied_nm"]) <= 3.0))

    def test_full_state_baseline_has_no_estimator_diagnostics(self):
        log = simulate_lqg(self.config, mode="full_state_lqr")

        self.assertTrue(np.all(np.isnan(log["nis"])))
        self.assertEqual(log["controller_mode"][0], "full_state_lqr")


if __name__ == "__main__":
    unittest.main()
