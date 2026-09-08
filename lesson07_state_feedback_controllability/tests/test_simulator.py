"""Tests for noisy, saturation-aware closed-loop simulation and metrics."""

import unittest

import numpy as np

from lesson07_state_feedback_controllability.src.model import (
    build_continuous_model,
    discretize_zoh,
    load_config,
)
from lesson07_state_feedback_controllability.src.pole_design import (
    design_discrete_feedback,
    second_order_poles,
)


class ClosedLoopSimulationTest(unittest.TestCase):
    def test_simulation_logs_clipped_torque_and_aligned_signals(self) -> None:
        from lesson07_state_feedback_controllability.src.simulator import (
            simulate_closed_loop,
        )

        config = load_config("lesson07_state_feedback_controllability/config/controller.yaml")
        A, B = build_continuous_model(config["model"]["inertia"], config["model"]["damping"])
        Ad, Bd = discretize_zoh(A, B, config["simulation"]["dt_s"])
        K = design_discrete_feedback(
            Ad,
            Bd,
            second_order_poles(0.8, 0.8),
            config["simulation"]["dt_s"],
        )["K"]

        log = simulate_closed_loop(config, K, plant_inertia=config["model"]["inertia"])

        required = {
            "time_s",
            "q_ref_rad",
            "q_rad",
            "q_measured_rad",
            "dq_est_rad_s",
            "torque_unsat_nm",
            "torque_applied_nm",
            "saturated",
        }
        self.assertTrue(required.issubset(log))
        self.assertTrue(np.all(np.abs(log["torque_applied_nm"]) <= 3.0))
        self.assertTrue(all(values.size == log["time_s"].size for values in log.values()))


class MetricsTest(unittest.TestCase):
    def test_metrics_use_applied_torque_and_final_reference(self) -> None:
        from lesson07_state_feedback_controllability.src.metrics import calculate_metrics

        log = {
            "time_s": np.array([0.0, 0.1, 0.2]),
            "q_ref_rad": np.array([1.0, 1.0, 1.0]),
            "q_rad": np.array([0.0, 1.0, 1.0]),
            "torque_applied_nm": np.array([0.0, 2.0, 0.0]),
            "saturated": np.array([False, True, False]),
        }

        metrics = calculate_metrics(log)

        self.assertAlmostEqual(metrics["tracking_rmse_rad"], np.sqrt(1.0 / 3.0))
        self.assertAlmostEqual(metrics["peak_torque_nm"], 2.0)
        self.assertAlmostEqual(metrics["rms_torque_nm"], np.sqrt(4.0 / 3.0))
        self.assertAlmostEqual(metrics["saturation_ratio_percent"], 100.0 / 3.0)
        self.assertAlmostEqual(metrics["final_error_rad"], 0.0)
        self.assertAlmostEqual(metrics["settling_time_s"], 0.1)


if __name__ == "__main__":
    unittest.main()
