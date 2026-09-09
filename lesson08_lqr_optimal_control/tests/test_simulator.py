"""Tests for the noisy LQR simulation and its cost metrics."""

import unittest

import numpy as np

from lesson08_lqr_optimal_control.src.controller import LQRController
from lesson08_lqr_optimal_control.src.lqr_design import (
    build_bryson_weights,
    design_dlqr,
)
from lesson08_lqr_optimal_control.src.model import (
    build_continuous_model,
    discretize_zoh,
    load_config,
)


class LqrSimulationTest(unittest.TestCase):
    def test_simulation_keeps_command_within_limit_and_logs_costs(self) -> None:
        from lesson08_lqr_optimal_control.src.simulator import simulate_closed_loop

        config = load_config("lesson08_lqr_optimal_control/config/lqr.yaml")
        A, B = build_continuous_model(
            config["model"]["inertia"],
            config["model"]["damping"],
        )
        Ad, Bd = discretize_zoh(A, B, config["simulation"]["dt_s"])
        Q, R = build_bryson_weights(5.0, 1.5, 3.0, 1.0, 1.0, 1.0)
        K = design_dlqr(Ad, Bd, Q, R)["K"]
        controller = LQRController(K, torque_limit_nm=3.0)

        log = simulate_closed_loop(
            config,
            controller,
            Q,
            R,
            plant_inertia=config["model"]["inertia"],
        )

        required = {
            "time_s",
            "q_ref_rad",
            "q_rad",
            "dq_ref_rad_s",
            "dq_rad_s",
            "q_measured_rad",
            "dq_measured_rad_s",
            "position_error_rad",
            "velocity_error_rad_s",
            "torque_unsat_nm",
            "torque_cmd_nm",
            "saturated",
            "state_cost",
            "input_cost",
            "total_stage_cost",
        }
        self.assertTrue(required.issubset(log))
        self.assertTrue(np.all(np.abs(log["torque_cmd_nm"]) <= 3.0))
        np.testing.assert_allclose(
            log["total_stage_cost"],
            log["state_cost"] + log["input_cost"],
        )
        self.assertTrue(all(values.size == log["time_s"].size for values in log.values()))


class LqrMetricsTest(unittest.TestCase):
    def test_metrics_sum_logged_lqr_costs(self) -> None:
        from lesson08_lqr_optimal_control.src.metrics import calculate_metrics

        log = {
            "time_s": np.array([0.0, 0.1]),
            "q_ref_rad": np.ones(2),
            "q_rad": np.ones(2),
            "torque_cmd_nm": np.zeros(2),
            "saturated": np.zeros(2, dtype=bool),
            "state_cost": np.array([2.0, 3.0]),
            "input_cost": np.array([5.0, 7.0]),
            "total_stage_cost": np.array([7.0, 10.0]),
        }

        metrics = calculate_metrics(log)

        self.assertEqual(metrics["state_cost_total"], 5.0)
        self.assertEqual(metrics["input_cost_total"], 12.0)
        self.assertEqual(metrics["lqr_cost_total"], 17.0)


if __name__ == "__main__":
    unittest.main()
