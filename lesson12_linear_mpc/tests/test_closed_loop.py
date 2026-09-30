"""Causal timing tests for Lesson 12's multi-rate MPC/LQR simulation."""

from importlib import import_module
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from lesson08_lqr_optimal_control.src.lqr_design import build_bryson_weights, dlqr_from_dare
from lesson08_lqr_optimal_control.src.model import discretize_zoh
from lesson10_kalman_filter.src.kalman_filter import DiscreteKalmanFilter
from lesson10_kalman_filter.src.noise_model import build_noise_covariances
from lesson12_linear_mpc.src.model_loader import build_mpc_model, load_config
from lesson12_linear_mpc.src.mpc_controller import LinearMPCController


class MultiRateClosedLoopTest(TestCase):
    def simulator(self):
        try:
            return import_module("lesson12_linear_mpc.src.closed_loop").simulate_closed_loop
        except ImportError as exc:
            self.fail(f"Closed-loop simulator is unavailable: {exc}")

    def config(self, duration_s: float = 0.04):
        path = Path(__file__).resolve().parents[1] / "config" / "mpc.yaml"
        config = load_config(path)
        config.setdefault("experiment_b", {})["duration_s"] = duration_s
        return config

    def test_fast_plant_grid_and_ten_step_command_hold(self):
        simulate = self.simulator()
        config = self.config()
        trace = simulate(config, "lqr", measurement_noise=np.zeros(40))

        self.assertEqual(len(trace["time_s"]), 40)
        np.testing.assert_allclose(trace["time_s"], np.arange(40) * 0.001)
        np.testing.assert_array_equal(np.flatnonzero(trace["control_update"]), [0, 10, 20, 30])
        for start in (0, 10, 20, 30):
            np.testing.assert_allclose(
                trace["torque_applied_nm"][start:start + 10],
                trace["torque_applied_nm"][start],
                atol=0.0,
            )

        model = build_mpc_model(config)
        Ad_fast, Bd_fast = discretize_zoh(model["A"], model["B"], 0.001)
        for index in range(39):
            before = np.array([trace["q_true_rad"][index], trace["dq_true_rad_s"][index]])
            after = np.array([trace["q_true_rad"][index + 1], trace["dq_true_rad_s"][index + 1]])
            expected = Ad_fast @ before + Bd_fast[:, 0] * trace["torque_applied_nm"][index]
            np.testing.assert_allclose(after, expected, atol=1e-12)

    def test_initial_measurement_corrects_without_a_fictitious_prediction(self):
        simulate = self.simulator()
        config = self.config(duration_s=0.02)
        noise = np.zeros(20)
        noise[0] = 0.01
        trace = simulate(config, "lqr", measurement_noise=noise)

        model = build_mpc_model(config)
        Ad_fast, Bd_fast = discretize_zoh(model["A"], model["B"], 0.001)
        covariance = build_noise_covariances(config, Bd_fast)
        filter_ = DiscreteKalmanFilter(
            Ad_fast, Bd_fast, np.array([[1.0, 0.0]]), covariance["Gd"],
            covariance["Q_process"], covariance["R_measurement"],
            np.zeros(2), covariance["P0"],
        )
        expected_initial = filter_.update(float(trace["q_measured_rad"][0]))["x_hat"]
        np.testing.assert_allclose(
            [trace["q_hat_rad"][0], trace["dq_hat_rad_s"][0]],
            expected_initial, atol=1e-12,
        )
        filter_.predict(float(trace["torque_applied_nm"][0]))
        expected_second = filter_.update(float(trace["q_measured_rad"][1]))["x_hat"]
        np.testing.assert_allclose(
            [trace["q_hat_rad"][1], trace["dq_hat_rad_s"][1]],
            expected_second, atol=1e-12,
        )

    def test_mpc_applies_first_input_computed_from_estimated_state(self):
        simulate = self.simulator()
        config = self.config(duration_s=0.02)
        noise = np.zeros(20)
        noise[0] = 0.02
        trace = simulate(config, "mpc", horizon=5, measurement_noise=noise)

        model = build_mpc_model(config)
        Q, R = build_bryson_weights(5.0, 1.5, 3.0, 1.0, 1.0, 1.0)
        K, P = dlqr_from_dare(model["Ad"], model["Bd"], Q, R)
        estimated = np.array([trace["q_hat_rad"][0], trace["dq_hat_rad_s"][0]])
        true = np.array([trace["q_true_rad"][0], trace["dq_true_rad_s"][0]])
        reference = np.array([np.deg2rad(30.0), 0.0])
        planned = LinearMPCController(model["Ad"], model["Bd"], Q, R, P, 5).solve(
            estimated, reference
        )
        self.assertGreater(abs(float((K @ (estimated - true)).item())), 0.01)
        self.assertAlmostEqual(trace["torque_applied_nm"][0], planned["u0"][0], places=5)
        self.assertNotAlmostEqual(trace["torque_applied_nm"][0], planned["U"][0, 1], places=5)
        np.testing.assert_allclose(trace["torque_applied_nm"][:10], planned["u0"][0])
        self.assertEqual(trace["qp_status"][0], "optimal")
        self.assertEqual(trace["qp_status"][1], "")

    def test_lqr_and_mpc_consume_the_exact_supplied_noise(self):
        simulate = self.simulator()
        config = self.config(duration_s=0.02)
        noise = np.linspace(-0.001, 0.001, 20)
        lqr = simulate(config, "lqr", measurement_noise=noise)
        mpc = simulate(config, "mpc", horizon=1, measurement_noise=noise)
        np.testing.assert_array_equal(lqr["measurement_noise_rad"], noise)
        np.testing.assert_array_equal(mpc["measurement_noise_rad"], noise)
        np.testing.assert_allclose(lqr["q_measured_rad"] - lqr["q_true_rad"], noise)
        np.testing.assert_allclose(mpc["q_measured_rad"] - mpc["q_true_rad"], noise)

    def test_solver_failure_stops_the_scenario(self):
        simulate = self.simulator()
        config = self.config(duration_s=0.02)
        failure = {"status": "solver_error", "u0": None, "error": "injected failure"}
        with patch.object(LinearMPCController, "solve", return_value=failure):
            with self.assertRaisesRegex(RuntimeError, "solver_error.*injected failure"):
                simulate(config, "mpc", horizon=1, measurement_noise=np.zeros(20))

    def test_invalid_noise_length_is_rejected(self):
        simulate = self.simulator()
        with self.assertRaises(ValueError):
            simulate(self.config(), "lqr", measurement_noise=np.zeros(39))
