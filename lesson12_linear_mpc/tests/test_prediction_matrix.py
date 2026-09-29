"""Behavioral tests for Lesson 12 prediction and model sampling."""

import unittest
from pathlib import Path

import numpy as np

from lesson08_lqr_optimal_control.src.model import (
    build_continuous_model,
    discretize_zoh,
)
from lesson12_linear_mpc.src.model_loader import build_mpc_model, load_config
from lesson12_linear_mpc.src.prediction import (
    build_prediction_matrices,
    predict_states_matrix,
    rollout_states,
)


LESSON_DIR = Path(__file__).resolve().parents[1]


class MpcModelTest(unittest.TestCase):
    def test_mpc_model_uses_mpc_period_and_lesson08_zoh(self):
        config = load_config(LESSON_DIR / "config" / "mpc.yaml")
        model = build_mpc_model(config)
        inertia = config["model"]["inertia_kg_m2"]
        damping = config["model"]["damping_nm_s_rad"]
        A, B = build_continuous_model(inertia, damping)
        expected_Ad, expected_Bd = discretize_zoh(A, B, 0.01)
        plant_Ad, plant_Bd = discretize_zoh(A, B, 0.001)

        np.testing.assert_allclose(model["Ad"], expected_Ad, rtol=0, atol=1e-13)
        np.testing.assert_allclose(model["Bd"], expected_Bd, rtol=0, atol=1e-13)
        self.assertGreater(np.max(np.abs(model["Ad"] - plant_Ad)), 1e-5)
        self.assertGreater(np.max(np.abs(model["Bd"] - plant_Bd)), 1e-5)


class PredictionMatrixTest(unittest.TestCase):
    def test_seeded_robot_prediction_matches_independent_rollout(self):
        config = load_config(LESSON_DIR / "config" / "mpc.yaml")
        model = build_mpc_model(config)
        horizon = config["mpc"]["horizon_steps"]
        rng = np.random.default_rng(config["validation"]["seed"])
        x0 = rng.normal(size=model["Ad"].shape[0])
        U = rng.uniform(-0.5, 0.5, size=(horizon, model["Bd"].shape[1]))

        F, G = build_prediction_matrices(model["Ad"], model["Bd"], horizon)
        X_matrix = predict_states_matrix(model["Ad"], model["Bd"], x0, U)
        X_rollout = rollout_states(model["Ad"], model["Bd"], x0, U)

        self.assertEqual(F.shape, (horizon * len(x0), len(x0)))
        self.assertEqual(G.shape, (horizon * len(x0), horizon * U.shape[1]))
        self.assertEqual(X_matrix.shape, (horizon, len(x0)))
        self.assertEqual(X_rollout.shape, (horizon + 1, len(x0)))
        np.testing.assert_allclose(X_matrix, X_rollout[1:], rtol=0, atol=1e-10)
        np.testing.assert_array_equal(X_rollout[0], x0)

    def test_two_input_blocks_use_time_major_c_order(self):
        Ad = np.array([[1.0, 1.0], [0.0, 1.0]])
        Bd = np.array([[0.0, 1.0], [1.0, 0.0]])
        horizon = 3
        x0 = np.array([2.0, -1.0])
        U = np.array([[3.0, 5.0], [7.0, 11.0], [13.0, 17.0]])
        expected_F = np.array(
            [[1, 1], [0, 1], [1, 2], [0, 1], [1, 3], [0, 1]], dtype=float
        )
        expected_G = np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [2, 1, 1, 1, 0, 1],
                [1, 0, 1, 0, 1, 0],
            ],
            dtype=float,
        )
        expected_X = np.array([[6.0, 2.0], [19.0, 9.0], [45.0, 22.0]])

        F, G = build_prediction_matrices(Ad, Bd, horizon)
        X = predict_states_matrix(Ad, Bd, x0, U)

        np.testing.assert_array_equal(F, expected_F)
        np.testing.assert_array_equal(G, expected_G)
        np.testing.assert_array_equal(X, expected_X)
        np.testing.assert_allclose(X, rollout_states(Ad, Bd, x0, U)[1:], rtol=0, atol=0)

    def test_invalid_horizon_is_rejected(self):
        Ad = np.eye(2)
        Bd = np.ones((2, 1))
        for invalid in (0, -1, 1.5, True):
            with self.subTest(horizon=invalid):
                with self.assertRaises(ValueError):
                    build_prediction_matrices(Ad, Bd, invalid)

    def test_invalid_model_shapes_and_nonfinite_values_are_rejected(self):
        valid_A = np.eye(2)
        valid_B = np.ones((2, 1))
        cases = (
            (np.ones((2, 3)), valid_B),
            (valid_A, np.ones(2)),
            (valid_A, np.array([[1.0], [np.inf]])),
            (np.array([[1.0, np.nan], [0.0, 1.0]]), valid_B),
        )
        for Ad, Bd in cases:
            with self.subTest(Ad=Ad, Bd=Bd):
                with self.assertRaises(ValueError):
                    build_prediction_matrices(Ad, Bd, 2)

    def test_invalid_initial_state_and_input_sequence_are_rejected(self):
        Ad = np.eye(2)
        Bd = np.ones((2, 1))
        x0_bad = (np.ones((2, 1)), np.array([np.nan, 0.0]))
        for x0 in x0_bad:
            with self.subTest(x0=x0):
                with self.assertRaises(ValueError):
                    predict_states_matrix(Ad, Bd, x0, np.zeros((2, 1)))

        U_bad = (
            np.zeros(2),
            np.zeros((2, 2)),
            np.zeros((0, 1)),
            np.array([[0.0], [np.inf]]),
        )
        for U in U_bad:
            with self.subTest(U=U):
                with self.assertRaises(ValueError):
                    predict_states_matrix(Ad, Bd, np.zeros(2), U)


if __name__ == "__main__":
    unittest.main()
