"""Behavioral tests for the non-condensed Lesson 12 MPC optimizer."""

from importlib import import_module
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from lesson08_lqr_optimal_control.src.lqr_design import (
    build_bryson_weights,
    dlqr_from_dare,
)
from lesson12_linear_mpc.src.model_loader import build_mpc_model, load_config


class LinearMPCControllerTest(TestCase):
    def controller_class(self):
        try:
            module = import_module("lesson12_linear_mpc.src.mpc_controller")
        except ImportError as exc:
            self.fail(f"MPC controller is unavailable: {exc}")
        return module.LinearMPCController

    def joint_design(self):
        config_path = Path(__file__).resolve().parents[1] / "config" / "mpc.yaml"
        model = build_mpc_model(load_config(config_path))
        Q, R = build_bryson_weights(5.0, 1.5, 3.0, 1.0, 1.0, 1.0)
        K, P = dlqr_from_dare(model["Ad"], model["Bd"], Q, R)
        return model["Ad"], model["Bd"], Q, R, K, P

    def test_scalar_problem_has_hand_computed_first_action(self):
        controller = self.controller_class()(
            Ad=np.array([[1.0]]),
            Bd=np.array([[1.0]]),
            Q=np.array([[1.0]]),
            R=np.array([[1.0]]),
            P_terminal=np.array([[1.0]]),
            horizon=1,
        )

        result = controller.solve(x_hat=1.0, x_ref=0.0)

        self.assertEqual(result["status"], "optimal")
        self.assertEqual(controller.X.shape, (1, 2))
        self.assertEqual(controller.U.shape, (1, 1))
        np.testing.assert_allclose(result["u0"], [-0.5], atol=1e-7)
        np.testing.assert_allclose(result["X"], [[1.0, 0.5]], atol=1e-7)
        np.testing.assert_allclose(result["U"], [[-0.5]], atol=1e-7)
        self.assertAlmostEqual(result["objective"], 1.5, places=6)

    def test_planned_trajectories_satisfy_dynamics_for_each_horizon(self):
        Ad, Bd, Q, R, _, P = self.joint_design()
        for horizon in (1, 5, 20):
            with self.subTest(horizon=horizon):
                controller = self.controller_class()(Ad, Bd, Q, R, P, horizon)
                x0 = np.array([0.2, -0.1])
                result = controller.solve(x_hat=x0, x_ref=np.array([0.4, 0.0]))
                self.assertIn(result["status"], ("optimal", "optimal_inaccurate"))
                self.assertEqual(controller.X.shape, (2, horizon + 1))
                self.assertEqual(controller.U.shape, (1, horizon))
                self.assertEqual(result["X"].shape, (2, horizon + 1))
                self.assertEqual(result["U"].shape, (1, horizon))
                np.testing.assert_allclose(result["X"][:, 0], x0, atol=1e-7)
                for k in range(horizon):
                    np.testing.assert_allclose(
                        result["X"][:, k + 1],
                        Ad @ result["X"][:, k] + Bd @ result["U"][:, k],
                        atol=1e-6,
                    )
                np.testing.assert_allclose(result["u0"], result["U"][:, 0], atol=1e-9)
                self.assertTrue(np.isfinite(result["objective"]))
                self.assertIsNotNone(result["solve_time_s"])
                self.assertGreaterEqual(result["solve_time_s"], 0.0)
                self.assertIsNotNone(result["iterations"])

    def test_first_action_matches_dare_lqr_for_each_horizon(self):
        Ad, Bd, Q, R, K, P = self.joint_design()
        cases = (
            (np.array([0.0, 0.0]), np.array([np.deg2rad(30.0), 0.0])),
            (np.array([0.4, 0.2]), np.array([np.deg2rad(30.0), 0.0])),
            (np.array([-0.1, -0.3]), np.array([0.2, 0.0])),
        )
        for horizon in (1, 5, 20):
            controller = self.controller_class()(Ad, Bd, Q, R, P, horizon)
            for x_hat, x_ref in cases:
                with self.subTest(horizon=horizon, x_hat=x_hat.tolist()):
                    result = controller.solve(x_hat=x_hat, x_ref=x_ref)
                    expected = -K @ (x_hat - x_ref)
                    np.testing.assert_allclose(result["u0"], expected, atol=1e-4, rtol=0.0)

    def test_parameterized_qp_can_reuse_its_canonical_form(self):
        controller = self.controller_class()(
            np.array([[1.0]]), np.array([[1.0]]),
            np.array([[1.0]]), np.array([[1.0]]),
            np.array([[1.0]]), 1,
        )
        self.assertTrue(controller.problem.is_dpp(quad_form_dpp='qp'))

    def test_invalid_design_inputs_are_rejected(self):
        cls = self.controller_class()
        good = dict(
            Ad=np.eye(2),
            Bd=np.array([[0.0], [1.0]]),
            Q=np.eye(2),
            R=np.array([[1.0]]),
            P_terminal=np.eye(2),
            horizon=2,
        )
        invalid = (
            ("horizon", 0),
            ("horizon", -1),
            ("horizon", 1.5),
            ("horizon", True),
            ("Ad", np.zeros((2, 3))),
            ("Bd", np.ones((3, 1))),
            ("Q", np.ones((3, 3))),
            ("Q", np.array([[1.0, 1.0], [0.0, 1.0]])),
            ("Q", -np.eye(2)),
            ("R", np.array([[0.0]])),
            ("P_terminal", -np.eye(2)),
            ("Ad", np.array([[np.nan, 0.0], [0.0, 1.0]])),
        )
        for name, value in invalid:
            with self.subTest(name=name, value=str(value)):
                with self.assertRaises(ValueError):
                    cls(**{**good, name: value})

    def test_nonfinite_or_wrong_sized_states_are_rejected(self):
        cls = self.controller_class()
        controller = cls(np.eye(2), np.array([[0.0], [1.0]]), np.eye(2),
                         np.array([[1.0]]), np.eye(2), 2)
        for x_hat, x_ref in (
            ([np.nan, 0.0], [0.0, 0.0]),
            ([0.0], [0.0, 0.0]),
            ([0.0, 0.0], [np.inf, 0.0]),
            ([0.0, 0.0], [0.0]),
        ):
            with self.subTest(x_hat=x_hat, x_ref=x_ref):
                with self.assertRaises(ValueError):
                    controller.solve(x_hat=x_hat, x_ref=x_ref)

    def test_solver_exception_never_reuses_previous_action(self):
        cls = self.controller_class()
        controller = cls(np.array([[1.0]]), np.array([[1.0]]),
                         np.array([[1.0]]), np.array([[1.0]]),
                         np.array([[1.0]]), 1)
        self.assertIsNotNone(controller.solve(1.0, 0.0)["u0"])
        with patch.object(controller.problem, "solve", side_effect=RuntimeError("solver broke")):
            result = controller.solve(1.0, 0.0)
        self.assertEqual(result["status"], "solver_error")
        self.assertIsNone(result["u0"])
        self.assertIsNone(result["X"])
        self.assertIsNone(result["U"])
        self.assertIn("solver broke", result["error"])

    def test_nonoptimal_status_has_no_executable_action(self):
        cls = self.controller_class()
        controller = cls(np.array([[1.0]]), np.array([[1.0]]),
                         np.array([[1.0]]), np.array([[1.0]]),
                         np.array([[1.0]]), 1)

        def report_infeasible(*args, **kwargs):
            controller.problem._status = "infeasible"

        with patch.object(controller.problem, "solve", side_effect=report_infeasible):
            result = controller.solve(1.0, 0.0)
        self.assertEqual(result["status"], "infeasible")
        self.assertIsNone(result["u0"])
        self.assertIsNone(result["X"])
        self.assertIsNone(result["U"])
