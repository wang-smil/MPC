"""Tests for Bryson weights, discrete LQR, and fair pole placement."""

import unittest

import numpy as np

from lesson08_lqr_optimal_control.src.model import (
    build_continuous_model,
    discretize_zoh,
)


class LqrDesignTest(unittest.TestCase):
    def setUp(self) -> None:
        A, B = build_continuous_model(inertia=0.02, damping=0.08)
        self.Ad, self.Bd = discretize_zoh(A, B, dt_s=0.001)

    def test_bryson_weights_use_radians_and_input_limit(self) -> None:
        from lesson08_lqr_optimal_control.src.lqr_design import build_bryson_weights

        Q, R = build_bryson_weights(5.0, 1.5, 3.0, 1.0, 1.0, 1.0)

        np.testing.assert_allclose(
            Q,
            np.diag([1.0 / np.deg2rad(5.0) ** 2, 1.0 / 1.5**2]),
        )
        np.testing.assert_allclose(R, [[1.0 / 9.0]])

    def test_dlqr_matches_independent_dare_solution(self) -> None:
        from lesson08_lqr_optimal_control.src.lqr_design import (
            build_bryson_weights,
            design_dlqr,
            dlqr_from_dare,
        )

        Q, R = build_bryson_weights(5.0, 1.5, 3.0, 1.0, 1.0, 1.0)
        design = design_dlqr(self.Ad, self.Bd, Q, R)
        K_dare, _ = dlqr_from_dare(self.Ad, self.Bd, Q, R)

        np.testing.assert_allclose(design["K"], K_dare, rtol=1e-6, atol=1e-8)
        self.assertTrue(np.all(np.abs(design["closed_loop_poles"]) < 1.0))

    def test_pole_placement_is_recomputed_at_one_millisecond(self) -> None:
        from lesson08_lqr_optimal_control.src.lqr_design import design_pole_placement

        design = design_pole_placement(
            self.Ad,
            self.Bd,
            damping_ratio=0.8,
            settling_time_s=0.8,
            dt_s=0.001,
        )

        np.testing.assert_allclose(
            np.sort_complex(design["computed_poles"]),
            np.sort_complex(design["requested_poles"]),
        )


if __name__ == "__main__":
    unittest.main()
