"""Tests for the identified model used by the observer."""

import unittest

import numpy as np

from lesson09_state_observer.src.model_loader import (
    build_balanced_lqr_gain,
    build_identified_model,
)


class ModelLoaderTest(unittest.TestCase):
    """The observer must use the known identified joint model."""

    def test_identified_model_has_expected_shapes_and_one_ms_zoh(self) -> None:
        """A 1 ms ZOH conversion keeps the two-state, one-input structure."""

        config = {
            "simulation": {"dt_s": 0.001},
            "model": {"inertia": 0.019762, "damping": 0.080257},
        }

        A, B, Ad, Bd = build_identified_model(config)

        self.assertEqual(A.shape, (2, 2))
        self.assertEqual(B.shape, (2, 1))
        self.assertEqual(Ad.shape, (2, 2))
        self.assertEqual(Bd.shape, (2, 1))
        np.testing.assert_allclose(A[1, 1], -0.080257 / 0.019762)
        self.assertFalse(np.allclose(Ad, np.eye(2)))

    def test_balanced_lqr_gain_is_one_by_two_and_stabilizes_model(self) -> None:
        """The reused balanced design produces a stable state-feedback gain."""

        config = {
            "simulation": {"dt_s": 0.001},
            "model": {"inertia": 0.019762, "damping": 0.080257},
            "controller": {
                "max_position_error_deg": 5.0,
                "max_velocity_error_rad_s": 1.5,
                "torque_limit_nm": 3.0,
            },
        }

        _, _, Ad, Bd = build_identified_model(config)
        K = build_balanced_lqr_gain(Ad, Bd, config)

        self.assertEqual(K.shape, (1, 2))
        self.assertTrue(np.all(np.abs(np.linalg.eigvals(Ad - Bd @ K)) < 1.0))


if __name__ == "__main__":
    unittest.main()
