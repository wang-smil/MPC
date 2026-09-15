"""Tests for discrete Luenberger observer pole placement."""

import unittest

import numpy as np

from lesson09_state_observer.src.model_loader import (
    build_balanced_lqr_gain,
    build_identified_model,
)
from lesson09_state_observer.src.observer_design import design_discrete_observer


class ObserverDesignTest(unittest.TestCase):
    """Observer poles must be placed in the discrete unit circle."""

    def test_observer_places_requested_stable_discrete_poles(self) -> None:
        """A 4x observer maps continuous-time speed correctly to z-plane poles."""

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
        controller_poles_z = np.linalg.eigvals(Ad - Bd @ K)

        result = design_discrete_observer(
            Ad=Ad,
            C=np.array([[1.0, 0.0]]),
            controller_poles_z=controller_poles_z,
            speed_factor=4.0,
            dt_s=0.001,
        )

        self.assertEqual(result["L"].shape, (2, 1))
        self.assertTrue(np.all(np.abs(result["achieved_poles"]) < 1.0))
        np.testing.assert_allclose(
            np.sort_complex(result["achieved_poles"]),
            np.sort_complex(result["requested_poles"]),
            atol=1e-7,
        )

    def test_nonpositive_speed_or_sample_time_is_rejected(self) -> None:
        """Observer speed and sampling period define a valid time-scale mapping."""

        with self.assertRaises(ValueError):
            design_discrete_observer(
                Ad=np.eye(2),
                C=np.array([[1.0, 0.0]]),
                controller_poles_z=np.array([0.9, 0.8]),
                speed_factor=0.0,
                dt_s=0.001,
            )


if __name__ == "__main__":
    unittest.main()
