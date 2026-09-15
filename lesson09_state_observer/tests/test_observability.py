"""Tests for discrete observability of the position-only joint model."""

import unittest

import numpy as np

from lesson09_state_observer.src.model_loader import build_identified_model
from lesson09_state_observer.src.observability import (
    observability_matrix,
    observability_report,
)


class ObservabilityTest(unittest.TestCase):
    """Position sensing must contain information about both joint states."""

    def test_position_measurement_observes_both_joint_states(self) -> None:
        """The 2-state identified joint has rank-2 observability from position."""

        config = {
            "simulation": {"dt_s": 0.001},
            "model": {"inertia": 0.019762, "damping": 0.080257},
        }
        _, _, Ad, _ = build_identified_model(config)

        report = observability_report(Ad, np.array([[1.0, 0.0]]))

        self.assertEqual(report["rank"], 2)
        self.assertEqual(report["state_dimension"], 2)
        self.assertGreater(report["sigma_min"], 0.0)

    def test_invalid_matrix_shapes_are_rejected(self) -> None:
        """A non-square state matrix cannot define an observability matrix."""

        with self.assertRaises(ValueError):
            observability_matrix(np.ones((2, 3)), np.ones((1, 2)))


if __name__ == "__main__":
    unittest.main()
