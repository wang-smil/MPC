"""Tests for the Lesson 07 identified state-space model."""

import unittest

import numpy as np

from lesson07_state_feedback_controllability.src.model import (
    build_continuous_model,
    discretize_zoh,
)


class ModelTest(unittest.TestCase):
    def test_build_continuous_model_uses_identified_inertia_and_damping(self) -> None:
        A, B = build_continuous_model(inertia=0.02, damping=0.08)

        np.testing.assert_allclose(A, [[0.0, 1.0], [0.0, -4.0]])
        np.testing.assert_allclose(B, [[0.0], [50.0]])

    def test_zoh_discretization_returns_state_and_input_matrices(self) -> None:
        A, B = build_continuous_model(inertia=0.02, damping=0.08)

        Ad, Bd = discretize_zoh(A, B, dt_s=0.01)

        self.assertEqual(Ad.shape, (2, 2))
        self.assertEqual(Bd.shape, (2, 1))
        self.assertLess(Ad[1, 1], 1.0)


if __name__ == "__main__":
    unittest.main()
