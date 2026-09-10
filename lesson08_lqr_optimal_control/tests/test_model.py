"""Tests for the Lesson 08 identified joint model."""

import unittest

import numpy as np


class ModelTest(unittest.TestCase):
    def test_identified_joint_model_and_zoh_shapes(self) -> None:
        from lesson08_lqr_optimal_control.src.model import (
            build_continuous_model,
            discretize_zoh,
        )

        A, B = build_continuous_model(inertia=0.02, damping=0.08)
        Ad, Bd = discretize_zoh(A, B, dt_s=0.001)

        np.testing.assert_allclose(A, [[0.0, 1.0], [0.0, -4.0]])
        np.testing.assert_allclose(B, [[0.0], [50.0]])
        self.assertEqual(Ad.shape, (2, 2))
        self.assertEqual(Bd.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
