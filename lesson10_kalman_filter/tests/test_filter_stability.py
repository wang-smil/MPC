"""Tests for recursive/steady-state Kalman gain consistency."""

import unittest

import numpy as np

from lesson10_kalman_filter.src.kalman_filter import DiscreteKalmanFilter
from lesson10_kalman_filter.src.steady_state_kf import design_steady_state_kf


class KalmanStabilityTest(unittest.TestCase):
    """The recursive filter must approach the documented predictor gain."""

    def test_recursive_posterior_gain_maps_to_steady_state_predictor_gain(self) -> None:
        Ad = np.array([[1.0, 0.01], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        C = np.array([[1.0, 0.0]])
        Gd = Bd.copy()
        Q = np.array([[0.04]])
        R = np.array([[0.01]])
        filter_ = DiscreteKalmanFilter(Ad, Bd, C, Gd, Q, R, [0.0, 0.0], np.eye(2))

        for _ in range(3000):
            filter_.predict(0.0)
            latest = filter_.update(0.0)

        steady = design_steady_state_kf(Ad, Gd, C, Q, R)
        np.testing.assert_allclose(
            Ad @ latest["K"],
            steady["predictor_gain"],
            rtol=0.05,
            atol=1e-5,
        )
        self.assertTrue(np.all(np.abs(steady["poles"]) < 1.0))


if __name__ == "__main__":
    unittest.main()
