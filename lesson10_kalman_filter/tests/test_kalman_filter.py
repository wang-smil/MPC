"""Tests for recursive Kalman predict/update behavior."""

import unittest

import numpy as np

from lesson10_kalman_filter.src.kalman_filter import DiscreteKalmanFilter


class DiscreteKalmanFilterTest(unittest.TestCase):
    """Joseph-form covariance must remain a usable covariance."""

    def setUp(self) -> None:
        self.filter = DiscreteKalmanFilter(
            Ad=np.array([[1.0, 0.01], [0.0, 0.95]]),
            Bd=np.array([[0.0], [0.1]]),
            C=np.array([[1.0, 0.0]]),
            Gd=np.array([[0.0], [0.1]]),
            Q_process=np.array([[0.04]]),
            R_measurement=np.array([[0.01]]),
            x0=np.array([0.0, 0.0]),
            P0=np.diag([1.0, 1.0]),
        )

    def test_joseph_update_keeps_covariance_symmetric_and_psd(self) -> None:
        for _ in range(200):
            self.filter.predict(0.1)
            result = self.filter.update(0.2)

        np.testing.assert_allclose(result["P"], result["P"].T, atol=1e-10)
        self.assertGreaterEqual(np.min(np.linalg.eigvalsh(result["P"])), -1e-10)
        self.assertGreaterEqual(result["nis"], 0.0)

    def test_vector_measurement_and_input_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.filter.predict([0.1])
        with self.assertRaises(ValueError):
            self.filter.update([0.2])


if __name__ == "__main__":
    unittest.main()
