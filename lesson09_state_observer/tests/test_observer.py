"""Tests for the position-only discrete observer runtime."""

import unittest

import numpy as np

from lesson09_state_observer.src.observer import DiscreteObserver


class DiscreteObserverTest(unittest.TestCase):
    """The runtime observer exposes prediction error and corrected state."""

    def setUp(self) -> None:
        self.observer = DiscreteObserver(
            Ad=np.eye(2),
            Bd=np.array([[0.0], [1.0]]),
            C=np.array([[1.0, 0.0]]),
            L=np.array([[0.5], [0.0]]),
            x0_hat=np.zeros(2),
        )

    def test_update_uses_only_measurement_and_applied_input(self) -> None:
        """One update combines scalar encoder innovation with scalar input."""

        result = self.observer.update(measurement=2.0, control_input=3.0)

        self.assertEqual(set(result), {"x_hat", "y_hat", "innovation"})
        self.assertEqual(result["y_hat"], 0.0)
        self.assertEqual(result["innovation"], 2.0)
        np.testing.assert_allclose(result["x_hat"], [1.0, 3.0])

    def test_vector_measurement_is_rejected(self) -> None:
        """The observer interface intentionally forbids true-state vectors."""

        with self.assertRaises(ValueError):
            self.observer.update(measurement=np.zeros(2), control_input=0.0)


if __name__ == "__main__":
    unittest.main()
