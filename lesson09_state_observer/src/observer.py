"""Position-only discrete Luenberger observer runtime."""

import numpy as np


class DiscreteObserver:
    """Estimate position and velocity from scalar encoder position and input."""

    def __init__(
        self,
        Ad: np.ndarray,
        Bd: np.ndarray,
        C: np.ndarray,
        L: np.ndarray,
        x0_hat: np.ndarray,
    ) -> None:
        self.Ad = np.asarray(Ad, dtype=float)
        self.Bd = np.asarray(Bd, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.L = np.asarray(L, dtype=float)
        self.x_hat = np.asarray(x0_hat, dtype=float).copy()

        if self.Ad.shape != (2, 2) or self.Bd.shape != (2, 1):
            raise ValueError("Ad must be 2-by-2 and Bd must be 2-by-1.")
        if self.C.shape != (1, 2) or self.L.shape != (2, 1):
            raise ValueError("C must be 1-by-2 and L must be 2-by-1.")
        if self.x_hat.shape != (2,):
            raise ValueError("x0_hat must have shape (2,).")

    @staticmethod
    def _scalar(value: float, name: str) -> float:
        """Convert a scalar-like input while rejecting state vectors."""

        array = np.asarray(value, dtype=float)
        if array.shape != ():
            raise ValueError(f"{name} must be scalar.")
        return float(array)

    def update(self, measurement: float, control_input: float) -> dict[str, np.ndarray | float]:
        """Predict one step and correct it using the encoder innovation."""

        y = self._scalar(measurement, "measurement")
        u_applied = self._scalar(control_input, "control_input")
        y_hat = float((self.C @ self.x_hat).item())
        innovation = y - y_hat
        self.x_hat = (
            self.Ad @ self.x_hat
            + self.Bd[:, 0] * u_applied
            + self.L[:, 0] * innovation
        )
        return {
            "x_hat": self.x_hat.copy(),
            "y_hat": y_hat,
            "innovation": innovation,
        }
