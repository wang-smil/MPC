"""Numerically robust posterior-form discrete Kalman filter."""

import numpy as np


class DiscreteKalmanFilter:
    """Estimate a two-state joint from scalar position and applied torque."""

    def __init__(
        self,
        Ad: np.ndarray,
        Bd: np.ndarray,
        C: np.ndarray,
        Gd: np.ndarray,
        Q_process: np.ndarray,
        R_measurement: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
    ) -> None:
        self.Ad = np.asarray(Ad, dtype=float)
        self.Bd = np.asarray(Bd, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.Gd = np.asarray(Gd, dtype=float)
        self.Q = np.asarray(Q_process, dtype=float)
        self.R = np.asarray(R_measurement, dtype=float)
        self.x = np.asarray(x0, dtype=float).reshape(-1, 1)
        self.P = np.asarray(P0, dtype=float)

        if self.Ad.shape != (2, 2) or self.Bd.shape != (2, 1):
            raise ValueError("Ad must be 2-by-2 and Bd must be 2-by-1.")
        if self.C.shape != (1, 2) or self.Gd.shape != (2, 1):
            raise ValueError("C must be 1-by-2 and Gd must be 2-by-1.")
        if self.Q.shape != (1, 1) or self.R.shape != (1, 1):
            raise ValueError("Q_process and R_measurement must be scalar matrices.")
        if self.x.shape != (2, 1) or self.P.shape != (2, 2):
            raise ValueError("x0 must have two states and P0 must be 2-by-2.")
        if self.Q.item() < 0.0 or self.R.item() <= 0.0:
            raise ValueError("Q_process must be non-negative and R_measurement positive.")
        self._assert_covariance(self.P, "P0")

    @staticmethod
    def _scalar(value: float, name: str) -> float:
        array = np.asarray(value, dtype=float)
        if array.shape != ():
            raise ValueError(f"{name} must be scalar.")
        return float(array)

    @staticmethod
    def _assert_covariance(covariance: np.ndarray, name: str) -> None:
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValueError(f"{name} must be symmetric.")
        if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-10:
            raise ValueError(f"{name} must be positive semidefinite.")

    @property
    def x_hat(self) -> np.ndarray:
        """Return the current state estimate as a convenient one-dimensional vector."""

        return self.x[:, 0].copy()

    def predict(self, applied_input: float) -> dict[str, np.ndarray]:
        """Propagate posterior state/covariance with the last applied torque."""

        u_applied = self._scalar(applied_input, "applied_input")
        self.x = self.Ad @ self.x + self.Bd * u_applied
        self.P = self.Ad @ self.P @ self.Ad.T + self.Gd @ self.Q @ self.Gd.T
        self.P = 0.5 * (self.P + self.P.T)
        return {"x_prior": self.x.copy(), "P_prior": self.P.copy()}

    def update(self, measurement: float) -> dict[str, np.ndarray | float]:
        """Correct the predicted estimate with a scalar encoder measurement."""

        y = np.array([[self._scalar(measurement, "measurement")]])
        innovation = y - self.C @ self.x
        S = self.C @ self.P @ self.C.T + self.R
        PCt = self.P @ self.C.T
        gain = np.linalg.solve(S.T, PCt.T).T
        self.x = self.x + gain @ innovation

        identity = np.eye(self.P.shape[0])
        I_KC = identity - gain @ self.C
        self.P = I_KC @ self.P @ I_KC.T + gain @ self.R @ gain.T
        self.P = 0.5 * (self.P + self.P.T)
        self._assert_covariance(self.P, "posterior covariance")
        nis = float((innovation.T @ np.linalg.solve(S, innovation)).item())

        return {
            "x_hat": self.x_hat,
            "P": self.P.copy(),
            "K": gain.copy(),
            "innovation": float(innovation.item()),
            "innovation_covariance": float(S.item()),
            "nis": nis,
        }
