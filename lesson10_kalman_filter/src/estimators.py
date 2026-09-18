"""Common estimator interfaces for the fair Lesson 10 comparison."""

import numpy as np

from lesson09_state_observer.src.observer_design import design_discrete_observer

from .kalman_filter import DiscreteKalmanFilter


def _scalar(value: float, name: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar.")
    return float(array)


class DifferenceEstimator:
    """Position encoder plus raw or low-pass differentiated velocity."""

    def __init__(self, dt_s: float, alpha: float | None = None) -> None:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        if alpha is not None and not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be in [0, 1).")
        self.dt_s = float(dt_s)
        self.alpha = alpha
        self.previous_position: float | None = None
        self.filtered_velocity = 0.0

    def step(self, measurement: float, applied_input: float) -> dict[str, np.ndarray | float]:
        """Estimate state; torque is accepted to match the common interface."""

        q = _scalar(measurement, "measurement")
        _scalar(applied_input, "applied_input")
        raw_velocity = 0.0 if self.previous_position is None else (q - self.previous_position) / self.dt_s
        self.previous_position = q
        if self.alpha is None:
            velocity = raw_velocity
        else:
            self.filtered_velocity = self.alpha * self.filtered_velocity + (1.0 - self.alpha) * raw_velocity
            velocity = self.filtered_velocity
        return {
            "x_hat": np.array([q, velocity]),
            "P": np.full((2, 2), np.nan),
            "K": np.full((2, 1), np.nan),
            "innovation": float("nan"),
            "innovation_covariance": float("nan"),
            "nis": float("nan"),
        }


class FixedGainLuenbergerEstimator:
    """Posterior-form adapter using the Lesson 09 fixed observer pole design."""

    def __init__(
        self,
        Ad: np.ndarray,
        Bd: np.ndarray,
        C: np.ndarray,
        controller_gain: np.ndarray,
        dt_s: float,
        x0: np.ndarray,
        speed_factor: float = 4.0,
    ) -> None:
        self.Ad = np.asarray(Ad, dtype=float)
        self.Bd = np.asarray(Bd, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.x = np.asarray(x0, dtype=float).reshape(2, 1)
        controller_poles = np.linalg.eigvals(self.Ad - self.Bd @ np.asarray(controller_gain, dtype=float))
        predictor = design_discrete_observer(
            self.Ad, self.C, controller_poles, speed_factor=speed_factor, dt_s=dt_s
        )
        self.predictor_gain = np.asarray(predictor["L"], dtype=float)
        self.posterior_gain = np.linalg.solve(self.Ad, self.predictor_gain)

    def step(self, measurement: float, applied_input: float) -> dict[str, np.ndarray | float]:
        q = _scalar(measurement, "measurement")
        u = _scalar(applied_input, "applied_input")
        x_prior = self.Ad @ self.x + self.Bd * u
        innovation = q - float((self.C @ x_prior).item())
        self.x = x_prior + self.posterior_gain * innovation
        return {
            "x_hat": self.x[:, 0].copy(),
            "P": np.full((2, 2), np.nan),
            "K": self.posterior_gain.copy(),
            "innovation": innovation,
            "innovation_covariance": float("nan"),
            "nis": float("nan"),
        }


class KalmanEstimator:
    """Adapter that exposes the same step contract as the comparison baselines."""

    def __init__(self, filter_: DiscreteKalmanFilter) -> None:
        self.filter = filter_

    def step(self, measurement: float, applied_input: float) -> dict[str, np.ndarray | float]:
        self.filter.predict(applied_input)
        return self.filter.update(measurement)
