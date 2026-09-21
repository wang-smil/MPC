"""Causal composition of an estimator, an LQR gain, and actuator clipping."""

import numpy as np


def _scalar(value: object, name: str) -> float:
    """Return a finite scalar and reject vectors at the public boundary."""

    array = np.asarray(value, dtype=float)
    if array.shape != () or not np.isfinite(array.item()):
        raise ValueError(f"{name} must be a finite scalar.")
    return float(array.item())


def _state_vector(value: object, name: str) -> np.ndarray:
    """Return the two-state vector used by the shared one-axis model."""

    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape (2,).")
    return array


class LQGController:
    """Use a supplied estimator and offline LQR gain without reimplementing either."""

    def __init__(self, estimator: object, K_controller: object, torque_limit: object):
        gain = np.asarray(K_controller, dtype=float)
        if gain.shape != (1, 2) or not np.all(np.isfinite(gain)):
            raise ValueError("K_controller must be a finite array with shape (1, 2).")
        limit = _scalar(torque_limit, "torque_limit")
        if limit <= 0.0:
            raise ValueError("torque_limit must be positive.")
        if not callable(getattr(estimator, "step", None)):
            raise ValueError("estimator must provide a callable step method.")

        self.estimator = estimator
        self.Kc = gain
        self.limit = limit

    def step(
        self,
        measurement: object,
        previous_applied_torque: object,
        x_ref: object,
        torque_ff: object = 0.0,
    ) -> dict:
        """Estimate state, compute feedback request, then apply actuator clipping."""

        measurement_scalar = _scalar(measurement, "measurement")
        previous_applied_scalar = _scalar(
            previous_applied_torque,
            "previous_applied_torque",
        )
        reference = _state_vector(x_ref, "x_ref")
        feedforward = _scalar(torque_ff, "torque_ff")

        estimate = self.estimator.step(measurement_scalar, previous_applied_scalar)
        if not isinstance(estimate, dict) or "x_hat" not in estimate:
            raise ValueError("estimator.step must return a mapping containing x_hat.")
        x_hat = _state_vector(estimate["x_hat"], "estimator x_hat")
        torque_request = feedforward - float((self.Kc @ (x_hat - reference)).item())
        torque_applied = float(np.clip(torque_request, -self.limit, self.limit))

        return {
            "x_hat": x_hat,
            "torque_request": torque_request,
            "torque_applied": torque_applied,
            "saturated": abs(torque_request) > self.limit,
            "estimator": estimate,
        }
