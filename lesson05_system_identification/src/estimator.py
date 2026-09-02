"""Least-squares inertia and viscous-damping estimation."""

import numpy as np


def identify_j_b(
    velocity: np.ndarray,
    acceleration: np.ndarray,
    torque: np.ndarray,
) -> dict[str, float | int | np.ndarray]:
    """Fit ``torque = J * acceleration + b * velocity`` by least squares."""

    velocity_array = np.asarray(velocity, dtype=float)
    acceleration_array = np.asarray(acceleration, dtype=float)
    torque_array = np.asarray(torque, dtype=float)
    if (
        velocity_array.ndim != 1
        or acceleration_array.ndim != 1
        or torque_array.ndim != 1
        or not (
            len(velocity_array) == len(acceleration_array) == len(torque_array)
        )
    ):
        raise ValueError("velocity, acceleration, and torque must be matching vectors.")
    if len(torque_array) < 2:
        raise ValueError("at least two samples are required.")

    phi = np.column_stack([acceleration_array, velocity_array])
    theta_hat, _, rank, _ = np.linalg.lstsq(phi, torque_array, rcond=None)
    if rank < 2:
        raise ValueError("Identification regression is rank deficient.")

    torque_predicted = phi @ theta_hat
    residual = torque_array - torque_predicted
    return {
        "inertia_hat": float(theta_hat[0]),
        "damping_hat": float(theta_hat[1]),
        "rank": int(rank),
        "condition_number": float(np.linalg.cond(phi)),
        "torque_predicted_nm": torque_predicted,
        "residual_nm": residual,
        "torque_rmse": float(np.sqrt(np.mean(residual**2))),
    }
