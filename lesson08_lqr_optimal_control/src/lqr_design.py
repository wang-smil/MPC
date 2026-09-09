"""Bryson weights, discrete LQR design, and a pole-placement baseline."""

import control as ct
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import place_poles


def build_bryson_weights(
    max_position_error_deg: float,
    max_velocity_error_rad_s: float,
    max_torque_nm: float,
    position_scale: float,
    velocity_scale: float,
    torque_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build diagonal Q and scalar R from acceptable state/input scales."""

    values = (
        max_position_error_deg,
        max_velocity_error_rad_s,
        max_torque_nm,
        position_scale,
        velocity_scale,
        torque_scale,
    )
    if any(value <= 0.0 for value in values):
        raise ValueError("Bryson limits and scales must be positive.")
    max_position_error_rad = np.deg2rad(max_position_error_deg)
    Q = np.diag(
        [
            position_scale / max_position_error_rad**2,
            velocity_scale / max_velocity_error_rad_s**2,
        ]
    )
    R = np.array([[torque_scale / max_torque_nm**2]])
    return Q, R


def _validate_design_matrices(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the single-input two-state design inputs."""

    Ad = np.asarray(Ad, dtype=float)
    Bd = np.asarray(Bd, dtype=float)
    if Ad.shape != (2, 2) or Bd.shape != (2, 1):
        raise ValueError("Lesson 08 requires a 2-by-2 Ad and 2-by-1 Bd.")
    if Q is not None and np.asarray(Q, dtype=float).shape != (2, 2):
        raise ValueError("Q must be 2-by-2.")
    if R is not None:
        R = np.asarray(R, dtype=float)
        if R.shape != (1, 1) or R.item() <= 0.0:
            raise ValueError("R must be a positive 1-by-1 matrix.")
    return Ad, Bd


def dlqr_from_dare(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the DARE independently and recover the feedback gain."""

    Ad, Bd = _validate_design_matrices(Ad, Bd, Q, R)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, P


def design_dlqr(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> dict[str, np.ndarray]:
    """Design discrete LQR with python-control and expose its closed-loop poles."""

    Ad, Bd = _validate_design_matrices(Ad, Bd, Q, R)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    K, S, E = ct.dlqr(Ad, Bd, Q, R)
    K = np.asarray(K, dtype=float)
    return {
        "K": K,
        "S": np.asarray(S, dtype=float),
        "closed_loop_poles": np.asarray(E),
    }


def design_pole_placement(
    Ad: np.ndarray,
    Bd: np.ndarray,
    damping_ratio: float,
    settling_time_s: float,
    dt_s: float,
) -> dict[str, np.ndarray]:
    """Recreate the Lesson 07 pole specification at the current sample period."""

    Ad, Bd = _validate_design_matrices(Ad, Bd)
    if not 0.0 < damping_ratio < 1.0:
        raise ValueError("damping_ratio must be between 0 and 1.")
    if settling_time_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("settling_time_s and dt_s must be positive.")
    natural_frequency = 4.0 / (damping_ratio * settling_time_s)
    real = -damping_ratio * natural_frequency
    imaginary = natural_frequency * np.sqrt(1.0 - damping_ratio**2)
    requested_poles = np.exp(np.array([real + 1j * imaginary, real - 1j * imaginary]) * dt_s)
    K = np.asarray(place_poles(Ad, Bd, requested_poles).gain_matrix, dtype=float)
    return {
        "K": K,
        "requested_poles": requested_poles,
        "computed_poles": np.linalg.eigvals(Ad - Bd @ K),
    }
