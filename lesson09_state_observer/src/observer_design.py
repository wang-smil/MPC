"""Discrete Luenberger observer gain design by dual pole placement."""

import numpy as np
from scipy.signal import place_poles


def continuous_poles_from_discrete(
    discrete_poles: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Map discrete poles to their continuous-time equivalents."""

    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive.")
    poles = np.asarray(discrete_poles, dtype=complex)
    if poles.ndim != 1 or poles.size == 0:
        raise ValueError("discrete_poles must be a non-empty vector.")
    return np.log(poles) / dt_s


def design_discrete_observer(
    Ad: np.ndarray,
    C: np.ndarray,
    controller_poles_z: np.ndarray,
    speed_factor: float,
    dt_s: float,
) -> dict[str, np.ndarray]:
    """Design L so the observer is faster than the controller in real time."""

    Ad = np.asarray(Ad, dtype=float)
    C = np.asarray(C, dtype=float)
    if Ad.ndim != 2 or Ad.shape[0] != Ad.shape[1]:
        raise ValueError("Ad must be square.")
    if C.ndim != 2 or C.shape != (1, Ad.shape[0]):
        raise ValueError("C must have shape (1, state_dimension).")
    if speed_factor <= 0.0:
        raise ValueError("speed_factor must be positive.")

    controller_poles_s = continuous_poles_from_discrete(controller_poles_z, dt_s)
    requested_poles = np.exp(speed_factor * controller_poles_s * dt_s)
    result = place_poles(Ad.T, C.T, requested_poles)
    L = np.asarray(result.gain_matrix.T)
    achieved_poles = np.linalg.eigvals(Ad - L @ C)

    return {
        "L": L,
        "requested_poles": requested_poles,
        "achieved_poles": achieved_poles,
    }
