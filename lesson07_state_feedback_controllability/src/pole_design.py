"""Continuous-pole targets and discrete state-feedback pole placement."""

import numpy as np
from scipy.signal import place_poles

from .controllability import controllability_report


def second_order_poles(
    settling_time_s: float,
    damping_ratio: float,
) -> np.ndarray:
    """Return a conjugate pair of continuous poles from settling-time targets."""

    if settling_time_s <= 0.0:
        raise ValueError("settling_time_s must be positive.")
    if not 0.0 < damping_ratio < 1.0:
        raise ValueError("damping_ratio must be between 0 and 1.")

    natural_frequency = 4.0 / (damping_ratio * settling_time_s)
    real = -damping_ratio * natural_frequency
    imaginary = natural_frequency * np.sqrt(1.0 - damping_ratio**2)
    return np.array([real + 1j * imaginary, real - 1j * imaginary])


def design_discrete_feedback(
    Ad: np.ndarray,
    Bd: np.ndarray,
    continuous_poles: np.ndarray,
    dt_s: float,
) -> dict[str, np.ndarray]:
    """Map desired poles into the z-plane and place them in ``Ad - Bd @ K``."""

    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive.")
    report = controllability_report(Ad, Bd)
    if report["rank"] != report["state_dimension"]:
        raise ValueError("Cannot place poles: the discrete model is not controllable.")

    continuous_poles = np.asarray(continuous_poles, dtype=complex)
    if continuous_poles.size != Ad.shape[0]:
        raise ValueError("One desired continuous pole is required per state.")

    requested_poles = np.exp(continuous_poles * dt_s)
    placement = place_poles(Ad, Bd, requested_poles)
    K = np.asarray(placement.gain_matrix, dtype=float)
    computed_poles = np.linalg.eigvals(Ad - Bd @ K)
    return {
        "K": K,
        "requested_poles": requested_poles,
        "computed_poles": computed_poles,
    }
