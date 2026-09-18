"""Steady-state predictor-form reference for the recursive filter."""

import control as ct
import numpy as np


def design_steady_state_kf(
    Ad: np.ndarray,
    Gd: np.ndarray,
    C: np.ndarray,
    Q_process: np.ndarray,
    R_measurement: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return python-control's predictor-form discrete estimator solution.

    ``dlqe`` returns the gain used by a one-step predictor.  The hand-written
    filter exposes posterior correction gain ``K_k``; at steady state the
    matching predictor correction is approximately ``Ad @ K_k``.
    """

    Ad = np.asarray(Ad, dtype=float)
    Gd = np.asarray(Gd, dtype=float)
    C = np.asarray(C, dtype=float)
    Q_process = np.asarray(Q_process, dtype=float)
    R_measurement = np.asarray(R_measurement, dtype=float)
    if Ad.shape != (2, 2) or Gd.shape != (2, 1) or C.shape != (1, 2):
        raise ValueError("expected a two-state, scalar-noise, scalar-measurement model.")
    if Q_process.shape != (1, 1) or R_measurement.shape != (1, 1):
        raise ValueError("Q_process and R_measurement must be scalar matrices.")
    if Q_process.item() < 0.0 or R_measurement.item() <= 0.0:
        raise ValueError("Q_process must be non-negative and R_measurement positive.")

    predictor_gain, covariance, poles = ct.dlqe(Ad, Gd, C, Q_process, R_measurement)
    return {
        "predictor_gain": np.asarray(predictor_gain, dtype=float),
        "covariance": np.asarray(covariance, dtype=float),
        "poles": np.asarray(poles),
    }
