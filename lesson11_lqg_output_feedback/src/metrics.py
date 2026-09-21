"""Metrics for LQG tracking, estimation, innovation, and saturation studies."""

import numpy as np


def _finite_rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def _finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _window_rms(values: np.ndarray, mask: np.ndarray) -> float:
    return _finite_rms(np.asarray(values, dtype=float)[mask]) if np.any(mask) else float("nan")


def calculate_metrics(log: dict[str, np.ndarray], startup_end_s: float = 0.5) -> dict[str, float]:
    """Summarize all controller modes, preserving unavailable baseline fields as NaN."""

    required = {
        "time_s", "q_true_rad", "dq_true_rad_s", "q_ref_rad", "q_hat_rad",
        "dq_hat_rad_s", "innovation_rad", "nis", "torque_applied_nm", "saturated",
    }
    missing = required.difference(log)
    if missing:
        raise ValueError(f"log is missing fields: {sorted(missing)}")
    if startup_end_s <= 0.0:
        raise ValueError("startup_end_s must be positive.")

    time_s = np.asarray(log["time_s"], dtype=float)
    q_error = np.asarray(log["q_true_rad"], dtype=float) - np.asarray(log["q_hat_rad"], dtype=float)
    dq_error = np.asarray(log["dq_true_rad_s"], dtype=float) - np.asarray(log["dq_hat_rad_s"], dtype=float)
    startup = time_s <= startup_end_s
    steady = time_s > startup_end_s
    torque = np.asarray(log["torque_applied_nm"], dtype=float)

    return {
        "tracking_rmse_rad": _finite_rms(np.asarray(log["q_true_rad"]) - np.asarray(log["q_ref_rad"])),
        "q_estimation_rmse_rad": _finite_rms(q_error),
        "dq_estimation_rmse_rad_s": _finite_rms(dq_error),
        "startup_q_estimation_rmse_rad": _window_rms(q_error, startup),
        "startup_dq_estimation_rmse_rad_s": _window_rms(dq_error, startup),
        "steady_q_estimation_rmse_rad": _window_rms(q_error, steady),
        "steady_dq_estimation_rmse_rad_s": _window_rms(dq_error, steady),
        "innovation_rms_rad": _finite_rms(log["innovation_rad"]),
        "nis_mean": _finite_mean(log["nis"]),
        "control_rms_nm": _finite_rms(torque),
        "peak_torque_nm": float(np.max(np.abs(torque))),
        "saturation_ratio_percent": float(100.0 * np.mean(np.asarray(log["saturated"], dtype=bool))),
    }
