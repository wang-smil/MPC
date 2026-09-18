"""Metrics shared by Kalman tuning and estimator-comparison experiments."""

import numpy as np


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _finite_mean(values: np.ndarray) -> float:
    values = _finite(values)
    return float(np.mean(values)) if values.size else float("nan")


def _finite_rms(values: np.ndarray) -> float:
    values = _finite(values)
    return _rms(values) if values.size else float("nan")


def _finite_percentile(values: np.ndarray, percentile: float) -> float:
    values = _finite(values)
    return float(np.percentile(values, percentile)) if values.size else float("nan")


def calculate_metrics(log: dict[str, np.ndarray]) -> dict[str, float]:
    """Summarize accuracy, control effort, and available estimator diagnostics."""

    required = {
        "q_true_rad", "q_ref_rad", "position_error_rad", "velocity_error_rad_s",
        "innovation_rad", "nis", "torque_applied_nm", "saturated", "kalman_gain_q",
        "kalman_gain_dq",
    }
    missing = required.difference(log)
    if missing:
        raise ValueError(f"log is missing fields: {sorted(missing)}")
    q_true = np.asarray(log["q_true_rad"], dtype=float)
    q_ref = np.asarray(log["q_ref_rad"], dtype=float)
    velocity_error = np.asarray(log["velocity_error_rad_s"], dtype=float)
    return {
        "q_rmse_rad": _rms(log["position_error_rad"]),
        "dq_rmse_rad_s": _rms(velocity_error),
        "estimate_noise_std_rad_s": float(np.std(velocity_error)),
        "innovation_rms_rad": _finite_rms(log["innovation_rad"]),
        "nis_mean": _finite_mean(log["nis"]),
        "nis_p95": _finite_percentile(log["nis"], 95.0),
        "tracking_rmse_rad": _rms(q_true - q_ref),
        "control_rms_nm": _rms(log["torque_applied_nm"]),
        "peak_torque_nm": float(np.max(np.abs(np.asarray(log["torque_applied_nm"], dtype=float)))),
        "saturation_ratio_percent": float(100.0 * np.mean(np.asarray(log["saturated"], dtype=bool))),
        "mean_kalman_gain_q": _finite_mean(log["kalman_gain_q"]),
        "mean_kalman_gain_dq": _finite_mean(log["kalman_gain_dq"]),
    }