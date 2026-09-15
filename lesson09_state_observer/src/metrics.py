"""Engineering metrics computed from observer closed-loop logs."""

import numpy as np


def _rms(values: np.ndarray) -> float:
    """Return root-mean-square magnitude for one recorded signal."""

    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _convergence_time(
    time_s: np.ndarray,
    position_error_rad: np.ndarray,
    velocity_error_rad_s: np.ndarray,
) -> float:
    """Return the first time both errors stay within 2% of their initial size."""

    position_limit = max(0.02 * abs(position_error_rad[0]), 1e-6)
    velocity_limit = max(0.02 * abs(velocity_error_rad_s[0]), 1e-6)
    within_limits = (
        (np.abs(position_error_rad) <= position_limit)
        & (np.abs(velocity_error_rad_s) <= velocity_limit)
    )
    for index in range(within_limits.size):
        if bool(np.all(within_limits[index:])):
            return float(time_s[index])
    return float("nan")


def calculate_metrics(log: dict[str, np.ndarray]) -> dict[str, float]:
    """Calculate estimation, tracking, innovation, and torque statistics."""

    required = {
        "time_s",
        "q_true_rad",
        "q_ref_rad",
        "q_estimation_error_rad",
        "dq_estimation_error_rad_s",
        "innovation_rad",
        "torque_cmd_nm",
        "saturated",
    }
    missing = required.difference(log)
    if missing:
        raise ValueError(f"log is missing fields: {sorted(missing)}")

    time_s = np.asarray(log["time_s"], dtype=float)
    position_error_rad = np.asarray(log["q_estimation_error_rad"], dtype=float)
    velocity_error_rad_s = np.asarray(log["dq_estimation_error_rad_s"], dtype=float)
    innovation_rad = np.asarray(log["innovation_rad"], dtype=float)
    torque_cmd_nm = np.asarray(log["torque_cmd_nm"], dtype=float)
    tracking_error_rad = np.asarray(log["q_true_rad"], dtype=float) - np.asarray(
        log["q_ref_rad"], dtype=float
    )
    if time_s.size == 0:
        raise ValueError("log signals must not be empty.")

    metrics = {
        "q_est_rmse_rad": _rms(position_error_rad),
        "dq_est_rmse_rad_s": _rms(velocity_error_rad_s),
        "observer_convergence_time_s": _convergence_time(
            time_s, position_error_rad, velocity_error_rad_s
        ),
        "innovation_rms_rad": _rms(innovation_rad),
        "tracking_rmse_rad": _rms(tracking_error_rad),
        "rms_torque_nm": _rms(torque_cmd_nm),
        "peak_torque_nm": float(np.max(np.abs(torque_cmd_nm))),
        "saturation_ratio_percent": float(
            100.0 * np.mean(np.asarray(log["saturated"], dtype=bool))
        ),
    }
    if "dq_control_estimation_error_rad_s" in log:
        control_velocity_error = np.asarray(
            log["dq_control_estimation_error_rad_s"], dtype=float
        )
        metrics["control_velocity_rmse_rad_s"] = _rms(control_velocity_error)
        metrics["control_velocity_error_std_rad_s"] = float(
            np.std(control_velocity_error)
        )
    return metrics
