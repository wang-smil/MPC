"""Performance metrics calculated from closed-loop simulation logs."""

import numpy as np


def calculate_metrics(log: dict[str, np.ndarray]) -> dict[str, float]:
    """Calculate tracking, settling, torque, and saturation measurements."""

    time_s = np.asarray(log["time_s"], dtype=float)
    reference = np.asarray(log["q_ref_rad"], dtype=float)
    position = np.asarray(log["q_rad"], dtype=float)
    applied_torque = np.asarray(log["torque_applied_nm"], dtype=float)
    saturated = np.asarray(log["saturated"], dtype=bool)
    if time_s.size == 0 or not all(array.size == time_s.size for array in (reference, position, applied_torque, saturated)):
        raise ValueError("All metric log arrays must be non-empty and aligned.")

    final_reference = float(reference[-1])
    tracking_error = position - reference
    settling_band = 0.02 * max(abs(final_reference), 1e-12)
    within_band_from_here = np.array(
        [np.all(np.abs(tracking_error[index:]) <= settling_band) for index in range(time_s.size)]
    )
    settling_indices = np.flatnonzero(within_band_from_here)
    settling_time_s = float(time_s[settling_indices[0]]) if settling_indices.size else float("nan")
    overshoot_percent = 0.0
    if abs(final_reference) > 1e-12:
        overshoot_percent = float(max(0.0, (np.max(position) - final_reference) / abs(final_reference) * 100.0))

    return {
        "final_error_rad": float(abs(tracking_error[-1])),
        "tracking_rmse_rad": float(np.sqrt(np.mean(tracking_error**2))),
        "overshoot_percent": overshoot_percent,
        "settling_time_s": settling_time_s,
        "peak_torque_nm": float(np.max(np.abs(applied_torque))),
        "rms_torque_nm": float(np.sqrt(np.mean(applied_torque**2))),
        "saturation_ratio_percent": float(100.0 * np.mean(saturated)),
    }
