"""Performance and LQR-cost metrics from simulation logs."""

import numpy as np


def calculate_metrics(log: dict[str, np.ndarray]) -> dict[str, float]:
    """Summarize tracking, actuator use, saturation, and logged LQR costs."""

    time_s = np.asarray(log["time_s"], dtype=float)
    reference = np.asarray(log["q_ref_rad"], dtype=float)
    position = np.asarray(log["q_rad"], dtype=float)
    torque = np.asarray(log["torque_cmd_nm"], dtype=float)
    saturated = np.asarray(log["saturated"], dtype=bool)
    state_cost = np.asarray(log["state_cost"], dtype=float)
    input_cost = np.asarray(log["input_cost"], dtype=float)
    total_stage_cost = np.asarray(log["total_stage_cost"], dtype=float)
    arrays = (reference, position, torque, saturated, state_cost, input_cost, total_stage_cost)
    if time_s.size == 0 or not all(values.size == time_s.size for values in arrays):
        raise ValueError("Metric arrays must be non-empty and aligned.")

    error = position - reference
    final_reference = float(reference[-1])
    settling_band = 0.02 * max(abs(final_reference), 1e-12)
    settled = np.array([np.all(np.abs(error[index:]) <= settling_band) for index in range(time_s.size)])
    indices = np.flatnonzero(settled)
    settling_time_s = float(time_s[indices[0]]) if indices.size else float("nan")
    overshoot_percent = 0.0
    if abs(final_reference) > 1e-12:
        overshoot_percent = float(max(0.0, (np.max(position) - final_reference) / abs(final_reference) * 100.0))

    return {
        "final_error_rad": float(abs(error[-1])),
        "tracking_rmse_rad": float(np.sqrt(np.mean(error**2))),
        "overshoot_percent": overshoot_percent,
        "settling_time_s": settling_time_s,
        "peak_torque_nm": float(np.max(np.abs(torque))),
        "rms_torque_nm": float(np.sqrt(np.mean(torque**2))),
        "saturation_ratio_percent": float(100.0 * np.mean(saturated)),
        "state_cost_total": float(np.sum(state_cost)),
        "input_cost_total": float(np.sum(input_cost)),
        "lqr_cost_total": float(np.sum(total_stage_cost)),
    }
