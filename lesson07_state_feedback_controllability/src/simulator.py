"""Noisy, saturation-aware discrete simulation for one rotary joint."""

import numpy as np

from .state_feedback import feedback_torque


def simulate_closed_loop(
    config: dict,
    K: np.ndarray,
    plant_inertia: float,
) -> dict[str, np.ndarray]:
    """Simulate a fixed state-feedback controller against a viscous joint plant."""

    dt_s = float(config["simulation"]["dt_s"])
    duration_s = float(config["simulation"]["duration_s"])
    damping = float(config["model"]["damping"])
    target_rad = np.deg2rad(float(config["reference"]["target_deg"]))
    noise_std_rad = np.deg2rad(float(config["sensor"]["position_noise_std_deg"]))
    alpha = float(config["sensor"]["velocity_filter_alpha"])
    torque_limit_nm = float(config["actuator"]["torque_limit_nm"])

    if dt_s <= 0.0 or duration_s <= 0.0 or plant_inertia <= 0.0:
        raise ValueError("dt_s, duration_s, and plant_inertia must be positive.")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("velocity_filter_alpha must be in [0, 1).")

    time_s = np.arange(0.0, duration_s + 0.5 * dt_s, dt_s)
    sample_count = time_s.size
    rng = np.random.default_rng(int(config["simulation"]["seed"]))

    q_rad = 0.0
    dq_rad_s = 0.0
    previous_measured_q = 0.0
    estimated_dq = 0.0
    previous_estimated_dq = 0.0
    logs = {
        "time_s": [],
        "q_ref_rad": [],
        "q_rad": [],
        "dq_rad_s": [],
        "q_measured_rad": [],
        "dq_est_rad_s": [],
        "state_error_q_rad": [],
        "state_error_dq_rad_s": [],
        "torque_unsat_nm": [],
        "torque_applied_nm": [],
        "saturated": [],
    }

    for index, current_time in enumerate(time_s):
        q_measured = q_rad + rng.normal(0.0, noise_std_rad)
        raw_dq = 0.0 if index == 0 else (q_measured - previous_measured_q) / dt_s
        estimated_dq = alpha * previous_estimated_dq + (1.0 - alpha) * raw_dq
        state_estimate = np.array([q_measured, estimated_dq])
        reference_state = np.array([target_rad, 0.0])
        torque = feedback_torque(K, state_estimate, reference_state, torque_limit_nm)

        logs["time_s"].append(current_time)
        logs["q_ref_rad"].append(target_rad)
        logs["q_rad"].append(q_rad)
        logs["dq_rad_s"].append(dq_rad_s)
        logs["q_measured_rad"].append(q_measured)
        logs["dq_est_rad_s"].append(estimated_dq)
        logs["state_error_q_rad"].append(q_measured - target_rad)
        logs["state_error_dq_rad_s"].append(estimated_dq)
        logs["torque_unsat_nm"].append(torque["torque_unsat_nm"])
        logs["torque_applied_nm"].append(torque["torque_applied_nm"])
        logs["saturated"].append(torque["saturated"])

        if index < sample_count - 1:
            acceleration = (torque["torque_applied_nm"] - damping * dq_rad_s) / plant_inertia
            dq_rad_s += acceleration * dt_s
            q_rad += dq_rad_s * dt_s
            previous_measured_q = q_measured
            previous_estimated_dq = estimated_dq

    return {name: np.asarray(values) for name, values in logs.items()}
