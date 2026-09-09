"""Noisy, load-aware single-joint simulation for discrete LQR."""

import numpy as np

from .controller import LQRController


def simulate_closed_loop(
    config: dict,
    controller: LQRController,
    Q: np.ndarray,
    R: np.ndarray,
    plant_inertia: float,
    load_torque_nm: float = 0.0,
    load_start_s: float = float("inf"),
) -> dict[str, np.ndarray]:
    """Simulate measured-state LQR against a hidden viscous joint plant."""

    dt_s = float(config["simulation"]["dt_s"])
    duration_s = float(config["simulation"]["duration_s"])
    damping = float(config["model"]["damping"])
    target_rad = float(np.deg2rad(config["reference"]["target_deg"]))
    position_noise_rad = float(np.deg2rad(config["sensor"]["position_noise_std_deg"]))
    velocity_noise_rad_s = float(config["sensor"]["velocity_noise_std_rad_s"])
    if dt_s <= 0.0 or duration_s <= 0.0 or plant_inertia <= 0.0:
        raise ValueError("dt_s, duration_s, and plant_inertia must be positive.")
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    if Q.shape != (2, 2) or R.shape != (1, 1) or R.item() <= 0.0:
        raise ValueError("Q must be 2-by-2 and R must be positive 1-by-1.")

    time_s = np.arange(0.0, duration_s + 0.5 * dt_s, dt_s)
    rng = np.random.default_rng(int(config["simulation"]["seed"]))
    q_rad = 0.0
    dq_rad_s = 0.0
    logs = {
        "time_s": [],
        "q_ref_rad": [],
        "q_rad": [],
        "dq_ref_rad_s": [],
        "dq_rad_s": [],
        "q_measured_rad": [],
        "dq_measured_rad_s": [],
        "position_error_rad": [],
        "velocity_error_rad_s": [],
        "load_torque_nm": [],
        "torque_unsat_nm": [],
        "torque_cmd_nm": [],
        "saturated": [],
        "state_cost": [],
        "input_cost": [],
        "total_stage_cost": [],
    }

    for index, current_time_s in enumerate(time_s):
        q_measured_rad = q_rad + rng.normal(0.0, position_noise_rad)
        dq_measured_rad_s = dq_rad_s + rng.normal(0.0, velocity_noise_rad_s)
        measured_state = np.array([q_measured_rad, dq_measured_rad_s])
        reference_state = np.array([target_rad, 0.0])
        error = measured_state - reference_state
        output = controller.update(measured_state, reference_state)
        active_load_torque_nm = load_torque_nm if current_time_s >= load_start_s else 0.0
        state_cost = float(error @ Q @ error)
        input_cost = float(output["torque_unsat_nm"] ** 2 * R.item())
        total_stage_cost = state_cost + input_cost

        logs["time_s"].append(current_time_s)
        logs["q_ref_rad"].append(target_rad)
        logs["q_rad"].append(q_rad)
        logs["dq_ref_rad_s"].append(0.0)
        logs["dq_rad_s"].append(dq_rad_s)
        logs["q_measured_rad"].append(q_measured_rad)
        logs["dq_measured_rad_s"].append(dq_measured_rad_s)
        logs["position_error_rad"].append(error[0])
        logs["velocity_error_rad_s"].append(error[1])
        logs["load_torque_nm"].append(active_load_torque_nm)
        logs["torque_unsat_nm"].append(output["torque_unsat_nm"])
        logs["torque_cmd_nm"].append(output["torque_cmd_nm"])
        logs["saturated"].append(output["saturated"])
        logs["state_cost"].append(state_cost)
        logs["input_cost"].append(input_cost)
        logs["total_stage_cost"].append(total_stage_cost)

        if index < time_s.size - 1:
            acceleration_rad_s2 = (
                output["torque_cmd_nm"] - active_load_torque_nm - damping * dq_rad_s
            ) / plant_inertia
            dq_rad_s += acceleration_rad_s2 * dt_s
            q_rad += dq_rad_s * dt_s

    return {name: np.asarray(values) for name, values in logs.items()}
