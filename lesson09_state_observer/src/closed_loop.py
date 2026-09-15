"""Observer-based LQR closed-loop simulation with position-only sensing."""

import numpy as np

from .observer import DiscreteObserver
from .velocity_estimators import LowPassDifferenceVelocity, RawDifferenceVelocity


def simulate_observer_closed_loop(
    config: dict,
    observer: DiscreteObserver,
    K: np.ndarray,
    plant_inertia: float,
    noise_scale: float = 1.0,
    encoder_bias_rad: float = 0.0,
    velocity_source: str = "observer",
) -> dict[str, np.ndarray]:
    """Simulate plant, encoder, state observer, and clipped state feedback."""

    dt_s = float(config["simulation"]["dt_s"])
    duration_s = float(config["simulation"]["duration_s"])
    damping = float(config["model"]["damping"])
    torque_limit_nm = float(config["controller"]["torque_limit_nm"])
    target_rad = float(np.deg2rad(config["controller"]["target_deg"]))
    position_noise_rad = float(
        np.deg2rad(config["measurement"]["position_noise_std_deg"])
    )
    initial = config["initial_condition"]
    q_true_rad = float(np.deg2rad(initial["q_true_deg"]))
    dq_true_rad_s = float(initial["dq_true_rad_s"])
    K = np.asarray(K, dtype=float)

    if dt_s <= 0.0 or duration_s <= 0.0 or plant_inertia <= 0.0:
        raise ValueError("dt_s, duration_s, and plant_inertia must be positive.")
    if torque_limit_nm <= 0.0 or noise_scale < 0.0:
        raise ValueError("torque limit must be positive and noise_scale non-negative.")
    if K.shape != (1, 2):
        raise ValueError("K must have shape (1, 2).")
    if velocity_source not in {"observer", "raw_difference", "filtered_difference"}:
        raise ValueError("velocity_source is not supported.")

    raw_difference = RawDifferenceVelocity(dt_s)
    filtered_difference = LowPassDifferenceVelocity(
        dt_s,
        alpha=float(config["velocity_difference"]["lpf_alpha"]),
    )
    rng = np.random.default_rng(int(config["simulation"]["seed"]))
    reference_state = np.array([target_rad, 0.0])
    logs: dict[str, list[float | bool]] = {
        "time_s": [],
        "q_ref_rad": [],
        "q_true_rad": [],
        "q_measured_rad": [],
        "q_hat_rad": [],
        "dq_true_rad_s": [],
        "dq_raw_rad_s": [],
        "dq_filtered_rad_s": [],
        "dq_hat_rad_s": [],
        "q_control_estimate_rad": [],
        "dq_control_estimate_rad_s": [],
        "q_estimation_error_rad": [],
        "dq_estimation_error_rad_s": [],
        "q_control_estimation_error_rad": [],
        "dq_control_estimation_error_rad_s": [],
        "y_hat_rad": [],
        "innovation_rad": [],
        "torque_unsat_nm": [],
        "torque_cmd_nm": [],
        "saturated": [],
    }

    sample_count = int(round(duration_s / dt_s))
    for sample_index in range(sample_count):
        q_measured_rad = (
            q_true_rad
            + float(encoder_bias_rad)
            + rng.normal(0.0, noise_scale * position_noise_rad)
        )
        dq_raw_rad_s = raw_difference.update(q_measured_rad)
        dq_filtered_rad_s = filtered_difference.update(q_measured_rad)

        if velocity_source == "observer":
            controller_state = observer.x_hat.copy()
        elif velocity_source == "raw_difference":
            controller_state = np.array([q_measured_rad, dq_raw_rad_s])
        else:
            controller_state = np.array([q_measured_rad, dq_filtered_rad_s])
        q_control_estimate_rad = float(controller_state[0])
        dq_control_estimate_rad_s = float(controller_state[1])

        torque_unsat_nm = float(-(K @ (controller_state - reference_state)).item())
        torque_cmd_nm = float(
            np.clip(torque_unsat_nm, -torque_limit_nm, torque_limit_nm)
        )
        observer_result = observer.update(
            measurement=q_measured_rad,
            control_input=torque_cmd_nm,
        )

        acceleration_rad_s2 = (torque_cmd_nm - damping * dq_true_rad_s) / plant_inertia
        dq_true_rad_s += acceleration_rad_s2 * dt_s
        q_true_rad += dq_true_rad_s * dt_s
        x_hat = np.asarray(observer_result["x_hat"])

        logs["time_s"].append((sample_index + 1) * dt_s)
        logs["q_ref_rad"].append(target_rad)
        logs["q_true_rad"].append(q_true_rad)
        logs["q_measured_rad"].append(q_measured_rad)
        logs["q_hat_rad"].append(float(x_hat[0]))
        logs["dq_true_rad_s"].append(dq_true_rad_s)
        logs["dq_raw_rad_s"].append(dq_raw_rad_s)
        logs["dq_filtered_rad_s"].append(dq_filtered_rad_s)
        logs["dq_hat_rad_s"].append(float(x_hat[1]))
        logs["q_control_estimate_rad"].append(q_control_estimate_rad)
        logs["dq_control_estimate_rad_s"].append(dq_control_estimate_rad_s)
        logs["q_estimation_error_rad"].append(q_true_rad - float(x_hat[0]))
        logs["dq_estimation_error_rad_s"].append(dq_true_rad_s - float(x_hat[1]))
        logs["y_hat_rad"].append(float(observer_result["y_hat"]))
        logs["q_control_estimation_error_rad"].append(q_true_rad - q_control_estimate_rad)
        logs["dq_control_estimation_error_rad_s"].append(dq_true_rad_s - dq_control_estimate_rad_s)
        logs["innovation_rad"].append(float(observer_result["innovation"]))
        logs["torque_unsat_nm"].append(torque_unsat_nm)
        logs["torque_cmd_nm"].append(torque_cmd_nm)
        logs["saturated"].append(abs(torque_unsat_nm) > torque_limit_nm)

    return {name: np.asarray(values) for name, values in logs.items()}
