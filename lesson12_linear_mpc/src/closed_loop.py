"""One-millisecond plant/KF simulation with ten-millisecond held feedback.

This module owns simulated truth. Both controllers receive only the posterior
Kalman state estimate and execute one input per controller update.
"""

import numpy as np

from lesson08_lqr_optimal_control.src.lqr_design import (
    build_bryson_weights,
    dlqr_from_dare,
)
from lesson08_lqr_optimal_control.src.model import discretize_zoh
from lesson10_kalman_filter.src.kalman_filter import DiscreteKalmanFilter
from lesson10_kalman_filter.src.noise_model import build_noise_covariances

from .model_loader import build_mpc_model
from .mpc_controller import LinearMPCController
from .prediction import _finite_array, _validated_horizon


def _positive_period(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite number.")
    try:
        period = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number.") from exc
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")
    return period


def _exact_count(duration: float, period: float, name: str) -> int:
    count = int(round(duration / period))
    if count <= 0 or not np.isclose(count * period, duration, rtol=0.0, atol=1e-10):
        raise ValueError(f"{name} must be an integer multiple of the plant period.")
    return count


def build_closed_loop_design(config: dict) -> dict:
    """Reuse Lessons 08/10 to align the fast estimator and slow LQR/MPC."""

    model = build_mpc_model(config)
    plant_dt = _positive_period(config["plant"]["dt_s"], "plant.dt_s")
    estimator_dt = _positive_period(config["estimator"]["dt_s"], "estimator.dt_s")
    control_dt = _positive_period(config["mpc"]["dt_s"], "mpc.dt_s")
    duration = _positive_period(config["experiment_b"]["duration_s"], "experiment_b.duration_s")
    if not np.isclose(estimator_dt, plant_dt, rtol=0.0, atol=1e-12):
        raise ValueError("estimator.dt_s must equal plant.dt_s in Experiment B.")
    stride = _exact_count(control_dt, plant_dt, "mpc.dt_s")
    sample_count = _exact_count(duration, plant_dt, "experiment_b.duration_s")

    Ad_fast, Bd_fast = discretize_zoh(model["A"], model["B"], plant_dt)
    limits = config["design_limits"]
    scales = config["lqr"]
    Q, R = build_bryson_weights(
        float(limits["max_position_error_deg"]),
        float(limits["max_velocity_error_rad_s"]),
        float(limits["max_torque_nm"]),
        float(scales["position_weight_scale"]),
        float(scales["velocity_weight_scale"]),
        float(scales["torque_weight_scale"]),
    )
    K, P = dlqr_from_dare(model["Ad"], model["Bd"], Q, R)
    covariance = build_noise_covariances(config, Bd_fast)
    return {
        "Ad_control": model["Ad"],
        "Bd_control": model["Bd"],
        "Ad_fast": Ad_fast,
        "Bd_fast": Bd_fast,
        "Q": Q,
        "R": R,
        "K": K,
        "P_terminal": P,
        "C": np.array([[1.0, 0.0]]),
        "covariance": covariance,
        "plant_dt_s": plant_dt,
        "control_stride": stride,
        "sample_count": sample_count,
    }


def simulate_closed_loop(
    config: dict,
    mode: str,
    *,
    horizon: int | None = None,
    measurement_noise: np.ndarray | None = None,
    torque_limit_nm: float | None = None,
) -> dict[str, np.ndarray]:
    """Simulate output feedback; each control event holds one applied torque."""

    if mode not in ("lqr", "mpc"):
        raise ValueError("mode must be 'lqr' or 'mpc'.")
    design = build_closed_loop_design(config)
    if torque_limit_nm is not None:
        torque_limit_nm = _positive_period(torque_limit_nm, 'torque_limit_nm')
    count = design["sample_count"]
    if mode == "mpc":
        if horizon is None:
            raise ValueError("horizon is required for MPC mode.")
        horizon = _validated_horizon(horizon)
        controller = LinearMPCController(
            design["Ad_control"], design["Bd_control"],
            design["Q"], design["R"], design["P_terminal"], horizon,
            torque_bounds=(-torque_limit_nm, torque_limit_nm) if torque_limit_nm is not None else None,
        )
    else:
        controller = None

    if measurement_noise is None:
        rng = np.random.default_rng(int(config["experiment_b"]["seed"]))
        standard_deviation = np.deg2rad(float(config["measurement"]["position_noise_std_deg"]))
        noise = rng.normal(0.0, standard_deviation, count)
    else:
        noise = _finite_array(measurement_noise, "measurement_noise")
        if noise.shape != (count,):
            raise ValueError(f"measurement_noise must have shape ({count},).")

    initial_true = config["initial_condition"]
    true_state = np.array(
        [np.deg2rad(float(initial_true["q_true_deg"])), float(initial_true["dq_true_rad_s"])],
        dtype=float,
    )
    initial_estimate = config["initial_estimate"]
    estimate0 = np.array(
        [np.deg2rad(float(initial_estimate["position_deg"])),
         float(initial_estimate["velocity_rad_s"])],
        dtype=float,
    )
    reference = np.array([np.deg2rad(float(config["experiment_b"]["target_deg"])), 0.0])
    covariance = design["covariance"]
    filter_ = DiscreteKalmanFilter(
        design["Ad_fast"], design["Bd_fast"], design["C"],
        covariance["Gd"], covariance["Q_process"], covariance["R_measurement"],
        estimate0, covariance["P0"],
    )

    fields = (
        "time_s", "q_true_rad", "dq_true_rad_s", "q_measured_rad", "measurement_noise_rad",
        "q_hat_rad", "dq_hat_rad_s", "q_ref_rad", "dq_ref_rad_s", "torque_request_nm",
        "torque_applied_nm", "control_update", "qp_status", "qp_objective",
        'safety_clip_active', 'torque_constraint_active',
        "qp_solve_time_s", "qp_iterations", "innovation_rad", "nis",
    )
    log: dict[str, list] = {name: [] for name in fields}
    applied_torque = 0.0
    requested_torque = 0.0
    safety_clip_active = False
    torque_constraint_active = False
    for index in range(count):
        time_s = index * design["plant_dt_s"]
        measured_position = float(true_state[0] + noise[index])
        if index > 0:
            filter_.predict(applied_torque)
        estimate = filter_.update(measured_position)
        x_hat = np.asarray(estimate["x_hat"], dtype=float)

        control_update = index % design["control_stride"] == 0
        qp_status = ""
        qp_objective = np.nan
        qp_solve_time = np.nan
        qp_iterations = np.nan
        if control_update:
            torque_constraint_active = False
            if mode == "lqr":
                applied_torque = -float((design["K"] @ (x_hat - reference)).item())
                qp_status = "analytic"
            else:
                result = controller.solve(x_hat=x_hat, x_ref=reference)
                qp_status = str(result["status"])
                if qp_status not in ("optimal", "optimal_inaccurate") or result["u0"] is None:
                    raise RuntimeError(
                        f"MPC solve failed at index {index}, t={time_s:.3f}s: "
                        f"{qp_status}; {result.get('error')}"
                    )
                applied_torque = float(result["u0"][0])
                torque_constraint_active = bool(
                    torque_limit_nm is not None
                    and np.any(np.abs(result['U']) >= torque_limit_nm - 1e-5)
                )
                qp_objective = float(result["objective"])
                if result["solve_time_s"] is not None:
                    qp_solve_time = float(result["solve_time_s"])
                if result["iterations"] is not None:
                    qp_iterations = float(result["iterations"])
            requested_torque = applied_torque
            if torque_limit_nm is not None:
                applied_torque = float(np.clip(requested_torque, -torque_limit_nm, torque_limit_nm))
            safety_clip_active = bool(abs(requested_torque - applied_torque) > 1e-5)

        values = {
            "time_s": time_s,
            "q_true_rad": float(true_state[0]),
            "dq_true_rad_s": float(true_state[1]),
            "q_measured_rad": measured_position,
            "measurement_noise_rad": float(noise[index]),
            "q_hat_rad": float(x_hat[0]),
            "dq_hat_rad_s": float(x_hat[1]),
            "q_ref_rad": float(reference[0]),
            "dq_ref_rad_s": float(reference[1]),
            "torque_request_nm": requested_torque,
            "torque_applied_nm": applied_torque,
            'safety_clip_active': safety_clip_active,
            'torque_constraint_active': torque_constraint_active,
            "control_update": control_update,
            "qp_status": qp_status,
            "qp_objective": qp_objective,
            "qp_solve_time_s": qp_solve_time,
            "qp_iterations": qp_iterations,
            "innovation_rad": float(estimate["innovation"]),
            "nis": float(estimate["nis"]),
        }
        for name, value in values.items():
            log[name].append(value)
        true_state = (
            design["Ad_fast"] @ true_state
            + design["Bd_fast"][:, 0] * applied_torque
        )

    output = {name: np.asarray(values) for name, values in log.items()}
    output["controller_mode"] = np.full(count, mode)
    output["horizon_steps"] = np.full(count, horizon if mode == "mpc" else 0, dtype=int)
    return output
