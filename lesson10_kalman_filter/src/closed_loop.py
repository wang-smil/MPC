"""Causal plant, encoder, estimator, LQR, and actuator simulation."""

from collections.abc import Callable

import numpy as np

from .estimators import (
    DifferenceEstimator,
    FixedGainLuenbergerEstimator,
    KalmanEstimator,
)
from .kalman_filter import DiscreteKalmanFilter
from .model_loader import build_balanced_lqr_gain, build_identified_discrete_model
from .noise_model import build_noise_covariances


def _build_estimator(
    estimator_kind: str,
    config: dict,
    Ad: np.ndarray,
    Bd: np.ndarray,
    C: np.ndarray,
    lqr_gain: np.ndarray,
    covariance: dict[str, np.ndarray],
):
    """Construct one estimator behind the common causal ``step`` interface."""

    initial = config["initial_estimate"]
    x0 = np.array(
        [
            np.deg2rad(float(initial["position_deg"])),
            float(initial["velocity_rad_s"]),
        ]
    )
    dt_s = float(config["simulation"]["dt_s"])
    if estimator_kind == "kalman":
        return KalmanEstimator(
            DiscreteKalmanFilter(
                Ad, Bd, C, covariance["Gd"], covariance["Q_process"],
                covariance["R_measurement"], x0, covariance["P0"],
            )
        )
    if estimator_kind == "raw":
        return DifferenceEstimator(dt_s)
    if estimator_kind == "lpf":
        return DifferenceEstimator(
            dt_s, alpha=float(config["velocity_difference"]["lpf_alpha"])
        )
    if estimator_kind == "luenberger":
        return FixedGainLuenbergerEstimator(Ad, Bd, C, lqr_gain, dt_s, x0)
    raise ValueError("estimator_kind must be raw, lpf, luenberger, or kalman.")


def simulate_closed_loop(
    config: dict,
    estimator_kind: str = "kalman",
    assumed_q_scale: float = 1.0,
    assumed_r_scale: float = 1.0,
    payload_scale: float = 1.0,
    measurement_noise_scale: float = 1.0,
    load_torque: Callable[[float], float] | None = None,
) -> dict[str, np.ndarray]:
    """Simulate all estimators with one causal plant/actuator/noise boundary.

    Each sample predicts or advances the estimator with the torque actually
    applied in the prior interval, consumes the present encoder reading, then
    computes and clips the next torque.  Only this function owns true state.
    """

    if payload_scale <= 0.0 or measurement_noise_scale < 0.0:
        raise ValueError("payload_scale must be positive and noise scale non-negative.")
    model = build_identified_discrete_model(config)
    Ad, Bd, C = model["Ad"], model["Bd"], model["C"]
    covariance = build_noise_covariances(
        config, Bd, q_scale=assumed_q_scale, r_scale=assumed_r_scale
    )
    lqr_gain = build_balanced_lqr_gain(Ad, Bd, config)
    estimator = _build_estimator(estimator_kind, config, Ad, Bd, C, lqr_gain, covariance)

    dt_s = float(config["simulation"]["dt_s"])
    duration_s = float(config["simulation"]["duration_s"])
    torque_limit_nm = float(config["actuator"]["torque_limit_nm"])
    damping = float(config["model"]["damping"])
    inertia = payload_scale * float(config["model"]["inertia"])
    reference = np.array(
        [np.deg2rad(float(config["controller"]["target_deg"])), 0.0]
    )
    measurement_std_rad = measurement_noise_scale * np.deg2rad(
        float(config["measurement"]["position_noise_std_deg"])
    )
    disturbance_std_nm = float(config["process_noise"]["disturbance_torque_std_nm"])
    initial = config["initial_condition"]
    q_true = np.deg2rad(float(initial["q_true_deg"]))
    dq_true = float(initial["dq_true_rad_s"])
    applied_input_previous = 0.0
    rng = np.random.default_rng(int(config["simulation"]["seed"]))

    fields = (
        "time_s", "q_ref_rad", "q_true_rad", "q_measured_rad", "q_hat_rad",
        "dq_true_rad_s", "dq_hat_rad_s", "position_error_rad", "velocity_error_rad_s",
        "P_qq", "P_dqdq", "kalman_gain_q", "kalman_gain_dq", "innovation_rad",
        "innovation_variance", "nis", "torque_unsat_nm", "torque_applied_nm",
        "saturated", "process_disturbance_nm", "load_torque_nm",
    )
    logs: dict[str, list[float | bool]] = {field: [] for field in fields}

    for index in range(int(round(duration_s / dt_s))):
        time_s = index * dt_s
        q_measured = q_true + rng.normal(0.0, measurement_std_rad)
        estimate = estimator.step(q_measured, applied_input_previous)
        x_hat = np.asarray(estimate["x_hat"], dtype=float)
        torque_unsat = float(-(lqr_gain @ (x_hat - reference)).item())
        torque_applied = float(np.clip(torque_unsat, -torque_limit_nm, torque_limit_nm))
        process_disturbance = float(rng.normal(0.0, disturbance_std_nm))
        load = 0.0 if load_torque is None else float(load_torque(time_s))

        P = np.asarray(estimate["P"], dtype=float)
        gain = np.asarray(estimate["K"], dtype=float)
        logs["time_s"].append(time_s)
        logs["q_ref_rad"].append(float(reference[0]))
        logs["q_true_rad"].append(q_true)
        logs["q_measured_rad"].append(q_measured)
        logs["q_hat_rad"].append(float(x_hat[0]))
        logs["dq_true_rad_s"].append(dq_true)
        logs["dq_hat_rad_s"].append(float(x_hat[1]))
        logs["position_error_rad"].append(q_true - float(x_hat[0]))
        logs["velocity_error_rad_s"].append(dq_true - float(x_hat[1]))
        logs["P_qq"].append(float(P[0, 0]))
        logs["P_dqdq"].append(float(P[1, 1]))
        logs["kalman_gain_q"].append(float(gain[0, 0]))
        logs["kalman_gain_dq"].append(float(gain[1, 0]))
        logs["innovation_rad"].append(float(estimate["innovation"]))
        logs["innovation_variance"].append(float(estimate["innovation_covariance"]))
        logs["nis"].append(float(estimate["nis"]))
        logs["torque_unsat_nm"].append(torque_unsat)
        logs["torque_applied_nm"].append(torque_applied)
        logs["saturated"].append(abs(torque_unsat) > torque_limit_nm)
        logs["process_disturbance_nm"].append(process_disturbance)
        logs["load_torque_nm"].append(load)

        acceleration = (torque_applied + process_disturbance + load - damping * dq_true) / inertia
        dq_true += acceleration * dt_s
        q_true += dq_true * dt_s
        applied_input_previous = torque_applied

    return {name: np.asarray(values) for name, values in logs.items()}