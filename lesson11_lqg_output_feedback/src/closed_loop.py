"""Fair causal LQG simulations; this module alone owns simulated plant truth."""

from collections.abc import Callable

import numpy as np

from lesson10_kalman_filter.src.estimators import KalmanEstimator
from lesson10_kalman_filter.src.kalman_filter import DiscreteKalmanFilter

from .lqg_controller import LQGController
from .model_loader import build_lqg_design


class SteadyStateKalmanEstimator:
    """Posterior-state adapter for Lesson 10's fixed predictor-form ``dlqe`` gain.

    The steady-state gain returned by ``dlqe`` corrects a one-step predictor.
    This controller needs a posterior state after the current measurement, so
    ``K_post = A_d^{-1} L_predictor`` makes ``A_d K_post = L_predictor``.
    """

    def __init__(
        self,
        Ad: np.ndarray,
        Bd: np.ndarray,
        C: np.ndarray,
        L_predictor: np.ndarray,
        x0: np.ndarray,
        P_steady: np.ndarray,
        R_measurement: np.ndarray,
    ) -> None:
        self.Ad = np.asarray(Ad, dtype=float)
        self.Bd = np.asarray(Bd, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.L_predictor = np.asarray(L_predictor, dtype=float)
        self.K_post = np.linalg.solve(self.Ad, self.L_predictor)
        self.x = np.asarray(x0, dtype=float).reshape(2, 1)
        self.P = np.asarray(P_steady, dtype=float)
        self.R = np.asarray(R_measurement, dtype=float)

    def step(self, measurement: float, applied_input: float) -> dict:
        x_prior = self.Ad @ self.x + self.Bd * float(applied_input)
        innovation = float(measurement) - float((self.C @ x_prior).item())
        innovation_variance = float((self.C @ self.P @ self.C.T + self.R).item())
        self.x = x_prior + self.K_post * innovation
        return {
            "x_hat": self.x[:, 0].copy(),
            "P": self.P.copy(),
            "K": self.K_post.copy(),
            "innovation": innovation,
            "innovation_covariance": innovation_variance,
            "nis": innovation**2 / innovation_variance,
        }


def _initial_estimate(config: dict) -> np.ndarray:
    initial = config["initial_estimate"]
    return np.array(
        [
            np.deg2rad(float(initial["position_deg"])),
            float(initial["velocity_rad_s"]),
        ]
    )


def _load_value(load_torque: Callable[[float], float] | float | None, time_s: float) -> float:
    if load_torque is None:
        return 0.0
    return float(load_torque(time_s) if callable(load_torque) else load_torque)


def _build_estimator(config: dict, design: dict, mode: str):
    x0 = _initial_estimate(config)
    if mode == "recursive_lqg":
        return KalmanEstimator(
            DiscreteKalmanFilter(
                design["Ad"],
                design["Bd"],
                design["C"],
                design["Gd"],
                design["Q_process"],
                design["R_measurement"],
                x0,
                design["P0"],
            )
        )
    if mode == "steady_state_lqg":
        return SteadyStateKalmanEstimator(
            design["Ad"],
            design["Bd"],
            design["C"],
            design["L_predictor"],
            x0,
            design["P_steady"],
            design["R_measurement"],
        )
    raise ValueError("mode must be full_state_lqr, recursive_lqg, or steady_state_lqg.")


def simulate_lqg(
    config: dict,
    mode: str,
    q_scale: float = 1.0,
    torque_limit: float | None = None,
    load_torque: Callable[[float], float] | float | None = None,
) -> dict[str, np.ndarray]:
    """Simulate an ideal full-state baseline or an output-feedback LQG controller.

    All modes use identical deterministic encoder and process-noise draws when
    called with the same configuration.  Only ``full_state_lqr`` accesses the
    simulated true state, and it is a labelled, non-deployable reference.
    """

    if mode not in {"full_state_lqr", "recursive_lqg", "steady_state_lqg"}:
        raise ValueError("mode must be full_state_lqr, recursive_lqg, or steady_state_lqg.")
    if q_scale <= 0.0:
        raise ValueError("q_scale must be positive.")

    design = build_lqg_design(config)
    if q_scale != 1.0:
        from lesson10_kalman_filter.src.noise_model import build_noise_covariances
        from lesson10_kalman_filter.src.steady_state_kf import design_steady_state_kf

        covariance = build_noise_covariances(config, design["Bd"], q_scale=q_scale)
        steady = design_steady_state_kf(
            design["Ad"], covariance["Gd"], design["C"],
            covariance["Q_process"], covariance["R_measurement"],
        )
        design = {**design, **covariance, "L_predictor": steady["predictor_gain"], "P_steady": steady["covariance"]}

    limit = float(config["actuator"]["torque_limit_nm"]) if torque_limit is None else float(torque_limit)
    if limit <= 0.0:
        raise ValueError("torque_limit must be positive.")
    dt_s = float(config["simulation"]["dt_s"])
    duration_s = float(config["simulation"]["duration_s"])
    if dt_s <= 0.0 or duration_s <= 0.0:
        raise ValueError("simulation dt_s and duration_s must be positive.")

    reference = np.array([np.deg2rad(float(config["controller"]["target_deg"])), 0.0])
    initial = config["initial_condition"]
    q_true = np.deg2rad(float(initial["q_true_deg"]))
    dq_true = float(initial["dq_true_rad_s"])
    inertia = float(config["model"]["inertia"])
    damping = float(config["model"]["damping"])
    measurement_std = np.deg2rad(float(config["measurement"]["position_noise_std_deg"]))
    disturbance_std = float(config["process_noise"]["disturbance_torque_std_nm"])
    rng = np.random.default_rng(int(config["simulation"]["seed"]))
    previous_applied = 0.0

    controller = None
    if mode != "full_state_lqr":
        controller = LQGController(_build_estimator(config, design, mode), design["K_controller"], limit)

    fields = (
        "time_s", "q_ref_rad", "q_true_rad", "dq_true_rad_s", "q_measured_rad",
        "q_hat_rad", "dq_hat_rad_s", "position_error_rad", "velocity_error_rad_s",
        "innovation_rad", "innovation_variance", "nis", "P_qq", "P_dqdq",
        "kalman_gain_q", "kalman_gain_dq", "torque_request_nm", "torque_applied_nm",
        "saturated", "process_disturbance_nm", "load_torque_nm",
    )
    logs: dict[str, list[float | bool | str]] = {field: [] for field in fields}
    controller_modes: list[str] = []

    for index in range(int(round(duration_s / dt_s))):
        time_s = index * dt_s
        measurement = q_true + float(rng.normal(0.0, measurement_std))
        if mode == "full_state_lqr":
            torque_request = -float((design["K_controller"] @ (np.array([q_true, dq_true]) - reference)).item())
            torque_applied = float(np.clip(torque_request, -limit, limit))
            x_hat = np.array([np.nan, np.nan])
            diagnostics = {"innovation": np.nan, "innovation_covariance": np.nan, "nis": np.nan,
                           "P": np.full((2, 2), np.nan), "K": np.full((2, 1), np.nan)}
        else:
            result = controller.step(measurement, previous_applied, reference)
            torque_request = result["torque_request"]
            torque_applied = result["torque_applied"]
            x_hat = result["x_hat"]
            diagnostics = result["estimator"]

        disturbance = float(rng.normal(0.0, disturbance_std))
        load = _load_value(load_torque, time_s)
        P = np.asarray(diagnostics["P"], dtype=float)
        gain = np.asarray(diagnostics["K"], dtype=float)
        logs["time_s"].append(time_s)
        logs["q_ref_rad"].append(float(reference[0]))
        logs["q_true_rad"].append(q_true)
        logs["dq_true_rad_s"].append(dq_true)
        logs["q_measured_rad"].append(measurement)
        logs["q_hat_rad"].append(float(x_hat[0]))
        logs["dq_hat_rad_s"].append(float(x_hat[1]))
        logs["position_error_rad"].append(q_true - float(x_hat[0]))
        logs["velocity_error_rad_s"].append(dq_true - float(x_hat[1]))
        logs["innovation_rad"].append(float(diagnostics["innovation"]))
        logs["innovation_variance"].append(float(diagnostics["innovation_covariance"]))
        logs["nis"].append(float(diagnostics["nis"]))
        logs["P_qq"].append(float(P[0, 0]))
        logs["P_dqdq"].append(float(P[1, 1]))
        logs["kalman_gain_q"].append(float(gain[0, 0]))
        logs["kalman_gain_dq"].append(float(gain[1, 0]))
        logs["torque_request_nm"].append(torque_request)
        logs["torque_applied_nm"].append(torque_applied)
        logs["saturated"].append(abs(torque_request) > limit)
        logs["process_disturbance_nm"].append(disturbance)
        logs["load_torque_nm"].append(load)
        controller_modes.append(mode)

        acceleration = (torque_applied + disturbance + load - damping * dq_true) / inertia
        dq_true += acceleration * dt_s
        q_true += dq_true * dt_s
        previous_applied = torque_applied

    output = {name: np.asarray(values) for name, values in logs.items()}
    output["controller_mode"] = np.asarray(controller_modes)
    return output
