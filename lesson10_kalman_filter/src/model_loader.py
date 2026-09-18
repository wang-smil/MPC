"""Reuse the identified Lesson 06 joint and Lesson 08 LQR conventions."""

from pathlib import Path

import numpy as np
import yaml

from lesson08_lqr_optimal_control.src.lqr_design import build_bryson_weights, design_dlqr
from lesson08_lqr_optimal_control.src.model import build_continuous_model, discretize_zoh


def load_config(path: Path | str) -> dict:
    """Load the Lesson 10 configuration mapping."""

    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Kalman configuration must be a mapping.")
    return config


def build_identified_discrete_model(config: dict) -> dict[str, np.ndarray]:
    """Build the 1 ms ZOH model from the identified inertia and damping."""

    inertia = float(config["model"]["inertia"])
    damping = float(config["model"]["damping"])
    dt_s = float(config["simulation"]["dt_s"])
    A, B = build_continuous_model(inertia, damping)
    Ad, Bd = discretize_zoh(A, B, dt_s)
    return {"A": A, "B": B, "Ad": Ad, "Bd": Bd, "C": np.array([[1.0, 0.0]])}


def build_balanced_lqr_gain(Ad: np.ndarray, Bd: np.ndarray, config: dict) -> np.ndarray:
    """Recreate Lesson 08's balanced discrete LQR gain."""

    controller = config["controller"]
    Q, R = build_bryson_weights(
        max_position_error_deg=float(controller["max_position_error_deg"]),
        max_velocity_error_rad_s=float(controller["max_velocity_error_rad_s"]),
        max_torque_nm=float(controller["torque_limit_nm"]),
        position_scale=1.0,
        velocity_scale=1.0,
        torque_scale=1.0,
    )
    return np.asarray(design_dlqr(Ad, Bd, Q, R)["K"], dtype=float)
