"""Identified joint model and ZOH discretization for Lesson 09."""

from pathlib import Path

import control as ct
import numpy as np
import yaml
from scipy.signal import cont2discrete


def load_config(path: Path | str) -> dict:
    """Load an observer configuration mapping from YAML."""

    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Observer configuration must be a mapping.")
    return config


def build_identified_model(
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the Lesson 05/06 joint model and discretize it with ZOH."""

    inertia = float(config["model"]["inertia"])
    damping = float(config["model"]["damping"])
    dt_s = float(config["simulation"]["dt_s"])
    if inertia <= 0.0:
        raise ValueError("model.inertia must be positive.")
    if damping < 0.0:
        raise ValueError("model.damping must be non-negative.")
    if dt_s <= 0.0:
        raise ValueError("simulation.dt_s must be positive.")

    A = np.array([[0.0, 1.0], [0.0, -damping / inertia]])
    B = np.array([[0.0], [1.0 / inertia]])
    C = np.eye(2)
    D = np.zeros((2, 1))
    Ad, Bd, _, _, _ = cont2discrete((A, B, C, D), dt_s, method="zoh")
    return A, B, np.asarray(Ad), np.asarray(Bd)

def build_balanced_lqr_gain(Ad: np.ndarray, Bd: np.ndarray, config: dict) -> np.ndarray:
    """Recompute the Lesson 08 balanced discrete LQR gain from Bryson limits."""

    Ad = np.asarray(Ad, dtype=float)
    Bd = np.asarray(Bd, dtype=float)
    if Ad.shape != (2, 2) or Bd.shape != (2, 1):
        raise ValueError("Ad must be 2-by-2 and Bd must be 2-by-1.")

    controller = config["controller"]
    max_position_error_rad = float(np.deg2rad(controller["max_position_error_deg"]))
    max_velocity_error_rad_s = float(controller["max_velocity_error_rad_s"])
    max_torque_nm = float(controller["torque_limit_nm"])
    if (
        max_position_error_rad <= 0.0
        or max_velocity_error_rad_s <= 0.0
        or max_torque_nm <= 0.0
    ):
        raise ValueError("Bryson limits must be positive.")

    Q = np.diag(
        [
            1.0 / max_position_error_rad**2,
            1.0 / max_velocity_error_rad_s**2,
        ]
    )
    R = np.array([[1.0 / max_torque_nm**2]])
    K, _, _ = ct.dlqr(Ad, Bd, Q, R)
    return np.asarray(K)
