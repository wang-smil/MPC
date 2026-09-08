"""Identified continuous and ZOH-discrete single-joint models."""

from pathlib import Path

import numpy as np
import yaml
from scipy.signal import cont2discrete


def build_continuous_model(
    inertia: float,
    damping: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the state model for ``x=[position, velocity]``."""

    if inertia <= 0.0 or damping < 0.0:
        raise ValueError("inertia must be positive and damping cannot be negative.")
    A = np.array([[0.0, 1.0], [0.0, -damping / inertia]])
    B = np.array([[0.0], [1.0 / inertia]])
    return A, B


def discretize_zoh(
    A: np.ndarray,
    B: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize a continuous state model with zero-order hold."""

    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive.")
    matrix_a = np.asarray(A, dtype=float)
    matrix_b = np.asarray(B, dtype=float)
    if matrix_a.ndim != 2 or matrix_a.shape[0] != matrix_a.shape[1]:
        raise ValueError("A must be a square matrix.")
    if matrix_b.ndim != 2 or matrix_b.shape[0] != matrix_a.shape[0]:
        raise ValueError("B must have the same row count as A.")
    C = np.eye(matrix_a.shape[0])
    D = np.zeros((matrix_a.shape[0], matrix_b.shape[1]))
    Ad, Bd, _, _, _ = cont2discrete((matrix_a, matrix_b, C, D), dt_s, method="zoh")
    return np.asarray(Ad), np.asarray(Bd)


def load_config(path: Path) -> dict:
    """Load and validate the Lesson 07 controller configuration."""

    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required_sections = {
        "simulation",
        "model",
        "controller",
        "reference",
        "actuator",
        "sensor",
        "stress",
    }
    if not isinstance(config, dict) or not required_sections.issubset(config):
        raise ValueError("controller configuration is missing required sections.")
    if config["simulation"]["dt_s"] <= 0.0 or config["simulation"]["duration_s"] <= 0.0:
        raise ValueError("simulation timing must be positive.")
    if config["model"]["inertia"] <= 0.0 or config["model"]["damping"] < 0.0:
        raise ValueError("model inertia must be positive and damping cannot be negative.")
    if not 0.0 < config["controller"]["damping_ratio"] < 1.0:
        raise ValueError("damping_ratio must lie between zero and one.")
    if config["controller"]["settling_time_s"] <= 0.0:
        raise ValueError("settling_time_s must be positive.")
    if config["actuator"]["torque_limit_nm"] <= 0.0:
        raise ValueError("torque_limit_nm must be positive.")
    if config["sensor"]["position_noise_std_deg"] < 0.0:
        raise ValueError("position_noise_std_deg cannot be negative.")
    alpha = config["sensor"]["velocity_filter_alpha"]
    if not 0.0 <= alpha < 1.0:
        raise ValueError("velocity_filter_alpha must lie in [0, 1).")
    if config["stress"]["payload_inertia_scale"] <= 0.0:
        raise ValueError("payload_inertia_scale must be positive.")
    return config
