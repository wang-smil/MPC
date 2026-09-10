"""Identified joint model and ZOH discretization for Lesson 08."""

from pathlib import Path

import numpy as np
import yaml
from scipy.signal import cont2discrete


def load_config(path: Path | str) -> dict:
    """Load the Lesson 08 YAML configuration as a mapping."""

    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("LQR configuration must be a mapping.")
    return config


def build_continuous_model(inertia: float, damping: float) -> tuple[np.ndarray, np.ndarray]:
    """Build the continuous position-velocity model from identified parameters."""

    if inertia <= 0.0:
        raise ValueError("inertia must be positive.")
    if damping < 0.0:
        raise ValueError("damping must be non-negative.")
    A = np.array([[0.0, 1.0], [0.0, -damping / inertia]])
    B = np.array([[0.0], [1.0 / inertia]])
    return A, B


def discretize_zoh(A: np.ndarray, B: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Discretize a state model with zero-order hold at the supplied period."""

    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive.")
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError("A must be square and B must have matching rows.")
    C = np.eye(A.shape[0])
    D = np.zeros((A.shape[0], B.shape[1]))
    Ad, Bd, _, _, _ = cont2discrete((A, B, C, D), dt_s, method="zoh")
    return np.asarray(Ad), np.asarray(Bd)
