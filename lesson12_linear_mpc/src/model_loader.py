"""Load Lesson 12 settings and reuse the identified Lesson 08 joint model."""

from pathlib import Path

import numpy as np
import yaml

from lesson08_lqr_optimal_control.src.model import (
    build_continuous_model,
    discretize_zoh,
)


def load_config(path: Path | str) -> dict:
    """Load a YAML configuration and require a top-level mapping."""

    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("MPC configuration must be a mapping.")
    return config


def _finite_number(config: dict, section: str, name: str) -> float:
    try:
        value = float(config[section][name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid configuration value: {section}.{name}.") from exc
    if not np.isfinite(value):
        raise ValueError(f"Configuration value {section}.{name} must be finite.")
    return value


def build_mpc_model(config: dict) -> dict[str, np.ndarray]:
    """Build continuous and discrete models using the MPC sample period.

    The continuous model and zero-order-hold implementation are deliberately
    reused from Lesson 08 so the identified parameter convention stays shared.
    """

    if not isinstance(config, dict):
        raise ValueError("MPC configuration must be a mapping.")
    inertia = _finite_number(config, "model", "inertia_kg_m2")
    damping = _finite_number(config, "model", "damping_nm_s_rad")
    dt_s = _finite_number(config, "mpc", "dt_s")
    if inertia <= 0.0:
        raise ValueError("model.inertia_kg_m2 must be positive.")
    if damping < 0.0:
        raise ValueError("model.damping_nm_s_rad must be non-negative.")
    if dt_s <= 0.0:
        raise ValueError("mpc.dt_s must be positive.")

    A, B = build_continuous_model(inertia, damping)
    Ad, Bd = discretize_zoh(A, B, dt_s)
    return {"A": A, "B": B, "Ad": Ad, "Bd": Bd}
