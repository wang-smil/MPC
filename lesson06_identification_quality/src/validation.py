"""Independent trajectory validation for identified joint models."""

from __future__ import annotations

import numpy as np

from lesson05_system_identification.src.excitation import MultiSine
from lesson05_system_identification.src.plant import SingleAxisPlant
from .quality_experiments import compare_friction_models


def _simulate_model(
    torque_nm: np.ndarray,
    dt_s: float,
    inertia: float,
    damping: float,
    coulomb_friction: float,
) -> np.ndarray:
    """Simulate one identified model with semi-implicit Euler integration."""

    position = 0.0
    velocity = 0.0
    positions = np.empty_like(torque_nm, dtype=float)
    for index, torque in enumerate(torque_nm):
        friction = coulomb_friction * np.tanh(velocity / 0.02)
        acceleration = (torque - damping * velocity - friction) / inertia
        velocity += acceleration * dt_s
        position += velocity * dt_s
        positions[index] = position
    return positions


def _simulate_true_plant(
    torque_nm: np.ndarray,
    dt_s: float,
    inertia: float,
) -> np.ndarray:
    plant = SingleAxisPlant(
        inertia=inertia,
        damping=0.080,
        coulomb_friction=0.10,
        torque_limit_nm=2.0,
    )
    records = [plant.step(torque, dt_s) for torque in torque_nm]
    return np.array([record["position_true_rad"] for record in records])


def validate_friction_models(duration_s: float, dt_s: float) -> dict:
    """Fit once, then validate nominal and +30% inertia plants on new input."""

    if duration_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("duration_s and dt_s must be positive.")
    window_length = min(61, len(np.arange(0.0, duration_s + 0.5 * dt_s, dt_s)))
    if window_length % 2 == 0:
        window_length -= 1
    fit = compare_friction_models(
        duration_s=duration_s,
        dt_s=dt_s,
        position_noise_std_deg=0.02,
        window_length=window_length,
    )
    time_s = np.arange(0.0, duration_s + 0.5 * dt_s, dt_s)
    signal = MultiSine(
        frequencies_hz=[0.8, 1.9, 3.4], amplitudes_nm=[0.40, 0.25, 0.10]
    )
    torque_nm = np.asarray(signal.evaluate(time_s), dtype=float)
    model_a = fit["model_a"]
    model_b = fit["model_b"]
    scenarios = {"nominal": 0.020, "payload_plus_30_percent": 0.026}
    results: dict[str, dict[str, float]] = {}
    for name, true_inertia in scenarios.items():
        position_true = _simulate_true_plant(torque_nm, dt_s, true_inertia)
        position_a = _simulate_model(
            torque_nm,
            dt_s,
            float(model_a["inertia_hat"]),
            float(model_a["damping_hat"]),
            0.0,
        )
        position_b = _simulate_model(
            torque_nm,
            dt_s,
            float(model_b["inertia_hat"]),
            float(model_b["damping_hat"]),
            float(model_b["coulomb_friction_hat"]),
        )
        results[name] = {
            "model_a_position_rmse_rad": float(np.sqrt(np.mean((position_true - position_a) ** 2))),
            "model_b_position_rmse_rad": float(np.sqrt(np.mean((position_true - position_b) ** 2))),
        }
    return results
