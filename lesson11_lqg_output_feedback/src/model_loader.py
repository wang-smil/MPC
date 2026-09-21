"""Thin adapters that compose the Lesson 08 LQR and Lesson 10 KF designs."""

from pathlib import Path

import numpy as np

from lesson10_kalman_filter.src.model_loader import (
    build_balanced_lqr_gain,
    build_identified_discrete_model,
    load_config as _load_lesson10_style_config,
)
from lesson10_kalman_filter.src.noise_model import build_noise_covariances
from lesson10_kalman_filter.src.steady_state_kf import design_steady_state_kf


def load_config(path: Path | str) -> dict:
    """Load the Lesson 11 YAML configuration."""

    return _load_lesson10_style_config(path)


def build_lqg_design(config: dict) -> dict[str, np.ndarray]:
    """Build the shared identified model, LQR gain, and predictor-form KF gain."""

    model = build_identified_discrete_model(config)
    covariance = build_noise_covariances(config, model["Bd"])
    K_controller = build_balanced_lqr_gain(model["Ad"], model["Bd"], config)
    steady = design_steady_state_kf(
        model["Ad"],
        covariance["Gd"],
        model["C"],
        covariance["Q_process"],
        covariance["R_measurement"],
    )
    return {
        **model,
        **covariance,
        "K_controller": K_controller,
        "L_predictor": steady["predictor_gain"],
        "P_steady": steady["covariance"],
        "estimator_poles": steady["poles"],
    }
