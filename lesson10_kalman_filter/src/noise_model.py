"""Physical process, measurement, and initial covariance construction."""

import numpy as np


def build_noise_covariances(
    config: dict,
    Bd: np.ndarray,
    q_scale: float = 1.0,
    r_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Convert torque and encoder standard deviations into KF covariances."""

    if q_scale <= 0.0 or r_scale <= 0.0:
        raise ValueError("q_scale and r_scale must be positive.")
    Bd = np.asarray(Bd, dtype=float)
    if Bd.shape != (2, 1):
        raise ValueError("Bd must have shape (2, 1).")

    torque_std_nm = float(config["process_noise"]["disturbance_torque_std_nm"])
    position_std_deg = float(config["measurement"]["position_noise_std_deg"])
    initial_position_std_deg = float(config["initial_uncertainty"]["position_std_deg"])
    initial_velocity_std = float(config["initial_uncertainty"]["velocity_std_rad_s"])
    if min(torque_std_nm, position_std_deg, initial_position_std_deg, initial_velocity_std) <= 0.0:
        raise ValueError("standard deviations must be positive.")

    return {
        "Gd": Bd.copy(),
        "Q_process": np.array([[(q_scale * torque_std_nm**2)]], dtype=float),
        "R_measurement": np.array([[(r_scale * np.deg2rad(position_std_deg) ** 2)]], dtype=float),
        "P0": np.diag(
            [np.deg2rad(initial_position_std_deg) ** 2, initial_velocity_std**2]
        ),
    }
