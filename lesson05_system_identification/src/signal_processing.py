"""Measurement processing for the system-identification pipeline."""

import numpy as np
from scipy.signal import butter, savgol_filter, sosfiltfilt


def add_position_noise(
    position_rad: np.ndarray,
    noise_std_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return encoder-like position measurements with zero-mean Gaussian noise."""

    if noise_std_deg < 0.0:
        raise ValueError("noise_std_deg cannot be negative.")
    noise_std_rad = np.deg2rad(noise_std_deg)
    return np.asarray(position_rad, dtype=float) + rng.normal(
        0.0, noise_std_rad, size=len(position_rad)
    )


def estimate_derivatives(
    position_rad: np.ndarray, dt_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate velocity and acceleration by central numerical differentiation."""

    position = np.asarray(position_rad, dtype=float)
    if position.ndim != 1 or len(position) < 3:
        raise ValueError("position_rad must have at least three one-dimensional samples.")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive.")
    velocity = np.gradient(position, dt_s)
    acceleration = np.gradient(velocity, dt_s)
    return velocity, acceleration


def lowpass(signal: np.ndarray, dt_s: float, cutoff_hz: float) -> np.ndarray:
    """Apply a zero-phase second-order Butterworth low-pass filter."""

    values = np.asarray(signal, dtype=float)
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive.")
    nyquist_hz = 0.5 / dt_s
    if not 0.0 < cutoff_hz < nyquist_hz:
        raise ValueError("cutoff_hz must lie between zero and the Nyquist frequency.")
    if values.ndim != 1 or len(values) < 10:
        raise ValueError("signal must have at least ten one-dimensional samples.")
    sos = butter(2, cutoff_hz, btype="lowpass", fs=1.0 / dt_s, output="sos")
    return sosfiltfilt(sos, values)


def discard_before(
    time_s: np.ndarray,
    arrays: list[np.ndarray],
    discard_start_s: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Apply the same start-time mask to time and every aligned signal."""

    time = np.asarray(time_s, dtype=float)
    if discard_start_s < 0.0:
        raise ValueError("discard_start_s cannot be negative.")
    if any(len(array) != len(time) for array in arrays):
        raise ValueError("every array must have the same length as time_s.")
    mask = time >= discard_start_s
    if not np.any(mask):
        raise ValueError("discard_start_s removes all samples.")
    return time[mask], [np.asarray(array)[mask] for array in arrays]

def savgol_derivatives(
    position_rad: np.ndarray,
    dt_s: float,
    window_length: int,
    polyorder: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate derivatives with a centred, offline Savitzky-Golay filter."""
    position = np.asarray(position_rad, dtype=float)
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive.")
    if window_length % 2 == 0 or window_length <= polyorder:
        raise ValueError("window_length must be odd and greater than polyorder.")
    if position.ndim != 1 or len(position) < window_length:
        raise ValueError("position_rad must contain at least window_length samples.")

    velocity = savgol_filter(
        position,
        window_length=window_length,
        polyorder=polyorder,
        deriv=1,
        delta=dt_s,
    )
    acceleration = savgol_filter(
        position,
        window_length=window_length,
        polyorder=polyorder,
        deriv=2,
        delta=dt_s,
    )
    return velocity, acceleration

