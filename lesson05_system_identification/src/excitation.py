"""Torque excitation signals for system-identification experiments."""

import numpy as np


class MultiSineExcitation:
    """A deterministic sum of sine waves, expressed in N m."""

    def __init__(self, frequencies_hz: list[float], amplitudes_nm: list[float]) -> None:
        if not frequencies_hz or len(frequencies_hz) != len(amplitudes_nm):
            raise ValueError(
                "frequencies_hz and amplitudes_nm must be non-empty and equal length."
            )
        if any(frequency <= 0.0 for frequency in frequencies_hz):
            raise ValueError("frequencies_hz must be positive.")
        if any(amplitude < 0.0 for amplitude in amplitudes_nm):
            raise ValueError("amplitudes_nm cannot be negative.")

        self.frequencies_hz = np.asarray(frequencies_hz, dtype=float)
        self.amplitudes_nm = np.asarray(amplitudes_nm, dtype=float)

    def evaluate(self, time_s: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the torque command at one or more times in seconds."""

        time = np.asarray(time_s, dtype=float)
        values = np.sum(
            self.amplitudes_nm
            * np.sin(2.0 * np.pi * self.frequencies_hz * time[..., None]),
            axis=-1,
        )
        return float(values) if time.ndim == 0 else values
