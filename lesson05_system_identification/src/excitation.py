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

class ExcitationSignal:
    """Common interface for torque signals expressed in N m."""

    def evaluate(self, time_s: float | np.ndarray) -> float | np.ndarray:
        raise NotImplementedError


def _validate_time(time_s: float | np.ndarray) -> np.ndarray:
    time = np.asarray(time_s, dtype=float)
    if np.any(time < 0.0):
        raise ValueError("time_s cannot be negative.")
    return time


def _return_value(time: np.ndarray, values: np.ndarray) -> float | np.ndarray:
    return float(values) if time.ndim == 0 else values


class ConstantTorque(ExcitationSignal):
    def __init__(self, amplitude_nm: float) -> None:
        if amplitude_nm < 0.0:
            raise ValueError("amplitude_nm cannot be negative.")
        self.amplitude_nm = float(amplitude_nm)

    def evaluate(self, time_s: float | np.ndarray) -> float | np.ndarray:
        time = _validate_time(time_s)
        values = np.full_like(time, self.amplitude_nm, dtype=float)
        return _return_value(time, values)


class SingleSine(ExcitationSignal):
    def __init__(
        self,
        amplitude_nm: float,
        frequency_hz: float,
        phase_rad: float = 0.0,
    ) -> None:
        if amplitude_nm < 0.0 or frequency_hz <= 0.0:
            raise ValueError("amplitude_nm cannot be negative and frequency_hz must be positive.")
        self.amplitude_nm = float(amplitude_nm)
        self.frequency_hz = float(frequency_hz)
        self.phase_rad = float(phase_rad)

    def evaluate(self, time_s: float | np.ndarray) -> float | np.ndarray:
        time = _validate_time(time_s)
        values = self.amplitude_nm * np.sin(
            2.0 * np.pi * self.frequency_hz * time + self.phase_rad
        )
        return _return_value(time, values)


class MultiSine(ExcitationSignal):
    def __init__(
        self,
        frequencies_hz: list[float],
        amplitudes_nm: list[float],
        phases_rad: list[float] | None = None,
    ) -> None:
        if not frequencies_hz or len(frequencies_hz) != len(amplitudes_nm):
            raise ValueError("frequencies_hz and amplitudes_nm must be non-empty and equal length.")
        if any(frequency <= 0.0 for frequency in frequencies_hz):
            raise ValueError("frequencies_hz must be positive.")
        if any(amplitude < 0.0 for amplitude in amplitudes_nm):
            raise ValueError("amplitudes_nm cannot be negative.")
        if phases_rad is not None and len(phases_rad) != len(frequencies_hz):
            raise ValueError("phases_rad must match frequencies_hz length.")
        self.frequencies_hz = np.asarray(frequencies_hz, dtype=float)
        self.amplitudes_nm = np.asarray(amplitudes_nm, dtype=float)
        self.phases_rad = (
            np.zeros_like(self.frequencies_hz)
            if phases_rad is None
            else np.asarray(phases_rad, dtype=float)
        )

    def evaluate(self, time_s: float | np.ndarray) -> float | np.ndarray:
        time = _validate_time(time_s)
        values = np.sum(
            self.amplitudes_nm
            * np.sin(
                2.0 * np.pi * self.frequencies_hz * time[..., None]
                + self.phases_rad
            ),
            axis=-1,
        )
        return _return_value(time, values)


class PRBSExcitation(ExcitationSignal):
    def __init__(self, amplitude_nm: float, hold_time_s: float, seed: int) -> None:
        if amplitude_nm <= 0.0 or hold_time_s <= 0.0:
            raise ValueError("amplitude_nm and hold_time_s must be positive.")
        self.amplitude_nm = float(amplitude_nm)
        self.hold_time_s = float(hold_time_s)
        self._rng = np.random.default_rng(seed)
        self._signs = np.empty(0, dtype=float)

    def _ensure_intervals(self, max_index: int) -> None:
        missing = max_index + 1 - len(self._signs)
        if missing > 0:
            signs = self._rng.choice(np.array([-1.0, 1.0]), size=missing)
            self._signs = np.concatenate([self._signs, signs])

    def evaluate(self, time_s: float | np.ndarray) -> float | np.ndarray:
        time = _validate_time(time_s)
        indices = np.floor(time / self.hold_time_s).astype(int)
        self._ensure_intervals(int(np.max(indices)))
        return _return_value(time, self.amplitude_nm * self._signs[indices])

