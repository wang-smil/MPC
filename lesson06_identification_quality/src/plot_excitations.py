"""Visualise candidate torque-excitation signals before identification."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lesson05_system_identification.src.excitation import ExcitationSignal


def plot_excitation_comparison(
    signals: dict[str, ExcitationSignal],
    duration_s: float,
    dt_s: float,
    output_path: Path,
) -> None:
    """Plot each named signal on the same time axis and save a PNG."""
    if not signals:
        raise ValueError("signals cannot be empty.")
    if duration_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("duration_s and dt_s must be positive.")

    time_s = np.arange(0.0, duration_s + dt_s * 0.5, dt_s)
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for name, signal in signals.items():
        torque_nm = np.asarray(signal.evaluate(time_s), dtype=float)
        axis.plot(time_s, torque_nm, label=name)

    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        title="Identification excitation comparison",
        xlabel="time / s",
        ylabel="torque command / N m",
    )
    axis.grid(True)
    axis.legend(ncol=4)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
