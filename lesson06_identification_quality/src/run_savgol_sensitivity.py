"""Run Lesson 06 Experiment B: Savitzky-Golay window sensitivity."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .quality_experiments import compare_savgol_windows


ROOT = Path(__file__).resolve().parents[1]
WINDOW_LENGTHS = [11, 31, 61, 101]


def run_experiment(
    duration_s: float = 12.0,
    dt_s: float = 0.001,
    output_root: Path | None = None,
) -> dict:
    """Run a fixed noisy multisine record through four filter windows."""

    results = compare_savgol_windows(
        window_lengths=WINDOW_LENGTHS,
        duration_s=duration_s,
        dt_s=dt_s,
        position_noise_std_deg=0.02,
    )
    root = ROOT if output_root is None else Path(output_root)
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "savgol_window_sensitivity.png"

    windows = list(results)
    inertia_hats = [results[window]["inertia_hat"] for window in windows]
    damping_hats = [results[window]["damping_hat"] for window in windows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(windows, inertia_hats, marker="o", label="J_hat")
    axes[0].axhline(0.020, color="black", linestyle="--", label="J true")
    axes[0].set(xlabel="window length / samples", ylabel="J / kg m^2", title="Inertia estimate")
    axes[1].plot(windows, damping_hats, marker="o", color="#F58518", label="b_hat")
    axes[1].axhline(0.080, color="black", linestyle="--", label="b true")
    axes[1].set(xlabel="window length / samples", ylabel="b / N m s/rad", title="Damping estimate")
    for axis in axes:
        axis.grid(True)
        axis.legend()
    figure.suptitle("Experiment B: Savitzky-Golay window sensitivity")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return {"results": results, "figure_path": figure_path}


def main() -> None:
    """Print each filter-window estimate and the generated figure location."""

    result = run_experiment()
    for window_length, estimate in result["results"].items():
        print(f"===== window {window_length} samples =====")
        print(f"window_ms: {window_length:.1f}")
        print(f"J_hat: {estimate['inertia_hat']:.6f}")
        print(f"b_hat: {estimate['damping_hat']:.6f}")
        print(f"J_error: {estimate['inertia_abs_error']:.6f}")
        print(f"b_error: {estimate['damping_abs_error']:.6f}")
    print(f"figure: {result['figure_path']}")


if __name__ == "__main__":
    main()
