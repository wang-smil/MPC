"""Run Lesson 06 Experiment A: excitation quality comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .quality_experiments import compare_excitation_quality, compare_noisy_identification


ROOT = Path(__file__).resolve().parents[1]


def run_experiment(
    duration_s: float = 12.0,
    dt_s: float = 0.001,
    output_root: Path | None = None,
) -> dict:
    """Run four excitation cases and save their regressor-conditioning comparison."""

    noisy_estimates = compare_noisy_identification(duration_s, dt_s, position_noise_std_deg=0.02)
    diagnostics = compare_excitation_quality(duration_s=duration_s, dt_s=dt_s)
    root = ROOT if output_root is None else Path(output_root)
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "excitation_quality.png"

    names = list(diagnostics)
    condition_numbers = [float(diagnostics[name]["condition_number"]) for name in names]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(names, condition_numbers, color="#4C78A8")
    axis.set_yscale("log")
    axis.set(
        title="Experiment A: Regressor conditioning by excitation",
        xlabel="torque excitation",
        ylabel="condition number of Phi (log scale)",
    )
    axis.grid(True, axis="y", which="both", alpha=0.35)
    for bar, value in zip(bars, condition_numbers):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    parameter_error_figure_path = figures_dir / "excitation_parameter_errors.png"
    inertia_errors = [float(noisy_estimates[name]["inertia_abs_error"]) for name in names]
    damping_errors = [float(noisy_estimates[name]["damping_abs_error"]) for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(names, inertia_errors, color="#4C78A8")
    axes[0].set(title="Inertia error", ylabel="absolute error / kg m^2")
    axes[1].bar(names, damping_errors, color="#F58518")
    axes[1].set(title="Damping error", ylabel="absolute error / N m s/rad")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(True, axis="y", alpha=0.35)
    figure.suptitle("Experiment A: Identification error with encoder noise")
    figure.tight_layout()
    figure.savefig(parameter_error_figure_path, dpi=180)
    plt.close(figure)
    return {
        "diagnostics": diagnostics,
        "figure_path": figure_path,
        "noisy_estimates": noisy_estimates,
        "parameter_error_figure_path": parameter_error_figure_path,
    }


def main() -> None:
    """Print the Experiment A diagnostics and saved figure location."""

    result = run_experiment()
    for name, diagnostic in result["diagnostics"].items():
        print(f"===== {name} =====")
        print(f"rank: {diagnostic['rank']}")
        print(f"condition_number: {diagnostic['condition_number']:.3f}")
        print(f"sigma_min: {diagnostic['sigma_min']:.6f}")
    print("===== noisy identification =====")
    for name, estimate in result["noisy_estimates"].items():
        print(f"{name}: J_hat={estimate['inertia_hat']:.6f}, b_hat={estimate['damping_hat']:.6f}")
        print(f"  J_error={estimate['inertia_abs_error']:.6f}, b_error={estimate['damping_abs_error']:.6f}")
        print(f"  Phi_measured: rank={estimate['rank']}, cond={estimate['condition_number']:.3f}, "
              f"sigma_min={estimate['sigma_min']:.6f}")
    print(f"conditioning_figure: {result['figure_path']}")
    print(f"parameter_error_figure: {result['parameter_error_figure_path']}")


if __name__ == "__main__":
    main()
