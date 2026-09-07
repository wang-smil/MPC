"""Run Lesson 06 Experiment C: linear versus friction-aware model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .quality_experiments import compare_friction_models


ROOT = Path(__file__).resolve().parents[1]


def run_experiment(
    duration_s: float = 12.0,
    dt_s: float = 0.001,
    output_root: Path | None = None,
) -> dict:
    """Compare torque residuals for the two models on one frictional record."""

    results = compare_friction_models(
        duration_s=duration_s,
        dt_s=dt_s,
        position_noise_std_deg=0.02,
        window_length=61,
    )
    root = ROOT if output_root is None else Path(output_root)
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "friction_model_comparison.png"

    residual_a = np.asarray(results["model_a"]["residual_nm"])
    residual_b = np.asarray(results["model_b"]["residual_nm"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(residual_a, label="model A: J, b", alpha=0.85)
    axes[0].plot(residual_b, label="model B: J, b, tau_c", alpha=0.85)
    axes[0].set(
        title="Torque residual after the transient",
        xlabel="fitted sample index",
        ylabel="residual / N m",
    )
    rmse = [results["model_a"]["torque_rmse"], results["model_b"]["torque_rmse"]]
    axes[1].bar(["model A", "model B"], rmse, color=["#E45756", "#4C78A8"])
    axes[1].set(title="Residual RMSE", ylabel="RMSE / N m")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.35)
    axes[0].legend()
    figure.suptitle("Experiment C: Does modelling Coulomb friction help?")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return {"results": results, "figure_path": figure_path}


def main() -> None:
    """Print model parameters and residual quality."""

    result = run_experiment()
    model_a = result["results"]["model_a"]
    model_b = result["results"]["model_b"]
    print("===== model A: J, b =====")
    print(f"J_hat: {model_a['inertia_hat']:.6f}")
    print(f"b_hat: {model_a['damping_hat']:.6f}")
    print(f"torque_rmse_nm: {model_a['torque_rmse']:.6f}")
    print("===== model B: J, b, tau_c =====")
    print(f"J_hat: {model_b['inertia_hat']:.6f}")
    print(f"b_hat: {model_b['damping_hat']:.6f}")
    print(f"tau_c_hat_nm: {model_b['coulomb_friction_hat']:.6f}")
    print(f"torque_rmse_nm: {model_b['torque_rmse']:.6f}")
    print(f"figure: {result['figure_path']}")


if __name__ == "__main__":
    main()
