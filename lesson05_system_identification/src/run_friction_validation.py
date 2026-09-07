"""Run Lesson 06 Experiment D: independent payload validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .validation import validate_friction_models


ROOT = Path(__file__).resolve().parents[1]


def run_experiment(
    duration_s: float = 12.0,
    dt_s: float = 0.001,
    output_root: Path | None = None,
) -> dict:
    """Validate both models on nominal and +30% inertia plants."""

    results = validate_friction_models(duration_s=duration_s, dt_s=dt_s)
    root = ROOT if output_root is None else Path(output_root)
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "friction_validation.png"

    scenarios = list(results)
    model_a_rmse = [results[name]["model_a_position_rmse_rad"] for name in scenarios]
    model_b_rmse = [results[name]["model_b_position_rmse_rad"] for name in scenarios]
    positions = np.arange(len(scenarios))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(positions - width / 2, model_a_rmse, width, label="model A: J, b", color="#E45756")
    axis.bar(positions + width / 2, model_b_rmse, width, label="model B: J, b, tau_c", color="#4C78A8")
    axis.set(
        title="Experiment D: Independent validation",
        xlabel="true plant scenario",
        ylabel="position RMSE / rad",
        xticks=positions,
        xticklabels=["nominal", "+30% inertia"],
    )
    axis.grid(True, axis="y", alpha=0.35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return {"results": results, "figure_path": figure_path}


def main() -> None:
    """Print independent-validation errors and the figure path."""

    result = run_experiment()
    for scenario, metrics in result["results"].items():
        print(f"===== {scenario} =====")
        print(f"model_a_position_rmse_rad: {metrics['model_a_position_rmse_rad']:.6f}")
        print(f"model_b_position_rmse_rad: {metrics['model_b_position_rmse_rad']:.6f}")
    print(f"figure: {result['figure_path']}")


if __name__ == "__main__":
    main()
