"""Run Experiment A: compare X = F x0 + G U with recursive rollout."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lesson12_linear_mpc.src.model_loader import build_mpc_model, load_config
from lesson12_linear_mpc.src.prediction import (
    predict_states_matrix,
    rollout_states,
)


LESSON_DIR = Path(__file__).resolve().parents[1]


def run_validation(
    config_path: Path | str,
    output_dir: Path | str | None = None,
) -> dict:
    """Validate matrix prediction against recurrence and save a comparison plot."""

    config = load_config(config_path)
    model = build_mpc_model(config)
    horizon = config["mpc"]["horizon_steps"]
    seed = config["validation"]["seed"]
    rng = np.random.default_rng(seed)

    x0 = np.array([np.deg2rad(10.0), 0.0])
    U = rng.uniform(-0.4, 0.4, size=(horizon, model["Bd"].shape[1]))
    X_matrix = predict_states_matrix(model["Ad"], model["Bd"], x0, U)
    states_rollout = rollout_states(model["Ad"], model["Bd"], x0, U)
    X_rollout = states_rollout[1:]
    max_abs_error = float(np.max(np.abs(X_matrix - X_rollout)))

    figure_dir = Path(output_dir) if output_dir is not None else LESSON_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "prediction_validation.png"

    time_s = np.arange(horizon + 1) * float(config["mpc"]["dt_s"])
    matrix_states = np.vstack((x0, X_matrix))
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(time_s, states_rollout[:, 0], "o-", ms=3, label="Recursive rollout")
    axes[0].plot(time_s, matrix_states[:, 0], "x--", ms=4, label="F x0 + G U")
    axes[0].set_ylabel("Position / rad")
    axes[0].set_title("MPC prediction validation")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend()

    axes[1].plot(time_s, states_rollout[:, 1], "o-", ms=3, label="Recursive rollout")
    axes[1].plot(time_s, matrix_states[:, 1], "x--", ms=4, label="F x0 + G U")
    axes[1].set_xlabel("Prediction time / s")
    axes[1].set_ylabel("Velocity / rad/s")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend()
    fig.suptitle(f"Maximum absolute mismatch: {max_abs_error:.3e}")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print(f"Maximum absolute prediction error: {max_abs_error:.3e}")
    print(f"Figure: {figure_path}")
    return {
        "max_abs_error": max_abs_error,
        "X_matrix": X_matrix,
        "X_rollout": X_rollout,
        "figure_path": figure_path,
    }


if __name__ == "__main__":
    run_validation(LESSON_DIR / "config" / "mpc.yaml")
