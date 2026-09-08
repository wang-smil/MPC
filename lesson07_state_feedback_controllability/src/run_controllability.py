"""Run and visualize Lesson 07 controllability diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .controllability import (
    build_uncontrollable_model,
    controllability_matrix,
    controllability_report,
)
from .model import build_continuous_model, load_config


def calculate_controllability_cases(
    inertia: float,
    damping: float,
) -> dict[str, dict[str, float | int]]:
    """Return diagnostics for the controllable joint and a bad teaching model."""

    A_nominal, B_nominal = build_continuous_model(inertia, damping)
    A_bad, B_bad = build_uncontrollable_model(inertia, damping)
    return {
        "nominal": controllability_report(A_nominal, B_nominal),
        "uncontrollable": controllability_report(A_bad, B_bad),
    }


def _format_matrix(matrix: np.ndarray) -> str:
    return np.array2string(matrix, precision=3, suppress_small=True)


def main() -> None:
    lesson_dir = Path(__file__).resolve().parents[1]
    config = load_config(lesson_dir / "config" / "controller.yaml")
    inertia = config["model"]["inertia"]
    damping = config["model"]["damping"]

    A_nominal, B_nominal = build_continuous_model(inertia, damping)
    A_bad, B_bad = build_uncontrollable_model(inertia, damping)
    cases = calculate_controllability_cases(inertia, damping)

    for name, A, B in (
        ("nominal", A_nominal, B_nominal),
        ("uncontrollable", A_bad, B_bad),
    ):
        report = cases[name]
        print(f"===== {name} =====")
        print("A =")
        print(_format_matrix(A))
        print("B =")
        print(_format_matrix(B))
        print("controllability_matrix =")
        print(_format_matrix(controllability_matrix(A, B)))
        print(f"rank: {report['rank']} / {report['state_dimension']}")
        print(f"condition_number: {report['condition_number']:.3f}")
        print(f"sigma_min: {report['sigma_min']:.6f}")

    output_dir = lesson_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "controllability_comparison.png"

    labels = ["Normal joint\n(rank 2 / 2)", "Bad model\n(rank 2 / 3)"]
    sigma_min = [cases["nominal"]["sigma_min"], cases["uncontrollable"]["sigma_min"]]
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, sigma_min, color=["#4C78A8", "#E45756"])
    plt.ylabel("Smallest singular value of controllability matrix")
    plt.title("Controllability: full rank matters")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, sigma_min):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.show()
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
