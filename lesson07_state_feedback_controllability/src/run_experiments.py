"""Generate the A-D Lesson 07 state-feedback experiments and report."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .controllability import build_uncontrollable_model, controllability_report
from .metrics import calculate_metrics
from .model import build_continuous_model, discretize_zoh, load_config
from .pole_design import design_discrete_feedback, second_order_poles
from .simulator import simulate_closed_loop


def _design_controller(config: dict, settling_time_s: float | None = None) -> dict[str, np.ndarray]:
    """Design a discrete controller from the identified nominal model."""

    inertia = float(config["model"]["inertia"])
    damping = float(config["model"]["damping"])
    dt_s = float(config["simulation"]["dt_s"])
    A, B = build_continuous_model(inertia, damping)
    Ad, Bd = discretize_zoh(A, B, dt_s)
    poles = second_order_poles(
        settling_time_s or float(config["controller"]["settling_time_s"]),
        float(config["controller"]["damping_ratio"]),
    )
    return design_discrete_feedback(Ad, Bd, poles, dt_s)


def _with_pole_columns(log: dict[str, np.ndarray], design: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Attach requested and achieved closed-loop poles to each CSV row."""

    augmented = dict(log)
    count = log["time_s"].size
    for prefix, poles in (("requested", design["requested_poles"]), ("computed", design["computed_poles"])):
        for index, pole in enumerate(poles, start=1):
            augmented[f"{prefix}_pole_{index}_real"] = np.full(count, pole.real)
            augmented[f"{prefix}_pole_{index}_imag"] = np.full(count, pole.imag)
    return augmented


def _write_log(path: Path, log: dict[str, np.ndarray]) -> None:
    """Write aligned simulation arrays as a transparent CSV record."""

    names = list(log)
    matrix = np.column_stack([np.asarray(log[name], dtype=float) for name in names])
    np.savetxt(path, matrix, delimiter=",", header=",".join(names), comments="")


def _plot_pole_placement(path: Path, design: dict[str, np.ndarray]) -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 400)
    plt.figure(figsize=(6, 6))
    plt.plot(np.cos(angle), np.sin(angle), "k--", alpha=0.55, label="Unit circle")
    plt.scatter(design["requested_poles"].real, design["requested_poles"].imag, marker="x", s=90, label="Requested")
    plt.scatter(design["computed_poles"].real, design["computed_poles"].imag, marker="o", facecolors="none", edgecolors="#E45756", s=90, label="Achieved")
    plt.axhline(0.0, color="gray", linewidth=0.8)
    plt.axvline(0.0, color="gray", linewidth=0.8)
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.title("A. Requested and achieved discrete closed-loop poles")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_speed_tradeoff(path: Path, cases: dict[str, dict]) -> None:
    plt.figure(figsize=(8.5, 5))
    for name, result in cases.items():
        log = result["log"]
        plt.plot(log["time_s"], np.rad2deg(log["q_rad"]), label=name.replace("_", " "))
    reference = next(iter(cases.values()))["log"]
    plt.plot(reference["time_s"], np.rad2deg(reference["q_ref_rad"]), "k--", label="reference")
    plt.xlabel("Time / s")
    plt.ylabel("Position / deg")
    plt.title("B. Faster target poles trade torque demand for response speed")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_uncontrollable_case(path: Path, duration_s: float) -> None:
    time_s = np.linspace(0.0, duration_s, 400)
    z = np.exp(0.5 * time_s)
    plt.figure(figsize=(8, 4.5))
    plt.plot(time_s, z, color="#E45756", label=r"unactuated $z(t)=e^{0.5t}$")
    plt.xlabel("Time / s")
    plt.ylabel("Uncontrolled state z")
    plt.title("C. An unstable uncontrollable mode cannot be stabilized by torque")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_payload_robustness(path: Path, nominal: dict, payload: dict) -> None:
    plt.figure(figsize=(8.5, 5))
    for name, result, color in (("Nominal inertia", nominal, "#4C78A8"), ("Payload +30% inertia", payload, "#F58518")):
        log = result["log"]
        plt.plot(log["time_s"], np.rad2deg(log["q_rad"]), color=color, label=name)
    log = nominal["log"]
    plt.plot(log["time_s"], np.rad2deg(log["q_ref_rad"]), "k--", label="reference")
    plt.xlabel("Time / s")
    plt.ylabel("Position / deg")
    plt.title("D. Fixed nominal K under inertia mismatch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _metric_line(metrics: dict[str, float]) -> str:
    return (
        f"final error={np.rad2deg(metrics['final_error_rad']):.3f} deg, "
        f"settling={metrics['settling_time_s']:.3f} s, "
        f"overshoot={metrics['overshoot_percent']:.2f}%, "
        f"RMS torque={metrics['rms_torque_nm']:.3f} N m, "
        f"saturation={metrics['saturation_ratio_percent']:.2f}%"
    )


def _write_report(
    path: Path,
    controllability: dict[str, float | int],
    bad_controllability: dict[str, float | int],
    nominal_design: dict[str, np.ndarray],
    cases: dict[str, dict],
) -> None:
    K = nominal_design["K"].ravel()
    requested = nominal_design["requested_poles"]
    achieved = nominal_design["computed_poles"]
    lines = [
        "# Lesson 07: State Feedback and Controllability Report",
        "",
        "## Design model and controllability",
        "",
        "The design model uses the Lesson 05 identified values `J_hat=0.019762 kg m^2` and `b_hat=0.080257 N m s/rad`.",
        f"The nominal controllability rank is {controllability['rank']} / {controllability['state_dimension']}; its smallest singular value is {controllability['sigma_min']:.6f}.",
        f"The teaching failure model has rank {bad_controllability['rank']} / {bad_controllability['state_dimension']}, so its unstable z mode cannot be placed by torque.",
        "",
        "## Pole placement",
        "",
        f"Nominal state-feedback gain: `K = [{K[0]:.6f}, {K[1]:.6f}]`.",
        f"Requested discrete poles: `{requested[0]:.6f}`, `{requested[1]:.6f}`.",
        f"Achieved discrete poles: `{achieved[0]:.6f}`, `{achieved[1]:.6f}`.",
        "",
        "## Experiment results",
        "",
    ]
    for name, result in cases.items():
        lines.append(f"- **{name}**: {_metric_line(result['metrics'])}")
    lines.extend(
        [
            "",
            "## Engineering interpretation",
            "",
            "The controller operates on noisy position and a causal filtered velocity estimate, not the plant's hidden true states. CSV logs preserve `torque_unsat_nm` and `torque_applied_nm` separately, so actuator saturation is visible rather than hidden. The payload case deliberately keeps the nominal K unchanged: degraded performance is therefore a robustness result, not a redesigned-controller result.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_all(config_path: Path, output_root: Path | None = None) -> dict:
    """Run A-D experiments and write figures, CSV logs, and a Markdown report."""

    config = load_config(config_path)
    base_dir = Path(output_root) if output_root is not None else config_path.parent.parent
    figure_dir = base_dir / "figures"
    log_dir = base_dir / "logs"
    report_dir = base_dir / "reports"
    for directory in (figure_dir, log_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    nominal_design = _design_controller(config)
    inertia = float(config["model"]["inertia"])
    damping = float(config["model"]["damping"])
    A, B = build_continuous_model(inertia, damping)
    controllability = controllability_report(A, B)
    A_bad, B_bad = build_uncontrollable_model(inertia, damping)
    bad_controllability = controllability_report(A_bad, B_bad)

    speed_targets = {"slow_1p5s": 1.5, "nominal_0p8s": 0.8, "aggressive_0p3s": 0.3}
    cases: dict[str, dict] = {}
    for name, settling_time_s in speed_targets.items():
        design = _design_controller(config, settling_time_s)
        log = simulate_closed_loop(config, design["K"], plant_inertia=inertia)
        cases[name] = {"design": design, "log": _with_pole_columns(log, design), "metrics": calculate_metrics(log)}

    payload_log = simulate_closed_loop(
        config,
        nominal_design["K"],
        plant_inertia=inertia * float(config["stress"]["payload_inertia_scale"]),
    )
    cases["payload_plus_30_percent"] = {
        "design": nominal_design,
        "log": _with_pole_columns(payload_log, nominal_design),
        "metrics": calculate_metrics(payload_log),
    }

    log_paths = {}
    for name, result in cases.items():
        path = log_dir / f"{name}.csv"
        _write_log(path, result["log"])
        log_paths[name] = path

    figure_paths = {
        "pole_placement": figure_dir / "pole_placement.png",
        "speed_tradeoff": figure_dir / "speed_tradeoff.png",
        "uncontrollable_case": figure_dir / "uncontrollable_case.png",
        "payload_robustness": figure_dir / "payload_robustness.png",
    }
    _plot_pole_placement(figure_paths["pole_placement"], nominal_design)
    _plot_speed_tradeoff(figure_paths["speed_tradeoff"], {name: cases[name] for name in speed_targets})
    _plot_uncontrollable_case(figure_paths["uncontrollable_case"], float(config["simulation"]["duration_s"]))
    _plot_payload_robustness(
        figure_paths["payload_robustness"], cases["nominal_0p8s"], cases["payload_plus_30_percent"]
    )

    report_path = report_dir / "state_feedback_report.md"
    _write_report(report_path, controllability, bad_controllability, nominal_design, cases)
    return {
        "figures": figure_paths,
        "logs": log_paths,
        "report_path": report_path,
        "controllability": controllability,
        "bad_controllability": bad_controllability,
        "cases": cases,
    }


def main() -> None:
    lesson_dir = Path(__file__).resolve().parents[1]
    result = run_all(lesson_dir / "config" / "controller.yaml")
    for name, case in result["cases"].items():
        print(f"===== {name} =====")
        for metric, value in case["metrics"].items():
            print(f"{metric}: {value}")
        print(f"log: {result['logs'][name]}")
    for name, path in result["figures"].items():
        print(f"{name}: {path}")
    print(f"report: {result['report_path']}")


if __name__ == "__main__":
    main()
