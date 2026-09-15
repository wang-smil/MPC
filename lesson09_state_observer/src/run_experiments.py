"""Run, visualize, and report the Lesson 09 observer experiments."""

from copy import deepcopy
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .closed_loop import simulate_observer_closed_loop
from .metrics import calculate_metrics
from .model_loader import build_balanced_lqr_gain, build_identified_model, load_config
from .observability import observability_report
from .observer import DiscreteObserver
from .observer_design import design_discrete_observer


def _make_observer(
    config: dict,
    speed_factor: float,
) -> tuple[DiscreteObserver, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Build a fresh observer and its matching balanced LQR design."""

    _, _, Ad, Bd = build_identified_model(config)
    K = build_balanced_lqr_gain(Ad, Bd, config)
    C = np.array([[1.0, 0.0]])
    design = design_discrete_observer(
        Ad=Ad,
        C=C,
        controller_poles_z=np.linalg.eigvals(Ad - Bd @ K),
        speed_factor=speed_factor,
        dt_s=float(config["simulation"]["dt_s"]),
    )
    initial = config["initial_condition"]
    x0_hat = np.array(
        [np.deg2rad(initial["q_hat_deg"]), initial["dq_hat_rad_s"]],
        dtype=float,
    )
    observer = DiscreteObserver(Ad, Bd, C, design["L"], x0_hat)
    return observer, K, design, Ad


def _write_csv(path: Path, log: dict[str, np.ndarray]) -> None:
    """Write an aligned simulation log with one row per sample."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(log)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index in range(len(log[fields[0]])):
            writer.writerow({field: log[field][index] for field in fields})


def _run_case(
    config: dict,
    output_root: Path,
    label: str,
    speed_factor: float,
    plant_inertia: float | None = None,
    noise_scale: float = 1.0,
    encoder_bias_rad: float = 0.0,
    velocity_source: str = "observer",
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray], np.ndarray]:
    """Execute one independently initialized scenario and persist its log."""

    observer, K, design, Ad = _make_observer(config, speed_factor)
    inertia = float(config["model"]["inertia"]) if plant_inertia is None else plant_inertia
    log = simulate_observer_closed_loop(
        config=config,
        observer=observer,
        K=K,
        plant_inertia=inertia,
        noise_scale=noise_scale,
        encoder_bias_rad=encoder_bias_rad,
        velocity_source=velocity_source,
    )
    _write_csv(output_root / "logs" / f"{label}.csv", log)
    return log, calculate_metrics(log), design, Ad


def _plot_convergence(log: dict[str, np.ndarray], path: Path) -> None:
    """Plot the four observer signals required for experiment A."""

    time_s = log["time_s"]
    figure, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    axes[0].plot(time_s, np.rad2deg(log["q_true_rad"]), label="q true")
    axes[0].plot(time_s, np.rad2deg(log["q_hat_rad"]), label="q hat", linestyle="--")
    axes[0].set_ylabel("Position / deg")
    axes[1].plot(time_s, log["dq_true_rad_s"], label="dq true")
    axes[1].plot(time_s, log["dq_hat_rad_s"], label="dq hat", linestyle="--")
    axes[1].set_ylabel("Velocity / rad/s")
    axes[2].plot(time_s, np.rad2deg(log["q_estimation_error_rad"]), label="q error")
    axes[2].plot(time_s, log["dq_estimation_error_rad_s"], label="dq error")
    axes[2].set_ylabel("Estimation error")
    axes[3].plot(time_s, np.rad2deg(log["innovation_rad"]), label="innovation")
    axes[3].set_ylabel("Innovation / deg")
    axes[3].set_xlabel("Time / s")
    for axis in axes:
        axis.grid(True)
        axis.legend()
    figure.suptitle("Observer Convergence from Mismatched Initial State")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_speed_tradeoff(
    logs: dict[str, dict[str, np.ndarray]],
    metrics: dict[str, dict[str, float]],
    path: Path,
) -> None:
    """Show the speed-versus-noise/control trade-off from experiment B."""

    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=False)
    for label, log in logs.items():
        axes[0].plot(log["time_s"], log["dq_hat_rad_s"], label=label)
    axes[0].set_xlabel("Time / s")
    axes[0].set_ylabel("Estimated velocity / rad/s")
    axes[0].grid(True)
    axes[0].legend()

    labels = list(metrics)
    x = np.arange(len(labels))
    convergence = [metrics[label]["observer_convergence_time_s"] for label in labels]
    torque = [metrics[label]["rms_torque_nm"] for label in labels]
    axes[1].bar(x - 0.2, convergence, width=0.4, label="convergence / s")
    axes[1].bar(x + 0.2, torque, width=0.4, label="torque RMS / Nm")
    axes[1].set_xticks(x, labels)
    axes[1].grid(True, axis="y")
    axes[1].legend()
    figure.suptitle("Observer Speed Trade-off")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_velocity_sources(logs: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    """Compare velocity estimates and torque for experiment C."""

    columns = {
        "raw_difference": "dq_raw_rad_s",
        "filtered_difference": "dq_filtered_rad_s",
        "observer": "dq_hat_rad_s",
    }
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for label, log in logs.items():
        axes[0].plot(log["time_s"], log[columns[label]], label=label)
        axes[1].plot(log["time_s"], log["torque_cmd_nm"], label=label)
    axes[0].plot(
        logs["observer"]["time_s"],
        logs["observer"]["dq_true_rad_s"],
        label="true velocity",
        color="black",
        linewidth=1.2,
    )
    axes[0].set_ylabel("Velocity / rad/s")
    axes[1].set_ylabel("Applied torque / Nm")
    axes[1].set_xlabel("Time / s")
    for axis in axes:
        axis.grid(True)
        axis.legend()
    figure.suptitle("Finite Difference versus Observer")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_robustness(
    logs: dict[str, dict[str, np.ndarray]],
    path: Path,
) -> None:
    """Compare mismatch, high-noise, and encoder-bias diagnostics."""

    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for label, log in logs.items():
        axes[0].plot(
            log["time_s"],
            np.rad2deg(log["q_estimation_error_rad"]),
            label=label,
        )
        axes[1].plot(
            log["time_s"],
            np.rad2deg(log["innovation_rad"]),
            label=label,
        )
    axes[0].set_ylabel("Position estimate error / deg")
    axes[1].set_ylabel("Innovation / deg")
    axes[1].set_xlabel("Time / s")
    for axis in axes:
        axis.grid(True)
        axis.legend()
    figure.suptitle("Observer Robustness and Limits")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _metric_lines(metrics: dict[str, float], prefix: str = "") -> list[str]:
    """Format one metrics mapping for the Markdown report."""

    return [f"- {prefix}{name}: {value:.6g}" for name, value in metrics.items()]


def _write_report(
    path: Path,
    observability: dict[str, float | int],
    nominal_design: dict[str, np.ndarray],
    results: dict[str, object],
) -> None:
    """Write the observer engineering findings and all numeric summaries."""

    speed_results = results["speed_tradeoff"]
    velocity_results = results["velocity_sources"]
    normal_results = results["normal"]
    stress_results = results["stress"]
    bias_results = results["bias"]
    lines = [
        "# Lesson 09: Observer Engineering Report",
        "",
        "## Observability",
        f"- rank: {observability['rank']} / {observability['state_dimension']}",
        f"- condition_number: {observability['condition_number']:.6g}",
        f"- sigma_min: {observability['sigma_min']:.6g}",
        "",
        "## Nominal observer poles",
        f"- requested: {np.array2string(nominal_design['requested_poles'], precision=7)}",
        f"- achieved: {np.array2string(nominal_design['achieved_poles'], precision=7)}",
        "",
        "## Findings",
        "",
        "1. Increasing observer speed reduces initial-state convergence time but can amplify encoder noise into velocity and torque.",
        "2. The observer uses the applied torque after saturation; prediction with requested torque would violate plant-input consistency.",
        "3. A constant encoder bias is not a state in this model, so innovation and estimated position can remain biased; use bias augmentation or a Kalman-style estimator later.",
        "",
        "## How to interpret these results",
        "",
        f"- The model is observable because rank is {observability['rank']}, but condition number {observability['condition_number']:.1f} and sigma_min {observability['sigma_min']:.3g} show that velocity information is numerically weak at a 1 ms interval.",
        f"- Observer 2x gives velocity RMSE {speed_results['observer_2x']['control_velocity_rmse_rad_s']:.3f} rad/s, while 10x reaches {speed_results['observer_10x']['control_velocity_rmse_rad_s']:.3f} rad/s. Faster poles improve position correction but inject more encoder noise into velocity and torque.",
        "- A late or NaN convergence time in noisy runs means the estimate did not remain inside the strict 2% error band; it does not mean that the observer poles are unstable.",
        f"- Raw difference produces {velocity_results['raw_difference']['rms_torque_nm']:.3f} Nm torque RMS and {velocity_results['raw_difference']['saturation_ratio_percent']:.2f}% saturation. The configured LPF gives the best velocity RMSE in this simple plant, so a pole-placed observer is not automatically the best noise filter.",
        f"- Ten-times encoder noise raises innovation RMS from {normal_results['innovation_rms_rad']:.4f} to {stress_results['innovation_rms_rad']:.4f} rad and torque RMS from {normal_results['rms_torque_nm']:.3f} to {stress_results['rms_torque_nm']:.3f} Nm.",
        f"- A 0.5 degree encoder offset produces {np.rad2deg(bias_results['q_est_rmse_rad']):.3f} degree position-estimate RMSE and no 2% convergence, because bias is absent from the observer state.",
        "",
        "The full-run RMSE includes the deliberately large initial mismatch q=20 deg and q_hat=0 deg. Compare convergence time and steady curves in the figures before selecting an observer speed.",
        "",
        "## Enterprise deployment guidance",
        "",
        "1. Initialize estimated position from the first validated encoder sample and start estimated velocity conservatively.",
        "2. Tune observer bandwidth offline from recorded q_measured and u_applied; increase speed only while innovation, velocity noise, and torque remain acceptable.",
        "3. Feed actuator-applied torque after current and torque limits into the observer model.",
        "4. Treat persistent innovation mean, periodic structure, and spikes as diagnostics for bias, missing dynamics, disturbances, or encoder faults.",
        "5. Use bias-state augmentation and Kalman/EKF sensor fusion when payload variation, process noise, or multiple sensors dominate.",
        "",
        "## Metrics",
    ]
    for name, value in results.items():
        lines.extend(["", f"### {name}"])
        if isinstance(value, dict) and all(
            isinstance(metric_value, (int, float)) for metric_value in value.values()
        ):
            lines.extend(_metric_lines(value))
        elif isinstance(value, dict):
            for child_name, child_metrics in value.items():
                lines.append(f"#### {child_name}")
                lines.extend(_metric_lines(child_metrics, prefix=""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(
    config_path: Path | str,
    output_root: Path | str | None = None,
) -> dict[str, object]:
    """Run experiments A-E plus normal, disturbance, and stress acceptance runs."""

    config = load_config(config_path)
    root = Path(output_root) if output_root is not None else Path(__file__).parents[1]
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    nominal_speed = float(config["observer"]["nominal_speed_factor"])
    nominal_inertia = float(config["model"]["inertia"])
    payload_inertia = float(config["stress"]["payload_scale"]) * nominal_inertia

    clean_config = deepcopy(config)
    clean_config["measurement"]["position_noise_std_deg"] = 0.0
    convergence_log, convergence_metrics, _, _ = _run_case(
        clean_config, root, "convergence", nominal_speed
    )

    speed_logs: dict[str, dict[str, np.ndarray]] = {}
    speed_metrics: dict[str, dict[str, float]] = {}
    for factor in config["observer"]["speed_factors"]:
        label = f"observer_{int(factor)}x"
        log, metrics, _, _ = _run_case(config, root, label, float(factor))
        speed_logs[label] = log
        speed_metrics[label] = metrics

    velocity_logs: dict[str, dict[str, np.ndarray]] = {}
    velocity_metrics: dict[str, dict[str, float]] = {}
    for source in ("raw_difference", "filtered_difference", "observer"):
        log, metrics, _, _ = _run_case(
            config, root, f"velocity_{source}", nominal_speed, velocity_source=source
        )
        velocity_logs[source] = log
        velocity_metrics[source] = metrics

    payload_log, payload_metrics, _, _ = _run_case(
        config, root, "payload_plus_30_percent", nominal_speed, plant_inertia=payload_inertia
    )
    bias_log, bias_metrics, _, _ = _run_case(
        config,
        root,
        "encoder_bias",
        nominal_speed,
        encoder_bias_rad=float(np.deg2rad(config["measurement"]["encoder_bias_deg"])),
    )
    normal_log, normal_metrics, nominal_design, Ad = _run_case(
        config, root, "normal", nominal_speed
    )
    disturbance_log, disturbance_metrics, _, _ = _run_case(
        config, root, "disturbance", nominal_speed, plant_inertia=payload_inertia
    )
    stress_log, stress_metrics, _, _ = _run_case(
        config,
        root,
        "stress",
        nominal_speed,
        noise_scale=float(config["stress"]["high_noise_scale"]),
    )

    _plot_convergence(convergence_log, root / "figures" / "observer_convergence.png")
    _plot_speed_tradeoff(speed_logs, speed_metrics, root / "figures" / "observer_speed_tradeoff.png")
    _plot_velocity_sources(velocity_logs, root / "figures" / "velocity_source_comparison.png")
    _plot_robustness(
        {"payload": payload_log, "bias": bias_log, "stress": stress_log},
        root / "figures" / "observer_robustness.png",
    )

    results: dict[str, object] = {
        "convergence": convergence_metrics,
        "speed_tradeoff": speed_metrics,
        "velocity_sources": velocity_metrics,
        "payload": payload_metrics,
        "bias": bias_metrics,
        "normal": normal_metrics,
        "disturbance": disturbance_metrics,
        "stress": stress_metrics,
    }
    report_observability = observability_report(Ad, np.array([[1.0, 0.0]]))
    _write_report(
        root / "reports" / "observer_engineering_report.md",
        report_observability,
        nominal_design,
        results,
    )
    return results


def main() -> None:
    """Run Lesson 09 experiments using its committed YAML configuration."""

    config_path = Path(__file__).parents[1] / "config" / "observer.yaml"
    results = run_all(config_path)
    for name, value in results.items():
        print(f"===== {name} =====")
        print(value)


if __name__ == "__main__":
    main()
