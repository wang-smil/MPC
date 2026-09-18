"""Run all Lesson 10 Kalman-filter experiments and write teaching artifacts."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .closed_loop import simulate_closed_loop
from .metrics import calculate_metrics
from .model_loader import load_config


def _save_log(log: dict[str, np.ndarray], path: Path) -> None:
    """Write one simulation log as a reviewable CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(log)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(len(log[fields[0]])):
            row: dict[str, float | bool] = {}
            for field in fields:
                value = log[field][index]
                row[field] = bool(value) if isinstance(value, np.bool_) else float(value)
            writer.writerow(row)


def _step_load(config: dict):
    """Build the course's unknown load-torque step at the configured time."""

    start_s = float(config["disturbance"]["step_time_s"])
    magnitude_nm = float(config["disturbance"]["step_torque_nm"])
    return lambda time_s: magnitude_nm if time_s >= start_s else 0.0


def _plot_baseline(log: dict[str, np.ndarray], path: Path) -> None:
    time = log["time_s"]
    q_hat = np.rad2deg(log["q_hat_rad"])
    dq_hat = log["dq_hat_rad_s"]
    q_sigma = 3.0 * np.rad2deg(np.sqrt(log["P_qq"]))
    dq_sigma = 3.0 * np.sqrt(log["P_dqdq"])
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(time, np.rad2deg(log["q_true_rad"]), label="true position")
    axes[0].plot(time, np.rad2deg(log["q_measured_rad"]), alpha=0.35, label="encoder")
    axes[0].plot(time, q_hat, label="KF position")
    axes[0].fill_between(time, q_hat - q_sigma, q_hat + q_sigma, alpha=0.2, label="KF ±3σ")
    axes[0].set_ylabel("Position / deg")
    axes[0].legend(loc="best")
    axes[0].grid(True)
    axes[1].plot(time, log["dq_true_rad_s"], label="true velocity")
    axes[1].plot(time, dq_hat, label="KF velocity")
    axes[1].fill_between(time, dq_hat - dq_sigma, dq_hat + dq_sigma, alpha=0.2, label="KF ±3σ")
    axes[1].set_xlabel("Time / s")
    axes[1].set_ylabel("Velocity / rad/s")
    axes[1].legend(loc="best")
    axes[1].grid(True)
    figure.suptitle("Kalman Baseline: Estimate and Confidence")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_r_q_tuning(
    r_metrics: dict[float, dict[str, float]],
    q_metrics: dict[float, dict[str, float]],
    path: Path,
) -> None:
    scales = [0.1, 1.0, 10.0]
    labels = ["0.1×", "1×", "10×"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].bar(labels, [r_metrics[s]["q_rmse_rad"] for s in scales])
    axes[0, 0].set_title("Assumed R: position RMSE")
    axes[0, 0].set_ylabel("rad")
    axes[0, 1].bar(labels, [r_metrics[s]["mean_kalman_gain_q"] for s in scales])
    axes[0, 1].set_title("Assumed R: mean position gain")
    axes[1, 0].bar(labels, [q_metrics[s]["q_rmse_rad"] for s in scales])
    axes[1, 0].set_title("Assumed Q with load step: position RMSE")
    axes[1, 0].set_ylabel("rad")
    axes[1, 1].bar(labels, [q_metrics[s]["mean_kalman_gain_q"] for s in scales])
    axes[1, 1].set_title("Assumed Q: mean position gain")
    for axis in axes.ravel():
        axis.grid(True, axis="y")
    figure.suptitle("Kalman Q/R Tuning Trade-offs")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_estimator_comparison(
    metrics: dict[str, dict[str, float]], path: Path
) -> None:
    names = ["raw", "lpf", "luenberger", "kalman"]
    labels = ["Raw diff", "LPF diff", "Luenberger", "Kalman"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for axis, key, title in (
        (axes[0, 0], "q_rmse_rad", "Position RMSE"),
        (axes[0, 1], "dq_rmse_rad_s", "Velocity RMSE"),
        (axes[1, 0], "control_rms_nm", "Control RMS"),
        (axes[1, 1], "tracking_rmse_rad", "Tracking RMSE"),
    ):
        axis.bar(labels, [metrics[name][key] for name in names])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=18)
        axis.grid(True, axis="y")
    figure.suptitle("Same Plant, Noise, LQR, Saturation, and Load")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_robustness(logs: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for name, log in logs.items():
        mask = log["time_s"] >= 0.1
        axes[0].plot(
            log["time_s"][mask], np.rad2deg(log["innovation_rad"][mask]), label=name
        )
        axes[1].plot(log["time_s"][mask], log["nis"][mask], label=name)
    axes[0].set_ylabel("Innovation / deg")
    axes[0].set_title("Startup excluded after 0.1 s to reveal steady behavior")
    axes[1].set_ylabel("NIS")
    axes[1].set_xlabel("Time / s")
    for axis in axes:
        axis.grid(True)
        axis.legend(loc="best")
    figure.suptitle("Innovation and NIS: Model/Measurement Health")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _format_metrics_table(metrics: dict[str, dict[str, float]]) -> str:
    rows = [
        "| Scenario | q RMSE (rad) | dq RMSE (rad/s) | innovation RMS (rad) | NIS mean | control RMS (N·m) | tracking RMSE (rad) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in metrics.items():
        rows.append(
            "| {name} | {q:.6f} | {dq:.6f} | {innovation:.6f} | {nis:.3f} | {control:.6f} | {tracking:.6f} |".format(
                name=name,
                q=values["q_rmse_rad"],
                dq=values["dq_rmse_rad_s"],
                innovation=values["innovation_rms_rad"],
                nis=values["nis_mean"],
                control=values["control_rms_nm"],
                tracking=values["tracking_rmse_rad"],
            )
        )
    return "\n".join(rows)


def _write_report(
    path: Path,
    formal_metrics: dict[str, dict[str, float]],
    r_metrics: dict[float, dict[str, float]],
    q_metrics: dict[float, dict[str, float]],
    comparison_metrics: dict[str, dict[str, float]],
) -> None:
    """Write an evidence-linked engineering interpretation, not only a table."""

    report = f"""# Lesson 10 Kalman Filter Engineering Report

## Formal Acceptance

{_format_metrics_table(formal_metrics)}

## R: How Much Does the Filter Trust the Encoder?

With real encoder noise held fixed, assumed `R=0.1R`, `R`, and `10R` produced position gains of {r_metrics[0.1]["mean_kalman_gain_q"]:.4f}, {r_metrics[1.0]["mean_kalman_gain_q"]:.4f}, and {r_metrics[10.0]["mean_kalman_gain_q"]:.4f}.  Smaller R makes the estimate correct toward each encoder sample more aggressively; it can reduce lag but also carries more measurement noise into the estimated state and LQR torque.

## Q: How Much Does the Filter Trust the Model?

The 2 s unknown load step was applied with the same physical noise in each Q case.  The `0.1Q`, `Q`, and `10Q` position RMSE values are {q_metrics[0.1]["q_rmse_rad"]:.6f}, {q_metrics[1.0]["q_rmse_rad"]:.6f}, and {q_metrics[10.0]["q_rmse_rad"]:.6f} rad.  Q is not a sensor-noise setting: it represents unmodelled torque, friction, payload, calibration, and discretization uncertainty.

## Estimator Comparison

{_format_metrics_table(comparison_metrics)}

The four methods shared the same model, initial state, deterministic encoder noise, LQR, torque limit, and load step.  Kalman filtering must not be declared universally best from a single table: its result depends on Q/R calibration, model quality, and which operational risk—noise, lag, control effort, or fault diagnosis—matters most.

## Industrial Reading

`innovation = y - C x_hat_prior` is the raw encoder/model disagreement.  `NIS = innovation² / S` normalizes it by the filter's declared uncertainty.  Persistent large NIS can indicate underestimated Q/R, payload mismatch, encoder fault, or model bias; persistent innovation bias particularly suggests offset or model bias.  On hardware, estimate R first from a static encoder recording, then tune Q against dynamic residuals, NIS, torque saturation, and an independent validation motion.  The filter must use the applied torque, never the torque request before saturation.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def run_all(config_path: Path | str, output_root: Path | str | None = None) -> dict:
    """Run every course experiment and return metrics and artifact paths."""

    config = load_config(config_path)
    root = Path(output_root) if output_root is not None else Path(config_path).resolve().parent.parent
    figure_dir, log_dir, report_dir = root / "figures", root / "logs", root / "reports"
    for directory in (figure_dir, log_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    load_step = _step_load(config)
    baseline = simulate_closed_loop(config, "kalman")
    r_logs = {
        scale: simulate_closed_loop(config, "kalman", assumed_r_scale=scale)
        for scale in (0.1, 1.0, 10.0)
    }
    q_logs = {
        scale: simulate_closed_loop(config, "kalman", assumed_q_scale=scale, load_torque=load_step)
        for scale in (0.1, 1.0, 10.0)
    }
    comparison_logs = {
        name: simulate_closed_loop(config, name, load_torque=load_step)
        for name in ("raw", "lpf", "luenberger", "kalman")
    }
    formal_logs = {
        "normal": simulate_closed_loop(config, "kalman"),
        "disturbance": simulate_closed_loop(config, "kalman", load_torque=load_step),
        "stress": simulate_closed_loop(
            config, "kalman", measurement_noise_scale=float(config["stress"]["high_noise_scale"])
        ),
        "payload_plus_30_percent": simulate_closed_loop(
            config, "kalman", payload_scale=float(config["stress"]["payload_scale"])
        ),
    }
    for name, log in {"baseline": baseline, **formal_logs}.items():
        _save_log(log, log_dir / f"{name}.csv")
    for scale, log in r_logs.items():
        _save_log(log, log_dir / f"r_assumed_{str(scale).replace('.', 'p')}.csv")
    for scale, log in q_logs.items():
        _save_log(log, log_dir / f"q_assumed_{str(scale).replace('.', 'p')}.csv")
    for name, log in comparison_logs.items():
        _save_log(log, log_dir / f"comparison_{name}.csv")

    r_metrics = {scale: calculate_metrics(log) for scale, log in r_logs.items()}
    q_metrics = {scale: calculate_metrics(log) for scale, log in q_logs.items()}
    comparison_metrics = {name: calculate_metrics(log) for name, log in comparison_logs.items()}
    formal_metrics = {name: calculate_metrics(log) for name, log in formal_logs.items()}

    _plot_baseline(baseline, figure_dir / "baseline_confidence.png")
    _plot_r_q_tuning(r_metrics, q_metrics, figure_dir / "r_q_tuning.png")
    _plot_estimator_comparison(comparison_metrics, figure_dir / "estimator_comparison.png")
    _plot_robustness(formal_logs, figure_dir / "robustness.png")
    report_path = report_dir / "kalman_engineering_report.md"
    _write_report(report_path, formal_metrics, r_metrics, q_metrics, comparison_metrics)
    return {
        "metrics": formal_metrics,
        "r_metrics": r_metrics,
        "q_metrics": q_metrics,
        "comparison_metrics": comparison_metrics,
        "report": report_path,
        "figures": {name: figure_dir / name for name in (
            "baseline_confidence.png", "r_q_tuning.png", "estimator_comparison.png", "robustness.png"
        )},
    }


def main() -> None:
    """Run the checked-in configuration from the repository root."""

    config_path = Path(__file__).resolve().parents[1] / "config" / "kalman.yaml"
    result = run_all(config_path)
    for name, values in result["metrics"].items():
        print(f"===== {name} =====")
        for metric, value in values.items():
            print(f"{metric}: {value}")
    print(f"report: {result['report']}")


if __name__ == "__main__":
    main()
