"""Generate the five reproducible Lesson 11 LQG experiments and report."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .closed_loop import simulate_lqg
from .metrics import calculate_metrics
from .model_loader import build_lqg_design, load_config
from .separation_analysis import analyze_separation_principle


def _step_load(config: dict):
    step_time = float(config["disturbance"]["step_time_s"])
    step_torque = float(config["disturbance"]["step_torque_nm"])
    return lambda time_s: step_torque if time_s >= step_time else 0.0


def _write_log(path: Path, log: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(log)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(zip(*(log[field] for field in fields)))


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_separation(result: dict, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 6.0))
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    axis.plot(np.cos(theta), np.sin(theta), "k--", linewidth=1.0, label="unit circle")
    for poles, label, marker in (
        (result["eig_controller"], "LQR: $A_d-B_dK_c$", "o"),
        (result["eig_estimator"], "KF predictor: $A_d-L_eC$", "s"),
        (result["eig_augmented"], "augmented LQG", "x"),
    ):
        axis.scatter(np.real(poles), np.imag(poles), label=label, marker=marker, s=70)
    axis.axhline(0.0, color="0.7", linewidth=0.8)
    axis.axvline(0.0, color="0.7", linewidth=0.8)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Real")
    axis.set_ylabel("Imaginary")
    axis.set_title("Experiment A: Separation-Principle Poles (Unsaturated Theory)")
    axis.grid(True)
    axis.legend()
    _save_figure(figure, path)


def _plot_full_state_vs_lqg(full: dict, lqg: dict, path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    for label, log, color in (("full-state LQR (simulation-only)", full, "tab:blue"), ("recursive LQG", lqg, "tab:orange")):
        axes[0].plot(log["time_s"], np.rad2deg(log["q_true_rad"]), label=label, color=color)
        axes[1].plot(log["time_s"], log["torque_applied_nm"], label=label, color=color)
    axes[0].axhline(np.rad2deg(full["q_ref_rad"][0]), color="k", linestyle="--", label="reference")
    axes[0].set_ylabel("Position / deg")
    axes[1].set_ylabel("Applied torque / N m")
    axes[1].set_xlabel("Time / s")
    axes[0].set_title("Experiment B: Ideal Full-State LQR vs Implementable LQG")
    for axis in axes:
        axis.grid(True)
        axis.legend()
    _save_figure(figure, path)


def _plot_recursive_vs_steady(recursive: dict, steady: dict, path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    for label, log, color in (("recursive KF", recursive, "tab:blue"), ("steady-state KF", steady, "tab:green")):
        axes[0].plot(log["time_s"], np.rad2deg(log["q_true_rad"] - log["q_hat_rad"]), label=label, color=color)
        axes[1].plot(log["time_s"], log["dq_true_rad_s"] - log["dq_hat_rad_s"], label=label, color=color)
    for axis in axes:
        axis.axvspan(0.0, 0.5, color="0.9", label="startup window")
        axis.grid(True)
        axis.legend()
    axes[0].set_ylabel("Position estimate error / deg")
    axes[1].set_ylabel("Velocity estimate error / rad s$^{-1}$")
    axes[1].set_xlabel("Time / s")
    axes[0].set_title("Experiment C: Recursive vs Steady-State LQG")
    _save_figure(figure, path)


def _plot_q_tuning(q_logs: dict[float, dict], path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    for scale, log in q_logs.items():
        label = f"assumed Q = {scale:g}x"
        axes[0].plot(log["time_s"], np.rad2deg(log["q_true_rad"] - log["q_hat_rad"]), label=label)
        axes[1].plot(log["time_s"], log["nis"], label=label)
    axes[0].set_ylabel("Position estimate error / deg")
    axes[1].set_ylabel("NIS")
    axes[1].set_xlabel("Time / s")
    axes[0].set_title("Experiment D: Fixed LQR, Varying KF Process Covariance")
    for axis in axes:
        axis.grid(True)
        axis.legend()
    _save_figure(figure, path)


def _plot_saturation_boundary(boundary: dict[str, dict], path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    for name, log in boundary.items():
        axes[0].plot(log["time_s"], np.rad2deg(log["q_true_rad"] - log["q_ref_rad"]), label=name)
        axes[1].plot(log["time_s"], log["torque_applied_nm"], label=name)
    axes[0].set_ylabel("Tracking error / deg")
    axes[1].set_ylabel("Applied torque / N m")
    axes[1].set_xlabel("Time / s")
    axes[0].set_title("Experiment E: Saturation Boundary (Nonlinear Regime)")
    for axis in axes:
        axis.grid(True)
        axis.legend()
    _save_figure(figure, path)


def _format_metrics(metrics: dict[str, float], keys: tuple[str, ...]) -> str:
    return " | ".join(f"{key}={metrics[key]:.6g}" for key in keys)


def _write_report(path: Path, separation: dict, metrics: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    window_rows = []
    for name, values in metrics.items():
        rows.append(
            f"| {name} | {values['tracking_rmse_rad']:.6g} | {values['q_estimation_rmse_rad']:.6g} | "
            f"{values['dq_estimation_rmse_rad_s']:.6g} | {values['control_rms_nm']:.6g} | "
            f"{values['saturation_ratio_percent']:.4g} | {values['nis_mean']:.6g} |"
        )
        if name in {"recursive", "steady_state"}:
            window_rows.extend(
                [
                    f"| {name} startup | {values['startup_q_estimation_rmse_rad']:.6g} | {values['startup_dq_estimation_rmse_rad_s']:.6g} |",
                    f"| {name} steady | {values['steady_q_estimation_rmse_rad']:.6g} | {values['steady_dq_estimation_rmse_rad_s']:.6g} |",
                ]
            )
    content = """# Lesson 11 — LQG Output Feedback Engineering Report

## Scope

The controller is a composition of the existing Lesson 08 LQR gain and Lesson 10 Kalman estimator.  The simulation alone owns plant truth; deployable LQG receives only encoder position and the previous **applied** torque.

## A. Separation principle

The `dlqe` gain is in predictor form, so the analysis uses `A_e = A_d - L_e C`.  The augmented linear, unsaturated matrix is `[[A_d-B_dK_c, B_dK_c], [0, A_d-L_eC]]`.

| union pole error | all augmented poles inside unit circle |
| ---: | :--- |
| {union_error:.3e} | {stable} |

This confirms the eigenvalue union only for the matched, linear, unsaturated model; it is not a guarantee for a clipped actuator.

## B–E quantitative results

| Scenario | tracking RMSE / rad | q-hat RMSE / rad | dq-hat RMSE / rad/s | control RMS / N m | saturation / % | mean NIS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

## C. Startup versus steady-state estimator error

| Estimator and window | q-hat RMSE / rad | dq-hat RMSE / rad/s |
| --- | ---: | ---: |
{window_rows}

`full_state` is a simulation-only upper/reference baseline and deliberately has no estimator statistics.  Compare `recursive` against `steady_state` over the logged startup (0–0.5 s) and steady windows; a transient difference is expected because only the recursive covariance/gain evolves.

The `q_scale_*` rows keep the LQR gain fixed while changing only the assumed process covariance.  Any changed tracking/control behavior demonstrates that independent LQR/KF design does not make runtime performance independent.

The `reduced_limit` row is deliberately nonlinear because torque clipping is active.  Interpret high saturation, innovation, and NIS as a boundary diagnostic rather than attempting to explain the response solely with `A_d-B_dK_c` and `A_d-L_eC` poles.

## Hardware transition

Replace only the simulated encoder/plant boundary with hardware I/O.  Keep `LQGController.step(measurement, previous_applied_torque, reference)` unchanged, log the actual prior actuator torque/current-derived torque, start at nominal limits, and investigate persistent innovation bias or abnormal NIS before retuning.
""".format(
        union_error=separation["union_error"],
        stable="yes" if separation["is_stable"] else "no",
        rows="\n".join(rows),
        window_rows="\n".join(window_rows),
    )
    path.write_text(content, encoding="utf-8")


def run_all(config_path: Path | str, output_root: Path | str | None = None) -> dict:
    """Run experiments A–E and write CSV logs, five figures, and a report."""

    config = load_config(config_path)
    root = Path(output_root) if output_root is not None else Path(__file__).parents[1]
    figures = root / "figures"
    logs = root / "logs"
    reports = root / "reports"
    design = build_lqg_design(config)
    separation = analyze_separation_principle(
        design["Ad"], design["Bd"], design["C"], design["K_controller"], design["L_predictor"]
    )
    step_load = _step_load(config)

    full_state = simulate_lqg(config, "full_state_lqr", load_torque=step_load)
    recursive = simulate_lqg(config, "recursive_lqg", load_torque=step_load)
    steady = simulate_lqg(config, "steady_state_lqg", load_torque=step_load)
    q_logs = {
        scale: simulate_lqg(config, "recursive_lqg", q_scale=scale, load_torque=step_load)
        for scale in config["estimator_tuning"]["q_scales"]
    }
    boundary = {
        "normal": simulate_lqg(config, "recursive_lqg"),
        "load": simulate_lqg(config, "recursive_lqg", load_torque=step_load),
        "reduced_limit": simulate_lqg(
            config,
            "recursive_lqg",
            torque_limit=float(config["actuator"]["reduced_torque_limit_nm"]),
            load_torque=step_load,
        ),
    }

    named_logs = {
        "full_state": full_state,
        "recursive": recursive,
        "steady_state": steady,
        **{f"q_scale_{scale:g}": log for scale, log in q_logs.items()},
        **boundary,
    }
    for name, log in named_logs.items():
        _write_log(logs / f"{name}.csv", log)

    _plot_separation(separation, figures / "separation_poles.png")
    _plot_full_state_vs_lqg(full_state, recursive, figures / "full_state_vs_lqg.png")
    _plot_recursive_vs_steady(recursive, steady, figures / "recursive_vs_steady_state.png")
    _plot_q_tuning(q_logs, figures / "q_tuning_coupling.png")
    _plot_saturation_boundary(boundary, figures / "saturation_boundary.png")

    summary_metrics = {name: calculate_metrics(log) for name, log in named_logs.items()}
    report_path = reports / "lqg_engineering_report.md"
    _write_report(report_path, separation, summary_metrics)
    return {
        "separation": separation,
        "comparison_metrics": {name: summary_metrics[name] for name in ("full_state", "recursive", "steady_state")},
        "q_tuning_metrics": {name: summary_metrics[name] for name in summary_metrics if name.startswith("q_scale_")},
        "boundary_metrics": {name: summary_metrics[name] for name in boundary},
        "report": report_path,
    }


def main() -> None:
    root = Path(__file__).parents[1]
    result = run_all(root / "config" / "lqg.yaml", output_root=root)
    print("===== separation =====")
    print(f"union_error: {result['separation']['union_error']:.3e}")
    print(f"is_stable: {result['separation']['is_stable']}")
    for group_name in ("comparison_metrics", "q_tuning_metrics", "boundary_metrics"):
        print(f"===== {group_name} =====")
        for name, metrics in result[group_name].items():
            print(name)
            print(_format_metrics(metrics, ("tracking_rmse_rad", "control_rms_nm", "saturation_ratio_percent", "nis_mean")))
    print(f"report: {result['report']}")


if __name__ == "__main__":
    main()
