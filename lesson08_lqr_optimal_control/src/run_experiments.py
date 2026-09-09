"""Run the A-D discrete LQR experiments and create learning artifacts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .controller import LQRController
from .lqr_design import (
    build_bryson_weights,
    design_dlqr,
    design_pole_placement,
)
from .metrics import calculate_metrics
from .model import build_continuous_model, discretize_zoh, load_config
from .simulator import simulate_closed_loop


def _nominal_matrices(config: dict) -> tuple[np.ndarray, np.ndarray]:
    A, B = build_continuous_model(config["model"]["inertia"], config["model"]["damping"])
    return discretize_zoh(A, B, config["simulation"]["dt_s"])


def _weights(config: dict, position_scale: float, velocity_scale: float, torque_scale: float) -> tuple[np.ndarray, np.ndarray]:
    limits = config["design_limits"]
    return build_bryson_weights(
        limits["max_position_error_deg"],
        limits["max_velocity_error_rad_s"],
        limits["max_torque_nm"],
        position_scale,
        velocity_scale,
        torque_scale,
    )


def _run_case(
    config: dict,
    K: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    plant_inertia: float,
    torque_limit_nm: float,
    load_torque_nm: float = 0.0,
    load_start_s: float = float("inf"),
) -> dict:
    controller = LQRController(K, torque_limit_nm)
    log = simulate_closed_loop(
        config,
        controller,
        Q,
        R,
        plant_inertia=plant_inertia,
        load_torque_nm=load_torque_nm,
        load_start_s=load_start_s,
    )
    return {"K": np.asarray(K), "Q": Q, "R": R, "log": log, "metrics": calculate_metrics(log)}


def _augment_log(log: dict[str, np.ndarray], result: dict, poles: np.ndarray) -> dict[str, np.ndarray]:
    augmented = dict(log)
    length = log["time_s"].size
    K = result["K"].ravel()
    augmented["Q_position"] = np.full(length, result["Q"][0, 0])
    augmented["Q_velocity"] = np.full(length, result["Q"][1, 1])
    augmented["R_torque"] = np.full(length, result["R"].item())
    augmented["K_position"] = np.full(length, K[0])
    augmented["K_velocity"] = np.full(length, K[1])
    for index, pole in enumerate(poles, start=1):
        augmented[f"closed_loop_pole_{index}_real"] = np.full(length, pole.real)
        augmented[f"closed_loop_pole_{index}_imag"] = np.full(length, pole.imag)
    return augmented


def _write_csv(path: Path, log: dict[str, np.ndarray]) -> None:
    names = list(log)
    values = np.column_stack([np.asarray(log[name], dtype=float) for name in names])
    np.savetxt(path, values, delimiter=",", header=",".join(names), comments="")


def _plot_qr_tradeoff(path: Path, cases: dict[str, dict]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, result in cases.items():
        log = result["log"]
        axes[0].plot(log["time_s"], np.rad2deg(log["q_rad"]), label=name)
        axes[1].plot(log["time_s"], log["torque_cmd_nm"], label=name)
    reference = next(iter(cases.values()))["log"]
    axes[0].plot(reference["time_s"], np.rad2deg(reference["q_ref_rad"]), "k--", label="reference")
    axes[0].set_ylabel("Position / deg")
    axes[1].set_ylabel("Torque command / N m")
    axes[1].set_xlabel("Time / s")
    axes[0].set_title("A. Q/R weight trade-off")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _plot_lqr_vs_pp(path: Path, lqr: dict, pole_placement: dict) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, result in (("Balanced LQR", lqr), ("Pole placement", pole_placement)):
        log = result["log"]
        axes[0].plot(log["time_s"], np.rad2deg(log["q_rad"]), label=name)
        axes[1].plot(log["time_s"], log["torque_cmd_nm"], label=name)
    reference = lqr["log"]
    axes[0].plot(reference["time_s"], np.rad2deg(reference["q_ref_rad"]), "k--", label="reference")
    axes[0].set_ylabel("Position / deg")
    axes[1].set_ylabel("Torque command / N m")
    axes[1].set_xlabel("Time / s")
    axes[0].set_title("B. Balanced LQR versus pole placement at 1 ms")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _plot_payload(path: Path, nominal: dict, payload: dict) -> None:
    plt.figure(figsize=(9, 5))
    for name, result in (("Nominal inertia", nominal), ("Payload +30%", payload)):
        log = result["log"]
        plt.plot(log["time_s"], np.rad2deg(log["q_rad"]), label=name)
    log = nominal["log"]
    plt.plot(log["time_s"], np.rad2deg(log["q_ref_rad"]), "k--", label="reference")
    plt.xlabel("Time / s")
    plt.ylabel("Position / deg")
    plt.title("C. Fixed LQR gain under payload mismatch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_stress(path: Path, stress: dict) -> None:
    log = stress["log"]
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(log["time_s"], np.rad2deg(log["q_rad"]), label="position")
    axes[0].plot(log["time_s"], np.rad2deg(log["q_ref_rad"]), "k--", label="reference")
    axes[0].axvline(1.0, color="gray", linestyle=":", label="load applied")
    axes[1].plot(log["time_s"], log["torque_unsat_nm"], label="unsaturated request")
    axes[1].plot(log["time_s"], log["torque_cmd_nm"], label="applied command")
    axes[1].plot(log["time_s"], log["load_torque_nm"], label="load torque")
    axes[0].set_ylabel("Position / deg")
    axes[1].set_ylabel("Torque / N m")
    axes[1].set_xlabel("Time / s")
    axes[0].set_title("D. Constant load and 1.5 N m actuator stress")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _summary_table(cases: dict[str, dict]) -> str:
    header = "| Case | Kq | Kdq | Settling / s | Overshoot / % | Peak torque | RMS torque | Saturation / % | State cost | Input cost | Total cost |"
    divider = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    rows = [header, divider]
    for name, result in cases.items():
        metrics = result["metrics"]
        K = result["K"].ravel()
        rows.append(
            f"| {name} | {K[0]:.4f} | {K[1]:.4f} | {metrics['settling_time_s']:.3f} | "
            f"{metrics['overshoot_percent']:.3f} | {metrics['peak_torque_nm']:.3f} | "
            f"{metrics['rms_torque_nm']:.3f} | {metrics['saturation_ratio_percent']:.3f} | "
            f"{metrics['state_cost_total']:.2f} | {metrics['input_cost_total']:.2f} | {metrics['lqr_cost_total']:.2f} |"
        )
    return "\n".join(rows) + "\n"


def _write_report(path: Path, summary: str, lqr: dict, pp: dict, payload: dict, stress: dict) -> None:
    content = "\n".join(
        [
            "# Lesson 08: Discrete LQR Optimal Control Report",
            "",
            "## A. Q/R trade-off",
            "",
            summary,
            "## B. LQR versus pole placement",
            "",
            "LQR is optimal for its selected Q/R cost; it is not guaranteed to minimize every external metric more than pole placement.",
            f"Balanced LQR total cost: {lqr['metrics']['lqr_cost_total']:.2f}. Pole-placement cost evaluated with the balanced Q/R: {pp['metrics']['lqr_cost_total']:.2f}.",
            "",
            "## C. Payload mismatch",
            "",
            f"The nominal LQR gain was kept fixed while real inertia changed by +30%. Nominal tracking RMSE: {lqr['metrics']['tracking_rmse_rad']:.6f} rad; payload RMSE: {payload['metrics']['tracking_rmse_rad']:.6f} rad.",
            "",
            "## D. Constant load and saturation stress",
            "",
            f"A 1 N m load begins at 1 s and torque is limited to 1.5 N m. Final error: {np.rad2deg(stress['metrics']['final_error_rad']):.3f} deg; saturation: {stress['metrics']['saturation_ratio_percent']:.2f}%.",
            "The residual error illustrates that plain state-feedback LQR has no integral action for an unknown constant load. The clipped torque trace shows that an R penalty is not a hard input constraint.",
        ]
    )
    path.write_text(content, encoding="utf-8")


def run_all(config_path: Path, output_root: Path | None = None) -> dict:
    """Run all LQR learning experiments and write logs, figures, and reports."""

    config = load_config(config_path)
    base_dir = Path(output_root) if output_root is not None else config_path.parent.parent
    figure_dir, log_dir, report_dir = base_dir / "figures", base_dir / "logs", base_dir / "reports"
    for directory in (figure_dir, log_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    Ad, Bd = _nominal_matrices(config)
    nominal_inertia = float(config["model"]["inertia"])
    torque_limit = float(config["design_limits"]["max_torque_nm"])
    weight_cases = {
        "balanced": (1.0, 1.0, 1.0),
        "position_priority": (10.0, 1.0, 1.0),
        "effort_saving": (1.0, 1.0, 10.0),
        "velocity_priority": (1.0, 10.0, 1.0),
    }
    cases: dict[str, dict] = {}
    poles: dict[str, np.ndarray] = {}
    for name, scales in weight_cases.items():
        Q, R = _weights(config, *scales)
        design = design_dlqr(Ad, Bd, Q, R)
        cases[name] = _run_case(config, design["K"], Q, R, nominal_inertia, torque_limit)
        poles[name] = design["closed_loop_poles"]

    balanced = cases["balanced"]
    balanced_Q, balanced_R = balanced["Q"], balanced["R"]
    pp_spec = config["pole_placement"]
    pp_design = design_pole_placement(Ad, Bd, pp_spec["damping_ratio"], pp_spec["settling_time_s"], config["simulation"]["dt_s"])
    cases["pole_placement"] = _run_case(config, pp_design["K"], balanced_Q, balanced_R, nominal_inertia, torque_limit)
    poles["pole_placement"] = pp_design["computed_poles"]

    cases["payload_plus_30_percent"] = _run_case(
        config,
        balanced["K"],
        balanced_Q,
        balanced_R,
        nominal_inertia * float(config["stress"]["payload_scale"]),
        torque_limit,
    )
    poles["payload_plus_30_percent"] = poles["balanced"]

    cases["load_saturation_stress"] = _run_case(
        config,
        balanced["K"],
        balanced_Q,
        balanced_R,
        nominal_inertia,
        float(config["stress"]["torque_limit_nm"]),
        load_torque_nm=float(config["stress"]["load_torque_nm"]),
        load_start_s=float(config["stress"]["load_start_s"]),
    )
    poles["load_saturation_stress"] = poles["balanced"]

    log_paths: dict[str, Path] = {}
    for name, result in cases.items():
        path = log_dir / f"{name}.csv"
        _write_csv(path, _augment_log(result["log"], result, poles[name]))
        log_paths[name] = path

    figures = {
        "qr_tradeoff": figure_dir / "qr_tradeoff.png",
        "lqr_vs_pole_placement": figure_dir / "lqr_vs_pole_placement.png",
        "payload_mismatch": figure_dir / "payload_mismatch.png",
        "load_saturation_stress": figure_dir / "load_saturation_stress.png",
    }
    qr_cases = {name: cases[name] for name in weight_cases}
    _plot_qr_tradeoff(figures["qr_tradeoff"], qr_cases)
    _plot_lqr_vs_pp(figures["lqr_vs_pole_placement"], balanced, cases["pole_placement"])
    _plot_payload(figures["payload_mismatch"], balanced, cases["payload_plus_30_percent"])
    _plot_stress(figures["load_saturation_stress"], cases["load_saturation_stress"])

    summary_path = report_dir / "qr_summary.md"
    summary = _summary_table(qr_cases)
    summary_path.write_text("# LQR Q/R Trade-off Summary\n\n" + summary, encoding="utf-8")
    report_path = report_dir / "lqr_report.md"
    _write_report(report_path, summary, balanced, cases["pole_placement"], cases["payload_plus_30_percent"], cases["load_saturation_stress"])
    return {
        "figures": figures,
        "logs": log_paths,
        "summary_path": summary_path,
        "report_path": report_path,
        "cases": cases,
    }


def main() -> None:
    lesson_dir = Path(__file__).resolve().parents[1]
    result = run_all(lesson_dir / "config" / "lqr.yaml")
    for name, case in result["cases"].items():
        print(f"===== {name} =====")
        for metric, value in case["metrics"].items():
            print(f"{metric}: {value}")
        print(f"log: {result['logs'][name]}")
    for name, path in result["figures"].items():
        print(f"{name}: {path}")
    print(f"summary: {result['summary_path']}")
    print(f"report: {result['report_path']}")


if __name__ == "__main__":
    main()
