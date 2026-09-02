"""Run Lesson 05 system-identification experiments."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from estimator import identify_j_b
from excitation import MultiSineExcitation
from metrics import calculate_metrics
from plant import SingleAxisPlant, simulate_linear_model
from signal_processing import add_position_noise, estimate_derivatives, lowpass


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_LOG_COLUMNS = [
    "timestamp_s",
    "torque_command_nm",
    "torque_applied_nm",
    "position_true_rad",
    "position_measured_rad",
    "velocity_true_rad_s",
    "velocity_est_rad_s",
    "acceleration_true_rad_s2",
    "acceleration_est_rad_s2",
    "torque_predicted_nm",
    "residual_nm",
]


def load_config(path: Path) -> dict:
    """Load the lesson configuration and validate basic input constraints."""

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _validate_config(config)
    return config


def _validate_config(config: dict) -> None:
    required_sections = {
        "simulation",
        "plant_true",
        "sensor",
        "actuator",
        "excitation",
        "identification",
    }
    if not required_sections.issubset(config):
        raise ValueError("sysid configuration is missing one or more required sections.")
    if config["simulation"]["dt"] <= 0.0 or config["simulation"]["duration_s"] <= 0.0:
        raise ValueError("simulation dt and duration_s must be positive.")
    if config["plant_true"]["inertia"] <= 0.0:
        raise ValueError("plant inertia must be positive.")
    if any(
        config["plant_true"][key] < 0.0 for key in ("damping", "coulomb_friction")
    ):
        raise ValueError("plant damping and coulomb_friction cannot be negative.")
    if config["sensor"]["position_noise_std_deg"] < 0.0:
        raise ValueError("position_noise_std_deg cannot be negative.")
    if config["actuator"]["torque_limit_nm"] <= 0.0:
        raise ValueError("torque_limit_nm must be positive.")
    if config["identification"]["discard_start_s"] < 0.0:
        raise ValueError("discard_start_s cannot be negative.")
    MultiSineExcitation(
        config["excitation"]["frequencies_hz"], config["excitation"]["amplitudes_nm"]
    )


def _case_configuration(case_name: str, config: dict) -> tuple[dict, bool]:
    case_config = copy.deepcopy(config)
    filtered = True
    if case_name in {"normal", "poor_excitation", "noise_raw", "noise_filtered"}:
        case_config["plant_true"]["coulomb_friction"] = 0.0
    if case_name == "poor_excitation":
        case_config["excitation"] = {
            "frequencies_hz": [0.5],
            "amplitudes_nm": [0.45],
        }
    elif case_name in {"noise_raw", "noise_filtered"}:
        case_config["sensor"]["position_noise_std_deg"] = 0.20
        filtered = case_name == "noise_filtered"
    elif case_name not in {"normal", "model_mismatch"}:
        raise ValueError(f"Unknown identification case: {case_name}")
    return case_config, filtered


def _simulate_case(config: dict) -> dict[str, np.ndarray]:
    simulation = config["simulation"]
    dt_s = float(simulation["dt"])
    time_s = np.arange(0.0, float(simulation["duration_s"]) + dt_s * 0.5, dt_s)
    excitation = MultiSineExcitation(
        config["excitation"]["frequencies_hz"], config["excitation"]["amplitudes_nm"]
    )
    plant_parameters = config["plant_true"]
    plant = SingleAxisPlant(
        plant_parameters["inertia"],
        plant_parameters["damping"],
        plant_parameters["coulomb_friction"],
        config["actuator"]["torque_limit_nm"],
    )
    torque_command_nm = np.asarray(excitation.evaluate(time_s), dtype=float)
    records = [plant.step(command, dt_s) for command in torque_command_nm]
    position_true_rad = np.array([record["position_true_rad"] for record in records])
    velocity_true_rad_s = np.array([record["velocity_true_rad_s"] for record in records])
    acceleration_true_rad_s2 = np.array(
        [record["acceleration_true_rad_s2"] for record in records]
    )
    torque_applied_nm = np.array([record["torque_applied_nm"] for record in records])
    rng = np.random.default_rng(int(simulation["seed"]))
    position_measured_rad = add_position_noise(
        position_true_rad, config["sensor"]["position_noise_std_deg"], rng
    )
    return {
        "time_s": time_s,
        "torque_command_nm": torque_command_nm,
        "torque_applied_nm": torque_applied_nm,
        "position_true_rad": position_true_rad,
        "position_measured_rad": position_measured_rad,
        "velocity_true_rad_s": velocity_true_rad_s,
        "acceleration_true_rad_s2": acceleration_true_rad_s2,
    }


def _write_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_LOG_COLUMNS)
        writer.writeheader()
        for index in range(len(data["time_s"])):
            writer.writerow(
                {
                    column: float(
                        data["time_s" if column == "timestamp_s" else column][index]
                    )
                    for column in REQUIRED_LOG_COLUMNS
                }
            )


def _plot_case(path: Path, case_name: str, data: dict[str, np.ndarray]) -> None:
    time_s = data["time_s"]
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time_s, np.rad2deg(data["position_true_rad"]), label="true position")
    axes[0].plot(
        time_s, np.rad2deg(data["position_measured_rad"]), alpha=0.55, label="measured position"
    )
    axes[0].set_ylabel("position / deg")
    axes[0].legend()
    axes[1].plot(time_s, data["velocity_true_rad_s"], label="true velocity")
    axes[1].plot(time_s, data["velocity_est_rad_s"], alpha=0.8, label="estimated velocity")
    axes[1].set_ylabel("velocity / rad s$^{-1}$")
    axes[1].legend()
    axes[2].plot(time_s, data["torque_applied_nm"], label="applied torque")
    axes[2].plot(time_s, data["torque_predicted_nm"], alpha=0.8, label="predicted torque")
    axes[2].set_xlabel("time / s")
    axes[2].set_ylabel("torque / N m")
    axes[2].legend()
    for axis in axes:
        axis.grid(True)
    figure.suptitle(f"System identification: {case_name}")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_residual_diagnostic(path: Path, data: dict[str, np.ndarray]) -> None:
    """Show whether a residual is random or tied to time and velocity."""

    valid = np.isfinite(data["residual_nm"])
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    axes[0].plot(data["time_s"][valid], data["residual_nm"][valid])
    axes[0].set(xlabel="time / s", ylabel="residual / N m")
    axes[1].scatter(
        data["velocity_est_rad_s"][valid], data["residual_nm"][valid], s=5, alpha=0.55
    )
    axes[1].set(xlabel="estimated velocity / rad s$^{-1}$", ylabel="residual / N m")
    for axis in axes:
        axis.grid(True)
    figure.suptitle("Model-mismatch residual diagnostic")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_case(
    case_name: str,
    config: dict,
    output_root: Path | None = None,
    use_filter: bool | None = None,
) -> dict:
    """Run one identification case and write its log, figure, and parameter JSON."""

    _validate_config(config)
    case_config, case_default_filter = _case_configuration(case_name, config)
    filtered = case_default_filter if use_filter is None else use_filter
    data = _simulate_case(case_config)
    dt_s = float(case_config["simulation"]["dt"])
    velocity_est, acceleration_est = estimate_derivatives(
        data["position_measured_rad"], dt_s
    )
    if filtered:
        cutoff_hz = float(case_config["identification"]["lowpass_cutoff_hz"])
        velocity_est = lowpass(velocity_est, dt_s, cutoff_hz)
        acceleration_est = lowpass(acceleration_est, dt_s, cutoff_hz)

    fit_mask = data["time_s"] >= float(case_config["identification"]["discard_start_s"])
    estimate = identify_j_b(
        velocity_est[fit_mask],
        acceleration_est[fit_mask],
        data["torque_applied_nm"][fit_mask],
    )
    predicted_full = np.full_like(data["time_s"], np.nan)
    residual_full = np.full_like(data["time_s"], np.nan)
    predicted_full[fit_mask] = estimate["torque_predicted_nm"]
    residual_full[fit_mask] = estimate["residual_nm"]
    data["velocity_est_rad_s"] = velocity_est
    data["acceleration_est_rad_s2"] = acceleration_est
    data["torque_predicted_nm"] = predicted_full
    data["residual_nm"] = residual_full

    root = ROOT if output_root is None else Path(output_root)
    logs_dir = root / "logs"
    figures_dir = root / "figures"
    logs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{case_name}.csv"
    parameters_path = logs_dir / f"{case_name}_identified_parameters.json"
    figure_path = figures_dir / f"{case_name}.png"
    _write_csv(log_path, data)
    _plot_case(figure_path, case_name, data)
    residual_figure_path = None
    if case_name == "model_mismatch":
        residual_figure_path = figures_dir / "model_mismatch_residual.png"
        _plot_residual_diagnostic(residual_figure_path, data)
    parameters = {
        "case": case_name,
        "inertia_hat": estimate["inertia_hat"],
        "damping_hat": estimate["damping_hat"],
        "rank": estimate["rank"],
        "condition_number": estimate["condition_number"],
        "torque_rmse": estimate["torque_rmse"],
    }
    parameters_path.write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    return {
        "case_name": case_name,
        "estimate": estimate,
        "data": data,
        "log_columns": REQUIRED_LOG_COLUMNS,
        "log_path": log_path,
        "parameters_path": parameters_path,
        "figure_path": figure_path,
        "residual_figure_path": residual_figure_path,
        "filtered": filtered,
    }


def run_validation(
    config: dict,
    inertia_hat: float,
    damping_hat: float,
    output_root: Path | None = None,
) -> dict:
    """Validate the identified linear model using a distinct torque excitation."""

    validation_config = copy.deepcopy(config)
    validation_config["excitation"] = {
        "frequencies_hz": [0.8, 1.9, 3.4],
        "amplitudes_nm": [0.40, 0.25, 0.10],
    }
    # Validation checks generalisation of the normal linear model; friction
    # mismatch is isolated to the dedicated model_mismatch experiment.
    validation_config["plant_true"]["coulomb_friction"] = 0.0
    data = _simulate_case(validation_config)
    position_model, _ = simulate_linear_model(
        data["torque_applied_nm"], data["time_s"], inertia_hat, damping_hat
    )
    root = ROOT if output_root is None else Path(output_root)
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "validation.png"
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(data["time_s"], np.rad2deg(data["position_true_rad"]), label="true plant")
    axis.plot(data["time_s"], np.rad2deg(position_model), label="identified model")
    axis.set(title="Independent validation", xlabel="time / s", ylabel="position / deg")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return {
        "true_position_rad": data["position_true_rad"],
        "model_position_rad": position_model,
        "figure_path": figure_path,
    }


def _write_report(path: Path, results: list[dict], validation_metrics: dict[str, float]) -> None:
    rows = []
    for result in results:
        estimate = result["estimate"]
        rows.append(
            "| {case} | {j:.5f} | {b:.5f} | {rmse:.5f} | {cond:.2f} |".format(
                case=result["case_name"],
                j=estimate["inertia_hat"],
                b=estimate["damping_hat"],
                rmse=estimate["torque_rmse"],
                cond=estimate["condition_number"],
            )
        )
    report = "\n".join(
        [
            "# Lesson 05 System Identification Report",
            "",
            "The estimator fits `torque = J * acceleration + b * velocity` using applied torque.",
            "",
            "| Case | J_hat / kg m^2 | b_hat / N m s/rad | torque RMSE / N m | cond(Phi) |",
            "| --- | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "## Independent validation",
            "",
            f"Position trajectory RMSE: {validation_metrics['validation_position_rmse_rad']:.6f} rad.",
            "",
            "Interpretation: condition number describes parameter sensitivity, not a standalone pass/fail result. "
            "The raw high-noise case exposes derivative noise amplification; filtering should make the fit more usable. "
            "Model-mismatch residuals can remain structured because Coulomb friction is absent from the fitted model.",
        ]
    )
    path.write_text(report, encoding="utf-8")


def main() -> None:
    """Run all Lesson 05 experiments and print their main diagnostics."""

    config = load_config(ROOT / "config" / "sysid.yaml")
    case_names = ["normal", "poor_excitation", "noise_raw", "noise_filtered", "model_mismatch"]
    results = [run_case(case_name, config) for case_name in case_names]
    normal = results[0]
    validation = run_validation(
        config,
        normal["estimate"]["inertia_hat"],
        normal["estimate"]["damping_hat"],
    )
    validation_metrics = calculate_metrics(
        normal["estimate"],
        config["plant_true"],
        validation["true_position_rad"],
        validation["model_position_rad"],
    )
    report_path = ROOT / "reports" / "identification_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(report_path, results, validation_metrics)
    for result in results:
        estimate = result["estimate"]
        print(f"===== {result['case_name']} =====")
        print(f"J_hat: {estimate['inertia_hat']:.6f}")
        print(f"b_hat: {estimate['damping_hat']:.6f}")
        print(f"torque_rmse_nm: {estimate['torque_rmse']:.6f}")
        print(f"condition_number: {estimate['condition_number']:.3f}")
        print(f"log: {result['log_path']}")
        print(f"figure: {result['figure_path']}")
    print(f"validation_position_rmse_rad: {validation_metrics['validation_position_rmse_rad']:.6f}")
    print(f"validation_figure: {validation['figure_path']}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
