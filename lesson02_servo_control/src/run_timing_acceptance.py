from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

import run_servo_test as servo


ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_CONFIG_PATH = ROOT / "config" / "timing_acceptance.yaml"


def load_acceptance_config() -> dict:
    """读取时序验收实验及阶段性阈值。"""

    with ACCEPTANCE_CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _build_case_config(base_config: dict, case: dict) -> dict:
    config = deepcopy(base_config)
    config["simulation"]["dt"] = float(case["dt"])
    config["simulation"]["jitter_std_ratio"] = float(
        case["jitter_std_ratio"]
    )
    config["actuator"]["delay_steps"] = int(case["delay_steps"])
    config["reference"]["type"] = "cubic"
    return config


def _collect_metrics(data: dict) -> dict[str, float]:
    metrics = servo.calculate_metrics(
        time=data["time"],
        position=data["position"],
        target=data["target"][-1],
        torque=data["torque_applied"],
        saturated=data["saturated"],
    )
    metrics.update(servo.engineering_metrics(data))
    metrics.update(servo.timing_metrics(data["actual_dt"][:-1]))
    metrics["safety_fault_count"] = int(
        np.sum(data["safety_fault"])
    )
    return metrics


def run_acceptance(
    base_config: dict,
    acceptance_config: dict,
) -> dict[str, dict]:
    """运行五组彼此独立的采样、延迟与抖动实验。"""

    results = {}

    for name, case in acceptance_config["experiments"].items():
        config = _build_case_config(base_config, case)
        data = servo.simulate(config, scenario="normal")
        nominal_dt = float(case["dt"])
        delay_steps = int(case["delay_steps"])

        results[name] = {
            "data": data,
            "metrics": _collect_metrics(data),
            "nominal_dt_ms": nominal_dt * 1000.0,
            "nominal_delay_ms": (
                nominal_dt * delay_steps * 1000.0
            ),
        }

    return results


def normal_passes(
    metrics: dict[str, float],
    limits: dict,
) -> tuple[bool, dict[str, bool]]:
    """逐项判断平滑轨迹基准是否满足阶段性验收阈值。"""

    checks = {
        "safety_fault_count": (
            metrics["safety_fault_count"]
            <= int(limits["safety_fault_count"])
        ),
        "final_error_deg": (
            metrics["final_error_deg"]
            < float(limits["final_error_deg"])
        ),
        "overshoot_percent": (
            metrics["overshoot_percent"]
            < float(limits["overshoot_percent"])
        ),
        "max_velocity_deg_s": (
            metrics["max_velocity_deg_s"]
            < float(limits["max_velocity_deg_s"])
        ),
        "saturation_ratio_percent": (
            metrics["saturation_ratio_percent"]
            < float(limits["saturation_ratio_percent"])
        ),
        "settling_time_s": (
            metrics["settling_time_s"]
            < float(limits["settling_time_s"])
        ),
    }

    return all(checks.values()), checks


def _step_comparison_data(
    base_config: dict,
    acceptance_config: dict,
) -> dict:
    normal_case = acceptance_config["experiments"][
        "acceptance_normal"
    ]
    config = _build_case_config(base_config, normal_case)
    config["reference"]["type"] = "step"
    return servo.simulate(config, scenario="normal")


def _plot_step_vs_cubic(
    step_data: dict,
    cubic_data: dict,
) -> Path:
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "step_vs_cubic.png"

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(
        step_data["time"],
        np.rad2deg(step_data["target"]),
        linestyle="--",
        label="Step target",
    )
    axes[0].plot(
        step_data["time"],
        np.rad2deg(step_data["position"]),
        label="Step position",
    )
    axes[0].plot(
        cubic_data["time"],
        np.rad2deg(cubic_data["target"]),
        linestyle="--",
        label="Cubic target",
    )
    axes[0].plot(
        cubic_data["time"],
        np.rad2deg(cubic_data["position"]),
        label="Cubic position",
    )
    axes[0].set_ylabel("Position / deg")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        step_data["time"],
        np.rad2deg(step_data["velocity"]),
        label="Step velocity",
    )
    axes[1].plot(
        cubic_data["time"],
        np.rad2deg(cubic_data["velocity"]),
        label="Cubic velocity",
    )
    axes[1].axhline(300.0, color="red", linestyle="--", label="Limit")
    axes[1].set_xlabel("Time / s")
    axes[1].set_ylabel("Velocity / deg/s")
    axes[1].set_xlim(0.0, 1.2)
    axes[1].grid(True)
    axes[1].legend()

    figure.suptitle("Position Step vs Cubic Reference")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return figure_path


def _plot_timing_comparison(results: dict[str, dict]) -> Path:
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "timing_comparison.png"

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for name, result in results.items():
        data = result["data"]
        axes[0].plot(
            data["time"],
            np.rad2deg(data["position"]),
            label=name,
        )
        axes[1].plot(
            data["time"],
            np.rad2deg(data["velocity"]),
            label=name,
        )

    axes[0].plot(
        results["acceptance_normal"]["data"]["time"],
        np.rad2deg(
            results["acceptance_normal"]["data"]["target"]
        ),
        color="black",
        linestyle="--",
        label="Target",
    )
    axes[0].set_ylabel("Position / deg")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].axhline(300.0, color="red", linestyle="--", label="Limit")
    axes[1].set_xlabel("Time / s")
    axes[1].set_ylabel("Velocity / deg/s")
    axes[1].set_xlim(0.0, 1.2)
    axes[1].grid(True)
    axes[1].legend()

    figure.suptitle("Sampling, Delay and Jitter Comparison")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return figure_path


def _write_report(
    results: dict[str, dict],
    limits: dict,
    step_data: dict,
) -> Path:
    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "engineering_acceptance.md"

    normal_metrics = results["acceptance_normal"]["metrics"]
    step_metrics = servo.engineering_metrics(step_data)
    passed, checks = normal_passes(normal_metrics, limits)
    status = "PASS" if passed else "FAIL"

    lines = [
        "# 单轴伺服工程验收与时序实验",
        "",
        f"## 正常工况验收：{status}",
        "",
        "原始位置阶跃在零时刻产生30°误差，使PD命令立即饱和并造成超速。",
        "正常工况现采用三次轨迹 `q=q0+(qf-q0)(3s²-2s³)`，",
        "其起点和终点目标速度均为零。",
        "",
        "| 参考指令 | 最大速度/(deg/s) | 安全采样点 |",
        "|---|---:|---:|",
        f"| 位置阶跃 | {step_metrics['max_velocity_deg_s']:.6f} | "
        f"{int(np.sum(step_data['safety_fault']))} |",
        f"| 三次轨迹 | {normal_metrics['max_velocity_deg_s']:.6f} | "
        f"{normal_metrics['safety_fault_count']} |",
        "",
        "| 验收项 | 结果 |",
        "|---|---:|",
    ]

    for name, check_passed in checks.items():
        lines.append(
            f"| {name} | {'PASS' if check_passed else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "## 采样、延迟与抖动结果",
            "",
            "| 工况 | 平均周期/ms | 最大周期/ms | 抖动标准差/ms | 名义延迟/ms | "
            "最大速度/(deg/s) | 超调/% | 安全采样点 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for name, result in results.items():
        metrics = result["metrics"]
        lines.append(
            f"| {name} | {metrics['mean_dt_ms']:.6f} | "
            f"{metrics['max_dt_ms']:.6f} | "
            f"{metrics['jitter_std_ms']:.6f} | "
            f"{result['nominal_delay_ms']:.3f} | "
            f"{metrics['max_velocity_deg_s']:.6f} | "
            f"{metrics['overshoot_percent']:.6f} | "
            f"{metrics['safety_fault_count']} |"
        )

    lines.extend(
        [
            "",
            "`delay_5steps`与`sample_5ms`的名义延迟都是5 ms，",
            "但前者仍以1 kHz读取传感器并更新控制器，后者仅以200 Hz更新，",
            "所以低频采样还会遗漏更多采样间动态，不能只看名义延迟。",
            "",
            "5%抖动实验的平均周期仍接近1 ms，但最大周期和周期标准差非零；",
            "当前参数下性能退化较轻，只能说明系统对这一级别抖动有一定裕量。",
            "",
            "## 结论边界",
            "",
            "本实验比较了相同控制逻辑下的采样周期、命令延迟和轻度周期抖动。",
            "数值仿真中的 `actual_dt` 会参与速度差分和动力学积分。",
            "",
            "本项目不能证明 Python 或普通 Windows 满足1 kHz硬实时要求。",
            "仿真时间由数值循环推进，没有验证操作系统调度、通信延迟、",
            "垃圾回收、长尾抖动或实机控制截止时间。",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def generate_artifacts(
    results: dict[str, dict],
    base_config: dict,
    acceptance_config: dict,
) -> dict:
    """保存五组日志、两张对比图和一份Markdown验收报告。"""

    logs = {
        name: servo.save_log(result["data"], name)
        for name, result in results.items()
    }
    step_data = _step_comparison_data(
        base_config,
        acceptance_config,
    )

    return {
        "logs": logs,
        "step_vs_cubic": _plot_step_vs_cubic(
            step_data,
            results["acceptance_normal"]["data"],
        ),
        "timing_comparison": _plot_timing_comparison(results),
        "report": _write_report(
            results,
            acceptance_config["normal_limits"],
            step_data,
        ),
    }


def main() -> None:
    base_config = servo.load_config()
    acceptance_config = load_acceptance_config()
    results = run_acceptance(base_config, acceptance_config)
    artifacts = generate_artifacts(
        results,
        base_config,
        acceptance_config,
    )

    for name, result in results.items():
        print(f"\n===== {name} =====")
        print(f"nominal_dt_ms: {result['nominal_dt_ms']}")
        print(f"nominal_delay_ms: {result['nominal_delay_ms']}")

        for metric_name, value in result["metrics"].items():
            print(f"{metric_name}: {value}")

    passed, checks = normal_passes(
        results["acceptance_normal"]["metrics"],
        acceptance_config["normal_limits"],
    )
    print(f"\nnormal_acceptance: {'PASS' if passed else 'FAIL'}")

    for name, check_passed in checks.items():
        print(f"{name}: {'PASS' if check_passed else 'FAIL'}")

    print(f"step_vs_cubic: {artifacts['step_vs_cubic']}")
    print(f"timing_comparison: {artifacts['timing_comparison']}")
    print(f"report: {artifacts['report']}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
