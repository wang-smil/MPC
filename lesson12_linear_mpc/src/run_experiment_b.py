"""Run Lesson 12 Experiment B: unconstrained MPC versus matching LQR."""

from collections import Counter
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .closed_loop import build_closed_loop_design, simulate_closed_loop
from .model_loader import load_config


LESSON_DIR = Path(__file__).resolve().parents[1]


def _metrics(trace: dict[str, np.ndarray], baseline: dict[str, np.ndarray]) -> dict:
    position_error = trace["q_true_rad"] - trace["q_ref_rad"]
    velocity_error = trace["dq_true_rad_s"] - trace["dq_ref_rad_s"]
    control_events = trace["control_update"]
    statuses = Counter(str(status) for status in trace["qp_status"][control_events])
    times = trace["qp_solve_time_s"][control_events]
    times = times[np.isfinite(times)]
    return {
        "tracking_rmse_rad": float(np.sqrt(np.mean(position_error**2))),
        "velocity_rmse_rad_s": float(np.sqrt(np.mean(velocity_error**2))),
        "max_abs_torque_difference_nm": float(
            np.max(np.abs(trace["torque_applied_nm"] - baseline["torque_applied_nm"]))
        ),
        "peak_abs_torque_nm": float(np.max(np.abs(trace["torque_applied_nm"]))),
        "solver_status_counts": dict(statuses),
        "solve_time_mean_s": float(np.mean(times)) if times.size else 0.0,
        "solve_time_p95_s": float(np.percentile(times, 95)) if times.size else 0.0,
        "solve_time_max_s": float(np.max(times)) if times.size else 0.0,
    }


def _write_csv(path: Path, traces: dict[str, dict[str, np.ndarray]]) -> None:
    first_trace = next(iter(traces.values()))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["case", *first_trace.keys()])
        writer.writeheader()
        for case, trace in traces.items():
            for index in range(len(trace["time_s"])):
                row = {name: values[index] for name, values in trace.items()}
                writer.writerow({"case": case, **row})


def _plot_comparison(path: Path, traces: dict[str, dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    styles = {"lqr": "-", "mpc_N1": "--", "mpc_N5": "-.", "mpc_N20": ":"}
    baseline = traces["lqr"]
    time_s = baseline["time_s"]

    for case, trace in traces.items():
        label = "LQR" if case == "lqr" else case.replace("mpc_N", "MPC N=")
        axes[0].plot(time_s, np.rad2deg(trace["q_true_rad"]), styles[case], label=label)
        axes[1].plot(time_s, trace["dq_hat_rad_s"], styles[case], label=label)
        axes[2].plot(time_s, trace["torque_applied_nm"], styles[case], label=label)
    axes[0].plot(time_s, np.rad2deg(baseline["q_ref_rad"]), "k--", alpha=0.5, label="Target")
    axes[0].set_ylabel("True position / deg")
    axes[1].set_ylabel("KF velocity estimate / rad/s")
    axes[2].set_ylabel("Applied torque / N m")
    axes[2].set_xlabel("Time / s")
    axes[0].set_title("Experiment B: unconstrained MPC and DARE LQR")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(path: Path, metrics: dict[str, dict], config: dict) -> None:
    rows = []
    for case, values in metrics.items():
        statuses = ", ".join(f"{name}: {count}" for name, count in values["solver_status_counts"].items())
        rows.append(
            f"| {case} | {values['tracking_rmse_rad']:.6f} | "
            f"{values['velocity_rmse_rad_s']:.6f} | "
            f"{values['max_abs_torque_difference_nm']:.8f} | "
            f"{values['peak_abs_torque_nm']:.5f} | {statuses} | "
            f"{values['solve_time_mean_s'] * 1e3:.4f} / "
            f"{values['solve_time_p95_s'] * 1e3:.4f} / "
            f"{values['solve_time_max_s'] * 1e3:.4f} |"
        )

    duration = float(config["experiment_b"]["duration_s"])
    text = f"""# 第 12 课实验 B：无约束 MPC 与 LQR

## 本次实验做了什么

同一个已辨识单轴关节模型，植物和卡尔曼滤波器每 1 ms 更新，LQR/MPC 每 10 ms 更新。仿真时长 {duration:g} s，目标位置 {config['experiment_b']['target_deg']:g}°，MPC 预测步数为 1、5、20。四组使用相同的初始状态、编码器噪声样本、模型和 Q/R 权重。

## 控制算法与时间顺序

状态是 x = [q, dq]，编码器只测位置 y = q + n。每个 1 ms 周期先用此前真实施加的转矩预测 KF，再用当前位置测量更新，得到后验估计 x_hat。t=0 只有测量更新。每逢 10 ms 控制时刻，控制器使用 x_hat；其余快周期保持上一次转矩。

非 Condensed MPC 把 x_0,…,x_N 和 u_0,…,u_(N-1) 一起作为优化变量，并加入等式 x_0=x_hat、x_(k+1)=Ad x_k+Bd u_k。目标函数是

Σ[k=0..N-1] ((x_k-x_ref)^T Q (x_k-x_ref) + u_k^T R u_k) + (x_N-x_ref)^T P (x_N-x_ref)。

这里 P 是同一 10 ms 模型、Q/R 的离散代数 Riccati 方程解。实验 A 的 X=F x0+G U 是把动力学等式消去后的等价预测关系；非 Condensed QP 直接保留这些等式，所以代码里无需显式构造 F/G。

每次求解只实际施加最优序列的第一个转矩 u_0，并在下一控制时刻重新测量、估计和规划。LQR 使用同一个 DARE 解得到 K，控制律为 u=-K(x_hat-x_ref)。在当前无约束、线性、静态平衡参考条件下，P 是有限时域 Riccati 递推的固定点，因此任何上述 N 的首个 MPC 输入都应与 LQR 输入相同；剩余差异是数值误差和闭环误差传播。

## 实测结果

| 工况 | 真实位置跟踪 RMSE / rad | 真实速度 RMSE / rad/s | 与 LQR 闭环转矩最大差 / N m | 峰值转矩 / N m | 求解状态计数 | 求解时间均值 / p95 / 最大值 (ms) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
{chr(10).join(rows)}

未舍入的四组闭环转矩最大差为 {max(values['max_abs_torque_difference_nm'] for values in metrics.values()):.3e} N m；表中的 0.00000000 是显示精度造成的，并非数学上严格为零。

上表的转矩差比较的是各自闭环轨迹同一时刻的输入；严格的“同一 x_hat 上首个输入相等”由 `test_mpc_controller.py` 单独检验，容差为 1e-4 N m。求解时间是本机 CVXPY/OSQP 报告的 solver time，不包含 Python 调度、问题构建、通信或硬件延迟。

## 工程边界

本次峰值转矩 {max(values['peak_abs_torque_nm'] for values in metrics.values()):.5f} N m，高于 Bryson 权重所用的 {float(config['design_limits']['max_torque_nm']):g} N m 转矩标尺。这个标尺只决定 R 的大小，不是执行器限幅；控制器没有 clip 转矩。

这是无约束的离线仿真。没有转矩限幅、速度/位置安全约束、负载扰动或真实电机；表中的峰值转矩不代表执行器可安全输出。DARE 终端代价下与 LQR 的等价性依赖上述线性、无约束和模型匹配假设，不能据此推断真实硬件性能。后续加入执行器约束时，MPC 与 LQR 的控制行为才可能明显分开。
"""
    path.write_text(text, encoding="utf-8")


def run_experiment_b(
    config_path: Path | str,
    output_root: Path | str | None = None,
    *,
    duration_s: float | None = None,
) -> dict:
    """Run the four fair scenarios and generate one aligned comparison set."""

    config = load_config(config_path)
    if duration_s is not None:
        config["experiment_b"]["duration_s"] = duration_s
    design = build_closed_loop_design(config)
    count = design["sample_count"]
    seed = int(config["experiment_b"]["seed"])
    std = np.deg2rad(float(config["measurement"]["position_noise_std_deg"]))
    measurement_noise = np.random.default_rng(seed).normal(0.0, std, count)

    traces = {"lqr": simulate_closed_loop(config, "lqr", measurement_noise=measurement_noise)}
    for horizon in config["experiment_b"]["horizons"]:
        name = f"mpc_N{horizon}"
        traces[name] = simulate_closed_loop(
            config, "mpc", horizon=horizon, measurement_noise=measurement_noise
        )

    baseline = traces["lqr"]
    metrics = {case: _metrics(trace, baseline) for case, trace in traces.items()}
    root = Path(output_root) if output_root is not None else LESSON_DIR
    paths = {
        "csv": root / "logs" / "experiment_b.csv",
        "figure": root / "figures" / "lqr_vs_mpc.png",
        "report": root / "reports" / "unconstrained_mpc_vs_lqr.md",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(paths["csv"], traces)
    _plot_comparison(paths["figure"], traces)
    _write_report(paths["report"], metrics, config)
    return {"traces": traces, "metrics": metrics, "paths": paths}


if __name__ == "__main__":
    output = run_experiment_b(LESSON_DIR / "config" / "mpc.yaml")
    for case, values in output["metrics"].items():
        print(f"{case}: tracking RMSE={values['tracking_rmse_rad']:.6f} rad, "
              f"max |delta torque|={values['max_abs_torque_difference_nm']:.8f} N m, "
              f"statuses={values['solver_status_counts']}")
    for name, path in output["paths"].items():
        print(f"{name}: {path}")
