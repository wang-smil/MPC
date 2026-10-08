"""Compare torque-clipped LQR with torque- and speed-constrained MPC."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .closed_loop import build_closed_loop_design, simulate_closed_loop
from .model_loader import load_config
from .run_experiment_b import _write_csv
from .run_experiment_c import _metrics as torque_metrics


LESSON_DIR = Path(__file__).resolve().parents[1]


def _metrics(trace: dict[str, np.ndarray], torque_limit: float, velocity_limit: float, band_deg: float, deadline_s: float) -> dict:
    values = torque_metrics(trace, torque_limit, band_deg)
    control = trace['control_update']
    predicted = trace['predicted_velocity_max_abs_rad_s'][control]
    predicted = predicted[np.isfinite(predicted)]
    solve_times = trace['qp_solve_time_s'][control]
    solve_times = solve_times[np.isfinite(solve_times)]
    peak_true = float(np.max(np.abs(trace['dq_true_rad_s'])))
    values.update({
        'peak_true_velocity_rad_s': peak_true,
        'max_true_velocity_excess_rad_s': max(0.0, peak_true - velocity_limit),
        'true_velocity_violation_samples': int(np.count_nonzero(
            np.abs(trace['dq_true_rad_s']) > velocity_limit + 1e-6
        )),
        'peak_predicted_velocity_rad_s': float(np.max(predicted)) if predicted.size else None,
        'predicted_velocity_violation_events': int(np.count_nonzero(
            predicted > velocity_limit + 1e-5
        )) if predicted.size else None,
        'velocity_constraint_active_count': int(np.count_nonzero(
            trace['velocity_constraint_active'][control]
        )),
        'solver_deadline_miss_count': int(np.count_nonzero(solve_times > deadline_s)) if solve_times.size else None,
    })
    return values


def _plot(path: Path, traces: dict[str, dict[str, np.ndarray]], torque_limit: float, velocity_limit: float) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for case, trace in traces.items():
        label = 'LQR + torque clip' if case == 'lqr_clip' else 'MPC + torque/velocity limits'
        color = 'tab:blue' if case == 'lqr_clip' else 'tab:orange'
        t = trace['time_s']
        axes[0].plot(t, np.rad2deg(trace['q_true_rad']), label=label, color=color)
        axes[1].plot(t, trace['dq_true_rad_s'], label=label, color=color)
        axes[2].plot(t, trace['torque_applied_nm'], label=label, color=color)
    baseline = traces['lqr_clip']
    axes[0].plot(baseline['time_s'], np.rad2deg(baseline['q_ref_rad']), 'k--', label='Target')
    for bound in (-velocity_limit, velocity_limit):
        axes[1].axhline(bound, linestyle='--', linewidth=1, color='gray')
    for bound in (-torque_limit, torque_limit):
        axes[2].axhline(bound, linestyle='--', linewidth=1, color='gray')
    axes[0].set_ylabel('True position / deg')
    axes[1].set_ylabel('True velocity / rad/s')
    axes[2].set_ylabel('Applied torque / N m')
    axes[2].set_xlabel('Time / s')
    axes[0].set_title('Experiment D: predicting the velocity limit')
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _report(path: Path, metrics: dict[str, dict], config: dict) -> None:
    velocity_limit = float(config['experiment_d']['velocity_limit_rad_s'])
    torque_limit = float(config['experiment_c']['torque_limit_nm'])
    rows = []
    for case, values in metrics.items():
        settle = '未进入并保持' if values['settling_time_s'] is None else f"{values['settling_time_s']:.3f}"
        predicted = '—' if values['peak_predicted_velocity_rad_s'] is None else f"{values['peak_predicted_velocity_rad_s']:.6f}"
        predicted_violations = '—' if values['predicted_velocity_violation_events'] is None else str(values['predicted_velocity_violation_events'])
        deadline_misses = '—' if values['solver_deadline_miss_count'] is None else str(values['solver_deadline_miss_count'])
        timing = '—' if values['solve_time_mean_ms'] is None else (
            f"{values['solve_time_mean_ms']:.4f} / {values['solve_time_p95_ms']:.4f} / {values['solve_time_max_ms']:.4f}"
        )
        status = ', '.join(f'{key}: {count}' for key, count in values['solver_status_counts'].items())
        rows.append(
            f"| {case} | {values['tracking_rmse_rad']:.6f} | {settle} | "
            f"{values['peak_true_velocity_rad_s']:.6f} | {values['max_true_velocity_excess_rad_s']:.6f} | "
            f"{values['true_velocity_violation_samples']} | "
            f"{predicted} | {predicted_violations} | "
            f"{values['velocity_constraint_active_count']} | {values['peak_applied_torque_nm']:.4f} | "
            f"{status} | {timing} | {deadline_misses} |"
        )
    text = f'''# 第 12 课实验 D：预测速度约束

目标位置 {float(config['experiment_b']['target_deg']):g}°，两组保持相同的单轴关节模型、Q/R、KF、初始状态和编码器噪声。植物/KF 周期 {float(config['plant']['dt_s']) * 1e3:g} ms，控制周期 {float(config['mpc']['dt_s']) * 1e3:g} ms，MPC 预测 N={int(config['mpc']['horizon_steps'])} 步。两组转矩均限制为 ±{torque_limit:g} N·m；仅 MPC 在 QP 内要求未来每一步预测速度（含终点）满足 ±{velocity_limit:g} rad/s。当前估计速度 x0 已经发生，不作为未来约束。

真实速度来自仿真植物，只用于评估，不输入控制器。MPC 的预测速度来自模型与 KF 后验估计，两者不必严格相等。真实越界按全体 1 ms 样本统计，容差 1e-6 rad/s；预测越界按每次控制更新的整个规划轨迹统计，容差 1e-5 rad/s。稳定时间沿用实验 C 的目标 ±{float(config['experiment_c']['settling_band_deg']):g}° 保持判据。

| 工况 | 角度 RMSE / rad | 稳定时间 / s | 真实速度峰值 / rad/s | 最大真实超速 / rad/s | 真实速度越界样本 | 预测速度峰值 / rad/s | 预测越界次数 | 速度约束活跃次数 | 转矩峰值 / N·m | 求解状态 | 求解均值 / p95 / 最大值 (ms) | 求解内核超控制周期次数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
{chr(10).join(rows)}

真实速度越界样本数反映持续时间，不反映严重程度；应同时看“最大真实超速”。两组均可能越界，但 5 rad/s 与 1.54 rad/s 的工程含义显然不同。

MPC 求解失败时本教学仿真会明确报错并停止，不会继续执行旧计划。求解时间是 OSQP 报告的内核求解时间，不包含 Python 调度或硬件通信；超时按配置中的控制周期 {float(config['mpc']['dt_s']) * 1e3:g} ms 判断，仅统计内核求解，完整控制循环耗时可能更长。这是无扰动、线性模型的离线教学实验；预测约束满足不构成真实硬件的安全保证，真实速度若发生越界须独立分析估计误差、模型误差和安全监控。
'''
    path.write_text(text, encoding='utf-8')


def run_experiment_d(config_path: Path | str, output_root: Path | str | None = None, *, duration_s: float | None = None) -> dict:
    config = load_config(config_path)
    if duration_s is not None:
        config['experiment_b']['duration_s'] = duration_s
    design = build_closed_loop_design(config)
    torque_limit = float(config['experiment_c']['torque_limit_nm'])
    velocity_limit = float(config['experiment_d']['velocity_limit_rad_s'])
    if not np.isfinite(torque_limit) or torque_limit <= 0:
        raise ValueError('experiment_c.torque_limit_nm must be positive and finite.')
    if not np.isfinite(velocity_limit) or velocity_limit <= 0:
        raise ValueError('experiment_d.velocity_limit_rad_s must be positive and finite.')
    band_deg = float(config['experiment_c']['settling_band_deg'])
    if not np.isfinite(band_deg) or band_deg <= 0:
        raise ValueError('experiment_c.settling_band_deg must be positive and finite.')
    std = np.deg2rad(float(config['measurement']['position_noise_std_deg']))
    noise = np.random.default_rng(int(config['experiment_d']['seed'])).normal(
        0.0, std, design['sample_count']
    )
    traces = {
        'lqr_clip': simulate_closed_loop(
            config, 'lqr', measurement_noise=noise, torque_limit_nm=torque_limit,
        ),
        'mpc_velocity_limited': simulate_closed_loop(
            config, 'mpc', horizon=int(config['mpc']['horizon_steps']),
            measurement_noise=noise, torque_limit_nm=torque_limit,
            velocity_limit_rad_s=velocity_limit,
        ),
    }
    metrics = {
        case: _metrics(
            trace, torque_limit, velocity_limit, band_deg, float(config['mpc']['dt_s'])
        )
        for case, trace in traces.items()
    }
    root = Path(output_root) if output_root is not None else LESSON_DIR
    paths = {
        'csv': root / 'logs' / 'experiment_d.csv',
        'figure': root / 'figures' / 'velocity_constraint.png',
        'report': root / 'reports' / 'velocity_constrained_mpc.md',
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(paths['csv'], traces)
    _plot(paths['figure'], traces, torque_limit, velocity_limit)
    _report(paths['report'], metrics, config)
    return {'traces': traces, 'metrics': metrics, 'paths': paths}


if __name__ == '__main__':
    output = run_experiment_d(LESSON_DIR / 'config' / 'mpc.yaml')
    for case, values in output['metrics'].items():
        print(case, values)
    for name, path in output['paths'].items():
        print(f'{name}: {path}')
