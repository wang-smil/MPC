"""Compare external LQR clipping with internal MPC torque constraints."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .closed_loop import build_closed_loop_design, simulate_closed_loop
from .model_loader import load_config
from .run_experiment_b import _write_csv


LESSON_DIR = Path(__file__).resolve().parents[1]


def _settling_time(trace: dict[str, np.ndarray], band_deg: float) -> float | None:
    error_deg = np.abs(np.rad2deg(trace['q_true_rad'] - trace['q_ref_rad']))
    outside = np.flatnonzero(error_deg > band_deg)
    if outside.size == 0:
        return 0.0
    last_outside = int(outside[-1])
    if last_outside == len(error_deg) - 1:
        return None
    return float(trace['time_s'][last_outside + 1])


def _metrics(trace: dict[str, np.ndarray], limit: float, band_deg: float) -> dict:
    control = trace['control_update']
    times = trace['qp_solve_time_s'][control]
    times = times[np.isfinite(times)]
    return {
        'tracking_rmse_rad': float(np.sqrt(np.mean((trace['q_true_rad'] - trace['q_ref_rad']) ** 2))),
        'settling_time_s': _settling_time(trace, band_deg),
        'peak_requested_torque_nm': float(np.max(np.abs(trace['torque_request_nm']))),
        'peak_applied_torque_nm': float(np.max(np.abs(trace['torque_applied_nm']))),
        'constraint_violation_count': int(np.count_nonzero(np.abs(trace['torque_applied_nm'][control]) > limit + 1e-6)),
        'safety_clip_count': int(np.count_nonzero(trace['safety_clip_active'][control])),
        'torque_constraint_active_count': int(np.count_nonzero(trace['torque_constraint_active'][control])),
        'solver_status_counts': dict(Counter(str(s) for s in trace['qp_status'][control])),
        'solve_time_mean_ms': float(np.mean(times) * 1e3) if times.size else None,
        'solve_time_p95_ms': float(np.percentile(times, 95) * 1e3) if times.size else None,
        'solve_time_max_ms': float(np.max(times) * 1e3) if times.size else None,
    }


def _plot(path: Path, traces: dict[str, dict[str, np.ndarray]], limit: float) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    colors = {'lqr_clip': 'tab:blue', 'mpc_constrained': 'tab:orange'}
    for case, trace in traces.items():
        t = trace['time_s']
        color = colors[case]
        label = 'LQR + clip' if case == 'lqr_clip' else 'Constrained MPC'
        axes[0].plot(t, np.rad2deg(trace['q_true_rad']), color=color, label=label)
        axes[1].plot(t, trace['dq_true_rad_s'], color=color, label=label)
        axes[2].plot(t, trace['torque_applied_nm'], color=color, label=f'{label}: applied')
        axes[2].plot(t, trace['torque_request_nm'], color=color, linestyle=':', alpha=0.65, label=f'{label}: requested')
    baseline = traces['lqr_clip']
    axes[0].plot(baseline['time_s'], np.rad2deg(baseline['q_ref_rad']), 'k--', label='Target')
    axes[2].axhline(limit, color='gray', linestyle='--', linewidth=1)
    axes[2].axhline(-limit, color='gray', linestyle='--', linewidth=1)
    axes[0].set_ylabel('True position / deg')
    axes[1].set_ylabel('True velocity / rad/s')
    axes[2].set_ylabel('Torque / N m')
    axes[2].set_xlabel('Time / s')
    axes[0].set_title('Experiment C: LQR + clip vs torque-constrained MPC')
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(loc='best', ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _report(path: Path, metrics: dict[str, dict], limit: float, band_deg: float, config: dict) -> None:
    rows = []
    for name, value in metrics.items():
        settling = '未进入并保持' if value['settling_time_s'] is None else f"{value['settling_time_s']:.3f}"
        solve = '—' if value['solve_time_mean_ms'] is None else (
            f"{value['solve_time_mean_ms']:.4f} / {value['solve_time_p95_ms']:.4f} / {value['solve_time_max_ms']:.4f}"
        )
        rows.append(
            f"| {name} | {value['tracking_rmse_rad']:.6f} | {settling} | "
            f"{value['peak_requested_torque_nm']:.4f} | {value['peak_applied_torque_nm']:.4f} | "
            f"{value['constraint_violation_count']} | {value['safety_clip_count']} | "
            f"{value['torque_constraint_active_count']} | {solve} |"
        )
    text = f'''# 第 12 课实验 C：LQR 外部限幅与 MPC 内部转矩约束

两组使用相同模型、Q/R、{float(config['experiment_b']['target_deg']):g}° 目标、初始状态和编码器噪声；植物/KF 每 {float(config['plant']['dt_s']) * 1e3:g} ms 更新，控制器每 {float(config['mpc']['dt_s']) * 1e3:g} ms 更新。MPC 预测 N={int(config['mpc']['horizon_steps'])} 步。真实状态仅用于仿真和评价，两个控制器都只接收 KF 的估计状态。

实际执行器教学限制为 [{-limit:g}, {limit:g}] N·m。LQR 先计算无约束请求，再由外部安全层截断；MPC 在 QP 内对未来每个转矩施加上下界，输出后仍经过独立安全限幅。设计 Q/R 时的 {float(config['design_limits']['max_torque_nm']):g} N·m 尺度保持不变，它不是这次的物理限值。

稳定时间定义为真实角度首次进入目标 ±{band_deg:g}° 后，直到仿真末尾都不再出界的时刻；若未满足则报告“未进入并保持”。约束违反次数只统计控制更新时刻实际转矩超过 ±{limit:g} N·m 的情况。

| 工况 | 角度跟踪 RMSE / rad | 稳定时间 / s | 请求峰值 / N·m | 实施峰值 / N·m | 约束违反次数 | 实质安全限幅次数 | MPC 约束活跃次数 | 求解均值 / p95 / 最大值 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

“MPC 约束活跃”表示该次规划的转矩序列至少一项接近边界；不等于外部安全层触发。“实质安全限幅”只统计请求与实际转矩相差超过 1e-5 N·m 的控制时刻；浮点求解可能造成更小的边界修正。求解时间仅为 OSQP 报告的求解时间，不含 Python 调度与硬件通信。这是离线、单输入、仅转矩约束的教学仿真；不预设 MPC 一定明显优于 LQR + clip，也不能据此宣称真实舵机安全。
'''
    path.write_text(text, encoding='utf-8')


def run_experiment_c(config_path: Path | str, output_root: Path | str | None = None, *, duration_s: float | None = None) -> dict:
    config = load_config(config_path)
    if duration_s is not None:
        config['experiment_b']['duration_s'] = duration_s
    design = build_closed_loop_design(config)
    settings = config['experiment_c']
    limit = float(settings['torque_limit_nm'])
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError('experiment_c.torque_limit_nm must be positive and finite.')
    band_deg = float(settings['settling_band_deg'])
    if not np.isfinite(band_deg) or band_deg <= 0:
        raise ValueError('experiment_c.settling_band_deg must be positive and finite.')
    std = np.deg2rad(float(config['measurement']['position_noise_std_deg']))
    noise = np.random.default_rng(int(settings['seed'])).normal(0.0, std, design['sample_count'])
    traces = {
        'lqr_clip': simulate_closed_loop(config, 'lqr', measurement_noise=noise, torque_limit_nm=limit),
        'mpc_constrained': simulate_closed_loop(
            config, 'mpc', horizon=int(config['mpc']['horizon_steps']),
            measurement_noise=noise, torque_limit_nm=limit,
        ),
    }
    metrics = {name: _metrics(trace, limit, band_deg) for name, trace in traces.items()}
    root = Path(output_root) if output_root is not None else LESSON_DIR
    paths = {
        'csv': root / 'logs' / 'experiment_c.csv',
        'figure': root / 'figures' / 'constrained_tracking.png',
        'report': root / 'reports' / 'torque_constrained_mpc.md',
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(paths['csv'], traces)
    _plot(paths['figure'], traces, limit)
    _report(paths['report'], metrics, limit, band_deg, config)
    return {'traces': traces, 'metrics': metrics, 'paths': paths}


if __name__ == '__main__':
    output = run_experiment_c(LESSON_DIR / 'config' / 'mpc.yaml')
    for name, values in output['metrics'].items():
        print(name, values)
    for name, path in output['paths'].items():
        print(f'{name}: {path}')
