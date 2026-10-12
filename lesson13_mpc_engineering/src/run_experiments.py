"""Run Lesson 13 A/B: planning smoothness versus external actuator protection."""

import argparse
import csv
import json
import platform
from pathlib import Path

import cvxpy
import numpy as np
import osqp

from lesson12_linear_mpc.src.model_loader import load_config
from ._validation import bounded_scalar, integer_steps
from .closed_loop import simulate_closed_loop
from .metrics import calculate_metrics


def _plots(results, config, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10})
    palette = ['#2171b5', '#d18a00', '#8c548c']
    styles = ['-', '--', '-.']
    for group, names, title, filename in (
        ('a', [name for name in results if name.startswith('a_')],
         'A: Input-move penalty (same nonideal actuator and protection)', 'input_move_penalty.png'),
        ('b', [name for name in results if name.startswith('b_')],
         'B: Hard rate planning versus external protection only', 'input_rate_constraint.png')):
        # Columns retain both the full record and the startup detail: no hidden late behavior.
        fig, axes = plt.subplots(4, 2, figsize=(14, 11), layout='constrained')
        for number, name in enumerate(names):
            log = results[name]['log']
            mask = log['control_update'].astype(bool)
            color, style = palette[number], styles[number]
            label = name.replace('a_s', 'S=').replace('p', '.') if group == 'a' else name.replace('b_', '')
            for column in range(2):
                ax = axes[:, column]
                ax[0].plot(log['time_s'], np.rad2deg(log['q_true_rad']), color=color, ls=style, label=label)
                if group == 'a':
                    ax[1].step(log['time_s'], log['torque_request_nm'], where='post', color=color, ls=style)
                    ax[2].step(log['time_s'], log['torque_actual_nm'], where='post', color=color, ls=style)
                else:
                    ax[1].plot(log['time_s'], log['dq_true_rad_s'], color=color, ls=style)
                    ax[2].step(log['time_s'], log['torque_request_nm'], where='post', color=color, ls=':', alpha=.8)
                    ax[2].step(log['time_s'], log['torque_actual_nm'], where='post', color=color, ls=style)
                ax[3].step(log['time_s'][mask], log['request_delta_nm'][mask], where='post', color=color, ls=style)
                if group == 'b':
                    ax[3].plot(log['time_s'][mask], log['qp_first_delta_nm'][mask], color=color,
                               ls=':', alpha=.8)
        last_time = max(float(result['log']['time_s'][-1]) for result in results.values())
        target = config['experiment_ab']['target_deg']
        delta = config['actuator']['slew_rate_nm_s'] * config['mpc']['dt_s']
        for column in range(2):
            axes[0, column].axhline(target, color='#374151', ls=':', linewidth=1)
            axes[0, column].legend(loc='best')
            axes[0, column].set_title('Full record' if column == 0 else 'Startup detail (0-0.6 s)')
            labels = ['Position / deg', 'Requested torque / N m', 'Actual torque / N m', 'Request delta / N m']
            if group == 'b':
                labels[1:3] = ['Velocity / rad/s', 'Torque / N m']
                labels[3] = 'Input move / N m'
            for row, label in enumerate(labels):
                axes[row, column].set_ylabel(label)
                axes[row, column].grid(alpha=.2)
                axes[row, column].spines[['top', 'right']].set_visible(False)
                axes[row, column].set_xlim(0, last_time if column == 0 else min(.6, last_time))
            axes[3, column].set_xlabel('Time / s')
            if group == 'b':
                for bound in (-delta, delta):
                    axes[3, column].axhline(bound, color='#374151', ls='--', linewidth=1)
        fig.suptitle(title + ('\nB torque: dotted=request; other=actual. B moves: dotted=request-feedback; other=request-request_prev.' if group == 'b' else ''))
        fig.savefig(directory / filename, dpi=160)
        plt.close(fig)


def _write_report(summary, root):
    rows = ['# 第13课实验A/B：输入变化量', '',
            '本文件由运行入口按实际结果生成。不是实机测量；软约束、预算降级和综合验收尚未实现。', '',
            '## 设置与公式', '',
            '`J = sum(e.T Q e + u.T R u + S * delta_u²) + e_N.T P e_N`；',
            '`delta_u[0] = U[0] - previous_actual_feedback`，后续为相邻规划输入之差。',
            'S是代价，不能保证硬边界；B硬约束为每10 ms最多0.5 N·m（默认配置）。', '',
            '同一名义模型、KF、参考、噪声实现、执行器和外部保护。Q/R沿用第12课Bryson尺度，P是旧课固定终端权重，未声称是变化量增广系统的最优终端权重。',
            '对象有库仑摩擦；执行器有延迟、滞后和真实变化率限制，预测仍是两状态线性模型，存在显式失配。扭矩反馈暂按无噪声同步驱动器反馈假设。',
            '速度约束在本阶段关闭，位置未来硬边界与真实越界监测启用；不宣称真实位置安全保证。', '',
            '完整配置和库版本见 `../logs/experiment_ab_summary.json`，每组CSV和计划NPZ可复核。', '',
            '## 实测结果', '',
            '| Case | Status | Position RMSE/rad | Actual torque RMS/Nm | Request delta RMS/Nm | Request delta max/Nm | QP first delta max/Nm | Guard changes |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for name, case in summary['cases'].items():
        m = case['metrics']
        def fmt(key):
            value = m[key]
            return f'{value:.6f}' if value is not None else 'unavailable'
        rows.append(f"| {name} | {case['status']} | {fmt('tracking_rmse_rad')} | {fmt('rms_actual_torque_nm')} | {fmt('rms_request_delta_nm')} | {fmt('max_request_delta_nm')} | {fmt('max_qp_first_delta_nm')} | {m['guard_modified_count']} |")
        if case['failure_reason']:
            rows.append(f"\n{name} stopped: {case['failure_reason']}\n")
    rows += ['', '## 本次数据的结论', '']
    for name, case in summary['cases'].items():
        m = case['metrics']
        if case['status'] == 'complete':
            rows.append(f"- {name}：位置RMSE={m['tracking_rmse_rad']:.6f} rad；请求变化量RMS={m['rms_request_delta_nm']:.6f} N·m；外部保护修正{m['guard_modified_count']}次。")
    rows += ['', '应同时比较误差与平滑度：增加S不是免费提升，平滑可能以跟踪性能为代价。硬约束组若保护修正更少，说明优化器更主动地考虑了输入变化边界；这不等于消除了执行器延迟或模型失配。', '', '## 图与指标怎样解读', '',
             '![A](../figures/input_move_penalty.png)', '', '![B](../figures/input_rate_constraint.png)', '',
             '图左列保留完整4 s（默认）记录，右列放大启动0–0.6 s。A依次显示位置、请求、实际、相邻请求差；B显示位置、速度、请求/实际、两种输入差。',
             'request_delta是相邻两次请求之差；QP首步delta是本次请求减当前反馈；有执行器滞后时两者不等。B的±0.5边界严格用于QP的DU，而不是机械套用到相邻请求差。',
             'RMS request_delta只统计控制事件，并包含首步相对初始输入；实际变化率按1 ms物理区间计算。所有组都有外部保护，实际平滑不能单独证明规划约束生效。',
             '比较A时同时看跟踪RMSE和请求delta，不能提前认定S增大后所有指标都单调改善。比较B时检查计划NPZ中全部DU和保护修正次数，而不只看实际扭矩。',
             '计时包括首轮编译开销；墙钟和求解器内部计时分列在JSON中，未实施超预算切换，因此不是实时保证。失败组保留部分日志且明确标记，不能当作完整实验。', '',
             '## 核心代码入口', '',
             '`mpc_controller.py`扩展旧课QP；`closed_loop.py`负责过去输入/KF/控制/执行器的因果时序；`constraint_manager.py`仅负责外部命令保护；`metrics.py`区分两种采样率。', '',
             '后续C/D/E研究软约束和不可行；F/G才实现计算预算与fallback。']
    (root / 'reports').mkdir(parents=True, exist_ok=True)
    (root / 'reports/input_move_report.md').write_text('\n'.join(rows) + '\n', encoding='utf-8')


def run_experiments(config: dict, output_dir: Path) -> dict:
    root = Path(output_dir)
    for name in ('logs', 'figures'):
        (root / name).mkdir(parents=True, exist_ok=True)
    dt = bounded_scalar(config['plant']['dt_s'], 'plant.dt_s', positive=True)
    duration = bounded_scalar(config['experiment_ab']['duration_s'], 'duration_s', positive=True)
    count = integer_steps(duration, dt, 'duration_s')
    std = bounded_scalar(config['measurement']['position_noise_std_deg'], 'position_noise_std_deg', positive=True)
    noise = np.random.default_rng(config['experiment_ab']['seed']).normal(0., np.deg2rad(std), count)
    settings = [(f'a_s{float(weight):g}'.replace('.', 'p'), weight, False)
                for weight in config['experiment_ab']['input_rate_weights']]
    settings += [('b_no_rate', .5, False), ('b_hard_rate', .5, True)]
    summary = {'config': config, 'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                                            'cvxpy': cvxpy.__version__, 'osqp': osqp.__version__},
               'scope': 'Phase2 A/B only; no velocity constraint, budget fallback or hardware safety guarantee',
               'cases': {}}
    results = {}
    for name, weight, hard_rate in settings:
        result = simulate_closed_loop(config, input_rate_weight=weight, hard_rate=hard_rate, measurement_noise=noise)
        results[name] = result
        log = result['log']
        csv_relative = f'logs/experiment_{name}.csv'
        with (root / csv_relative).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(log.keys())
            writer.writerows(zip(*log.values()))
        plan_relative = f'logs/experiment_{name}_plans.npz'
        plans = result['plans']
        np.savez_compressed(root / plan_relative,
                            time_s=np.array([plan['time_s'] for plan in plans]),
                            u_prev=np.array([plan['u_prev'] for plan in plans]),
                            **{field: np.array([plan[field] for plan in plans]) for field in ('X', 'U', 'DU')})
        summary['cases'][name] = {'S': float(weight), 'hard_rate': hard_rate, 'status': result['status'],
                                  'failure_reason': result['failure_reason'], 'csv': csv_relative,
                                  'plans': plan_relative, 'expected_samples': result['expected_samples'],
                                  'metrics': calculate_metrics(log, config['mpc']['dt_s'])}
    (root / 'logs/experiment_ab_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                                    allow_nan=False), encoding='utf-8')
    _plots(results, config, root / 'figures')
    _write_report(summary, root)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['ab'], default='ab')
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    summary = run_experiments(load_config(args.config or root / 'config/mpc_engineering.yaml'), root)
    for name, case in summary['cases'].items():
        m = case['metrics']
        print(f"{name}: {case['status']}, tracking_rmse={m['tracking_rmse_rad']}, request_delta_rms={m['rms_request_delta_nm']}, guard_changes={m['guard_modified_count']}")
    print(f'report: {root / "reports/input_move_report.md"}')


if __name__ == '__main__':
    main()
