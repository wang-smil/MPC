"""C: pure-QP infeasibility diagnosis; D/E: velocity softening on the nonideal joint."""

from copy import deepcopy
import csv
import json
from numbers import Integral
import platform
from pathlib import Path

import cvxpy
import numpy as np
import osqp

from ._validation import bounded_scalar, finite_scalar, integer_steps
from .closed_loop import simulate_closed_loop
from .feasibility_metrics import calculate_feasibility_metrics
from .model_loader import build_design
from .mpc_controller import EngineeringMPCController


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_json_value(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def _diagnose(config, velocity_limit):
    design = build_design(config)
    actuator = config['actuator']
    bound = np.deg2rad(config['constraints']['position_limit_deg'])
    delta = actuator['slew_rate_nm_s'] * config['mpc']['dt_s']
    initial = [0., config['initial_estimate']['velocity_rad_s']]
    reference = [np.deg2rad(config['experiment_ab']['target_deg']), 0.]
    diagnostic = {'kind': 'Pure QP diagnostic; no physical trajectory for C', 'x0': initial,
                  'x_ref': reference, 'u_prev': 0., 'velocity_limit_rad_s': velocity_limit,
                  'known_current_conflict_rad_s': abs(initial[1]) - velocity_limit,
                  'velocity_indices': list(range(config['mpc']['horizon_steps'] + 1))}
    for mode in ('hard', 'soft'):
        controller = EngineeringMPCController(design['Ad'], design['Bd'], design['Q'], design['R'],
                                              design['P'], config['mpc']['horizon_steps'], input_rate_weight=.5,
                                              torque_bounds=(-actuator['torque_limit_nm'], actuator['torque_limit_nm']),
                                              delta_bounds=(-delta, delta), position_bounds=(-bound, bound),
                                              velocity_mode=mode, velocity_bounds=(-velocity_limit, velocity_limit),
                                              include_current_velocity=True, slack_weight=1e4)
        diagnostic[mode] = controller.solve(initial, reference, 0.)
        diagnostic[mode]['objective_scale'] = controller.objective_scale
    return diagnostic


def _save_case(root, name, result, horizon):
    csv_path, plan_path = f'logs/experiment_{name}.csv', f'logs/experiment_{name}_plans.npz'
    log, plans = result['log'], result['plans']
    with (root / csv_path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(log)
        writer.writerows(zip(*log.values()))
    shapes = {'X': (2, horizon + 1), 'U': (1, horizon), 'DU': (1, horizon), 'EPS': (1, horizon + 1)}
    arrays = {field: (np.stack([plan[field] for plan in plans]) if plans else np.empty((0, *shape)))
              for field, shape in shapes.items()}
    np.savez_compressed(root / plan_path, time_s=np.array([plan['time_s'] for plan in plans]),
                        u_prev=np.array([plan['u_prev'] for plan in plans]),
                        velocity_indices=np.arange(horizon + 1), **arrays)
    return csv_path, plan_path


def _plot_results(results, summary, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 10})
    limit = summary['velocity_limit_rad_s']
    duration = summary['config']['experiment_cde']['duration_s']
    nominal = results['d_soft']
    log = nominal['log']
    mask = log['control_update'].astype(bool)
    fig, axes = plt.subplots(4, 2, figsize=(14, 11), layout='constrained')
    for col in range(2):
        ax = axes[:, col]
        ax[0].plot(log['time_s'], np.rad2deg(log['q_true_rad']), color='#2171b5', label='Actual position')
        ax[0].axhline(summary['config']['experiment_cde']['target_deg'], color='#374151', ls=':', label='Target')
        ax[1].plot(log['time_s'], log['dq_true_rad_s'], color='#2171b5', label='Actual velocity')
        ax[1].plot(log['time_s'], log['dq_hat_rad_s'], color='#d18a00', ls='--', label='KF velocity')
        for bound in (-limit, limit):
            ax[1].axhline(bound, color='#374151', ls=':')
        for field, label, color, style in (
            ('torque_request_nm', 'Request', '#2171b5', ':'),
            ('torque_command_nm', 'Guarded command', '#d18a00', '--'),
            ('torque_actual_nm', 'Actual', '#8c548c', '-')):
            ax[2].step(log['time_s'], log[field], where='post', color=color, ls=style, label=label)
        ax[3].step(log['time_s'][mask], log['qp_initial_slack_rad_s'][mask], where='post',
                   color='#2171b5', label='Initial-state slack')
        ax[3].step(log['time_s'][mask], log['max_predicted_slack_rad_s'][mask], where='post',
                   color='#d18a00', ls='--', label='Max horizon slack')
        ax[0].set_title('Full requested record' if col == 0 else 'Startup detail (0-0.6 s)')
        for row, ylabel in enumerate(('Position / deg', 'Velocity / rad/s', 'Torque / N m', 'QP slack / rad/s')):
            ax[row].set_ylabel(ylabel)
            ax[row].set_xlim(0, duration if col == 0 else min(.6, duration))
            ax[row].grid(alpha=.2)
            ax[row].spines[['top', 'right']].set_visible(False)
            ax[row].legend(loc='best', fontsize=9)
            if nominal['status'] != 'complete':
                ax[row].axvline(nominal['final_state_time_s'], color='#9c3c48', ls='--')
        ax[3].set_xlabel('Time / s')
    fig.suptitle(f"D: Soft velocity constraint, rho=10000 | {nominal['status']}\n"
                 f"C is a pure-QP diagnosis: {summary['diagnostic']['hard']['status']} (no hard-case recovery curve)")
    fig.savefig(root / 'figures/velocity_softening.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), layout='constrained')
    colors, styles = ['#2171b5', '#d18a00', '#8c548c'], ['-', '--', '-.']
    names = [name for name in results if name.startswith('e_')]
    for number, name in enumerate(names):
        result, case = results[name], summary['cases'][name]
        data = result['log']
        events = data['control_update'].astype(bool)
        label = f"rho={case['rho']:g} | {case['status']}"
        for col in range(2):
            axes[0, col].plot(data['time_s'], data['dq_true_rad_s'], color=colors[number], ls=styles[number], label=label)
            axes[1, col].step(data['time_s'][events], data['qp_initial_slack_rad_s'][events], where='post',
                              color=colors[number], ls=styles[number])
        axes[2, 0].plot(data['time_s'], np.rad2deg(data['q_true_rad']), color=colors[number], ls=styles[number])
        wall = data['solver_wall_time_s'][events] * 1000
        valid = np.isfinite(wall)
        axes[2, 1].plot(data['time_s'][events][valid], wall[valid], color=colors[number], ls=styles[number], marker='.', ms=2)
        if result['status'] != 'complete':
            for ax in axes.flat:
                ax.axvline(result['final_state_time_s'], color=colors[number], ls=':', alpha=.7)
    for col in range(2):
        axes[0, col].axhline(limit, color='#374151', ls=':')
        axes[0, col].axhline(-limit, color='#374151', ls=':')
        axes[0, col].set_title('Full requested record' if col == 0 else 'Startup detail (0-0.6 s)')
        axes[0, col].legend(fontsize=8, loc='best')
    labels = [['Velocity / rad/s', 'Velocity / rad/s'], ['QP initial slack / rad/s', 'QP initial slack / rad/s'],
              ['Position / deg', 'Solver wall time / ms']]
    for row in range(3):
        for col in range(2):
            ax = axes[row, col]
            # Timing includes the cold first solve; keep it and the full timing record visible.
            ax.set_xlim(0, duration if col == 0 or row == 2 else min(.6, duration))
            ax.set_ylabel(labels[row][col])
            ax.grid(alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
            if row == 2:
                ax.set_xlabel('Time / s')
    axes[2, 1].set_title('All solver calls, including cold start')
    fig.suptitle('E: Slack-penalty comparison | same noise, plant, controller weights and protection')
    fig.savefig(root / 'figures/slack_penalty_comparison.png', dpi=160)
    plt.close(fig)


def _write_report(summary, root):
    diagnostic = summary['diagnostic']
    eps = diagnostic['soft']['EPS']
    config = summary['cases']['d_soft']['effective_config']
    dt_ms = config['plant']['dt_s'] * 1000
    control_ms = config['mpc']['dt_s'] * 1000
    actuator = config['actuator']
    speed0 = diagnostic['x0'][1]
    limit = summary['velocity_limit_rad_s']
    required_slack = diagnostic['known_current_conflict_rad_s']
    rows = ['# 第13课C/D/E：不可行、速度软约束与惩罚权重', '',
            '本报告由真实仿真运行生成，不是实机测量。F/G超时降级与fallback尚未实现。', '',
            '## C：为什么硬约束不可行', '',
            f'本次C/D/E显式约束x0..xN。初始速度已知为{speed0:g} rad/s，却要求当前速度绝对值不超过{limit:g} rad/s；不能靠未来扭矩改变已经发生的初值。',
            f"硬组原始状态：{diagnostic['hard']['status']}；纯QP诊断，不存在可执行的C闭环恢复轨迹。完整返回值见../logs/feasibility_diagnostic.json。", '',
            '## D：松弛量怎样进入代码', '',
            '`lower - EPS <= X[1, indices] <= upper + EPS; EPS >= 0`', '',
            '`cost = original_cost + S*sum(DU**2) + rho*sum(EPS**2)`', '',
            '软化的是速度边界，不是扭矩、变化量或位置。EPS单位rad/s，rho表示每单位速度松弛平方的代价尺度，与S的扭矩变化量平方尺度不同；数值大小不能直接比较。',
            f"同一静态输入soft状态：{diagnostic['soft']['status']}；首列EPS0={eps[0][0] if eps is not None else 'unavailable'} rad/s（由初值至少需要{required_slack:g}）。",
            'mpc_controller.py构造期创建EPS并追加不等式；solve复核索引、非负性、软边界及全部硬边界。EPS不是从真实速度中减去的修正量。',
            '为改善数值尺度，整个soft目标除以max(1,rho)，不改变相对权重或数学最优解；导出objective恢复原单位。容差仍为1e-5，不靠放宽验收制造成功。', '',
            '## 实验条件与信息边界', '',
            '|参数|本次有效值|', '|---|---|',
            f'|物理/KF周期|{dt_ms:g} ms|', f'|MPC周期|{control_ms:g} ms|',
            f"|预测步数|{config['mpc']['horizon_steps']}|",
            f"|扭矩幅值|±{actuator['torque_limit_nm']:g} N·m|",
            f"|位置边界|±{config['constraints']['position_limit_deg']:g}°|",
            f'|速度边界|±{limit:g} rad/s|', f'|初始速度|{speed0:g} rad/s|',
            f"|实际变化率上限|{actuator['slew_rate_nm_s']:g} N·m/s|",
            f"|QP变化量边界|±{actuator['slew_rate_nm_s'] * config['mpc']['dt_s']:g} N·m|",
            f"|执行器延迟/时间常数|{actuator['delay_s'] * 1000:g}/{actuator['time_constant_s'] * 1000:g} ms|",
            f"|库仑摩擦|{config['plant']['coulomb_friction']:g} N·m|", '',
            '同一预生成编码器噪声，KF只使用位置测量及过去区间实际扭矩；预测为两状态线性模型。摩擦和执行器动态未进入预测模型。扭矩反馈仍是假设准确同步的驱动器反馈。',
            '位置独立越界监测启用，主实验不额外注入随机扰动力矩。Q/R/P及S=.5固定，D的rho=1e4，E只改rho。配置/版本/终态和原始CSV/NPZ均保存在logs。', '',
            '## 实测指标', '',
            '|Case|Status|覆盖/s|位置RMSE/rad|真实超速峰值/(rad/s)|真实超速积分/rad|QP初始slack持有积分/rad|平均迭代|墙钟p95/ms|失败次数|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, case in summary['cases'].items():
        metrics = case['metrics']
        def fmt(field):
            value = metrics[field]
            return 'unavailable' if value is None else f'{value:.6g}'
        rows.append(f"|{name}|{case['status']}|{case['final_state']['time_s']:.6g}|{fmt('tracking_rmse_rad')}|"
                    f"{fmt('max_actual_velocity_violation_rad_s')}|{fmt('actual_overspeed_integral_rad')}|"
                    f"{fmt('initial_slack_hold_integral_rad')}|{fmt('mean_solver_iterations')}|"
                    f"{fmt('p95_solver_wall_ms')}|{metrics['solver_failure_count']}|")
        if case['failure_reason']:
            rows += ['', f"{name}仅有部分记录：{case['failure_reason']}；其RMSE不能与完整仿真直接比较。", '']
    e_cases = sorted((case for name, case in summary['cases'].items() if name.startswith('e_')), key=lambda case: case['rho'])
    if all(case['status'] == 'complete' for case in e_cases):
        middle, highest = e_cases[1:]
        before, after = middle['metrics'], highest['metrics']
        rows += ['', '## 本次权重取舍', '',
                 f"rho从{middle['rho']:g}增至{highest['rho']:g}，真实超速积分从{before['actual_overspeed_integral_rad']:.8g}变为{after['actual_overspeed_integral_rad']:.8g} rad；位置RMSE从{before['tracking_rmse_rad']:.8g}变为{after['tracking_rmse_rad']:.8g} rad；平均迭代从{before['mean_solver_iterations']:.6g}变为{after['mean_solver_iterations']:.6g}。这些数值仅对应本次配置，不代表所有工况。",
                 'rho惩罚的是每次QP整个预测时域的slack平方和；表中的初始slack持有积分是闭环过程中另一种线性统计量，初值/轨迹也随控制变化，所以它不必随rho单调下降。计时未隔离主机负载，不能把墙钟差异直接归因于rho，更不是F的受控性能基准。']
    rows += ['', '## 图与结论怎样看', '', '![D](../figures/velocity_softening.png)', '',
             'D左列保留全程，右列放大启动段。速度实线是真值、虚线是KF；扭矩分别是请求、保护后命令和实际输出。slack只在控制事件记录，不等于真实超速。', '',
             '![E](../figures/slack_penalty_comparison.png)', '',
             'E前三组使用同一噪声，仅rho变化。速度/slack分别显示全程和启动段；最后一行显示位置和完整求解墙钟记录（包含首轮编译开销），平均迭代见表。失败轨迹在停止时刻截断，不填补成恢复曲线。',
             f'真实超速积分按已施加输入的{dt_ms:g}ms物理区间计算；QP初始slack持有积分按控制事件持有区间计算，并在末尾/失败时截断。最大预测slack及其完整计划见NPZ。最终状态单独保存，不额外增加积分区间。',
             f'当前速度初始{speed0:g}、边界±{limit:g}，所以各组起点的真实超速至少{required_slack:g}；不能仅比较这个峰值就判断rho有效性。应继续看超速持续时间/积分、slack、跟踪与数值求解表现。',
             '有限二次惩罚可能在可行时仍使用少量slack，也不保证最终严格为零。rho更大并不保证每项真实指标单调改善：估计误差、未建模执行器和数值条件都会影响结果。',
             'D与E名义rho组重复运行用于一致性校验，不是独立随机重复实验。结果仅对应当前种子/模型；不构成实机速度安全或硬实时保证。', '',
             '## 阅读代码顺序', '',
             '先看mpc_controller.py的velocity_indices/EPS构造，再看closed_loop.py的控制事件slack记录，最后看feasibility_metrics.py为何区分两种积分。run_feasibility.py负责公平分组、导出和画图。', '',
             '后续F/G才处理预算、fallback和恢复；软化速度不能修复其他硬约束冲突。']
    (root / 'reports/feasibility_report.md').write_text('\n'.join(rows) + '\n', encoding='utf-8')


def run_feasibility_experiments(config: dict, output_dir: Path) -> dict:
    root = Path(output_dir)
    for directory in ('logs', 'figures', 'reports'):
        (root / directory).mkdir(parents=True, exist_ok=True)
    settings = config['experiment_cde']
    effective = deepcopy(config)
    dt = bounded_scalar(config['plant']['dt_s'], 'plant.dt_s', positive=True)
    duration = bounded_scalar(settings['duration_s'], 'duration_s', positive=True)
    count = integer_steps(duration, dt, 'duration_s')
    if count < 1:
        raise ValueError('duration_s must contain at least one physical step.')
    limit = bounded_scalar(config['constraints']['velocity_limit_rad_s'], 'velocity_limit_rad_s', positive=True)
    initial_velocity = finite_scalar(settings['initial_velocity_rad_s'], 'initial_velocity_rad_s')
    if abs(initial_velocity) <= limit:
        raise ValueError('C/D initial speed must exceed the velocity boundary.')
    seed = settings['seed']
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')
    weights = [bounded_scalar(value, 'slack_weight', positive=True) for value in settings['slack_weights']]
    if len(weights) != 3 or len(set(weights)) != 3:
        raise ValueError('E requires three distinct positive slack weights.')
    effective['experiment_ab'].update(duration_s=duration, target_deg=finite_scalar(settings['target_deg'], 'target_deg'), seed=seed)
    effective['initial_condition'] = {'q_true_deg': 0., 'dq_true_rad_s': initial_velocity}
    effective['initial_estimate'] = {'position_deg': 0., 'velocity_rad_s': initial_velocity}
    std = bounded_scalar(config['measurement']['position_noise_std_deg'], 'position_noise_std_deg', positive=True)
    noise = np.random.default_rng(seed).normal(0., np.deg2rad(std), count)
    diagnostic = _diagnose(effective, limit)
    _write_json(root / 'logs/feasibility_diagnostic.json', diagnostic)
    summary = {'config': deepcopy(config), 'diagnostic': _json_value(diagnostic), 'velocity_limit_rad_s': limit,
               'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                            'cvxpy': cvxpy.__version__, 'osqp': osqp.__version__},
               'scope': 'C/D/E only; C has no physical trajectory. No fallback, deadline or hardware-safety guarantee.',
               'cases': {}}
    # Float's round-trip representation keeps nearby valid rho values distinct.
    cases = [('d_soft', 1e4)] + [('e_rho' + ('1e6' if weight == 1e6 else str(weight).removesuffix('.0')), weight)
                                for weight in weights]
    results = {}
    for name, rho in cases:
        result = simulate_closed_loop(effective, input_rate_weight=.5, hard_rate=True, measurement_noise=noise,
                                      velocity_mode='soft', velocity_bounds=(-limit, limit),
                                      include_current_velocity=True, slack_weight=rho)
        results[name] = result
        csv_path, plan_path = _save_case(root, name, result, config['mpc']['horizon_steps'])
        summary['cases'][name] = {'rho': rho, 'S': .5, 'velocity_mode': 'soft', 'include_current_velocity': True,
                                  'effective_config': deepcopy(effective), 'status': result['status'],
                                  'failure_reason': result['failure_reason'], 'expected_samples': result['expected_samples'],
                                  'final_state': {'time_s': result['final_state_time_s'],
                                                  'position_rad': float(result['final_state'][0]),
                                                  'velocity_rad_s': float(result['final_state'][1])},
                                  'csv': csv_path, 'plans': plan_path,
                                  'metrics': calculate_feasibility_metrics(result, config['mpc']['dt_s'], dt, limit)}
    _write_json(root / 'logs/experiment_cde_summary.json', summary)
    _plot_results(results, summary, root)
    _write_report(summary, root)
    return summary
