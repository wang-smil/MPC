"""Inspectable F/G exports and standalone Matplotlib figures (no timing here)."""

import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_json_value(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def _write_rows(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def _plots(summary, benchmark, results, root):
    colors = {'warm_on': '#2267A7', 'warm_off': '#C57920'}
    budget_ms = summary['config']['runtime']['budget_s'] * 1000
    deadline_ms = summary['config']['mpc']['dt_s'] * 1000
    rows = benchmark['rows']
    with plt.rc_context({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
        for mode, marker in (('warm_on', 'o'), ('warm_off', 'x')):
            selected = [r for r in rows if r['mode'] == mode]
            axes[0, 0].scatter([r['sample_index'] for r in selected], [r['wall_time_s'] * 1000 for r in selected],
                               s=13, alpha=.55, color=colors[mode], marker=marker, label=mode)
            steady = sorted(r['wall_time_s'] * 1000 for r in selected if not r['cold'])
            if steady:
                axes[1, 0].plot(steady, np.arange(1, len(steady) + 1) / len(steady) * 100,
                                 color=colors[mode], ls='-' if mode == 'warm_on' else '--', label=f'{mode} (n={len(steady)})')
        axes[0, 0].set(yscale='log', title='All calls, including cold start (log y)', xlabel='Frozen replay sample index', ylabel='Measured wall time / ms')
        axes[0, 0].legend()
        for ax in (axes[0, 0], axes[1, 0]):
            if ax is axes[0, 0]:
                ax.axhline(budget_ms, color='#383838', ls='--', label='Budget')
                ax.axhline(deadline_ms, color='#777777', ls=':', label='Control deadline')
            else:
                ax.axvline(budget_ms, color='#383838', ls='--', label=f'Budget {budget_ms:g} ms')
                ax.axvline(deadline_ms, color='#777777', ls=':', label=f'Deadline {deadline_ms:g} ms')
        axes[1, 0].set(title='Steady calls: empirical distribution', xlabel='Measured wall time / ms', ylabel='Calls at or below / %', xlim=(0, None), ylim=(0, 102))
        axes[1, 0].legend(fontsize=9)
        cold = [r for r in rows if r['cold']]
        axes[0, 1].bar(np.arange(len(cold)), [r['wall_time_s'] * 1000 for r in cold],
                       color=[colors[r['mode']] for r in cold],
                       hatch=None)
        for i, row in enumerate(cold):
            if row['mode'] == 'warm_off':
                axes[0, 1].patches[i].set_hatch('//')
        axes[0, 1].set_xticks(np.arange(len(cold)), [f"r{r['repetition']}\n{r['mode']}" for r in cold])
        axes[0, 1].set(title='First solve of each new Problem', ylabel='Measured wall time / ms', ylim=(0, None))
        statistics = ['mean', 'p95', 'p99', 'max']
        for i, mode in enumerate(('warm_on', 'warm_off')):
            metric = benchmark['metrics'][mode]['steady']
            values = [metric[f'wall_{stat}_ms'] for stat in statistics]
            if all(value is not None for value in values):
                axes[1, 1].bar(np.arange(4) + (i - .5) * .34, values, width=.34,
                               color=colors[mode], hatch=None if i == 0 else '//', label=mode)
        axes[1, 1].set_xticks(np.arange(4), statistics)
        axes[1, 1].set(title='Steady calls: pooled wall statistics', ylabel='Measured wall time / ms', ylim=(0, None))
        if axes[1, 1].patches:
            axes[1, 1].legend()
        else:
            axes[1, 1].text(.5, .5, 'No steady samples (cold calls only)', transform=axes[1, 1].transAxes, ha='center')
        for ax in axes.flat:
            ax.grid(alpha=.2)
        fig.suptitle(f"F: CVXPY + OSQP wall time | budget {budget_ms:g} ms, control deadline {deadline_ms:g} ms", fontsize=15)
        fig.savefig(root / 'figures/solver_runtime.png', dpi=160)
        plt.close(fig)

        names = ['g_nominal', 'g_iteration_limit', 'g_deadline', 'g_error_recovery']
        fig, axes = plt.subplots(4, 3, figsize=(16, 13), layout='constrained')
        position_top = max(33., max(float(np.nanmax(np.rad2deg(results[name]['log']['q_true_rad']))) for name in names) + 2.)
        for row_index, name in enumerate(names):
            result, case = results[name], summary['cases'][name]
            log = result['log']
            time = log['time_s']
            faults = summary['fault_scenarios'][name]
            start = max(0., min(map(int, faults)) * summary['config']['mpc']['dt_s'] - .01) if faults else 0.
            stop = min(summary['config']['experiment_fg']['duration_s'], start + .11)
            transitions = [event['time_s'] for event in result['events'] if event['mode_before'] != event['mode_after']]
            if transitions:
                stop = min(summary['config']['experiment_fg']['duration_s'], max(stop, max(transitions) + .01))
            axes[row_index, 0].plot(time, np.rad2deg(log['q_true_rad']), color='#2267A7', label='True position')
            axes[row_index, 0].plot(time, np.rad2deg(log['q_ref_rad']), color='#555555', ls='--', label='Reference')
            axes[row_index, 0].set(title=f'{name}: full trajectory', ylabel='Position / deg', xlim=(0, summary['config']['experiment_fg']['duration_s']), ylim=(-2, position_top))
            axes[row_index, 1].plot(time, log['torque_command_nm'], color='#C57920', ls='--', label='Guarded command')
            axes[row_index, 1].plot(time, log['torque_actual_nm'], color='#2267A7', label='Actual torque')
            limit = summary['config']['actuator']['torque_limit_nm']
            axes[row_index, 1].set(title='Fault/startup window: torque', ylabel='Torque / N m', xlim=(start, stop), ylim=(-limit * 1.1, limit * 1.1))
            mode = (log['mode'] == 'fallback').astype(int)
            axes[row_index, 2].step(time, mode, where='post', color='#2267A7', label='Mode')
            for event in result['events']:
                if event.get('recovered'):
                    axes[row_index, 2].scatter(event['time_s'], 0, marker='o', facecolors='white', edgecolors='#2267A7', zorder=4)
                    if start <= event['time_s'] <= stop:
                        axes[row_index, 2].annotate('3 good: recover', (event['time_s'], 0), xytext=(0, 12), textcoords='offset points', ha='center', fontsize=8)
                elif event['event'] == 'late_result_discarded':
                    axes[row_index, 2].scatter(event['time_s'], 1, marker='x', color='#C57920', zorder=4)
            for cycle in faults:
                axes[row_index, 2].axvline(int(cycle) * summary['config']['mpc']['dt_s'], color='#999999', ls=':', lw=1)
            axes[row_index, 2].set(title=f"Mode: {case['metrics']['fallback_entry_count']} fallback / {case['metrics']['recovery_count']} recovery", xlim=(start, stop), ylim=(-.3, 1.45))
            axes[row_index, 2].set_yticks([0, 1], ['MPC', 'Fallback'])
            for ax in axes[row_index]:
                ax.set_xlabel('Physical simulation time / s')
                ax.grid(alpha=.2)
            if row_index == 0:
                axes[row_index, 0].legend(fontsize=9)
                axes[row_index, 1].legend(fontsize=9)
        fig.suptitle('G: injected completion events (NOT measured solver latency)\nDotted lines: fault submissions | x: late discarded | o: recovery after 3 good results', fontsize=15)
        fig.savefig(root / 'figures/fallback_recovery.png', dpi=160)
        plt.close(fig)


def _number(value):
    return '—' if value is None else f'{value:.6g}'


def _report(summary, root):
    c = summary['config']
    lines = ['# 第13课F/G：求解预算、降级与恢复', '',
             '本报告由本次实际运行生成。F是真实Python墙钟测量；G是明确注入完成时刻的确定性事件仿真。二者不能混成一张“实机实时性能”表。', '',
             '## 1. 条件和边界', '',
             f"名义J={c['nominal_model']['inertia']} kg·m²，b={c['nominal_model']['damping']} N·m·s/rad；物理/KF周期{c['plant']['dt_s']*1000:g} ms，控制周期{c['mpc']['dt_s']*1000:g} ms，N={c['mpc']['horizon_steps']}，时长{c['experiment_fg']['duration_s']:g} s，目标{c['experiment_fg']['target_deg']:g}°，seed={c['experiment_fg']['seed']}。",
             f"实际和命令扭矩±{c['actuator']['torque_limit_nm']:g} N·m，变化率{c['actuator']['slew_rate_nm_s']:g} N·m/s；QP首步相对当前实际反馈、后续相对相邻规划输入，变化量±{c['actuator']['slew_rate_nm_s']*c['mpc']['dt_s']:g} N·m。位置未来硬边界±{c['constraints']['position_limit_deg']:g}°，速度未来x₁…x_N软边界±{c['constraints']['velocity_limit_rad_s']:g} rad/s；S={c['experiment_fg']['input_rate_weight']:g}，rho={c['experiment_fg']['slack_weight']:g}。",
             f"执行器延迟{c['actuator']['delay_s']*1000:g} ms、一阶滞后{c['actuator']['time_constant_s']*1000:g} ms；库仑摩擦{c['plant']['coulomb_friction']:g} N·m，平滑速度{c['plant']['friction_smoothing']:g} rad/s。编码器噪声标准差{c['measurement']['position_noise_std_deg']:g}°；过程力矩标准差{c['process_noise']['disturbance_torque_std_nm']:g}只用于KF协方差，本阶段没有额外随机外力。",
             f"求解预算{c['runtime']['budget_s']*1000:g} ms，与控制截止{c['mpc']['dt_s']*1000:g} ms不同。max_iter={c['solver']['max_iter']}，eps_abs=eps_rel=1e-8，独立残差容差1e-5；warm on/off均显式polishing=True。",
             f"版本：{json.dumps(summary['versions'], ensure_ascii=False)}。完整有效配置、故障表、统计见[summary](../logs/experiment_fg_summary.json)。", '',
             '## 2. 算法代码串起来', '',
             '```text\n编码器位置 → KF当前估计 → 参数化QP → 本轮结果/残差/时间监控\n                                    ↓\n                    有效且按时：MPC；否则：fresh LQR\n                                    ↓\n                 公共幅值/变化率保护 → 延迟/滞后执行器 → 关节\n```', '',
             '- `qp_solver.py`：可配置OSQP调用，保存本轮原始状态、迭代数和可取得的计时。失败不回传上轮旧解。A–E未选择新设置时仍走旧路径。',
             '- `solver_monitor.py`：先检验状态、有限值和残差，再判断时间；`optimal`不等于可执行。连续有效且按时的结果才推进恢复计数。',
             '- `fallback_controller.py`：固定既有LQR增益，`u_request = -K @ (x_hat-x_ref)`；每次决策重新计算，不偷看真实状态，不执行上次MPC的U。',
             '- `managed_closed_loop.py`：一个worker、一个在途作业；先处理旧完成，再预算，再新提交。超时不假装取消求解；worker仍忙时跳过新提交并重置恢复计数。',
             f"- 恰好{c['runtime']['budget_s']*1000:g} ms完成可接受，超过即丢弃；降级后连续{c['runtime']['recovery_successes']}次有效及时结果才恢复。命令保护按距上次更新的真实事件间隔Δt限制`|Δu_command|≤slew_rate*Δt`，不一律套10 ms。",
             '- `runtime_metrics.py`：每物理采样、每控制触发、每提交作业分母分开。`events.csv`保存同一毫秒的多次决策，避免物理CSV把事件覆盖。', '',
             '## 3. F：真实求解时间', '',
             f"冻结{c['experiment_fg']['replay_samples']}组(x_hat,x_ref,u_prev)，每个模式重复{c['experiment_fg']['repetitions']}轮；逐轮交替模式顺序，每个模式/轮只构造一次Problem。输入为人工平滑参数序列，不宣称是真实轨迹；[原序列NPZ](../logs/runtime_parameter_sequence.npz)可重放。",
             '墙钟用perf_counter_ns，涵盖参数更新、求解、控制器独立验解，不含控制器构造；首调用含CVXPY编译/冷启动，单列不删除。计时运行没有并行跑其他性能测试。', '',
             '| 模式 | 样本组 | n | mean ms | p95 ms | p99 ms | max ms | >预算/总数 | >周期/总数 | 数值有效/总数 |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mode in ('warm_on', 'warm_off'):
        for group in ('cold', 'steady', 'all'):
            m = summary['benchmark']['metrics'][mode][group]
            lines.append(f"| {mode} | {group} | {m['count']} | {_number(m['wall_mean_ms'])} | {_number(m['wall_p95_ms'])} | {_number(m['wall_p99_ms'])} | {_number(m['wall_max_ms'])} | {m['budget_miss_count']}/{m['count']} | {m['deadline_miss_count']}/{m['count']} | {m['numerical_success_count']}/{m['count']} |")
    on, off = (summary['benchmark']['metrics'][mode]['steady'] for mode in ('warm_on', 'warm_off'))
    if on['wall_p95_ms'] is None or off['wall_p95_ms'] is None:
        comparison = '无稳态样本，不能比较稳态p95；缺失统计保留为null，不画成零。'
    else:
        relation = '低于' if on['wall_p95_ms'] < off['wall_p95_ms'] else '不低于'
        comparison = f"本次warm_on稳态p95（{_number(on['wall_p95_ms'])} ms）{relation}warm_off（{_number(off['wall_p95_ms'])} ms）。这只是当前机器、版本和输入序列的测量，不预设warm必然更快，也不是硬实时上界。"
    lines += ['', comparison + '重复轮次分列保存在summary，不把各轮p95平均冒充整体p95。',
              '本机CVXPY公开适配源码确认：`solver_stats.solve_time`映射OSQP `info.run_time`，不能当作纯求解时间。另存`backend_solve_time_s/setup_time_s/update_time_s/run_time_s`；不可取得为null/空，不填0。warm_start=False虽然复用CVXPY Problem，CVXPY会重建OSQP后端，因此此处比较的是CVXPY的warm开关，不是纯迭代初值的隔离实验。', '',
              '![真实计时](../figures/solver_runtime.png)', '',
              '## 4. G：故障注入与恢复', '',
              f"正常完成延迟固定注入{c['runtime']['nominal_completion_s']*1000:g} ms；真实solve仍调用，wall_time另存，不能用其覆盖注入完成时刻。各组使用完全相同的预生成编码器噪声。", '',
              '| 场景 | 周期→故障 |', '| --- | --- |']
    for name, faults in summary['fault_scenarios'].items():
        lines.append(f"| {name} | `{json.dumps(faults, ensure_ascii=False)}` |")
    lines += ['', '`max_iter=1`实际传给OSQP，原始返回由程序保存；error/nonfinite是明确的人为接口/数据损坏，不冒充自然发生的OSQP故障。delay只注入完成事件，不使用sleep、不冒充墙钟。', '',
              '| 场景 | 状态 | 提交/周期 | 预算miss | 截止miss | 忙跳过 | 迟到丢弃 | 降级进入/恢复 | 降级s | 跟踪RMSE rad | 实际速度超限峰值 rad/s |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, case in summary['cases'].items():
        m = case['metrics']
        lines.append(f"| {name} | {case['status']} | {m['submitted_job_count']}/{m['control_cycle_count']} | {m['budget_miss_count']} | {m['deadline_miss_count']} | {m['busy_skip_count']} | {m['late_discard_count']} | {m['fallback_entry_count']}/{m['recovery_count']} | {_number(m['fallback_duration_s'])} | {_number(m['tracking_rmse_rad'])} | {_number(m['actual_overspeed_peak_rad_s'])} |")
    for name, case in summary['cases'].items():
        jobs = json.loads((root / case['jobs']).read_text(encoding='utf-8'))
        unforced = [j for j in jobs if not j['candidate_valid'] and not j['fault']]
        if unforced:
            details = [(j['cycle_index'], j['diagnostics'].get('status'), j.get('result_category')) for j in unforced]
            lines += ['', f'{name}还记录到{len(unforced)}次未注入的数值候选失败，周期/原始状态/分类为`{details}`。默认迭代上限下也可能未收敛；complete表示仿真跑完，不代表每次QP成功或速度达标。']
    lines += ['', '注意：f_measured_sync行只表示阻塞返回后，按真实墙钟决定拒绝和降级；物理仿真并未在阻塞期间实时推进，不能从这行推出“5 ms准时切换”。G才是物理事件时序演示，也不是已经实现可抢占异步线程。', '',
              '![注入时序](../figures/fallback_recovery.png)', '',
              '## 5. 如何独立检查与迁移实机', '',
              '- 每个场景保存physical CSV、jobs JSON、events CSV、成功数值候选plans NPZ。`accepted`只表示及时有效，恢复前两次仍用LQR；`executed`表示MPC请求被选中，之后仍可能被共同guard修正，不表示实际电机扭矩等于该请求。',
              '- `discarded`作业不得executed；`late_result_discarded`不能推进恢复。作业预算/截止计数以提交数为分母，busy以控制触发数为分母；pending保持未完成，不能伪造成功。',
              '- 跟踪/扭矩指标是各自已观察时段的描述，不把提前终止场景并入成功平均。终态和时间单列；真实速度超限按1 ms矩形积分，峰值包含终态。位置越界独立检测后停仿真，不能称为硬件急停。',
              '- 实机需要单调时间戳、真正的独立求解worker、带序号/有效期的结果邮箱、不可重叠的提交策略、确定的命令保护与驱动器故障处理；求解器超时不意味着OS线程已取消。',
              '- 外层保护和LQR fallback并不保证速度/机械边界，也不使非线性饱和系统自动具有线性理论保证；还需硬件限位、看门狗、驱动级安全以及现场测试。',
              '- 本次完成F/G；综合多seed、LQR/基础MPC/工程MPC全场景验收仍是后续阶段，不能把这两组替代综合验收。', '',
              '运行：', '', '```powershell\nconda run -n robot-control python -m lesson13_mpc_engineering.src.run_experiments --experiment fg\n```', '']
    (root / 'reports/runtime_fallback_report.md').write_text('\n'.join(lines), encoding='utf-8')


def export_runtime_artifacts(summary, sequence, benchmark, results, root):
    for directory in ('logs', 'figures', 'reports'):
        (root / directory).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / 'logs/runtime_parameter_sequence.npz', **sequence)
    _write_rows(root / 'logs/runtime_benchmark.csv', benchmark['rows'])
    horizon = summary['config']['mpc']['horizon_steps']
    for name, result in results.items():
        case = summary['cases'][name]
        for key, suffix in (('csv', '.csv'), ('jobs', '_jobs.json'), ('events', '_events.csv'), ('plans', '_plans.npz')):
            case[key] = f'logs/managed_{name}{suffix}'
        log = result['log']
        _write_rows(root / case['csv'], [dict(zip(log, row)) for row in zip(*log.values())], list(log))
        _write_json(root / case['jobs'], result['jobs'])
        _write_rows(root / case['events'], result['events'])
        plans = result['plans']
        arrays = {key: np.array([plan[key] for plan in plans])
                  for key in ('job_id', 'time_s', 'u_prev', 'executed', 'discarded')}
        for key, shape in (('X', (2, horizon + 1)), ('U', (1, horizon)), ('DU', (1, horizon)), ('EPS', (1, horizon))):
            arrays[key] = np.stack([p[key] for p in plans]) if plans else np.empty((0, *shape))
        arrays['velocity_indices'] = np.arange(1, horizon + 1)
        np.savez_compressed(root / case['plans'], **arrays)
    summary = _json_value(summary)
    _write_json(root / 'logs/experiment_fg_summary.json', summary)
    _plots(summary, benchmark, results, root)
    _report(summary, root)
    return summary
