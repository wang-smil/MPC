"""F wall-clock statistics and G event/physical metrics use separate denominators."""

import numpy as np

from ._validation import bounded_scalar


def _values(items):
    values = np.asarray([x for x in items if x is not None], dtype=float)
    return values[np.isfinite(values)]


def _stats(items, prefix, scale=1.):
    values = _values(items) * scale
    return {f'{prefix}_{name}': float(operation(values)) if values.size else None
            for name, operation in (('mean', np.mean), ('p95', lambda x: np.percentile(x, 95)),
                                    ('p99', lambda x: np.percentile(x, 99)), ('max', np.max))}


def _sample_metrics(rows, budget, deadline):
    count = len(rows)
    wall = _values([row.get('wall_time_s') for row in rows])
    successful = sum(bool(row['candidate_valid']) for row in rows)
    result = {'count': count, 'wall_observation_count': len(wall), 'numerical_success_count': successful,
              'numerical_success_percent': 100 * successful / count if count else None,
              'budget_miss_count': int(np.sum(wall > budget)),
              'deadline_miss_count': int(np.sum(wall > deadline))}
    for key in ('budget', 'deadline'):
        result[f'{key}_miss_percent'] = 100 * result[f'{key}_miss_count'] / len(wall) if len(wall) else None
    for prefix, field, scale in (('wall', 'wall_time_s', 1000), ('iterations', 'iterations', 1),
                                  ('backend_solve', 'backend_solve_time_s', 1000),
                                  ('backend_run', 'run_time_s', 1000)):
        # Keep unit suffixes explicit without converting absent diagnostics to zero.
        stats = _stats([row.get(field) for row in rows], prefix, scale)
        result.update({f'{key}_ms' if scale == 1000 else key: value for key, value in stats.items()})
    return result


def calculate_runtime_metrics(rows, budget_s, deadline_s):
    budget = bounded_scalar(budget_s, 'budget_s', positive=True)
    deadline = bounded_scalar(deadline_s, 'deadline_s', positive=True)
    if budget >= deadline:
        raise ValueError('budget must be below the control deadline.')

    def groups(selected):
        return {name: _sample_metrics(subset, budget, deadline)
                for name, subset in (('all', selected), ('cold', [r for r in selected if r['cold']]),
                                     ('steady', [r for r in selected if not r['cold']]))}

    result = {}
    for mode in sorted({row['mode'] for row in rows}):
        selected = [row for row in rows if row['mode'] == mode]
        result[mode] = groups(selected)
        result[mode]['repetitions'] = {str(rep): groups([r for r in selected if r['repetition'] == rep])
                                       for rep in sorted({r['repetition'] for r in selected})}
    return result


def calculate_managed_metrics(result, config):
    log, jobs, events = result['log'], result['jobs'], result['events']
    dt = config['plant']['dt_s']
    end = result['final_state_time_s']
    error = _values(np.asarray(log['q_true_rad'], dtype=float) - np.asarray(log['q_ref_rad'], dtype=float))
    actual = _values(log['torque_actual_nm'])
    times = np.asarray(log['time_s'], dtype=float)
    applied = np.isfinite(np.asarray(log['torque_actual_nm'], dtype=float)) & (times < end)
    speed = _values(np.asarray(log['dq_true_rad_s'], dtype=float)[applied])
    excess = np.maximum(np.abs(speed) - config['constraints']['velocity_limit_rad_s'], 0.)
    final_q, final_speed = result['final_state']
    peak_excess = max(float(np.max(excess)) if excess.size else 0.,
                      max(0., abs(float(final_speed)) - config['constraints']['velocity_limit_rad_s']))
    position = np.r_[_values(np.asarray(log['q_true_rad'], dtype=float)[times < end]), final_q]
    entries = recoveries = 0
    mode, last_time, fallback_duration = 'mpc', 0., 0.
    for event in events:
        if mode == 'fallback':
            fallback_duration += max(0., event['time_s'] - last_time)
        if event['mode_after'] != mode and event['mode_after'] == 'fallback':
            entries += 1
        recoveries += bool(event.get('recovered'))
        mode, last_time = event['mode_after'], event['time_s']
    if mode == 'fallback':
        fallback_duration += max(0., end - last_time)
    cycles = int(np.sum(np.asarray(log['control_update']) == True))
    submitted = len(jobs)
    completed = sum(job['completed'] for job in jobs)
    command_moves = _values([e.get('command_delta_nm') for e in events])
    numerical_success = sum(job['candidate_valid'] for job in jobs)
    metrics = {'status': result['status'], 'observation_duration_s': end,
               'logged_samples': len(log['time_s']), 'expected_samples': result['expected_samples'],
               'control_cycle_count': cycles, 'submitted_job_count': submitted,
               'solver_call_count': sum(job['solver_called'] for job in jobs),
               'completed_job_count': completed, 'pending_job_count': submitted - completed,
               'numerical_success_count': numerical_success,
               'accepted_job_count': sum(job['accepted'] for job in jobs),
               'executed_mpc_request_count': sum(job['executed'] for job in jobs),
               'budget_miss_count': sum(job['budget_missed'] for job in jobs),
               'deadline_miss_count': sum(job['deadline_missed'] for job in jobs),
               'busy_skip_count': sum(e['event'] == 'busy_skip' for e in events),
               'late_discard_count': sum(e['event'] == 'late_result_discarded' for e in events),
               'fallback_entry_count': entries, 'fallback_duration_s': fallback_duration,
               'recovery_count': recoveries,
               'tracking_rmse_rad': float(np.sqrt(np.mean(error**2))) if error.size else None,
               'final_error_rad': abs(float(final_q) - np.deg2rad(config['experiment_fg']['target_deg'])),
               'rms_actual_torque_nm': float(np.sqrt(np.mean(actual**2))) if actual.size else None,
               'peak_actual_torque_nm': float(np.max(np.abs(actual))) if actual.size else None,
               'max_command_delta_nm': float(np.max(np.abs(command_moves))) if command_moves.size else None,
               'max_actual_slew_nm_s': float(np.max(np.abs(_values(log['delta_actual_nm'])))) / dt if actual.size else None,
               'guard_modified_count': sum(bool(e.get('guard_modified')) for e in events),
               'actual_overspeed_peak_rad_s': peak_excess,
               'actual_overspeed_duration_s': int(np.sum(excess > 1e-9)) * dt,
               'actual_overspeed_integral_rad': float(np.sum(excess) * dt),
               'position_violation_observation_count': int(np.sum(np.abs(position) > np.deg2rad(config['constraints']['position_limit_deg'])))}
    for key in ('budget', 'deadline'):
        metrics[f'{key}_miss_percent_submitted'] = 100 * metrics[f'{key}_miss_count'] / submitted if submitted else None
    metrics['busy_skip_percent_cycles'] = 100 * metrics['busy_skip_count'] / cycles if cycles else None
    metrics.update({f'{k}_ms': v for k, v in _stats([j['wall_time_s'] for j in jobs], 'wall', 1000).items()})
    return metrics
