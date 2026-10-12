"""Metrics with explicit physical-sample versus controller-event denominators."""

import numpy as np

from ._validation import bounded_scalar


def _finite(values):
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _peak(values):
    values = _finite(values)
    return float(np.max(np.abs(values))) if values.size else None


def _rms(values):
    values = _finite(values)
    return float(np.sqrt(np.mean(values**2))) if values.size else None


def calculate_metrics(log: dict, control_dt_s: float) -> dict:
    control_dt = bounded_scalar(control_dt_s, 'control_dt_s', positive=True)
    mask = np.asarray(log['control_update'], dtype=bool)
    requests = _finite(np.asarray(log['torque_request_nm'])[mask])
    initial = float(log['feedback_prev_nm'][0]) if len(mask) else 0.
    request_moves = np.diff(np.r_[initial, requests])
    error = np.asarray(log['q_true_rad']) - np.asarray(log['q_ref_rad'])
    result = {'tracking_rmse_rad': _rms(error),
              'rms_actual_torque_nm': _rms(log['torque_actual_nm']),
              'peak_actual_torque_nm': _peak(log['torque_actual_nm']),
              'rms_request_delta_nm': _rms(request_moves), 'max_request_delta_nm': _peak(request_moves),
              'max_command_delta_nm': _peak(np.asarray(log['command_delta_nm'])[mask]),
              'max_qp_first_delta_nm': _peak(np.asarray(log['qp_first_delta_nm'])[mask]),
              'max_planned_delta_nm': _peak(np.asarray(log['max_planned_delta_nm'])[mask]),
              'control_dt_s': control_dt, 'logged_samples': len(mask)}
    times = np.asarray(log['time_s'])
    physical_dt = float(np.median(np.diff(times))) if times.size > 1 else None
    result['physical_dt_s'] = physical_dt
    peak_delta = _peak(log['delta_actual_nm'])
    result['max_actual_slew_nm_s'] = peak_delta / physical_dt if peak_delta is not None and physical_dt else None
    statuses = np.asarray(log['solver_status'])[mask]
    called = statuses != ''
    successful = np.isin(statuses, ['optimal', 'optimal_inaccurate'])
    result.update(solver_call_count=int(np.sum(called)), solver_success_count=int(np.sum(successful)),
                  solver_failure_count=int(np.sum(called & ~successful)),
                  guard_modified_count=int(np.sum(np.asarray(log['guard_modified'])[mask] == 1)))
    for label, field in (('wall', 'solver_wall_time_s'), ('internal', 'solver_solve_time_s')):
        values = _finite(np.asarray(log[field])[mask]) * 1000
        for stat, value in (('mean', np.mean(values) if values.size else None),
                            ('p95', np.percentile(values, 95) if values.size else None),
                            ('p99', np.percentile(values, 99) if values.size else None),
                            ('max', np.max(values) if values.size else None)):
            result[f'{stat}_solver_{label}_ms'] = float(value) if value is not None else None
    iterations = _finite(np.asarray(log['solver_iterations'])[mask])
    result['mean_solver_iterations'] = float(np.mean(iterations)) if iterations.size else None
    return result
