"""Separate actual physical overspeed from the QP's predicted relaxation."""

import numpy as np

from ._validation import bounded_scalar
from .metrics import calculate_metrics, _peak


def calculate_feasibility_metrics(result: dict, control_dt_s: float,
                                  physical_dt_s: float, velocity_limit_rad_s: float) -> dict:
    control_dt = bounded_scalar(control_dt_s, 'control_dt_s', positive=True)
    dt = bounded_scalar(physical_dt_s, 'physical_dt_s', positive=True)
    limit = bounded_scalar(velocity_limit_rad_s, 'velocity_limit_rad_s', positive=True)
    log = result['log']
    metrics = calculate_metrics(log, control_dt)
    applied = np.isfinite(log['torque_actual_nm'])
    speeds = np.asarray(log['dq_true_rad_s'], dtype=float)
    excess = np.maximum(np.abs(speeds[applied]) - limit, 0.)
    # Terminal evidence contributes a peak, not a fictitious extra interval.
    all_speeds = np.r_[speeds, result['final_state'][1]]
    metrics.update(max_actual_velocity_violation_rad_s=float(np.max(np.maximum(np.abs(all_speeds) - limit, 0.))),
                   actual_overspeed_duration_s=float(np.count_nonzero(excess > 0) * dt),
                   actual_overspeed_integral_rad=float(np.sum(excess) * dt))
    mask = np.asarray(log['control_update'], dtype=bool)
    times = np.asarray(log['time_s'])[mask]
    slack = np.asarray(log['qp_initial_slack_rad_s'])[mask]
    future = np.asarray(log['max_predicted_slack_rad_s'])[mask]
    valid = np.isfinite(slack)
    metrics['max_predicted_slack_rad_s'] = _peak(future)
    metrics['max_initial_slack_rad_s'] = _peak(slack)
    metrics['initial_slack_hold_integral_rad'] = None
    if np.any(valid):
        end = float(result['final_state_time_s'])
        ends = np.minimum(np.r_[times[1:], end], end)
        widths = np.clip(ends - times, 0., control_dt)
        metrics['initial_slack_hold_integral_rad'] = float(np.sum(np.maximum(slack[valid], 0.) * widths[valid]))
    return metrics
