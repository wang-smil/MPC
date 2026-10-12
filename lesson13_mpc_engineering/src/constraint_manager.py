"""Shared command guard; distinct from planning constraints and motor dynamics."""

import numpy as np

from ._validation import bounded_scalar, finite_scalar


def guard_command(request: float, previous_command: float, torque_limit_nm: float,
                  slew_rate_nm_s: float, elapsed_s: float) -> float:
    request = finite_scalar(request, 'request')
    previous = finite_scalar(previous_command, 'previous_command')
    limit = bounded_scalar(torque_limit_nm, 'torque_limit_nm', positive=True)
    rate = bounded_scalar(slew_rate_nm_s, 'slew_rate_nm_s', positive=True)
    elapsed = bounded_scalar(elapsed_s, 'elapsed_s', positive=True)
    if abs(previous) > limit:
        raise ValueError('previous_command is outside the torque bound.')
    target = float(np.clip(request, -limit, limit))
    return float(np.clip(previous + np.clip(target - previous, -rate * elapsed, rate * elapsed),
                         -limit, limit))
