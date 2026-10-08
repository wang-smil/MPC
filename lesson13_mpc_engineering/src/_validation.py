"""Small shared numeric checks; never allow invalid values into dynamic state."""

from numbers import Real

import numpy as np


def finite_scalar(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real scalar.')
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f'{name} must be finite.')
    return result


def bounded_scalar(value, name: str, *, positive: bool = False) -> float:
    result = finite_scalar(value, name)
    if result < 0 or (positive and result == 0):
        raise ValueError(f'{name} must be {"positive" if positive else "nonnegative"}.')
    return result


def integer_steps(duration: float, dt_s: float, name: str) -> int:
    ratio = duration / dt_s
    if not np.isfinite(ratio):
        raise ValueError(f'{name}/dt_s must be finite.')
    steps = int(round(ratio))
    if not np.isclose(steps * dt_s, duration, atol=1e-12, rtol=1e-10):
        raise ValueError(f'{name} must be an integer multiple of dt_s.')
    return steps
