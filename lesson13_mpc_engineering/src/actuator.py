"""Delayed, lagged torque output with physical amplitude and slew limits."""

from collections import deque

import numpy as np

from ._validation import bounded_scalar, finite_scalar, integer_steps


class TorqueActuator:
    def __init__(self, dt_s: float = .001, torque_limit_nm: float = 2.,
                 slew_rate_nm_s: float = 50., delay_s: float = .002,
                 time_constant_s: float = .005, initial_torque_nm: float = 0.):
        self.dt_s = bounded_scalar(dt_s, 'dt_s', positive=True)
        self.torque_limit_nm = bounded_scalar(torque_limit_nm, 'torque_limit_nm', positive=True)
        self.slew_rate_nm_s = bounded_scalar(slew_rate_nm_s, 'slew_rate_nm_s', positive=True)
        delay = bounded_scalar(delay_s, 'delay_s')
        lag = bounded_scalar(time_constant_s, 'time_constant_s')
        initial = finite_scalar(initial_torque_nm, 'initial_torque_nm')
        if abs(initial) > self.torque_limit_nm:
            raise ValueError('initial_torque_nm exceeds torque_limit_nm.')
        self._pending = deque([initial] * integer_steps(delay, self.dt_s, 'delay_s'))
        self._alpha = 1. if lag == 0 else float(-np.expm1(-self.dt_s / lag))
        self._actual = initial

    @property
    def torque_actual_nm(self) -> float:
        return self._actual

    def step(self, command_nm: float) -> dict[str, float]:
        # Validate before touching the FIFO: bad inputs cannot consume a period.
        command = finite_scalar(command_nm, 'command_nm')
        command = float(np.clip(command, -self.torque_limit_nm, self.torque_limit_nm))
        self._pending.append(command)
        delayed = self._pending.popleft()
        candidate_delta = self._alpha * (delayed - self._actual)
        max_delta = self.slew_rate_nm_s * self.dt_s
        previous = self._actual
        self._actual = float(np.clip(previous + np.clip(candidate_delta, -max_delta, max_delta),
                                     -self.torque_limit_nm, self.torque_limit_nm))
        return {'command_nm': command, 'delayed_command_nm': delayed,
                'torque_actual_nm': self._actual, 'delta_actual_nm': self._actual - previous}
