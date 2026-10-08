"""Simulated joint truth: inertia, viscous damping, friction and unknown load."""

import numpy as np

from lesson08_lqr_optimal_control.src.model import build_continuous_model, discretize_zoh
from ._validation import bounded_scalar, finite_scalar


class JointPlant:
    def __init__(self, inertia: float = .019762, damping: float = .080257,
                 dt_s: float = .001, coulomb_friction: float = 0.,
                 friction_smoothing: float = .02, initial_state=(0., 0.),
                 integration: str = 'semi_implicit'):
        self.inertia = bounded_scalar(inertia, 'inertia', positive=True)
        self.damping = bounded_scalar(damping, 'damping')
        self.dt_s = bounded_scalar(dt_s, 'dt_s', positive=True)
        self.coulomb_friction = bounded_scalar(coulomb_friction, 'coulomb_friction')
        self.friction_smoothing = bounded_scalar(friction_smoothing, 'friction_smoothing', positive=True)
        raw = np.asarray(initial_state, dtype=object)
        if raw.shape != (2,):
            raise ValueError('initial_state must have shape (2,).')
        self._state = np.array([finite_scalar(item, 'initial_state') for item in raw])
        if integration not in ('semi_implicit', 'exact_zoh'):
            raise ValueError('integration must be semi_implicit or exact_zoh.')
        if integration == 'exact_zoh' and self.coulomb_friction != 0:
            raise ValueError('exact_zoh requires zero Coulomb friction.')
        self.integration = integration
        if integration == 'exact_zoh':
            A, B = build_continuous_model(self.inertia, self.damping)
            self._Ad, self._Bd = discretize_zoh(A, B, self.dt_s)

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    def step(self, torque_actual_nm: float, load_torque_nm: float = 0.) -> np.ndarray:
        torque = finite_scalar(torque_actual_nm, 'torque_actual_nm')
        load = finite_scalar(load_torque_nm, 'load_torque_nm')
        with np.errstate(over='ignore', invalid='ignore'):
            if self.integration == 'exact_zoh':
                next_state = self._Ad @ self._state + self._Bd[:, 0] * (torque - load)
            else:
                q, velocity = self._state
                friction = self.coulomb_friction * np.tanh(velocity / self.friction_smoothing)
                acceleration = (torque - load - self.damping * velocity - friction) / self.inertia
                # Semi-implicit Euler: first update velocity, then use NEW velocity.
                next_velocity = velocity + acceleration * self.dt_s
                next_state = np.array([q + next_velocity * self.dt_s, next_velocity])
        if not np.all(np.isfinite(next_state)):
            raise ValueError('Plant integration produced nonfinite state; state was not advanced.')
        self._state = next_state
        return self.state
