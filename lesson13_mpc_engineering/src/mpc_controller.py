"""Extend the Lesson 12 QP with input moves; old lessons remain unchanged."""

import cvxpy as cp
import numpy as np

from lesson12_linear_mpc.src.mpc_controller import LinearMPCController
from ._validation import bounded_scalar, finite_scalar


def finite_vector(value, length: int, name: str) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.shape != (length,):
        raise ValueError(f'{name} must have shape ({length},).')
    return np.array([finite_scalar(item, name) for item in raw])


def finite_bounds(value, name: str):
    if value is None:
        return None
    pair = finite_vector(value, 2, name)
    if pair[0] >= pair[1]:
        raise ValueError(f'{name} must be an increasing pair.')
    return tuple(pair)


class EngineeringMPCController(LinearMPCController):
    def __init__(self, Ad, Bd, Q, R, P_terminal, horizon, *, input_rate_weight=0.,
                 torque_bounds=(-2., 2.), delta_bounds=None, position_bounds=None):
        self.input_rate_weight = bounded_scalar(input_rate_weight, 'input_rate_weight')
        self.delta_bounds = finite_bounds(delta_bounds, 'delta_bounds')
        self.position_bounds = finite_bounds(position_bounds, 'position_bounds')
        torque_bounds = finite_bounds(torque_bounds, 'torque_bounds')
        if self.delta_bounds is not None and not self.delta_bounds[0] <= 0 <= self.delta_bounds[1]:
            raise ValueError('delta_bounds must contain zero.')
        super().__init__(Ad, Bd, Q, R, P_terminal, horizon, torque_bounds=torque_bounds)
        if (self.nx, self.nu) != (2, 1):
            raise ValueError('Lesson 13 requires two states and one input.')
        self.u_prev = cp.Parameter(1, name='u_prev_applied')
        moves = [self.U[:, 0:1] - cp.reshape(self.u_prev, (1, 1), order='C')]
        if self.horizon > 1:
            moves.append(self.U[:, 1:] - self.U[:, :-1])
        self.DU = cp.hstack(moves)
        constraints = list(self.problem.constraints)
        if self.delta_bounds is not None:
            constraints += [self.DU >= self.delta_bounds[0], self.DU <= self.delta_bounds[1]]
        if self.position_bounds is not None:
            constraints += [self.X[0, 1:] >= self.position_bounds[0],
                            self.X[0, 1:] <= self.position_bounds[1]]
        cost = self.problem.objective.expr + self.input_rate_weight * cp.sum_squares(self.DU)
        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, x_hat, x_ref, u_prev: float) -> dict:
        state = finite_vector(x_hat, 2, 'x_hat')
        reference = finite_vector(x_ref, 2, 'x_ref')
        previous = finite_scalar(u_prev, 'u_prev')
        self.u_prev.value = np.array([previous])
        result = super().solve(state, reference)
        result.update(DU=None, max_hard_violation=None)
        if result['u0'] is None:
            return result
        X, U = result['X'], result['U']
        DU = np.diff(np.concatenate(([previous], U[0])))[None, :]
        violations = [float(np.max(np.abs(X[:, 0] - state))),
                      float(np.max(np.abs(X[:, 1:] - self.Ad @ X[:, :-1] - self.Bd @ U)))]
        for bounds, values in ((self.torque_bounds, U), (self.delta_bounds, DU),
                               (self.position_bounds, X[0, 1:])):
            if bounds is not None:
                violations += [float(np.max(bounds[0] - values)), float(np.max(values - bounds[1]))]
        maximum = max(0., *violations)
        result['max_hard_violation'] = maximum
        if maximum > 1e-5 or not np.all(np.isfinite(DU)):
            result.update(status='invalid_solution', error='Plan failed extended constraint validation',
                          u0=None, X=None, U=None, DU=None)
        else:
            result['DU'] = DU
        return result
