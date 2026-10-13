"""Extend the Lesson 12 QP with input moves; old lessons remain unchanged."""

import cvxpy as cp
import numpy as np

from lesson12_linear_mpc.src.mpc_controller import LinearMPCController
from ._validation import bounded_scalar, finite_scalar
from .qp_solver import SolverSettings, solve_qp


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
                 torque_bounds=(-2., 2.), delta_bounds=None, position_bounds=None,
                 velocity_mode='off', velocity_bounds=None, include_current_velocity=False,
                 slack_weight=1e4, solver_settings=None):
        if solver_settings is not None and not isinstance(solver_settings, SolverSettings):
            raise ValueError('solver_settings must be SolverSettings or None.')
        self.solver_settings = solver_settings
        self.input_rate_weight = bounded_scalar(input_rate_weight, 'input_rate_weight')
        self.delta_bounds = finite_bounds(delta_bounds, 'delta_bounds')
        self.position_bounds = finite_bounds(position_bounds, 'position_bounds')
        torque_bounds = finite_bounds(torque_bounds, 'torque_bounds')
        if not isinstance(velocity_mode, str) or velocity_mode not in ('off', 'hard', 'soft'):
            raise ValueError('velocity_mode must be off, hard or soft.')
        if not isinstance(include_current_velocity, (bool, np.bool_)):
            raise ValueError('include_current_velocity must be boolean.')
        self.velocity_mode = velocity_mode
        # Do not set the parent's velocity_bounds: it validates HARD future bounds.
        self.engineering_velocity_bounds = finite_bounds(velocity_bounds, 'velocity_bounds')
        self.slack_weight = bounded_scalar(slack_weight, 'slack_weight', positive=True)
        if velocity_mode == 'off':
            if velocity_bounds is not None or include_current_velocity:
                raise ValueError('Off velocity mode cannot have bounds or current-state constraints.')
        elif self.engineering_velocity_bounds is None:
            raise ValueError('Enabled velocity mode requires velocity_bounds.')
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
        self.velocity_indices = (() if velocity_mode == 'off' else
                                 tuple(range(0 if include_current_velocity else 1, self.horizon + 1)))
        self.EPS = None
        if self.velocity_indices:
            speed = self.X[1:2, list(self.velocity_indices)]
            lower, upper = self.engineering_velocity_bounds
            if velocity_mode == 'soft':
                self.EPS = cp.Variable((1, len(self.velocity_indices)), nonneg=True, name='EPS')
                constraints += [speed >= lower - self.EPS, speed <= upper + self.EPS]
                cost += self.slack_weight * cp.sum_squares(self.EPS)
            else:
                constraints += [speed >= lower, speed <= upper]
        # Positive whole-objective scaling preserves the optimizer and all weight ratios.
        self.objective_scale = max(1., self.slack_weight) if velocity_mode == 'soft' else 1.
        self.problem = cp.Problem(cp.Minimize(cost / self.objective_scale), constraints)

    def solve(self, x_hat, x_ref, u_prev: float, *, solver_settings=None) -> dict:
        state = finite_vector(x_hat, 2, 'x_hat')
        reference = finite_vector(x_ref, 2, 'x_ref')
        previous = finite_scalar(u_prev, 'u_prev')
        self.u_prev.value = np.array([previous])
        settings = solver_settings if solver_settings is not None else self.solver_settings
        if settings is None:
            result = super().solve(state, reference)  # Preserve A-E's original invocation path.
        else:
            if not isinstance(settings, SolverSettings):
                raise ValueError('solver_settings must be SolverSettings or None.')
            self.x0.value, self.x_ref.value = state, reference
            result = solve_qp(self.problem, self.X, self.U, settings)
        result.update(DU=None, EPS=None, max_hard_violation=None, max_soft_violation=None)
        if result['u0'] is None:
            return result
        result['objective'] *= self.objective_scale  # Report cost in the original units.
        X, U = result['X'], result['U']
        DU = np.diff(np.concatenate(([previous], U[0])))[None, :]
        violations = [float(np.max(np.abs(X[:, 0] - state))),
                      float(np.max(np.abs(X[:, 1:] - self.Ad @ X[:, :-1] - self.Bd @ U)))]
        for bounds, values in ((self.torque_bounds, U), (self.delta_bounds, DU),
                               (self.position_bounds, X[0, 1:])):
            if bounds is not None:
                violations += [float(np.max(bounds[0] - values)), float(np.max(values - bounds[1]))]
        eps, soft_violation = None, 0.
        if self.velocity_indices:
            speed = X[1:2, list(self.velocity_indices)]
            lower, upper = self.engineering_velocity_bounds
            if self.velocity_mode == 'hard':
                violations += [float(np.max(lower - speed)), float(np.max(speed - upper))]
            else:
                raw = self.EPS.value
                if raw is None or np.shape(raw) != (1, len(self.velocity_indices)) or not np.all(np.isfinite(raw)):
                    soft_violation = np.inf
                else:
                    eps = np.array(raw, dtype=float, copy=True)
                    soft_violation = max(0., float(np.max(-eps)), float(np.max(lower - eps - speed)),
                                         float(np.max(speed - upper - eps)))
                result['max_soft_violation'] = soft_violation
        maximum = max(0., *violations)
        result['max_hard_violation'] = maximum
        if (maximum > 1e-5 or soft_violation > 1e-5 or not np.all(np.isfinite(DU))
                or not np.isfinite(result['objective'])):
            result.update(status='invalid_solution', error='Plan failed extended constraint validation',
                          u0=None, X=None, U=None, DU=None, EPS=None)
        else:
            result['DU'] = DU
            result['EPS'] = eps
        return result
