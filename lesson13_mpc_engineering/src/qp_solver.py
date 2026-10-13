"""Opt-in CVXPY/OSQP invocation; preserve original statuses and current-call diagnostics."""

from dataclasses import dataclass
from numbers import Integral
import warnings

import cvxpy as cp
import numpy as np
import osqp


@dataclass(frozen=True)
class SolverSettings:
    warm_start: bool = True
    max_iter: int = 4000

    def __post_init__(self):
        if not isinstance(self.warm_start, (bool, np.bool_)):
            raise ValueError('warm_start must be boolean.')
        if isinstance(self.max_iter, (bool, np.bool_)) or not isinstance(self.max_iter, Integral) or self.max_iter < 1:
            raise ValueError('max_iter must be a positive integer.')
        object.__setattr__(self, 'warm_start', bool(self.warm_start))
        object.__setattr__(self, 'max_iter', int(self.max_iter))


def _time(value):
    return float(value) if value is not None and np.isfinite(value) and value >= 0 else None


def solve_qp(problem, X_variable, U_variable, settings: SolverSettings) -> dict:
    if not isinstance(settings, SolverSettings):
        raise ValueError('settings must be SolverSettings.')
    result = dict.fromkeys(('status', 'u0', 'X', 'U', 'objective', 'iterations', 'solve_time_s', 'error',
                            'backend_status', 'backend_status_val', 'backend_solve_time_s',
                            'setup_time_s', 'update_time_s', 'run_time_s'))
    polish_key = 'polishing' if int(osqp.__version__.split('.')[0]) >= 1 else 'polish'
    with warnings.catch_warnings(record=True) as notices:
        try:
            problem.solve(solver=cp.OSQP, warm_start=settings.warm_start, max_iter=settings.max_iter,
                          eps_abs=1e-8, eps_rel=1e-8, **{polish_key: True})
        except Exception as exc:
            result.update(status='solver_error', error=str(exc), warnings=[str(item.message) for item in notices])
            return result  # Never consult solver_stats from a previous successful invocation.
    result['warnings'] = [str(item.message) for item in notices]
    result['status'] = problem.status
    stats = problem.solver_stats
    if stats is not None:
        result['solve_time_s'] = _time(stats.solve_time)  # CVXPY maps this to OSQP run_time.
        result['iterations'] = int(stats.num_iters) if stats.num_iters is not None else None
        info = getattr(getattr(stats, 'extra_stats', None), 'info', None)
        if info is not None:
            result['backend_status'] = getattr(info, 'status', None)
            result['backend_status_val'] = getattr(info, 'status_val', None)
            for field, source in (('setup_time_s', 'setup_time'), ('update_time_s', 'update_time'),
                                  ('backend_solve_time_s', 'solve_time'), ('run_time_s', 'run_time')):
                result[field] = _time(getattr(info, source, None))
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        result['error'] = f'OSQP did not solve the MPC problem: {problem.status}'
        return result
    try:
        X, U = np.asarray(X_variable.value, dtype=float), np.asarray(U_variable.value, dtype=float)
        valid = (X.shape == X_variable.shape and U.shape == U_variable.shape
                 and np.all(np.isfinite(X)) and np.all(np.isfinite(U))
                 and problem.value is not None and np.isfinite(problem.value))
    except (ValueError, TypeError):
        valid = False
    if not valid:
        result.update(status='invalid_solution', error='Missing, malformed or nonfinite QP candidate.')
        return result
    result.update(X=X.copy(), U=U.copy(), u0=U[:, 0].copy(), objective=float(problem.value))
    return result
