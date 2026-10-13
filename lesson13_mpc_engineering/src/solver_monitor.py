"""Pure acceptance assessment plus a small, explicit fallback/recovery state machine."""

from numbers import Integral

import numpy as np

from ._validation import bounded_scalar, finite_scalar


def _array(value):
    raw = np.asarray(value, dtype=object)
    if any(isinstance(item, (bool, np.bool_)) for item in raw.flat):
        raise ValueError('Boolean candidate data is not numeric state.')
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError('Nonfinite candidate data.')
    return array


def _valid_candidate(result):
    try:
        u0, X, U, DU = (_array(result.get(key)) for key in ('u0', 'X', 'U', 'DU'))
        if (u0.shape != (1,) or U.ndim != 2 or U.shape[0] != 1 or U.shape[1] < 1
                or X.shape != (2, U.shape[1] + 1) or DU.shape != U.shape):
            return False
        if not np.allclose(u0, U[:, 0], atol=1e-8, rtol=0):
            return False
        finite_scalar(result.get('objective'), 'objective')
        residual = finite_scalar(result.get('max_hard_violation'), 'max_hard_violation')
        if not 0 <= residual <= 1e-5:
            return False
        if result.get('EPS') is not None:
            eps = _array(result['EPS'])
            soft = finite_scalar(result.get('max_soft_violation'), 'max_soft_violation')
            if eps.shape not in ((1, U.shape[1]), (1, U.shape[1] + 1)) or np.min(eps) < -1e-5 or not 0 <= soft <= 1e-5:
                return False
        return True
    except (TypeError, ValueError, KeyError):
        return False


def assess_result(result: dict, elapsed_s: float, budget_s: float, deadline_s: float) -> dict:
    elapsed = bounded_scalar(elapsed_s, 'elapsed_s')
    budget = bounded_scalar(budget_s, 'budget_s', positive=True)
    deadline = bounded_scalar(deadline_s, 'deadline_s', positive=True)
    if budget >= deadline:
        raise ValueError('budget_s must be smaller than deadline_s.')
    raw = result.get('status')
    backend = str(result.get('backend_status') or '').lower()
    assessment = {'accepted': False, 'category': None, 'budget_overrun': elapsed > budget,
                  'deadline_overrun': elapsed > deadline, 'raw_status': raw,
                  'backend_status': result.get('backend_status'), 'reason': result.get('error')}
    if raw not in ('optimal', 'optimal_inaccurate'):
        if str(raw).startswith('infeasible'):
            category = 'infeasible'
        elif 'maximum iterations' in backend or 'max_iter' in backend:
            category = 'iteration_limit'
        elif 'time limit' in backend:
            category = 'backend_time_limit'
        elif raw in ('solver_error', 'invalid_solution'):
            category = raw
        else:
            category = 'unknown_status'
    elif not _valid_candidate(result):
        category = 'invalid_solution'
    elif elapsed > budget:
        category = 'budget_overrun'
    else:
        category = 'accepted_inaccurate' if raw == 'optimal_inaccurate' else 'accepted'
        assessment['accepted'] = True
    assessment['category'] = category
    assessment['reason'] = assessment['reason'] or category
    return assessment


class SolverMonitor:
    def __init__(self, recovery_successes=3):
        if isinstance(recovery_successes, (bool, np.bool_)) or not isinstance(recovery_successes, Integral) or recovery_successes < 1:
            raise ValueError('recovery_successes must be a positive integer.')
        self.recovery_successes = int(recovery_successes)
        self.mode, self.good_count = 'mpc', 0

    def observe(self, assessment: dict) -> dict:
        if not isinstance(assessment.get('accepted'), (bool, np.bool_)):
            raise ValueError('assessment.accepted must be boolean.')
        before, recovered = self.mode, False
        if not assessment['accepted']:
            self.mode, self.good_count = 'fallback', 0
        elif self.mode == 'fallback':
            self.good_count += 1
            if self.good_count >= self.recovery_successes:
                self.mode, recovered = 'mpc', True
        else:
            self.good_count = 0
        return {**assessment, 'mode_before': before, 'mode_after': self.mode,
                'good_count': self.good_count, 'recovered': recovered}

    def reject(self, category: str, reason: str) -> dict:
        return self.observe({'accepted': False, 'category': category, 'reason': reason,
                             'raw_status': None, 'backend_status': None,
                             'budget_overrun': category == 'budget_timeout', 'deadline_overrun': False})
