"""Non-condensed, unconstrained finite-horizon MPC solved with CVXPY/OSQP.

Both state and input trajectories are decision variables. Only ``u0`` from
the returned plan is intended to be applied before the next measurement.
"""

import cvxpy as cp
import numpy as np

from lesson12_linear_mpc.src.prediction import (
    _finite_array,
    _validated_horizon,
    _validated_model,
)


def _validated_weight(value, name: str, dimension: int, *, positive: bool) -> np.ndarray:
    matrix = _finite_array(value, name)
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension}).")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be symmetric.")
    matrix = (matrix + matrix.T) / 2.0
    smallest_eigenvalue = float(np.min(np.linalg.eigvalsh(matrix)))
    if positive and smallest_eigenvalue <= 0.0:
        raise ValueError(f"{name} must be positive definite.")
    if not positive and smallest_eigenvalue < -1e-12:
        raise ValueError(f"{name} must be positive semidefinite.")
    return matrix


def _validated_state(value, name: str, dimension: int) -> np.ndarray:
    state = _finite_array(value, name)
    if state.ndim == 0 and dimension == 1:
        state = state.reshape(1)
    if state.shape != (dimension,):
        raise ValueError(f"{name} must be a vector of length {dimension}.")
    return state


def _sqrt_weight(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return np.diag(np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T


class LinearMPCController:
    """Reusable non-condensed QP for a fixed linear model and horizon."""

    def __init__(
        self,
        Ad: np.ndarray,
        Bd: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        P_terminal: np.ndarray,
        horizon: int,
        torque_bounds: tuple[float, float] | None = None,
    ) -> None:
        self.Ad, self.Bd = _validated_model(Ad, Bd)
        self.horizon = _validated_horizon(horizon)
        self.nx, self.nu = self.Bd.shape
        if torque_bounds is not None:
            bounds = _finite_array(torque_bounds, 'torque_bounds')
            if bounds.shape != (2,) or bounds[0] >= bounds[1]:
                raise ValueError('torque_bounds must be a finite increasing pair.')
            self.torque_bounds = (float(bounds[0]), float(bounds[1]))
        else:
            self.torque_bounds = None
        self.Q = _validated_weight(Q, "Q", self.nx, positive=False)
        self.R = _validated_weight(R, "R", self.nu, positive=True)
        self.P_terminal = _validated_weight(
            P_terminal, "P_terminal", self.nx, positive=False
        )

        self.X = cp.Variable((self.nx, self.horizon + 1), name="X")
        self.U = cp.Variable((self.nu, self.horizon), name="U")
        self.x0 = cp.Parameter(self.nx, name="x0")
        self.x_ref = cp.Parameter(self.nx, name="x_ref")

        Q_factor = _sqrt_weight(self.Q)
        R_factor = _sqrt_weight(self.R)
        P_factor = _sqrt_weight(self.P_terminal)
        constraints = [self.X[:, 0] == self.x0]
        cost = 0
        for k in range(self.horizon):
            error = self.X[:, k] - self.x_ref
            cost += cp.sum_squares(Q_factor @ error) + cp.sum_squares(R_factor @ self.U[:, k])
            constraints.append(
                self.X[:, k + 1] == self.Ad @ self.X[:, k] + self.Bd @ self.U[:, k]
            )
        terminal_error = self.X[:, self.horizon] - self.x_ref
        if self.torque_bounds is not None:
            constraints.extend((self.U >= self.torque_bounds[0], self.U <= self.torque_bounds[1]))
        cost += cp.sum_squares(P_factor @ terminal_error)
        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, x_hat: np.ndarray, x_ref: np.ndarray) -> dict:
        """Solve once; return diagnostics and an executable first input only on success."""

        self.x0.value = _validated_state(x_hat, "x_hat", self.nx)
        self.x_ref.value = _validated_state(x_ref, "x_ref", self.nx)
        result = {
            "status": None,
            "u0": None,
            "X": None,
            "U": None,
            "objective": None,
            "solve_time_s": None,
            "iterations": None,
            "error": None,
        }
        try:
            self.problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                eps_abs=1e-8,
                eps_rel=1e-8,
            )
        except Exception as exc:
            result["status"] = "solver_error"
            result["error"] = str(exc)
            return result

        result["status"] = self.problem.status
        stats = self.problem.solver_stats
        if stats is not None:
            if stats.solve_time is not None:
                result["solve_time_s"] = float(stats.solve_time)
            if stats.num_iters is not None:
                result["iterations"] = int(stats.num_iters)

        if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            result["error"] = f"OSQP did not solve the MPC problem: {self.problem.status}"
            return result
        if self.X.value is None or self.U.value is None or self.problem.value is None:
            result["status"] = "invalid_solution"
            result["error"] = "OSQP reported success without a complete solution."
            return result

        X = np.asarray(self.X.value, dtype=float)
        U = np.asarray(self.U.value, dtype=float)
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(U)):
            result["status"] = "invalid_solution"
            result["error"] = "OSQP returned non-finite decision variables."
            return result
        result["X"] = X.copy()
        result["U"] = U.copy()
        result["u0"] = U[:, 0].copy()
        result["objective"] = float(self.problem.value)
        return result
