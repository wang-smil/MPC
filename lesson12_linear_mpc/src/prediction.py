"""Finite-horizon state prediction in explicit time-major order."""

from numbers import Integral

import numpy as np


def _finite_array(value, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric values.") from exc
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real numeric values, not complex values.")
    if np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{name} must contain real numeric values, not booleans.")
    try:
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric values.") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validated_model(Ad, Bd) -> tuple[np.ndarray, np.ndarray]:
    A = _finite_array(Ad, "Ad")
    B = _finite_array(Bd, "Bd")
    if A.ndim != 2 or A.shape[0] == 0 or A.shape[0] != A.shape[1]:
        raise ValueError("Ad must be a non-empty square matrix.")
    if B.ndim != 2 or B.shape[0] != A.shape[0] or B.shape[1] == 0:
        raise ValueError("Bd must be a 2D matrix with matching state rows and at least one input.")
    return A, B


def _validated_horizon(horizon) -> int:
    if isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, Integral):
        raise ValueError("horizon must be a positive integer.")
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")
    return int(horizon)


def _validated_sequence(Ad, Bd, x0, U) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A, B = _validated_model(Ad, Bd)
    state = _finite_array(x0, "x0")
    inputs = _finite_array(U, "U")
    if state.ndim != 1 or state.shape[0] != A.shape[0]:
        raise ValueError("x0 must be a vector with one value per state.")
    if inputs.ndim != 2 or inputs.shape[0] == 0 or inputs.shape[1] != B.shape[1]:
        raise ValueError("U must have shape (horizon, number_of_inputs) with a positive horizon.")
    return A, B, state


def build_prediction_matrices(Ad, Bd, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Build F and G for X = F x0 + G U with time-major state/input stacks."""

    A, B = _validated_model(Ad, Bd)
    N = _validated_horizon(horizon)
    nx, nu = A.shape[0], B.shape[1]

    F = np.zeros((N * nx, nx), dtype=float)
    G = np.zeros((N * nx, N * nu), dtype=float)
    powers = [np.eye(nx)]
    for exponent in range(1, N):
        powers.append(powers[-1] @ A)

    for i in range(N):
        row = slice(i * nx, (i + 1) * nx)
        F[row, :] = powers[i] @ A
        for j in range(i + 1):
            col = slice(j * nu, (j + 1) * nu)
            G[row, col] = powers[i - j] @ B
    return F, G


def predict_states_matrix(Ad, Bd, x0, U) -> np.ndarray:
    """Predict x1..xN via X = F x0 + G U; return shape ``(N, nx)``."""

    A, B, state = _validated_sequence(Ad, Bd, x0, U)
    inputs = _finite_array(U, "U")
    F, G = build_prediction_matrices(A, B, inputs.shape[0])
    stacked = F @ state + G @ inputs.reshape(-1, order="C")
    return stacked.reshape((inputs.shape[0], A.shape[0]), order="C")


def rollout_states(Ad, Bd, x0, U) -> np.ndarray:
    """Recursively simulate the discrete model, returning x0 through xN."""

    A, B, state = _validated_sequence(Ad, Bd, x0, U)
    inputs = _finite_array(U, "U")
    states = np.empty((inputs.shape[0] + 1, A.shape[0]), dtype=float)
    states[0] = state
    for k, control in enumerate(inputs):
        states[k + 1] = A @ states[k] + B @ control
    return states
