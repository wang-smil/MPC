"""Observability calculations for the discrete joint model."""

import numpy as np


def observability_matrix(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Stack C, CA, ..., CA^(n-1) for an n-state discrete system."""

    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if C.ndim != 2 or C.shape[1] != A.shape[0]:
        raise ValueError("C must have one column per state.")

    return np.vstack(
        [C @ np.linalg.matrix_power(A, power) for power in range(A.shape[0])]
    )


def observability_report(A: np.ndarray, C: np.ndarray) -> dict[str, float | int]:
    """Return rank and conditioning indicators for the observability matrix."""

    matrix = observability_matrix(A, C)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return {
        "rank": int(np.linalg.matrix_rank(matrix)),
        "state_dimension": int(A.shape[0]),
        "condition_number": float(np.linalg.cond(matrix)),
        "sigma_min": float(singular_values[-1]),
    }
