"""Numerical diagnostics for whether identification data are informative."""

import numpy as np


def analyse_regressor(phi: np.ndarray) -> dict[str, float | int]:
    """Return rank, condition number, and singular-value limits of Phi."""
    matrix = np.asarray(phi, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("phi must be a non-empty two-dimensional matrix.")

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return {
        "rank": int(np.linalg.matrix_rank(matrix)),
        "condition_number": float(np.linalg.cond(matrix)),
        "sigma_max": float(singular_values[0]),
        "sigma_min": float(singular_values[-1]),
    }
