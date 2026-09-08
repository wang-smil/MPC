"""Controllability diagnostics for the Lesson 07 state-feedback model."""

import numpy as np


def _validate_state_matrices(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return floating-point state matrices after checking compatible shapes."""

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError("B must have the same number of rows as A.")
    return A, B


def controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Build [B, AB, ..., A^(n-1)B] for an n-state linear system."""

    A, B = _validate_state_matrices(A, B)
    state_dimension = A.shape[0]
    return np.hstack([np.linalg.matrix_power(A, power) @ B for power in range(state_dimension)])


def controllability_report(A: np.ndarray, B: np.ndarray) -> dict[str, float | int]:
    """Calculate rank and SVD-based numerical controllability diagnostics."""

    matrix = controllability_matrix(A, B)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return {
        "rank": int(np.linalg.matrix_rank(matrix)),
        "state_dimension": int(A.shape[0]),
        "condition_number": float(np.linalg.cond(matrix)),
        "sigma_min": float(singular_values[-1]),
    }


def build_uncontrollable_model(inertia: float, damping: float) -> tuple[np.ndarray, np.ndarray]:
    """Create a teaching counterexample with an unactuated unstable third state."""

    if inertia <= 0:
        raise ValueError("inertia must be positive.")
    if damping < 0:
        raise ValueError("damping must be non-negative.")

    A_bad = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, -damping / inertia, 0.0],
            [0.0, 0.0, 0.5],
        ]
    )
    B_bad = np.array([[0.0], [1.0 / inertia], [0.0]])
    return A_bad, B_bad
