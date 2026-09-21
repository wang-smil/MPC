"""Unsaturated linear separation-principle analysis in predictor-gain form."""

import numpy as np


def _matrix(value: object, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return array


def analyze_separation_principle(
    Ad: object,
    Bd: object,
    C: object,
    K_controller: object,
    L_predictor: object,
) -> dict:
    """Analyze controller and predictor error poles for the nominal linear model.

    ``L_predictor`` is the gain returned by ``control.dlqe``.  This function
    therefore uses the predictor convention ``A_e = A_d - L_e C``.  The
    result does not cover actuator clipping or other nonlinear plant effects.
    """

    Ad_matrix = _matrix(Ad, (2, 2), "Ad")
    Bd_matrix = _matrix(Bd, (2, 1), "Bd")
    C_matrix = _matrix(C, (1, 2), "C")
    controller_gain = _matrix(K_controller, (1, 2), "K_controller")
    predictor_gain = _matrix(L_predictor, (2, 1), "L_predictor")

    controller_matrix = Ad_matrix - Bd_matrix @ controller_gain
    estimator_matrix = Ad_matrix - predictor_gain @ C_matrix
    augmented_matrix = np.block(
        [
            [controller_matrix, Bd_matrix @ controller_gain],
            [np.zeros_like(Ad_matrix), estimator_matrix],
        ]
    )
    eig_controller = np.linalg.eigvals(controller_matrix)
    eig_estimator = np.linalg.eigvals(estimator_matrix)
    eig_augmented = np.linalg.eigvals(augmented_matrix)
    expected = np.sort_complex(np.concatenate([eig_controller, eig_estimator]))
    actual = np.sort_complex(eig_augmented)

    return {
        "controller_matrix": controller_matrix,
        "estimator_matrix": estimator_matrix,
        "augmented_matrix": augmented_matrix,
        "eig_controller": eig_controller,
        "eig_estimator": eig_estimator,
        "eig_augmented": eig_augmented,
        "union_error": float(np.max(np.abs(actual - expected))),
        "is_stable": bool(np.all(np.abs(eig_augmented) < 1.0)),
    }
