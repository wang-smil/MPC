"""Quality metrics for system-identification results."""

import numpy as np


def relative_error_percent(estimate: float, truth: float) -> float:
    """Return the magnitude of a relative parameter error in percent."""

    if truth == 0.0:
        raise ValueError("truth must be non-zero for relative error.")
    return abs(estimate - truth) / abs(truth) * 100.0


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return root-mean-square error for matching one-dimensional signals."""

    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if actual_array.shape != predicted_array.shape or actual_array.size == 0:
        raise ValueError("actual and predicted must be matching non-empty arrays.")
    return float(np.sqrt(np.mean((actual_array - predicted_array) ** 2)))


def calculate_metrics(
    estimate: dict[str, float],
    truth: dict[str, float],
    validation_true_position: np.ndarray,
    validation_model_position: np.ndarray,
) -> dict[str, float]:
    """Collect fit and independent-validation metrics without pass/fail thresholds."""

    return {
        "inertia_error_percent": relative_error_percent(
            estimate["inertia_hat"], truth["inertia"]
        ),
        "damping_error_percent": relative_error_percent(
            estimate["damping_hat"], truth["damping"]
        ),
        "torque_rmse": float(estimate["torque_rmse"]),
        "condition_number": float(estimate["condition_number"]),
        "validation_position_rmse_rad": rmse(
            validation_true_position, validation_model_position
        ),
    }
