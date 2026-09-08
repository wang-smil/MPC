"""State-feedback control law with actuator saturation logging."""

import numpy as np


def feedback_torque(
    K: np.ndarray,
    state_estimate: np.ndarray,
    reference_state: np.ndarray,
    torque_limit_nm: float,
) -> dict[str, float | bool]:
    """Calculate ``u=-K(x-x_ref)`` and separately record actuator clipping."""

    if torque_limit_nm <= 0.0:
        raise ValueError("torque_limit_nm must be positive.")
    K = np.asarray(K, dtype=float)
    state_estimate = np.asarray(state_estimate, dtype=float)
    reference_state = np.asarray(reference_state, dtype=float)
    if K.shape != (1, state_estimate.size) or reference_state.shape != state_estimate.shape:
        raise ValueError("K and state vectors have incompatible shapes.")

    torque_unsat_nm = float((-K @ (state_estimate - reference_state)).item())
    torque_applied_nm = float(np.clip(torque_unsat_nm, -torque_limit_nm, torque_limit_nm))
    return {
        "torque_unsat_nm": torque_unsat_nm,
        "torque_applied_nm": torque_applied_nm,
        "saturated": bool(torque_applied_nm != torque_unsat_nm),
    }
