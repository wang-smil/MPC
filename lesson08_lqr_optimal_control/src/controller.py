"""Engineering LQR controller with explicit actuator clipping."""

import numpy as np


class LQRController:
    """Apply state feedback and preserve both requested and applied torque."""

    def __init__(self, K: np.ndarray, torque_limit_nm: float) -> None:
        K = np.asarray(K, dtype=float)
        if K.shape != (1, 2):
            raise ValueError("K must have shape (1, 2) for position and velocity.")
        if torque_limit_nm <= 0.0:
            raise ValueError("torque_limit_nm must be positive.")
        self.K = K
        self.torque_limit_nm = float(torque_limit_nm)

    def update(
        self,
        x: np.ndarray,
        x_ref: np.ndarray,
        torque_ff: float = 0.0,
    ) -> dict[str, float | bool]:
        """Return feedforward-plus-feedback torque before and after clipping."""

        x = np.asarray(x, dtype=float)
        x_ref = np.asarray(x_ref, dtype=float)
        if x.shape != (2,) or x_ref.shape != (2,):
            raise ValueError("x and x_ref must each have shape (2,).")
        error = x - x_ref
        torque_unsat_nm = float(torque_ff - (self.K @ error).item())
        torque_cmd_nm = float(
            np.clip(torque_unsat_nm, -self.torque_limit_nm, self.torque_limit_nm)
        )
        return {
            "torque_unsat_nm": torque_unsat_nm,
            "torque_cmd_nm": torque_cmd_nm,
            "saturated": bool(abs(torque_unsat_nm) > self.torque_limit_nm),
        }
