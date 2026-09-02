"""True and identified single-axis joint models for Lesson 05."""

import numpy as np


class SingleAxisPlant:
    """Torque-limited joint with viscous and Coulomb friction."""

    def __init__(
        self,
        inertia: float,
        damping: float,
        coulomb_friction: float,
        torque_limit_nm: float,
    ) -> None:
        if inertia <= 0.0:
            raise ValueError("inertia must be positive.")
        if damping < 0.0 or coulomb_friction < 0.0:
            raise ValueError("damping and coulomb_friction cannot be negative.")
        if torque_limit_nm <= 0.0:
            raise ValueError("torque_limit_nm must be positive.")

        self.inertia = float(inertia)
        self.damping = float(damping)
        self.coulomb_friction = float(coulomb_friction)
        self.torque_limit_nm = float(torque_limit_nm)
        self.position_rad = 0.0
        self.velocity_rad_s = 0.0

    def step(self, torque_command_nm: float, dt_s: float) -> dict[str, float]:
        """Advance one semi-implicit-Euler simulation sample."""

        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")

        torque_applied = float(
            np.clip(torque_command_nm, -self.torque_limit_nm, self.torque_limit_nm)
        )
        friction = self.coulomb_friction * np.sign(self.velocity_rad_s)
        acceleration = (
            torque_applied - self.damping * self.velocity_rad_s - friction
        ) / self.inertia
        self.velocity_rad_s += acceleration * dt_s
        self.position_rad += self.velocity_rad_s * dt_s

        return {
            "torque_applied_nm": torque_applied,
            "position_true_rad": self.position_rad,
            "velocity_true_rad_s": self.velocity_rad_s,
            "acceleration_true_rad_s2": float(acceleration),
        }


def simulate_linear_model(
    torque_nm: np.ndarray,
    time_s: np.ndarray,
    inertia: float,
    damping: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the friction-free model represented by estimated J and b."""

    torque = np.asarray(torque_nm, dtype=float)
    time = np.asarray(time_s, dtype=float)
    if torque.ndim != 1 or time.ndim != 1 or torque.shape != time.shape:
        raise ValueError("torque_nm and time_s must be matching one-dimensional arrays.")
    if len(time) < 2:
        raise ValueError("at least two time samples are required.")
    if inertia <= 0.0 or damping < 0.0:
        raise ValueError("inertia must be positive and damping cannot be negative.")

    position = np.zeros_like(time)
    velocity = np.zeros_like(time)
    for index in range(1, len(time)):
        dt_s = time[index] - time[index - 1]
        if dt_s <= 0.0:
            raise ValueError("time_s must be strictly increasing.")
        acceleration = (torque[index - 1] - damping * velocity[index - 1]) / inertia
        velocity[index] = velocity[index - 1] + acceleration * dt_s
        position[index] = position[index - 1] + velocity[index] * dt_s

    return position, velocity
