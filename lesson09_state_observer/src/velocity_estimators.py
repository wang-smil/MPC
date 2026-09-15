"""Velocity estimates formed from position encoder samples only."""


class RawDifferenceVelocity:
    """Backward-difference velocity estimator."""

    def __init__(self, dt_s: float) -> None:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        self.dt_s = float(dt_s)
        self.previous_position_rad: float | None = None

    def update(self, position_rad: float) -> float:
        """Return velocity from the current and preceding position sample."""

        position_rad = float(position_rad)
        if self.previous_position_rad is None:
            self.previous_position_rad = position_rad
            return 0.0
        velocity_rad_s = (position_rad - self.previous_position_rad) / self.dt_s
        self.previous_position_rad = position_rad
        return velocity_rad_s


class LowPassDifferenceVelocity(RawDifferenceVelocity):
    """Backward difference followed by a first-order low-pass filter."""

    def __init__(self, dt_s: float, alpha: float) -> None:
        super().__init__(dt_s)
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must satisfy 0 <= alpha < 1.")
        self.alpha = float(alpha)
        self.filtered_velocity_rad_s = 0.0

    def update(self, position_rad: float) -> float:
        """Filter each raw difference sample using the configured alpha."""

        raw_velocity_rad_s = super().update(position_rad)
        self.filtered_velocity_rad_s = (
            self.alpha * self.filtered_velocity_rad_s
            + (1.0 - self.alpha) * raw_velocity_rad_s
        )
        return self.filtered_velocity_rad_s
