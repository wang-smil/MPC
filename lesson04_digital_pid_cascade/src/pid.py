import numpy as np


class PIDController:
    def __init__(self, kp, ki, kd, output_limit, integral_limit=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_limit, self.integral_limit = output_limit, integral_limit
        self.reset()

    def reset(self):
        self.integral = 0.0

    def update(self, error, measurement_rate, dt):
        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = float(np.clip(self.integral, -self.integral_limit, self.integral_limit))
        raw = self.kp * error + self.ki * self.integral - self.kd * measurement_rate
        command = float(np.clip(raw, -self.output_limit, self.output_limit))
        return float(raw), command, command != raw
