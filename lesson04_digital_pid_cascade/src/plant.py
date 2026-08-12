"""第4课单轴关节模型；详细仿真循环位于 run_experiments.py。"""

class JointPlant:
    def __init__(self, inertia=0.020, damping=0.080):
        self.inertia, self.damping = inertia, damping
        self.position_rad = self.velocity_rad_s = 0.0

    def step(self, torque_cmd, load_torque, dt):
        acceleration = (torque_cmd - load_torque - self.damping * self.velocity_rad_s) / self.inertia
        self.velocity_rad_s += acceleration * dt
        self.position_rad += self.velocity_rad_s * dt
        return self.position_rad, self.velocity_rad_s
