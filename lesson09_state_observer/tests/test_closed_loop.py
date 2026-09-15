"""Tests for observer-based LQR closed-loop simulation."""

import unittest

import numpy as np

from lesson09_state_observer.src.closed_loop import simulate_observer_closed_loop
from lesson09_state_observer.src.model_loader import (
    build_balanced_lqr_gain,
    build_identified_model,
)
from lesson09_state_observer.src.observer import DiscreteObserver
from lesson09_state_observer.src.observer_design import design_discrete_observer


def build_nominal_log(noise_scale: float) -> dict[str, np.ndarray]:
    """Create one deterministic observer-based LQR simulation for tests."""

    config = {
        "simulation": {"dt_s": 0.001, "duration_s": 1.0, "seed": 2026},
        "model": {"inertia": 0.019762, "damping": 0.080257},
        "measurement": {"position_noise_std_deg": 0.05},
        "initial_condition": {
            "q_true_deg": 20.0,
            "dq_true_rad_s": 0.5,
            "q_hat_deg": 0.0,
            "dq_hat_rad_s": 0.0,
        },
        "controller": {
            "target_deg": 30.0,
            "max_position_error_deg": 5.0,
            "max_velocity_error_rad_s": 1.5,
            "torque_limit_nm": 3.0,
        },
        "velocity_difference": {"lpf_alpha": 0.90},
    }
    _, _, Ad, Bd = build_identified_model(config)
    K = build_balanced_lqr_gain(Ad, Bd, config)
    design = design_discrete_observer(
        Ad=Ad,
        C=np.array([[1.0, 0.0]]),
        controller_poles_z=np.linalg.eigvals(Ad - Bd @ K),
        speed_factor=4.0,
        dt_s=0.001,
    )
    observer = DiscreteObserver(
        Ad=Ad,
        Bd=Bd,
        C=np.array([[1.0, 0.0]]),
        L=design["L"],
        x0_hat=np.zeros(2),
    )
    return simulate_observer_closed_loop(
        config=config,
        observer=observer,
        K=K,
        plant_inertia=config["model"]["inertia"],
        noise_scale=noise_scale,
    )


class ClosedLoopTest(unittest.TestCase):
    """Observer-based control preserves physical signal boundaries."""

    def test_simulation_logs_required_signals_and_clipped_torque(self) -> None:
        """Logs include true, measured, estimated states and applied torque."""

        log = build_nominal_log(noise_scale=0.0)

        required = {
            "q_true_rad",
            "q_measured_rad",
            "q_hat_rad",
            "dq_true_rad_s",
            "dq_raw_rad_s",
            "dq_filtered_rad_s",
            "dq_hat_rad_s",
            "q_estimation_error_rad",
            "dq_estimation_error_rad_s",
            "innovation_rad",
            "torque_unsat_nm",
            "torque_cmd_nm",
            "saturated",
        }
        self.assertTrue(required.issubset(log))
        self.assertTrue(np.all(np.abs(log["torque_cmd_nm"]) <= 3.0 + 1e-12))

    def test_noise_free_observer_converges_from_mismatched_initial_state(self) -> None:
        """Without encoder noise, the designed observer eliminates state error."""

        log = build_nominal_log(noise_scale=0.0)

        self.assertLess(abs(log["q_estimation_error_rad"][-1]), 1e-3)
        self.assertLess(abs(log["dq_estimation_error_rad_s"][-1]), 1e-3)


if __name__ == "__main__":
    unittest.main()
