from pathlib import Path
import unittest

import numpy as np

from lesson11_lqg_output_feedback.src.closed_loop import simulate_lqg
from lesson11_lqg_output_feedback.src.model_loader import build_lqg_design, load_config


class PlantModelAlignmentTest(unittest.TestCase):
    def test_simulated_truth_uses_shared_zoh_transition_with_logged_input(self):
        config_path = Path(__file__).parents[1] / "config" / "lqg.yaml"
        config = load_config(config_path)
        config["simulation"]["duration_s"] = 0.004
        design = build_lqg_design(config)
        log = simulate_lqg(config, mode="recursive_lqg")

        x_current = np.array([log["q_true_rad"][0], log["dq_true_rad_s"][0]])
        total_input = (
            log["torque_applied_nm"][0]
            + log["process_disturbance_nm"][0]
            + log["load_torque_nm"][0]
        )
        expected_next = (design["Ad"] @ x_current.reshape(2, 1) + design["Bd"] * total_input).ravel()
        actual_next = np.array([log["q_true_rad"][1], log["dq_true_rad_s"][1]])

        np.testing.assert_allclose(actual_next, expected_next, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
