from pathlib import Path
import unittest

from lesson11_lqg_output_feedback.src.model_loader import (
    build_lqg_design,
    load_config,
)


class SharedDesignTest(unittest.TestCase):
    def test_shared_design_reuses_two_state_one_input_one_output_contract(self):
        config_path = Path(__file__).parents[1] / "config" / "lqg.yaml"
        design = build_lqg_design(load_config(config_path))

        self.assertEqual(design["Ad"].shape, (2, 2))
        self.assertEqual(design["Bd"].shape, (2, 1))
        self.assertEqual(design["C"].shape, (1, 2))
        self.assertEqual(design["K_controller"].shape, (1, 2))
        self.assertEqual(design["L_predictor"].shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
