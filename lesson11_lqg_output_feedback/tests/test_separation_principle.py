from pathlib import Path
import unittest

import numpy as np

from lesson11_lqg_output_feedback.src.model_loader import (
    build_lqg_design,
    load_config,
)
from lesson11_lqg_output_feedback.src.separation_analysis import (
    analyze_separation_principle,
)


class SeparationPrincipleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_path = Path(__file__).parents[1] / "config" / "lqg.yaml"
        cls.design = build_lqg_design(load_config(config_path))

    def test_augmented_poles_equal_controller_and_predictor_estimator_poles(self):
        result = analyze_separation_principle(
            self.design["Ad"],
            self.design["Bd"],
            self.design["C"],
            self.design["K_controller"],
            self.design["L_predictor"],
        )

        expected = np.sort_complex(
            np.concatenate([result["eig_controller"], result["eig_estimator"]])
        )
        np.testing.assert_allclose(
            np.sort_complex(result["eig_augmented"]),
            expected,
            atol=1e-9,
        )
        self.assertTrue(result["is_stable"])

    def test_invalid_gain_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze_separation_principle(
                self.design["Ad"],
                self.design["Bd"],
                self.design["C"],
                np.zeros((2, 1)),
                self.design["L_predictor"],
            )

    def test_analysis_uses_predictor_gain_directly(self):
        result = analyze_separation_principle(
            self.design["Ad"],
            self.design["Bd"],
            self.design["C"],
            self.design["K_controller"],
            self.design["L_predictor"],
        )
        np.testing.assert_allclose(
            result["estimator_matrix"],
            self.design["Ad"] - self.design["L_predictor"] @ self.design["C"],
        )


if __name__ == "__main__":
    unittest.main()
