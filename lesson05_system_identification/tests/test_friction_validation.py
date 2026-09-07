"""Tests for independent validation of models A and B."""

import unittest

from lesson05_system_identification.src.quality_experiments import (
    validate_friction_models,
)


class FrictionValidationTest(unittest.TestCase):
    def test_validation_reports_nominal_and_payload_position_errors(self) -> None:
        results = validate_friction_models(duration_s=2.0, dt_s=0.01)

        self.assertIn("nominal", results)
        self.assertIn("payload_plus_30_percent", results)
        for scenario in results.values():
            self.assertGreaterEqual(scenario["model_a_position_rmse_rad"], 0.0)
            self.assertGreaterEqual(scenario["model_b_position_rmse_rad"], 0.0)


if __name__ == "__main__":
    unittest.main()
