"""Tests for the Lesson 06 excitation-quality experiment."""

import unittest

from lesson05_system_identification.src.quality_experiments import (
    compare_excitation_quality,
)


class ExcitationQualityTest(unittest.TestCase):
    def test_comparison_returns_one_diagnostic_per_signal(self) -> None:
        results = compare_excitation_quality(duration_s=1.0, dt_s=0.01)

        self.assertEqual(set(results), {"constant", "single_sine", "multisine", "prbs"})
        for diagnostic in results.values():
            self.assertEqual(diagnostic["rank"], 2)
            self.assertGreater(diagnostic["condition_number"], 0.0)


if __name__ == "__main__":
    unittest.main()
