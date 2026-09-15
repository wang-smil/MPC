"""Tests for position-only velocity-estimation baselines."""

import unittest

from lesson09_state_observer.src.velocity_estimators import (
    LowPassDifferenceVelocity,
    RawDifferenceVelocity,
)


class VelocityEstimatorTest(unittest.TestCase):
    """Difference estimators provide deterministic comparison baselines."""

    def test_raw_difference_and_lpf_have_known_two_sample_values(self) -> None:
        """The filter uses raw difference then its configured recursive blend."""

        raw = RawDifferenceVelocity(dt_s=0.1)
        filtered = LowPassDifferenceVelocity(dt_s=0.1, alpha=0.8)

        self.assertEqual(raw.update(1.0), 0.0)
        self.assertEqual(filtered.update(1.0), 0.0)
        self.assertEqual(raw.update(1.5), 5.0)
        self.assertAlmostEqual(filtered.update(1.5), 1.0)

    def test_invalid_sampling_or_alpha_is_rejected(self) -> None:
        """Difference needs a positive sample time and a causal filter weight."""

        with self.assertRaises(ValueError):
            RawDifferenceVelocity(dt_s=0.0)
        with self.assertRaises(ValueError):
            LowPassDifferenceVelocity(dt_s=0.001, alpha=1.0)


if __name__ == "__main__":
    unittest.main()
