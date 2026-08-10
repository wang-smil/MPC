from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import discretize_demo


class DiscretizeModuleContractTest(unittest.TestCase):
    def test_discretize_module_exists(self) -> None:
        module_path = PROJECT_ROOT / "src" / "discretize_demo.py"

        self.assertTrue(
            module_path.is_file(),
            "discretize_demo.py 尚未创建",
        )


class ZohDiscretizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.build_continuous_model = getattr(
            discretize_demo,
            "build_continuous_model",
            None,
        )
        self.discretize_zoh = getattr(
            discretize_demo,
            "discretize_zoh",
            None,
        )

        self.assertIsNotNone(
            self.build_continuous_model,
            "build_continuous_model() 尚未实现",
        )
        self.assertIsNotNone(
            self.discretize_zoh,
            "discretize_zoh() 尚未实现",
        )

    def test_uses_lesson01_model_and_changes_with_sample_time(self) -> None:
        a, b, c, d = self.build_continuous_model()

        np.testing.assert_allclose(
            a,
            np.array([[0.0, 1.0], [-4.0, -0.8]]),
        )
        np.testing.assert_allclose(b, np.array([[0.0], [1.0]]))
        np.testing.assert_allclose(c, np.eye(2))
        np.testing.assert_allclose(d, np.zeros((2, 1)))

        ad_1ms, bd_1ms, cd_1ms, dd_1ms, dt_1ms = (
            self.discretize_zoh(0.001)
        )
        ad_10ms, bd_10ms, _, _, dt_10ms = (
            self.discretize_zoh(0.01)
        )

        self.assertEqual(ad_1ms.shape, (2, 2))
        self.assertEqual(bd_1ms.shape, (2, 1))
        self.assertEqual(cd_1ms.shape, (2, 2))
        self.assertEqual(dd_1ms.shape, (2, 1))
        self.assertAlmostEqual(dt_1ms, 0.001)
        self.assertAlmostEqual(dt_10ms, 0.01)
        self.assertFalse(np.allclose(ad_1ms, ad_10ms))
        self.assertFalse(np.allclose(bd_1ms, bd_10ms))

    def test_nonpositive_sample_time_is_rejected(self) -> None:
        for sample_time_s in (0.0, -0.001):
            with self.subTest(sample_time_s=sample_time_s):
                with self.assertRaises(ValueError):
                    self.discretize_zoh(sample_time_s)


if __name__ == "__main__":
    unittest.main()
