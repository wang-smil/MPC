import importlib.util
import unittest

import numpy as np

if importlib.util.find_spec('lesson13_mpc_engineering.src.fallback_controller'):
    from lesson13_mpc_engineering.src.fallback_controller import LqrFallbackController
else:
    LqrFallbackController = None


class FallbackControllerTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(LqrFallbackController, 'fallback not implemented')

    def test_fallback_uses_current_estimate(self):
        controller = LqrFallbackController([[2., .5]])
        self.assertAlmostEqual(controller.request([.2, .4], [0, 0], .1), -.5)
        self.assertAlmostEqual(controller.request([.1, .2], [0, 0]), -.3)
        for gain in ([[1.], [2.]], [[True, .5]], [[np.inf, 0]]):
            with self.assertRaises(ValueError):
                LqrFallbackController(gain)
        for args in (([True, 0], [0, 0], 0), ([0, np.nan], [0, 0], 0), ([0, 0], [0, 0], np.inf)):
            with self.assertRaises(ValueError):
                controller.request(*args)
        with self.assertRaises(ValueError):
            LqrFallbackController([[1e308, 0.]]).request([1e308, 0.], [0., 0.])


if __name__ == '__main__':
    unittest.main()
