import unittest

import numpy as np

from lesson11_lqg_output_feedback.src.lqg_controller import LQGController


class SpyEstimator:
    def __init__(self):
        self.seen = None

    def step(self, measurement, applied_input):
        self.seen = (measurement, applied_input)
        return {"x_hat": np.array([0.0, 0.0]), "nis": 0.0}


class LQGControllerTest(unittest.TestCase):
    def test_controller_passes_prior_applied_torque_and_clips_request(self):
        spy = SpyEstimator()
        controller = LQGController(spy, np.array([[20.0, 0.0]]), torque_limit=1.0)

        result = controller.step(
            0.2,
            previous_applied_torque=0.4,
            x_ref=np.array([1.0, 0.0]),
        )

        self.assertEqual(spy.seen, (0.2, 0.4))
        self.assertGreater(result["torque_request"], 1.0)
        self.assertEqual(result["torque_applied"], 1.0)

    def test_controller_rejects_vector_measurement_and_bad_reference_shape(self):
        spy = SpyEstimator()
        controller = LQGController(spy, np.array([[1.0, 0.0]]), torque_limit=1.0)

        with self.assertRaises(ValueError):
            controller.step([0.2], 0.0, np.array([0.0, 0.0]))
        with self.assertRaises(ValueError):
            controller.step(0.2, 0.0, np.array([0.0]))
        with self.assertRaises(ValueError):
            LQGController(spy, np.array([[1.0, 0.0]]), torque_limit=0.0)


if __name__ == "__main__":
    unittest.main()
