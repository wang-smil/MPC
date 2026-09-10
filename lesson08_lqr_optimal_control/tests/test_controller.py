"""Tests for the LQR control law and actuator clipping record."""

import unittest

import numpy as np


class LqrControllerTest(unittest.TestCase):
    def test_controller_records_feedforward_and_actuator_clipping(self) -> None:
        from lesson08_lqr_optimal_control.src.controller import LQRController

        controller = LQRController(np.array([[10.0, 2.0]]), torque_limit_nm=3.0)
        result = controller.update(
            x=np.array([1.0, 0.0]),
            x_ref=np.zeros(2),
            torque_ff=0.5,
        )

        self.assertEqual(result["torque_unsat_nm"], -9.5)
        self.assertEqual(result["torque_cmd_nm"], -3.0)
        self.assertTrue(result["saturated"])


if __name__ == "__main__":
    unittest.main()
