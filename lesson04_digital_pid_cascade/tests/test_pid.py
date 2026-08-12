from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pid import PIDController


class PIDTest(unittest.TestCase):
    def test_clamp_and_output_limit(self):
        controller = PIDController(10, 2, 0, 3, 0.5)
        raw, command, saturated = controller.update(1, 0, 0.1)
        self.assertEqual(raw, 10.2)
        self.assertEqual(command, 3.0)
        self.assertTrue(saturated)
        for _ in range(10): controller.update(1, 0, 0.1)
        self.assertEqual(controller.integral, 0.5)

