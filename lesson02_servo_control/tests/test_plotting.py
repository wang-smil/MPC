from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_servo_test


class PlotResultTest(unittest.TestCase):
    def test_plot_result_creates_png(self) -> None:
        plot_result = getattr(run_servo_test, "plot_result", None)
        self.assertIsNotNone(plot_result, "plot_result() 尚未实现")

        data = {
            "time": np.array([0.0, 0.1]),
            "target": np.deg2rad(np.array([30.0, 30.0])),
            "target_velocity": np.deg2rad(np.array([0.0, 0.0])),
            "position": np.deg2rad(np.array([0.0, 10.0])),
            "velocity": np.deg2rad(np.array([0.0, 100.0])),
            "position_measured": np.deg2rad(np.array([0.0, 10.1])),
            "velocity_estimated": np.deg2rad(np.array([0.0, 101.0])),
            "torque_command": np.array([3.5, 1.0]),
            "torque_applied": np.array([3.0, 1.0]),
            "load_torque": np.array([0.0, 0.0]),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(run_servo_test, "ROOT", Path(temp_dir)):
                figure_path = plot_result(data, "normal")

            self.assertTrue(figure_path.is_file())
            self.assertEqual(figure_path.suffix, ".png")


if __name__ == "__main__":
    unittest.main()
