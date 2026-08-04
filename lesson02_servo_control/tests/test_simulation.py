from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_servo_test


class SimulationTimeAxisTest(unittest.TestCase):
    def test_time_axis_ends_at_configured_duration(self) -> None:
        config = run_servo_test.load_config()
        data = run_servo_test.simulate(config, scenario="normal")

        duration = float(config["simulation"]["duration"])
        dt = float(config["simulation"]["dt"])
        expected_count = int(round(duration / dt)) + 1

        self.assertEqual(len(data["time"]), expected_count)
        self.assertAlmostEqual(data["time"][-1], duration)


if __name__ == "__main__":
    unittest.main()
