import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from .test_plant_actuator_demo import demo_config

if importlib.util.find_spec('lesson13_mpc_engineering.src.run_plant_actuator_demo') is not None:
    from lesson13_mpc_engineering.src.run_plant_actuator_demo import simulate_demo, write_outputs
else:
    simulate_demo = write_outputs = None


class DemoExportTest(unittest.TestCase):
    def test_export_csv_preserves_interval_alignment_and_creates_png(self):
        self.assertIsNotNone(write_outputs, 'write_outputs is not implemented yet')
        with tempfile.TemporaryDirectory() as directory:
            paths = write_outputs(simulate_demo(demo_config()), Path(directory), .001, 50.)
            with paths['log'].open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 500)
            self.assertEqual(float(rows[52]['dq_rad_s']), 0)
            self.assertAlmostEqual(float(rows[52]['torque_actual_nm']), .05)
            self.assertAlmostEqual(float(rows[-1]['time_s']), .499)
            self.assertEqual(paths['figure'].read_bytes()[:8], b'\x89PNG\r\n\x1a\n')


if __name__ == '__main__':
    unittest.main()
