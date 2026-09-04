from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from excitation import ConstantTorque, PRBSExcitation, SingleSine
from plot_excitations import plot_excitation_comparison


class ExcitationPlotTest(unittest.TestCase):
    def test_comparison_plot_is_written(self):
        signals = {
            "constant": ConstantTorque(0.5),
            "single_sine": SingleSine(0.5, 1.0),
            "prbs": PRBSExcitation(0.5, 0.1, 2026),
        }
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "excitation_compare.png"
            plot_excitation_comparison(
                signals,
                duration_s=1.0,
                dt_s=0.01,
                output_path=output_path,
            )

            self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
