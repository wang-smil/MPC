from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import run_experiments

class ArtifactTest(unittest.TestCase):
    def test_all_experiments_write_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(run_experiments,'ROOT',Path(directory)):
                artifacts=run_experiments.run_all_experiments()
            self.assertEqual(set(artifacts),{'pd_vs_pid_load','anti_windup','single_vs_cascade','report'})
            self.assertTrue(all(p.is_file() for p in artifacts.values()))

    def test_windup_case_separates_unbounded_and_clamped_integrals(self):
        no_anti_windup, integral_clamp = run_experiments.run_windup_cases()

        self.assertGreater(np.max(no_anti_windup["integral"]), 1.0)
        self.assertLessEqual(np.max(integral_clamp["integral"]), 0.1)
