from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

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
