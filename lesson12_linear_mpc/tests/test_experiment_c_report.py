"""Regression test for configuration-backed Experiment C reporting."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import yaml

from lesson12_linear_mpc.src.model_loader import load_config
from lesson12_linear_mpc.src.run_experiment_c import run_experiment_c


class ExperimentCReportTest(TestCase):
    def test_report_uses_the_actual_scenario_configuration(self):
        config = load_config(Path(__file__).resolve().parents[1] / 'config' / 'mpc.yaml')
        config['experiment_b']['target_deg'] = 20.0
        config['mpc']['horizon_steps'] = 5
        config['design_limits']['max_torque_nm'] = 4.0
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / 'custom.yaml'
            config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
            result = run_experiment_c(config_path, Path(directory) / 'output', duration_s=0.04)
            report = result['paths']['report'].read_text(encoding='utf-8')
        self.assertIn('20° 目标', report)
        self.assertIn('N=5 步', report)
        self.assertIn('4 N·m', report)
