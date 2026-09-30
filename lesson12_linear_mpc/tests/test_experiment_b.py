"""End-to-end checks for the unconstrained MPC/LQR comparison."""

import csv
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np


LESSON_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = LESSON_DIR / "config" / "mpc.yaml"


class ExperimentBTest(TestCase):
    def runner(self):
        try:
            return import_module("lesson12_linear_mpc.src.run_experiment_b").run_experiment_b
        except ImportError as exc:
            self.fail(f"Experiment B runner is unavailable: {exc}")

    def test_runner_writes_aligned_comparison_artifacts(self):
        run = self.runner()
        with TemporaryDirectory() as temporary_dir:
            result = run(CONFIG_PATH, output_root=temporary_dir, duration_s=0.05)
            for path in result["paths"].values():
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 0)

            with result["paths"]["csv"].open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 4 * 50)
            required = {
                "case", "time_s", "q_true_rad", "dq_true_rad_s", "q_measured_rad",
                "q_hat_rad", "dq_hat_rad_s", "q_ref_rad", "torque_applied_nm",
                "controller_mode", "horizon_steps", "control_update", "qp_status",
                "qp_objective", "qp_solve_time_s", "qp_iterations",
            }
            self.assertTrue(required.issubset(rows[0]))
            self.assertEqual(set(row["case"] for row in rows), {"lqr", "mpc_N1", "mpc_N5", "mpc_N20"})
            self.assertEqual(sum(row["control_update"] == "True" for row in rows), 4 * 5)

    def test_each_horizon_solves_and_tracks_the_lqr_baseline(self):
        run = self.runner()
        with TemporaryDirectory() as temporary_dir:
            result = run(CONFIG_PATH, output_root=temporary_dir, duration_s=0.05)
        for horizon in (1, 5, 20):
            with self.subTest(horizon=horizon):
                metrics = result["metrics"][f"mpc_N{horizon}"]
                self.assertEqual(sum(metrics["solver_status_counts"].values()), 5)
                self.assertTrue(
                    set(metrics["solver_status_counts"]).issubset(
                        {"optimal", "optimal_inaccurate"}
                    )
                )
                self.assertLessEqual(metrics["max_abs_torque_difference_nm"], 1e-4)
                self.assertGreaterEqual(metrics["solve_time_mean_s"], 0.0)
                self.assertGreaterEqual(metrics["solve_time_p95_s"], 0.0)
                self.assertGreaterEqual(metrics["solve_time_max_s"], 0.0)

    def test_all_cases_receive_identical_seeded_noise(self):
        run = self.runner()
        with TemporaryDirectory() as temporary_dir:
            first = run(CONFIG_PATH, output_root=temporary_dir, duration_s=0.05)
            second = run(CONFIG_PATH, output_root=temporary_dir, duration_s=0.05)
        baseline_noise = first["traces"]["lqr"]["measurement_noise_rad"]
        for case, trace in first["traces"].items():
            with self.subTest(case=case):
                np.testing.assert_array_equal(trace["measurement_noise_rad"], baseline_noise)
                np.testing.assert_allclose(
                    trace["torque_applied_nm"],
                    second["traces"][case]["torque_applied_nm"],
                    atol=1e-9, rtol=0.0,
                )
