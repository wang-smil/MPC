# Lesson 10 Kalman Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a complete discrete Kalman-filter lesson for the identified one-axis servo, including confidence, tuning, comparison, robustness, tests, logs, figures, and report.

**Architecture:** Reuse the identified joint and discrete LQR conventions from Lessons 08 and 09.  A recursive, posterior-form Kalman filter receives only encoder position plus applied torque; it uses Joseph covariance update.  One experiment runner shares a deterministic plant/noise loop across every estimator and produces teaching artifacts.

**Tech Stack:** Python 3.11, NumPy, SciPy, python-control, Matplotlib, PyYAML, unittest.

**Spec:** `docs/superpowers/specs/2026-09-18-lesson10-kalman-filter-design.md`

## Global Constraints

- Reuse Lesson 06 identified `J=0.019762`, `b=0.080257` through the Lesson 08 ZOH/LQR conventions; do not create an independent model.
- Use `dt_s=0.001`, scalar position measurement, scalar torque input, and the applied/clipped torque in estimator prediction.
- Use Joseph covariance update and test symmetry/positive semidefiniteness.
- All simulations use deterministic seeded noise and write only under `lesson10_kalman_filter/`.
- Work directly on `main` because the user explicitly requested no worktree.

---

### Task 1: Lesson scaffold, shared model, and physical noise covariances

**Files:**
- Create: `lesson10_kalman_filter/__init__.py`
- Create: `lesson10_kalman_filter/src/__init__.py`
- Create: `lesson10_kalman_filter/config/kalman.yaml`
- Create: `lesson10_kalman_filter/src/model_loader.py`
- Create: `lesson10_kalman_filter/src/noise_model.py`
- Create: `lesson10_kalman_filter/tests/__init__.py`
- Create: `lesson10_kalman_filter/tests/test_covariance.py`

**Interfaces:**
- Produces `load_config(path) -> dict`, `build_identified_discrete_model(config) -> dict`, `build_balanced_lqr_gain(Ad, Bd, config) -> np.ndarray`.
- Produces `build_noise_covariances(config, q_scale=1.0, r_scale=1.0) -> dict` with `Gd`, `Q_process`, `R_measurement`, `P0`.

- [ ] **Step 1: Write failing covariance tests**

```python
def test_physical_standard_deviations_become_scalar_variances():
    covariance = build_noise_covariances(config)
    assert covariance["Q_process"].shape == (1, 1)
    assert covariance["R_measurement"].shape == (1, 1)
    assert covariance["P0"].shape == (2, 2)
    assert covariance["R_measurement"].item() > 0.0

def test_nonpositive_noise_scales_are_rejected():
    with self.assertRaises(ValueError):
        build_noise_covariances(config, r_scale=0.0)
```

- [ ] **Step 2: Run the covariance test and verify RED**

Run: `conda run -n robot-control python -m unittest lesson10_kalman_filter.tests.test_covariance -v`

Expected: import failure because Lesson 10 does not yet exist.

- [ ] **Step 3: Implement model and covariance helpers**

```python
A, B = build_continuous_model(inertia, damping)
Ad, Bd = discretize_zoh(A, B, dt_s)
C = np.array([[1.0, 0.0]])
Gd = Bd.copy()
Q = np.array([[disturbance_torque_std_nm ** 2]])
R = np.array([[np.deg2rad(position_noise_std_deg) ** 2]])
P0 = np.diag([np.deg2rad(position_std_deg) ** 2, velocity_std_rad_s ** 2])
```

- [ ] **Step 4: Run the covariance test and verify GREEN**

Run: `conda run -n robot-control python -m unittest lesson10_kalman_filter.tests.test_covariance -v`

Expected: PASS.

- [ ] **Step 5: Commit scaffold and covariance helpers**

```bash
git add lesson10_kalman_filter
git commit -m "feat: scaffold lesson10 kalman model"
```

### Task 2: Recursive posterior-form Kalman filter

**Files:**
- Create: `lesson10_kalman_filter/src/kalman_filter.py`
- Create: `lesson10_kalman_filter/tests/test_kalman_convergence.py`
- Modify: `lesson10_kalman_filter/tests/test_covariance.py`

**Interfaces:**
- Produces `DiscreteKalmanFilter(Ad, Bd, C, Gd, Q_process, R_measurement, x0, P0)`.
- Produces `predict(applied_input: float) -> dict` and `update(measurement: float) -> dict`.
- `update` returns `x_hat`, `P`, `K`, `innovation`, `innovation_covariance`, and `nis`.

- [ ] **Step 1: Write failing filter behavior tests**

```python
def test_update_keeps_covariance_symmetric_and_positive_semidefinite():
    for _ in range(200):
        kf.predict(0.1)
        result = kf.update(0.2)
    np.testing.assert_allclose(result["P"], result["P"].T, atol=1e-10)
    self.assertGreaterEqual(np.min(np.linalg.eigvalsh(result["P"])), -1e-10)

def test_update_rejects_vector_measurement():
    with self.assertRaises(ValueError):
        kf.update([0.0, 1.0])
```

- [ ] **Step 2: Run filter tests and verify RED**

Run: `conda run -n robot-control python -m unittest lesson10_kalman_filter.tests.test_kalman_convergence -v`

Expected: import failure because `DiscreteKalmanFilter` is missing.

- [ ] **Step 3: Implement predict/update with Joseph form**

```python
innovation = y - C @ x_prior
S = C @ P_prior @ C.T + R
K = np.linalg.solve(S.T, (P_prior @ C.T).T).T
I_KC = np.eye(2) - K @ C
P_post = I_KC @ P_prior @ I_KC.T + K @ R @ K.T
P_post = 0.5 * (P_post + P_post.T)
nis = float(innovation.T @ np.linalg.solve(S, innovation))
```

- [ ] **Step 4: Run all current Lesson 10 tests and verify GREEN**

Run: `conda run -n robot-control python -m unittest discover lesson10_kalman_filter/tests -v`

Expected: all covariance and filter tests PASS.

- [ ] **Step 5: Commit recursive filter**

```bash
git add lesson10_kalman_filter
git commit -m "feat: add recursive kalman filter"
```

### Task 3: Steady-state reference and estimator adapters

**Files:**
- Create: `lesson10_kalman_filter/src/steady_state_kf.py`
- Create: `lesson10_kalman_filter/src/estimators.py`
- Create: `lesson10_kalman_filter/tests/test_filter_stability.py`

**Interfaces:**
- Produces `design_steady_state_kf(Ad, Gd, C, Q, R) -> dict` using `ct.dlqe`.
- Produces `build_estimator(kind, model, covariance, initial_state, lesson09_observer=None)` exposing `predict`, `update`, and common logged fields.

- [ ] **Step 1: Write failing steady-state and gain-convergence tests**

```python
def test_recursive_gain_converges_to_a_finite_steady_state_reference():
    reference = design_steady_state_kf(Ad, Gd, C, Q, R)
    for _ in range(3000):
        kf.predict(0.0)
        latest = kf.update(0.0)
    np.testing.assert_allclose(latest["K"], reference["filter_gain"], rtol=0.05, atol=1e-5)
```

- [ ] **Step 2: Run the stability test and verify RED**

Run: `conda run -n robot-control python -m unittest lesson10_kalman_filter.tests.test_filter_stability -v`

Expected: import failure because steady-state design is missing.

- [ ] **Step 3: Implement the explicit posterior/predictor timing conversion and adapters**

```python
L_predictor, P_ss, poles = ct.dlqe(Ad, Gd, C, Q, R)
# Document that dlqe gain is predictor-form; comparison tests use the matching gain convention.
```

- [ ] **Step 4: Run the stability test and verify GREEN**

Run: `conda run -n robot-control python -m unittest lesson10_kalman_filter.tests.test_filter_stability -v`

Expected: PASS with documented finite/stable gain.

- [ ] **Step 5: Commit steady-state and estimator layer**

```bash
git add lesson10_kalman_filter
git commit -m "feat: add kalman estimator comparisons"
```

### Task 4: Causal closed-loop simulation and metrics

**Files:**
- Create: `lesson10_kalman_filter/src/closed_loop.py`
- Create: `lesson10_kalman_filter/src/metrics.py`
- Modify: `lesson10_kalman_filter/tests/test_kalman_convergence.py`

**Interfaces:**
- Produces `simulate_closed_loop(config, estimator_kind, assumed_q_scale=1.0, assumed_r_scale=1.0, payload_scale=1.0, load_torque=None) -> dict[str, np.ndarray]`.
- Produces `calculate_metrics(log) -> dict[str, float]`.

- [ ] **Step 1: Write failing causal-loop tests**

```python
def test_closed_loop_logs_clipped_applied_input_and_kf_state():
    log = simulate_closed_loop(config, estimator_kind="kalman")
    self.assertTrue(np.all(np.abs(log["torque_applied_nm"]) <= 3.0))
    self.assertIn("P_qq", log)
    self.assertIn("nis", log)

def test_noise_free_kf_reduces_initial_position_error():
    log = simulate_closed_loop(noiseless_config, estimator_kind="kalman")
    self.assertLess(abs(log["position_error_rad"][-1]), abs(log["position_error_rad"][0]))
```

- [ ] **Step 2: Run the closed-loop tests and verify RED**

Run: `conda run -n robot-control python -m unittest lesson10_kalman_filter.tests.test_kalman_convergence -v`

Expected: failure because simulation is missing.

- [ ] **Step 3: Implement explicit sampled-data timing**

```python
kf.predict(applied_input_previous)
estimate = kf.update(q_measured)
torque_request = float(-(K_lqr @ (estimate["x_hat"] - reference)).item())
torque_applied = float(np.clip(torque_request, -limit, limit))
q_true, dq_true = plant_step(q_true, dq_true, torque_applied, disturbance)
applied_input_previous = torque_applied
```

- [ ] **Step 4: Run all Lesson 10 tests and verify GREEN**

Run: `conda run -n robot-control python -m unittest discover lesson10_kalman_filter/tests -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit simulation and metrics**

```bash
git add lesson10_kalman_filter
git commit -m "feat: add kalman closed loop"
```

### Task 5: Experiment runner, figures, logs, report, and README

**Files:**
- Create: `lesson10_kalman_filter/src/run_experiments.py`
- Create: `lesson10_kalman_filter/tests/test_experiments.py`
- Create: `lesson10_kalman_filter/README.md`
- Create: `lesson10_kalman_filter/figures/`
- Create: `lesson10_kalman_filter/logs/`
- Create: `lesson10_kalman_filter/reports/kalman_engineering_report.md`

**Interfaces:**
- Produces `run_all(config_path, output_root=None) -> dict`.
- Writes named logs for baseline, R sweep, Q sweep, four-estimator comparison, normal, disturbance, stress, and payload mismatch.

- [ ] **Step 1: Write failing artifact test**

```python
def test_runner_writes_required_figures_logs_and_report():
    result = run_all(config_path, output_root=temp_dir)
    self.assertTrue((temp_dir / "figures" / "baseline_confidence.png").exists())
    self.assertTrue((temp_dir / "figures" / "r_q_tuning.png").exists())
    self.assertTrue((temp_dir / "figures" / "estimator_comparison.png").exists())
    self.assertTrue((temp_dir / "reports" / "kalman_engineering_report.md").exists())
```

- [ ] **Step 2: Run artifact test and verify RED**

Run: `conda run -n robot-control python -m unittest lesson10_kalman_filter.tests.test_experiments -v`

Expected: import failure because runner is missing.

- [ ] **Step 3: Implement all five experiment groups and reporting**

```python
baseline = simulate_closed_loop(config, "kalman")
r_sweep = {scale: simulate_closed_loop(config, "kalman", assumed_r_scale=scale) for scale in (0.1, 1.0, 10.0)}
q_sweep = {scale: simulate_closed_loop(config, "kalman", assumed_q_scale=scale, load_torque=step_load) for scale in (0.1, 1.0, 10.0)}
comparison = {name: simulate_closed_loop(config, name) for name in ("raw", "lpf", "luenberger", "kalman")}
```

- [ ] **Step 4: Run runner and all Lesson 10 tests**

Run: `conda run -n robot-control python -m lesson10_kalman_filter.src.run_experiments`

Run: `conda run -n robot-control python -m unittest discover lesson10_kalman_filter/tests -v`

Expected: generated files exist and all tests PASS.

- [ ] **Step 5: Commit full Lesson 10 artifact**

```bash
git add lesson10_kalman_filter
git commit -m "feat: add lesson10 kalman filter experiments"
```

### Task 6: Final verification and teaching handoff

**Files:**
- Modify: `lesson10_kalman_filter/README.md`
- Modify: `lesson10_kalman_filter/reports/kalman_engineering_report.md`

- [ ] **Step 1: Verify all required files and artifact links**

Run: `Get-ChildItem lesson10_kalman_filter -Recurse -File`

Expected: configuration, source, tests, generated figures, logs, report, and README exist.

- [ ] **Step 2: Run final full test suite**

Run: `conda run -n robot-control python -m unittest discover lesson10_kalman_filter/tests -v`

Expected: all tests PASS with no failures.

- [ ] **Step 3: Summarize quantitative experimental results**

Include the observed R/Q tradeoffs, estimator-comparison results, payload/NIS behavior, and exact real-servo tuning order in the report and user handoff.

- [ ] **Step 4: Commit final documentation**

```bash
git add lesson10_kalman_filter
git commit -m "docs: complete lesson10 kalman report"
```
