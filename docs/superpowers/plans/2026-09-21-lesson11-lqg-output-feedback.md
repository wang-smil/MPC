# Lesson 11 LQG Output Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular LQG output-feedback lesson that composes the existing LQR and Kalman modules and verifies theory and limits through reproducible experiments.

**Architecture:** A thin `LQGController` receives an estimator and precomputed LQR gain; it owns only state-feedback composition and actuator clipping.  The simulator alone owns true state and passes position measurement plus prior applied torque to the controller.  Separation analysis is a separate mathematical module so its linear result cannot be mistaken for an assertion about saturated nonlinear behavior.

**Tech Stack:** Python 3.11, NumPy, SciPy, python-control, Matplotlib, PyYAML, unittest.

**Spec:** `docs/superpowers/specs/2026-09-21-lesson11-lqg-output-feedback-design.md`

## Global Constraints

- Import the Lesson 08 balanced LQR construction and Lesson 10 model, covariance, recursive KF, and steady-state `dlqe` solution; do not copy their algorithms.
- The estimator receives scalar encoder position and scalar previous **applied** torque only; it must never receive true state or unclipped requested torque.
- Define all separation matrices using the `dlqe` predictor-form estimator gain and document this convention.
- Use the same deterministic plant/noise realization for each controller comparison in a scenario.
- Keep saturation in the simulation boundary and explicitly mark the separation result as unsaturated linear theory.
- Create all source, tests, logs, figures, and reports under `lesson11_lqg_output_feedback/`.

## Review Focus

- A requested torque outside limits must reach the next estimator prediction only after clipping; test the exact scalar passed to a spy estimator.
- A vector measurement, vector previous torque, invalid reference shape, or nonpositive torque limit must fail with a clear `ValueError`.
- `dlqe` predictor gain and recursive posterior gain must not be compared without their time-index conversion; test the intended predictor convention directly.
- Full-state LQR must be labelled simulation-only and never instantiate or call an estimator.
- Reduced torque limit must increase measurable saturation exposure without using linear pole analysis as a performance guarantee.

---

### Task 1: Lesson scaffold and shared-model adapters

**Files:**
- Create: `lesson11_lqg_output_feedback/__init__.py`
- Create: `lesson11_lqg_output_feedback/src/__init__.py`
- Create: `lesson11_lqg_output_feedback/config/lqg.yaml`
- Create: `lesson11_lqg_output_feedback/src/model_loader.py`
- Create: `lesson11_lqg_output_feedback/tests/__init__.py`
- Create: `lesson11_lqg_output_feedback/tests/test_augmented_stability.py`

**Interfaces:**
- Consumes `lesson10_kalman_filter.src.model_loader.build_identified_discrete_model`, `build_balanced_lqr_gain`, `lesson10_kalman_filter.src.noise_model.build_noise_covariances`, and `design_steady_state_kf`.
- Produces `load_config(path: Path | str) -> dict` and `build_lqg_design(config: dict) -> dict[str, np.ndarray]` with `Ad`, `Bd`, `C`, `Gd`, `Q_process`, `R_measurement`, `P0`, `K_controller`, and `L_predictor`.

- [ ] **Step 1: Write the failing shared-design test**

```python
def test_shared_design_reuses_two_state_one_input_one_output_contract(self):
    design = build_lqg_design(config)
    self.assertEqual(design["Ad"].shape, (2, 2))
    self.assertEqual(design["Bd"].shape, (2, 1))
    self.assertEqual(design["C"].shape, (1, 2))
    self.assertEqual(design["K_controller"].shape, (1, 2))
    self.assertEqual(design["L_predictor"].shape, (2, 1))
```

- [ ] **Step 2: Run test to verify RED**

Run: `conda run -n robot-control python -m unittest lesson11_lqg_output_feedback.tests.test_augmented_stability -v`

Expected: import failure because Lesson 11 adapter does not exist.

- [ ] **Step 3: Implement thin configuration/model adapter**

```python
model = build_identified_discrete_model(config)
covariance = build_noise_covariances(config, model["Bd"])
K_controller = build_balanced_lqr_gain(model["Ad"], model["Bd"], config)
steady = design_steady_state_kf(
    model["Ad"], covariance["Gd"], model["C"],
    covariance["Q_process"], covariance["R_measurement"],
)
```

- [ ] **Step 4: Run test to verify GREEN**

Run: `conda run -n robot-control python -m unittest lesson11_lqg_output_feedback.tests.test_augmented_stability -v`

Expected: PASS.

- [ ] **Step 5: Commit shared adapters**

```bash
git add lesson11_lqg_output_feedback
git commit -m "feat: scaffold lesson11 lqg design"
```

### Task 2: Composition-only LQG controller and causal timing

**Files:**
- Create: `lesson11_lqg_output_feedback/src/lqg_controller.py`
- Create: `lesson11_lqg_output_feedback/tests/test_causal_timing.py`

**Interfaces:**
- Consumes any `estimator` with `step(measurement: float, applied_input: float) -> dict` returning a two-element `x_hat`.
- Produces `LQGController(estimator, K_controller, torque_limit)` and `step(measurement, previous_applied_torque, x_ref, torque_ff=0.0) -> dict`.

- [ ] **Step 1: Write failing controller timing and clipping tests**

```python
class SpyEstimator:
    def step(self, measurement, applied_input):
        self.seen = (measurement, applied_input)
        return {"x_hat": np.array([0.0, 0.0]), "nis": 0.0}

def test_controller_passes_prior_applied_torque_and_clips_request(self):
    spy = SpyEstimator()
    controller = LQGController(spy, np.array([[20.0, 0.0]]), torque_limit=1.0)
    result = controller.step(0.2, previous_applied_torque=0.4, x_ref=np.array([1.0, 0.0]))
    self.assertEqual(spy.seen, (0.2, 0.4))
    self.assertGreater(result["torque_request"], 1.0)
    self.assertEqual(result["torque_applied"], 1.0)

def test_controller_rejects_vector_measurement_and_bad_reference_shape(self):
    with self.assertRaises(ValueError):
        controller.step([0.2], 0.0, np.array([0.0, 0.0]))
    with self.assertRaises(ValueError):
        controller.step(0.2, 0.0, np.array([0.0]))
    with self.assertRaises(ValueError):
        LQGController(spy, np.array([[1.0, 0.0]]), torque_limit=0.0)
```

- [ ] **Step 2: Run timing test to verify RED**

Run: `conda run -n robot-control python -m unittest lesson11_lqg_output_feedback.tests.test_causal_timing -v`

Expected: import failure because `LQGController` is missing.

- [ ] **Step 3: Implement the composition layer without KF/LQR duplication**

```python
estimate = self.estimator.step(measurement_scalar, previous_applied_scalar)
x_hat = _state_vector(estimate["x_hat"])
torque_request = torque_ff_scalar - float((self.Kc @ (x_hat - x_ref_vector)).item())
torque_applied = float(np.clip(torque_request, -self.limit, self.limit))
return {"x_hat": x_hat, "torque_request": torque_request,
        "torque_applied": torque_applied, "saturated": abs(torque_request) > self.limit,
        "estimator": estimate}
```

- [ ] **Step 4: Run timing test to verify GREEN**

Run: `conda run -n robot-control python -m unittest lesson11_lqg_output_feedback.tests.test_causal_timing -v`

Expected: PASS.

- [ ] **Step 5: Commit controller**

```bash
git add lesson11_lqg_output_feedback
git commit -m "feat: add causal lqg controller"
```

### Task 3: Separation-principle mathematics

**Files:**
- Create: `lesson11_lqg_output_feedback/src/separation_analysis.py`
- Create: `lesson11_lqg_output_feedback/tests/test_separation_principle.py`
- Modify: `lesson11_lqg_output_feedback/tests/test_augmented_stability.py`

**Interfaces:**
- Consumes `Ad: (2,2)`, `Bd: (2,1)`, `C: (1,2)`, `K_controller: (1,2)`, and predictor `L_predictor: (2,1)`.
- Produces `analyze_separation_principle(...) -> dict` with `controller_matrix`, `estimator_matrix`, `augmented_matrix`, `eig_controller`, `eig_estimator`, `eig_augmented`, `union_error`, and `is_stable`.

- [ ] **Step 1: Write failing eigenvalue-union test**

```python
def test_augmented_poles_equal_controller_and_predictor_estimator_poles(self):
    result = analyze_separation_principle(Ad, Bd, C, Kc, Le)
    expected = np.sort_complex(np.concatenate([result["eig_controller"], result["eig_estimator"]]))
    np.testing.assert_allclose(np.sort_complex(result["eig_augmented"]), expected, atol=1e-9)
    self.assertTrue(result["is_stable"])

def test_invalid_gain_shape_is_rejected(self):
    with self.assertRaises(ValueError):
        analyze_separation_principle(Ad, Bd, C, np.zeros((2, 1)), Le)

def test_analysis_uses_predictor_gain_directly(self):
    result = analyze_separation_principle(Ad, Bd, C, Kc, Le)
    np.testing.assert_allclose(result["estimator_matrix"], Ad - Le @ C)
```

- [ ] **Step 2: Run separation test to verify RED**

Run: `conda run -n robot-control python -m unittest lesson11_lqg_output_feedback.tests.test_separation_principle -v`

Expected: import failure because analysis module is missing.

- [ ] **Step 3: Implement the predictor-form augmented system**

```python
Ac = Ad - Bd @ K_controller
Ae = Ad - L_predictor @ C
A_aug = np.block([[Ac, Bd @ K_controller],
                  [np.zeros_like(Ad), Ae]])
eig_augmented = np.linalg.eigvals(A_aug)
union_error = float(np.max(np.abs(
    np.sort_complex(eig_augmented) - np.sort_complex(np.concatenate([np.linalg.eigvals(Ac), np.linalg.eigvals(Ae)]))
)))
```

- [ ] **Step 4: Run all Lesson 11 unit tests to verify GREEN**

Run: `conda run -n robot-control python -m unittest discover lesson11_lqg_output_feedback/tests -v`

Expected: all current tests PASS.

- [ ] **Step 5: Commit separation analysis**

```bash
git add lesson11_lqg_output_feedback
git commit -m "feat: verify lqg separation principle"
```

### Task 4: Recursive/steady estimators, fair plant simulation, and metrics

**Files:**
- Create: `lesson11_lqg_output_feedback/src/closed_loop.py`
- Create: `lesson11_lqg_output_feedback/src/metrics.py`
- Modify: `lesson11_lqg_output_feedback/tests/test_causal_timing.py`

**Interfaces:**
- Produces `simulate_lqg(config, mode, q_scale=1.0, torque_limit=None, load_torque=None) -> dict[str, np.ndarray]`, where `mode` is `full_state_lqr`, `recursive_lqg`, or `steady_state_lqg`.
- Produces `calculate_metrics(log, startup_end_s=0.5) -> dict[str, float]`.

- [ ] **Step 1: Write failing fair-simulation tests**

```python
def test_lqg_log_uses_estimate_and_records_actual_limited_torque(self):
    log = simulate_lqg(config, mode="recursive_lqg")
    self.assertIn("q_hat_rad", log)
    self.assertIn("nis", log)
    self.assertTrue(np.all(np.abs(log["torque_applied_nm"]) <= 3.0))

def test_full_state_baseline_has_no_estimator_diagnostics(self):
    log = simulate_lqg(config, mode="full_state_lqr")
    self.assertTrue(np.all(np.isnan(log["nis"])))
    self.assertEqual(log["controller_mode"][0], "full_state_lqr")
```

- [ ] **Step 2: Run simulation test to verify RED**

Run: `conda run -n robot-control python -m unittest lesson11_lqg_output_feedback.tests.test_causal_timing -v`

Expected: failure because `simulate_lqg` is missing.

- [ ] **Step 3: Implement recursive and steady-state estimator adapters plus the three simulation modes**

```python
# recursive_lqg: reused DiscreteKalmanFilter predict/update adapter
# steady_state_lqg: x_prior = Ad @ x + Bd * u_prev;
#                    x = x_prior + K_post @ (y - C @ x_prior),
#                    with K_post obtained from the documented predictor gain convention.
# full_state_lqr: x_for_control = np.array([q_true, dq_true])  # simulation reference only
result = controller.step(q_measured, applied_input_previous, reference)
```

- [ ] **Step 4: Implement startup/steady metrics and run all tests**

Run: `conda run -n robot-control python -m unittest discover lesson11_lqg_output_feedback/tests -v`

Expected: PASS; metrics distinguish `[0, 0.5]` s and `(0.5, duration]` s using logged timestamps.

- [ ] **Step 5: Commit simulation layer**

```bash
git add lesson11_lqg_output_feedback
git commit -m "feat: add lqg closed loop experiments"
```

### Task 5: Experiment runner, report, figures, and README

**Files:**
- Create: `lesson11_lqg_output_feedback/src/run_experiments.py`
- Create: `lesson11_lqg_output_feedback/tests/test_experiments.py`
- Create: `lesson11_lqg_output_feedback/README.md`
- Create: `lesson11_lqg_output_feedback/figures/`
- Create: `lesson11_lqg_output_feedback/logs/`
- Create: `lesson11_lqg_output_feedback/reports/lqg_engineering_report.md`

**Interfaces:**
- Produces `run_all(config_path: Path | str, output_root: Path | str | None = None) -> dict`.
- Writes Experiment A–E logs, five requested figures, and Markdown report.

- [ ] **Step 1: Write failing artifact test with test-only short duration**

```python
def test_runner_writes_all_lqg_artifacts(self):
    config["simulation"]["duration_s"] = 0.05
    result = run_all(short_config_path, output_root=temp_dir)
    for name in ("separation_poles.png", "full_state_vs_lqg.png",
                 "recursive_vs_steady_state.png", "q_tuning_coupling.png",
                 "saturation_boundary.png"):
        self.assertTrue((temp_dir / "figures" / name).exists())
    self.assertTrue((temp_dir / "reports" / "lqg_engineering_report.md").exists())
    self.assertIn("separation", result)
    self.assertGreaterEqual(
        result["boundary_metrics"]["reduced_limit"]["saturation_ratio_percent"],
        result["boundary_metrics"]["normal"]["saturation_ratio_percent"],
    )
```

- [ ] **Step 2: Run artifact test to verify RED**

Run: `conda run -n robot-control python -m unittest lesson11_lqg_output_feedback.tests.test_experiments -v`

Expected: import failure because runner is missing.

- [ ] **Step 3: Implement experiments A–E with identical seeds within each comparison**

```python
separation = analyze_separation_principle(...)
full_state = simulate_lqg(config, "full_state_lqr", load_torque=step_load)
recursive = simulate_lqg(config, "recursive_lqg", load_torque=step_load)
steady = simulate_lqg(config, "steady_state_lqg", load_torque=step_load)
q_sweep = {scale: simulate_lqg(config, "recursive_lqg", q_scale=scale, load_torque=step_load)
           for scale in (0.1, 1.0, 10.0)}
boundary = {"normal": simulate_lqg(config, "recursive_lqg"),
            "load": simulate_lqg(config, "recursive_lqg", load_torque=step_load),
            "reduced_limit": simulate_lqg(config, "recursive_lqg", torque_limit=0.4, load_torque=step_load)}
```

- [ ] **Step 4: Generate formal artifacts and verify tests**

Run: `conda run -n robot-control python -m lesson11_lqg_output_feedback.src.run_experiments`

Run: `conda run -n robot-control python -m unittest discover lesson11_lqg_output_feedback/tests -v`

Expected: all requested logs, five figures, report, and tests exist/pass.

- [ ] **Step 5: Commit all Lesson 11 artifacts**

```bash
git add lesson11_lqg_output_feedback
git commit -m "feat: add lesson11 lqg output feedback"
```

### Task 6: Final verification and teaching handoff

**Files:**
- Modify: `lesson11_lqg_output_feedback/README.md`
- Modify: `lesson11_lqg_output_feedback/reports/lqg_engineering_report.md`

- [ ] **Step 1: Validate complete artifact tree**

Run: `Get-ChildItem lesson11_lqg_output_feedback -Recurse -File`

Expected: config, six source modules, three core test files plus experiment test, five figures, scenario logs, report, and README exist.

- [ ] **Step 2: Run final complete test suite**

Run: `conda run -n robot-control python -m unittest discover lesson11_lqg_output_feedback/tests -v`

Expected: all tests PASS with no failure or warning.

- [ ] **Step 3: Verify spec coverage in report**

Confirm the report includes numeric evidence for eigenvalue union, full-state/LQG loss, recursive/steady startup versus steady windows, fixed-\(K_c\) Q coupling, and saturation boundary diagnostics.

- [ ] **Step 4: Commit final documentation**

```bash
git add lesson11_lqg_output_feedback
git commit -m "docs: complete lesson11 lqg report"
```
