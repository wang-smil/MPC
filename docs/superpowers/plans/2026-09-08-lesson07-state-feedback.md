# Lesson 07 State Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Lesson 07 discrete state-feedback project, including controllability, pole placement, saturation-aware simulation, A–D experiments, figures, CSV logs, and a report.

**Architecture:** Lesson 07 owns controller-design, simulation, metrics, and reporting modules under `lesson07_state_feedback_controllability`. It imports only `J_hat=0.019762` and `b_hat=0.080257` as its nominal identified values; the controller is designed from that model while the simulated plant may have a +30% inertia mismatch.

**Tech Stack:** Python 3.11, NumPy, SciPy (`cont2discrete`, `place_poles`), Matplotlib, PyYAML, unittest.

**Spec:** `docs/superpowers/specs/2026-09-08-lesson07-state-feedback-design.md`

## Global Constraints

- Use ZOH discretization at the YAML `simulation.dt_s` period.
- Preserve requested and achieved closed-loop poles in logs and report.
- Log `torque_unsat_nm` separately from the clipped `torque_applied_nm`.
- Use a causal velocity estimate from noisy position; do not use non-causal S-G filtering in the control loop.
- Use the fixed nominal controller for the +30% payload experiment.
- Do not modify Lesson 05 or Lesson 06 source files.

---

### Task 1: Project scaffold and identified-model builder

**Files:**
- Create: `lesson07_state_feedback_controllability/config/controller.yaml`
- Create: `lesson07_state_feedback_controllability/src/model.py`
- Create: `lesson07_state_feedback_controllability/tests/test_model.py`
- Create: `lesson07_state_feedback_controllability/__init__.py`
- Create: `lesson07_state_feedback_controllability/src/__init__.py`
- Create: `lesson07_state_feedback_controllability/tests/__init__.py`

**Interfaces:**
- Produces: `build_continuous_model(inertia: float, damping: float) -> tuple[np.ndarray, np.ndarray]`
- Produces: `discretize_zoh(A: np.ndarray, B: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray]`
- Produces: `load_config(path: Path) -> dict`

- [ ] **Step 1: Write the failing model test**

```python
def test_build_continuous_model_uses_identified_inertia_and_damping():
    A, B = build_continuous_model(0.02, 0.08)
    np.testing.assert_allclose(A, [[0.0, 1.0], [0.0, -4.0]])
    np.testing.assert_allclose(B, [[0.0], [50.0]])
```

- [ ] **Step 2: Run the failing test**

Run: `conda run -n robot-control python -m unittest lesson07_state_feedback_controllability.tests.test_model -v`

Expected: import error for `model`.

- [ ] **Step 3: Create the configuration and implementation**

```python
def build_continuous_model(inertia: float, damping: float):
    if inertia <= 0.0 or damping < 0.0:
        raise ValueError("inertia must be positive and damping cannot be negative.")
    return np.array([[0.0, 1.0], [0.0, -damping / inertia]]), np.array([[0.0], [1.0 / inertia]])
```

Set YAML nominal values to `inertia: 0.019762`, `damping: 0.080257`, `dt_s: 0.01`, `duration_s: 4.0`, `damping_ratio: 0.8`, `settling_time_s: 0.8`, `target_deg: 30.0`, `torque_limit_nm: 3.0`, `position_noise_std_deg: 0.03`, `velocity_filter_alpha: 0.90`, and `payload_inertia_scale: 1.30`.

- [ ] **Step 4: Verify model and discretization tests pass**

Run: `conda run -n robot-control python -m unittest lesson07_state_feedback_controllability.tests.test_model -v`

Expected: all model tests pass, including `Ad.shape == (2,2)` and `Bd.shape == (2,1)`.

- [ ] **Step 5: Commit**

```powershell
git add lesson07_state_feedback_controllability/config lesson07_state_feedback_controllability/src/model.py lesson07_state_feedback_controllability/tests/test_model.py
git commit -m "feat: add lesson07 identified model"
```

### Task 2: Controllability diagnostics and failure model

**Files:**
- Create: `lesson07_state_feedback_controllability/src/controllability.py`
- Create: `lesson07_state_feedback_controllability/tests/test_controllability.py`

**Interfaces:**
- Consumes: `A`, `B` arrays from `model.py`.
- Produces: `controllability_matrix(A, B) -> np.ndarray`
- Produces: `controllability_report(A, B) -> dict[str, float | int]`
- Produces: `build_uncontrollable_model(inertia, damping) -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: Write failing nominal and failure-model tests**

```python
def test_nominal_joint_is_controllable():
    A, B = build_continuous_model(0.02, 0.08)
    assert controllability_report(A, B)["rank"] == 2

def test_uncontrolled_unstable_mode_is_not_controllable():
    A_bad, B_bad = build_uncontrollable_model(0.02, 0.08)
    assert controllability_report(A_bad, B_bad)["rank"] == 2
    assert A_bad.shape[0] == 3
```

- [ ] **Step 2: Run the failing tests**

Run: `conda run -n robot-control python -m unittest lesson07_state_feedback_controllability.tests.test_controllability -v`

Expected: import error for `controllability`.

- [ ] **Step 3: Implement diagnostics**

```python
def controllability_matrix(A, B):
    return np.hstack([np.linalg.matrix_power(A, index) @ B for index in range(A.shape[0])])

def controllability_report(A, B):
    matrix = controllability_matrix(A, B)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return {"rank": int(np.linalg.matrix_rank(matrix)), "state_dimension": A.shape[0],
            "condition_number": float(np.linalg.cond(matrix)), "sigma_min": float(singular_values[-1])}
```

Use `A_bad=[[0,1,0],[0,-b/J,0],[0,0,0.5]]` and `B_bad=[[0],[1/J],[0]]`.

- [ ] **Step 4: Verify diagnostics pass**

Run: `conda run -n robot-control python -m unittest lesson07_state_feedback_controllability.tests.test_controllability -v`

Expected: nominal rank 2; failure model rank 2 of state dimension 3.

- [ ] **Step 5: Commit**

```powershell
git add lesson07_state_feedback_controllability/src/controllability.py lesson07_state_feedback_controllability/tests/test_controllability.py
git commit -m "feat: add lesson07 controllability diagnostics"
```

### Task 3: Discrete pole design and state-feedback law

**Files:**
- Create: `lesson07_state_feedback_controllability/src/pole_design.py`
- Create: `lesson07_state_feedback_controllability/src/state_feedback.py`
- Create: `lesson07_state_feedback_controllability/tests/test_pole_placement.py`

**Interfaces:**
- Consumes: `Ad,Bd` from Task 1 and controllability report from Task 2.
- Produces: `second_order_poles(settling_time_s, damping_ratio) -> np.ndarray`
- Produces: `design_discrete_feedback(Ad, Bd, continuous_poles, dt_s) -> dict`
- Produces: `feedback_torque(K, state_estimate, reference_state, torque_limit_nm) -> dict[str, float | bool]`

- [ ] **Step 1: Write failing placement and saturation tests**

```python
def test_achieved_discrete_poles_match_requested_poles():
    poles = second_order_poles(0.8, 0.8)
    design = design_discrete_feedback(Ad, Bd, poles, 0.01)
    np.testing.assert_allclose(np.sort_complex(design["computed_poles"]), np.sort_complex(design["requested_poles"]))

def test_feedback_logs_unsaturated_and_applied_torque():
    result = feedback_torque(np.array([[10.0, 2.0]]), np.array([1.0, 0.0]), np.zeros(2), 3.0)
    assert result["torque_unsat_nm"] == -10.0
    assert result["torque_applied_nm"] == -3.0
    assert result["saturated"] is True
```

- [ ] **Step 2: Run the failing tests**

Run: `conda run -n robot-control python -m unittest lesson07_state_feedback_controllability.tests.test_pole_placement -v`

Expected: import error for `pole_design`.

- [ ] **Step 3: Implement pole mapping, placement, and clipping**

```python
def second_order_poles(settling_time_s, damping_ratio):
    wn = 4.0 / (damping_ratio * settling_time_s)
    real = -damping_ratio * wn
    imag = wn * np.sqrt(1.0 - damping_ratio**2)
    return np.array([real + 1j * imag, real - 1j * imag])
```

Map with `np.exp(continuous_poles * dt_s)` and use `scipy.signal.place_poles`. Reject rank-deficient controllability before calling placement.

- [ ] **Step 4: Verify pole and torque tests pass**

Run: `conda run -n robot-control python -m unittest lesson07_state_feedback_controllability.tests.test_pole_placement -v`

Expected: requested and achieved poles agree; clipped torque is recorded separately.

- [ ] **Step 5: Commit**

```powershell
git add lesson07_state_feedback_controllability/src/pole_design.py lesson07_state_feedback_controllability/src/state_feedback.py lesson07_state_feedback_controllability/tests/test_pole_placement.py
git commit -m "feat: add lesson07 pole placement feedback"
```

### Task 4: Saturation-aware noisy closed-loop simulator and metrics

**Files:**
- Create: `lesson07_state_feedback_controllability/src/simulator.py`
- Create: `lesson07_state_feedback_controllability/src/metrics.py`
- Create: `lesson07_state_feedback_controllability/tests/test_simulator.py`

**Interfaces:**
- Consumes: K from Task 3 and YAML plant/sensor/actuator values.
- Produces: `simulate_closed_loop(config, K, plant_inertia) -> dict[str, np.ndarray]`
- Produces: `calculate_metrics(log: dict[str, np.ndarray]) -> dict[str, float]`

- [ ] **Step 1: Write failing simulation tests**

```python
def test_simulation_exposes_unsaturated_and_applied_torque():
    log = simulate_closed_loop(config, K=np.array([[5.0, 0.5]]), plant_inertia=0.019762)
    assert {"torque_unsat_nm", "torque_applied_nm", "saturated"}.issubset(log)
    assert np.all(np.abs(log["torque_applied_nm"]) <= 3.0)
```

- [ ] **Step 2: Run the failing simulator test**

Run: `conda run -n robot-control python -m unittest lesson07_state_feedback_controllability.tests.test_simulator -v`

Expected: import error for `simulator`.

- [ ] **Step 3: Implement causal measurement and plant update**

At each sample: add position noise; estimate velocity with `raw=(q_measured-q_previous)/dt_s` and `dq_est=alpha*dq_est_previous+(1-alpha)*raw`; compute clipped feedback torque; update the true plant by semi-implicit Euler using `dq += ddq*dt_s`, `q += dq*dt_s`.

- [ ] **Step 4: Implement metrics**

Compute settling time using a 2% final-reference band, overshoot relative to the reference step, tracking RMSE, peak/RMS applied torque, and `100*mean(saturated)`.

- [ ] **Step 5: Verify simulator tests pass**

Run: `conda run -n robot-control python -m unittest lesson07_state_feedback_controllability.tests.test_simulator -v`

Expected: torque limit is never exceeded; all required log arrays align in length.

- [ ] **Step 6: Commit**

```powershell
git add lesson07_state_feedback_controllability/src/simulator.py lesson07_state_feedback_controllability/src/metrics.py lesson07_state_feedback_controllability/tests/test_simulator.py
git commit -m "feat: add lesson07 closed loop simulation"
```

### Task 5: A–D orchestration, figures, logs, and report

**Files:**
- Create: `lesson07_state_feedback_controllability/src/run_experiments.py`
- Create: `lesson07_state_feedback_controllability/tests/test_experiments.py`
- Create: `lesson07_state_feedback_controllability/README.md`

**Interfaces:**
- Consumes: all modules from Tasks 1–4.
- Produces: `run_all(config_path: Path, output_root: Path | None = None) -> dict`
- Produces: four PNG figures, one CSV per closed-loop case, and `state_feedback_report.md`.

- [ ] **Step 1: Write failing orchestration test**

```python
def test_run_all_writes_all_required_artifacts():
    result = run_all(config_path, output_root=temporary_directory)
    assert set(result["figures"]) == {"pole_placement", "speed_tradeoff", "uncontrollable_case", "payload_robustness"}
    assert result["report_path"].is_file()
```

- [ ] **Step 2: Run the failing orchestration test**

Run: `conda run -n robot-control python -m unittest lesson07_state_feedback_controllability.tests.test_experiments -v`

Expected: import error for `run_experiments`.

- [ ] **Step 3: Implement experiments A–D**

Run nominal placement; run settling-time cases `[1.5, 0.8, 0.3]`; generate the uncontrollable `z`-mode report without attempting placement; run the nominal K against `J_hat` and `1.3*J_hat` without redesigning K.

- [ ] **Step 4: Generate required artifacts**

Write CSV logs with every required field. Create `pole_placement.png`, `speed_tradeoff.png`, `uncontrollable_case.png`, and `payload_robustness.png`. Write a Markdown report containing controllability diagnostics, requested/achieved poles, K, experiment metrics, and the interpretation of saturation and payload mismatch.

- [ ] **Step 5: Verify orchestration test passes**

Run: `conda run -n robot-control python -m unittest lesson07_state_feedback_controllability.tests.test_experiments -v`

Expected: all figures, logs, and report exist in the selected output root.

- [ ] **Step 6: Commit**

```powershell
git add lesson07_state_feedback_controllability
git commit -m "feat: add lesson07 state feedback experiments"
```

### Task 6: Full engineering acceptance

**Files:**
- Modify: `lesson07_state_feedback_controllability/README.md`

**Interfaces:**
- Consumes: generated report and all test modules.
- Produces: documented commands for all four experiments and the full test suite.

- [ ] **Step 1: Run the complete project verification**

Run:

```powershell
conda run -n robot-control python -m unittest discover lesson05_system_identification/tests -v
conda run -n robot-control python -m unittest discover lesson06_identification_quality/tests -v
conda run -n robot-control python -m unittest discover lesson07_state_feedback_controllability/tests -v
```

Expected: no failures across all three lessons.

- [ ] **Step 2: Run the Lesson 07 experiment entry point**

Run: `conda run -n robot-control python -m lesson07_state_feedback_controllability.src.run_experiments`

Expected: console summary for A–D and paths to all figures, logs, and report.

- [ ] **Step 3: Update README usage and interpretation**

Document the four module commands, the identified parameter source, the distinction between `torque_unsat_nm` and `torque_applied_nm`, and why the fixed nominal K degrades on payload.

- [ ] **Step 4: Commit acceptance documentation**

```powershell
git add lesson07_state_feedback_controllability/README.md
git commit -m "docs: complete lesson07 acceptance guide"
```
