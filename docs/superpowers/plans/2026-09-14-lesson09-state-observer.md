# Lesson 09: Discrete State Observer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a discrete Luenberger observer-based LQR lesson that estimates joint position and velocity from a noisy position encoder and applied torque.

**Architecture:** The lesson owns a nominal identified position-velocity model, discrete observability and observer-pole design, a position-only observer, and a closed-loop simulator. The simulator keeps true state private to the plant and metrics; its observer receives only scalar encoder position and clipped actuator torque. A runner produces the convergence, speed trade-off, velocity-source, mismatch, and encoder-bias experiments.

**Tech Stack:** Python 3.11, NumPy, SciPy (`cont2discrete`, `place_poles`), PyYAML, Matplotlib, unittest.

**Spec:** `docs/superpowers/specs/2026-09-14-lesson09-state-observer-design.md`

## Global Constraints

- Create `lesson09_state_observer/`; do not modify earlier lesson implementations.
- Use `J_hat = 0.019762 kg m^2`, `b_hat = 0.080257 N m s/rad`, and `Ts = 1 ms`.
- Derive ZOH `Ad`, `Bd` and the lesson-08 balanced LQR gain; do not hard-code either gain.
- Observer API accepts only `measurement` and `control_input`; it cannot receive true plant state.
- Feed the observer `u_applied` after torque clipping, never `u_unsat`.
- Preserve true state only for offline metrics and plots.
- Use test-first development. Run commands from `D:\MPC_learn` with `conda run -n robot-control python`.

---

## File Structure

```text
lesson09_state_observer/
├── __init__.py
├── config/observer.yaml
├── src/
│   ├── __init__.py
│   ├── model_loader.py          # nominal continuous/ZOH model and balanced LQR gain
│   ├── observability.py         # observability matrix and conditioning report
│   ├── observer_design.py       # continuous-pole mapping and dual pole placement
│   ├── observer.py              # position-only DiscreteObserver
│   ├── velocity_estimators.py   # raw and LPF backward differences
│   ├── closed_loop.py           # plant/encoder/observer/controller data chain
│   ├── metrics.py               # estimate, tracking, torque, innovation statistics
│   └── run_experiments.py       # A–E experiments, CSV/figures/reports
├── tests/
│   ├── __init__.py
│   ├── test_model_loader.py
│   ├── test_observability.py
│   ├── test_observer_design.py
│   ├── test_observer.py
│   ├── test_velocity_estimators.py
│   ├── test_closed_loop.py
│   └── test_experiments.py
├── logs/
├── figures/
├── reports/
└── README.md
```

### Task 1: Lesson scaffold and identified model loader

**Files:**
- Create: `lesson09_state_observer/__init__.py`
- Create: `lesson09_state_observer/config/observer.yaml`
- Create: `lesson09_state_observer/src/__init__.py`
- Create: `lesson09_state_observer/src/model_loader.py`
- Create: `lesson09_state_observer/tests/__init__.py`
- Create: `lesson09_state_observer/tests/test_model_loader.py`

**Interfaces:**
- Produces `load_config(path: Path | str) -> dict`.
- Produces `build_identified_model(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`, returning `(A, B, Ad, Bd)`.
- Produces `build_balanced_lqr_gain(Ad: np.ndarray, Bd: np.ndarray, config: dict) -> np.ndarray`, returning shape `(1, 2)`.

- [ ] **Step 1: Write the failing model test**

```python
def test_identified_model_has_expected_shapes_and_one_ms_zoh():
    config = load_config(CONFIG_PATH)
    A, B, Ad, Bd = build_identified_model(config)
    assert A.shape == (2, 2)
    assert B.shape == (2, 1)
    assert Ad.shape == (2, 2)
    assert Bd.shape == (2, 1)
    np.testing.assert_allclose(A[1, 1], -0.080257 / 0.019762)
    assert not np.allclose(Ad, np.eye(2))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n robot-control python -m unittest lesson09_state_observer.tests.test_model_loader -v`

Expected: import failure because the lesson package does not exist.

- [ ] **Step 3: Add YAML and minimal loader implementation**

```yaml
simulation: {dt_s: 0.001, duration_s: 4.0, seed: 2026}
model: {inertia: 0.019762, damping: 0.080257}
measurement: {position_noise_std_deg: 0.05, encoder_bias_deg: 0.5}
observer: {speed_factors: [2.0, 4.0, 10.0], nominal_speed_factor: 4.0}
initial_condition: {q_true_deg: 20.0, dq_true_rad_s: 0.5, q_hat_deg: 0.0, dq_hat_rad_s: 0.0}
controller: {target_deg: 30.0, max_position_error_deg: 5.0, max_velocity_error_rad_s: 1.5, torque_limit_nm: 3.0}
stress: {payload_scale: 1.30, high_noise_scale: 10.0}
velocity_difference: {lpf_alpha: 0.90}
```

```python
def build_identified_model(config):
    inertia = float(config["model"]["inertia"])
    damping = float(config["model"]["damping"])
    dt_s = float(config["simulation"]["dt_s"])
    A = np.array([[0.0, 1.0], [0.0, -damping / inertia]])
    B = np.array([[0.0], [1.0 / inertia]])
    Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(2), np.zeros((2, 1))), dt_s, method="zoh")
    return A, B, np.asarray(Ad), np.asarray(Bd)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `conda run -n robot-control python -m unittest lesson09_state_observer.tests.test_model_loader -v`

Expected: `OK`.

- [ ] **Step 5: Add and test balanced LQR-gain derivation**

```python
def test_balanced_lqr_gain_is_one_by_two_and_stabilizes_model():
    config = load_config(CONFIG_PATH)
    _, _, Ad, Bd = build_identified_model(config)
    K = build_balanced_lqr_gain(Ad, Bd, config)
    assert K.shape == (1, 2)
    assert np.all(np.abs(np.linalg.eigvals(Ad - Bd @ K)) < 1.0)
```

Implement Bryson weights using 5 degrees, 1.5 rad/s, and 3 Nm, then call
`control.dlqr(Ad, Bd, Q, R)` and return `K`.

- [ ] **Step 6: Commit the scaffold and model**

```powershell
git add lesson09_state_observer
git commit -m "feat: add lesson09 identified observer model"
```

### Task 2: Observability analysis

**Files:**
- Create: `lesson09_state_observer/src/observability.py`
- Create: `lesson09_state_observer/tests/test_observability.py`

**Interfaces:**
- Consumes `Ad` from task 1 and `C = np.array([[1.0, 0.0]])`.
- Produces `observability_matrix(A: np.ndarray, C: np.ndarray) -> np.ndarray`.
- Produces `observability_report(A: np.ndarray, C: np.ndarray) -> dict[str, float | int]` with `rank`, `state_dimension`, `condition_number`, `sigma_min`.

- [ ] **Step 1: Write failing observability tests**

```python
def test_position_measurement_observes_both_joint_states():
    _, _, Ad, _ = build_identified_model(load_config(CONFIG_PATH))
    report = observability_report(Ad, np.array([[1.0, 0.0]]))
    assert report["rank"] == 2
    assert report["state_dimension"] == 2
    assert report["sigma_min"] > 0.0

def test_invalid_matrix_shapes_are_rejected():
    with self.assertRaises(ValueError):
        observability_matrix(np.ones((2, 3)), np.ones((1, 2)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n robot-control python -m unittest lesson09_state_observer.tests.test_observability -v`

Expected: import failure for `observability`.

- [ ] **Step 3: Implement the matrix and report**

```python
def observability_matrix(A, C):
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or C.ndim != 2 or C.shape[1] != A.shape[0]:
        raise ValueError("A must be square and C must have matching columns.")
    return np.vstack([C @ np.linalg.matrix_power(A, power) for power in range(A.shape[0])])
```

Compute singular values with `np.linalg.svd(O, compute_uv=False)` and expose
the final value as `sigma_min`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n robot-control python -m unittest lesson09_state_observer.tests.test_observability -v`

Expected: `OK`.

- [ ] **Step 5: Commit observability analysis**

```powershell
git add lesson09_state_observer/src/observability.py lesson09_state_observer/tests/test_observability.py
git commit -m "feat: add lesson09 observability analysis"
```

### Task 3: Discrete observer pole design

**Files:**
- Create: `lesson09_state_observer/src/observer_design.py`
- Create: `lesson09_state_observer/tests/test_observer_design.py`

**Interfaces:**
- Consumes `Ad`, `C`, controller closed-loop poles, `speed_factor`, and `dt_s`.
- Produces `continuous_poles_from_discrete(discrete_poles: np.ndarray, dt_s: float) -> np.ndarray`.
- Produces `design_discrete_observer(Ad, C, controller_poles_z, speed_factor, dt_s) -> dict[str, np.ndarray]` with `L`, `requested_poles`, and `achieved_poles`.

- [ ] **Step 1: Write the failing pole-placement tests**

```python
def test_observer_places_requested_stable_discrete_poles():
    config = load_config(CONFIG_PATH)
    _, _, Ad, Bd = build_identified_model(config)
    K = build_balanced_lqr_gain(Ad, Bd, config)
    controller_poles_z = np.linalg.eigvals(Ad - Bd @ K)
    result = design_discrete_observer(Ad, np.array([[1.0, 0.0]]), controller_poles_z, 4.0, 0.001)
    assert result["L"].shape == (2, 1)
    assert np.all(np.abs(result["achieved_poles"]) < 1.0)
    np.testing.assert_allclose(np.sort_complex(result["achieved_poles"]), np.sort_complex(result["requested_poles"]), atol=1e-7)

def test_nonpositive_speed_or_sample_time_is_rejected():
    with self.assertRaises(ValueError):
        design_discrete_observer(np.eye(2), np.array([[1.0, 0.0]]), np.array([0.9, 0.8]), 0.0, 0.001)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n robot-control python -m unittest lesson09_state_observer.tests.test_observer_design -v`

Expected: import failure for `observer_design`.

- [ ] **Step 3: Implement continuous-scale pole mapping and dual placement**

```python
def continuous_poles_from_discrete(discrete_poles, dt_s):
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive.")
    return np.log(np.asarray(discrete_poles, dtype=complex)) / dt_s

controller_poles_s = continuous_poles_from_discrete(controller_poles_z, dt_s)
requested_poles = np.exp(speed_factor * controller_poles_s * dt_s)
result = place_poles(Ad.T, C.T, requested_poles)
L = result.gain_matrix.T
achieved_poles = np.linalg.eigvals(Ad - L @ C)
return {"L": L, "requested_poles": requested_poles, "achieved_poles": achieved_poles}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n robot-control python -m unittest lesson09_state_observer.tests.test_observer_design -v`

Expected: `OK`.

- [ ] **Step 5: Commit observer design**

```powershell
git add lesson09_state_observer/src/observer_design.py lesson09_state_observer/tests/test_observer_design.py
git commit -m "feat: add lesson09 observer pole design"
```

### Task 4: Position-only observer runtime

**Files:**
- Create: `lesson09_state_observer/src/observer.py`
- Create: `lesson09_state_observer/tests/test_observer.py`

**Interfaces:**
- Consumes `Ad (2,2)`, `Bd (2,1)`, `C (1,2)`, `L (2,1)`, and `x0_hat (2,)`.
- Produces `DiscreteObserver.update(measurement: float, control_input: float) -> dict[str, np.ndarray | float]` with `x_hat`, `y_hat`, and `innovation`.

- [ ] **Step 1: Write failing API and one-step update tests**

```python
def test_update_uses_only_measurement_and_applied_input():
    observer = DiscreteObserver(np.eye(2), np.array([[0.0], [1.0]]), np.array([[1.0, 0.0]]), np.array([[0.5], [0.0]]), np.zeros(2))
    result = observer.update(measurement=2.0, control_input=3.0)
    assert set(result) == {"x_hat", "y_hat", "innovation"}
    assert result["y_hat"] == 0.0
    assert result["innovation"] == 2.0
    np.testing.assert_allclose(result["x_hat"], [1.0, 3.0])

def test_vector_measurement_is_rejected():
    with self.assertRaises(ValueError):
        observer.update(measurement=np.zeros(2), control_input=0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n robot-control python -m unittest lesson09_state_observer.tests.test_observer -v`

Expected: import failure for `observer`.

- [ ] **Step 3: Implement prediction plus innovation correction**

```python
def update(self, measurement, control_input):
    y = float(np.asarray(measurement, dtype=float))
    u = float(np.asarray(control_input, dtype=float))
    y_hat = float((self.C @ self.x_hat).item())
    innovation = y - y_hat
    self.x_hat = self.Ad @ self.x_hat + self.Bd[:, 0] * u + self.L[:, 0] * innovation
    return {"x_hat": self.x_hat.copy(), "y_hat": y_hat, "innovation": innovation}
```

Validate shapes in `__init__` and reject non-scalar `measurement` or
`control_input` before converting them.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n robot-control python -m unittest lesson09_state_observer.tests.test_observer -v`

Expected: `OK`.

- [ ] **Step 5: Commit the observer runtime**

```powershell
git add lesson09_state_observer/src/observer.py lesson09_state_observer/tests/test_observer.py
git commit -m "feat: add lesson09 discrete observer"
```

### Task 5: Velocity-estimator comparison baselines

**Files:**
- Create: `lesson09_state_observer/src/velocity_estimators.py`
- Create: `lesson09_state_observer/tests/test_velocity_estimators.py`

**Interfaces:**
- Produces `RawDifferenceVelocity(dt_s: float)` with `update(position_rad: float) -> float`.
- Produces `LowPassDifferenceVelocity(dt_s: float, alpha: float)` with `update(position_rad: float) -> float`.
- The first update returns zero because no previous encoder sample exists.

- [ ] **Step 1: Write failing estimator tests**

```python
def test_raw_difference_and_lpf_have_known_two_sample_values():
    raw = RawDifferenceVelocity(0.1)
    filtered = LowPassDifferenceVelocity(0.1, alpha=0.8)
    assert raw.update(1.0) == 0.0
    assert filtered.update(1.0) == 0.0
    assert raw.update(1.5) == 5.0
    assert filtered.update(1.5) == 1.0

def test_invalid_sampling_or_alpha_is_rejected():
    with self.assertRaises(ValueError):
        LowPassDifferenceVelocity(0.001, alpha=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n robot-control python -m unittest lesson09_state_observer.tests.test_velocity_estimators -v`

Expected: import failure for `velocity_estimators`.

- [ ] **Step 3: Implement backward difference and recursive LPF**

```python
raw_velocity = (position_rad - self.previous_position_rad) / self.dt_s
self.filtered_velocity_rad_s = self.alpha * self.filtered_velocity_rad_s + (1.0 - self.alpha) * raw_velocity
```

Use strict validation: `dt_s > 0` and `0 <= alpha < 1`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n robot-control python -m unittest lesson09_state_observer.tests.test_velocity_estimators -v`

Expected: `OK`.

- [ ] **Step 5: Commit velocity baselines**

```powershell
git add lesson09_state_observer/src/velocity_estimators.py lesson09_state_observer/tests/test_velocity_estimators.py
git commit -m "feat: add lesson09 velocity baselines"
```

### Task 6: Observer-based closed loop and metrics

**Files:**
- Create: `lesson09_state_observer/src/closed_loop.py`
- Create: `lesson09_state_observer/src/metrics.py`
- Create: `lesson09_state_observer/tests/test_closed_loop.py`

**Interfaces:**
- Produces `simulate_observer_closed_loop(config, observer, K, plant_inertia, noise_scale=1.0, encoder_bias_rad=0.0, velocity_source="observer") -> dict[str, np.ndarray]`.
- Produces `calculate_metrics(log: dict[str, np.ndarray], dt_s: float) -> dict[str, float]`.
def run_nominal_simulation(noise_scale):
    config = load_config(CONFIG_PATH)
    _, _, Ad, Bd = build_identified_model(config)
    K = build_balanced_lqr_gain(Ad, Bd, config)
    controller_poles_z = np.linalg.eigvals(Ad - Bd @ K)
    design = design_discrete_observer(
        Ad, np.array([[1.0, 0.0]]), controller_poles_z, 4.0, config["simulation"]["dt_s"]
    )
    observer = DiscreteObserver(Ad, Bd, np.array([[1.0, 0.0]]), design["L"], np.zeros(2))
    return simulate_observer_closed_loop(
        config, observer, K, config["model"]["inertia"], noise_scale=noise_scale
    )

- `velocity_source` accepts exactly `"raw_difference"`, `"filtered_difference"`, or `"observer"`.

- [ ] **Step 1: Write failing closed-loop safety tests**

```python
def test_simulation_logs_only_clipped_input_to_observer_and_all_required_signals():
    log = run_nominal_simulation(noise_scale=0.0)
    required = {"q_true_rad", "q_measured_rad", "q_hat_rad", "dq_true_rad_s", "dq_raw_rad_s", "dq_filtered_rad_s", "dq_hat_rad_s", "innovation_rad", "torque_unsat_nm", "torque_cmd_nm", "saturated"}
    assert required.issubset(log)
    assert np.all(np.abs(log["torque_cmd_nm"]) <= 3.0 + 1e-12)

def test_noise_free_observer_converges_from_mismatched_initial_state():
    log = run_nominal_simulation(noise_scale=0.0)
    assert abs(log["q_estimation_error_rad"][-1]) < 1e-3
    assert abs(log["dq_estimation_error_rad_s"][-1]) < 1e-3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n robot-control python -m unittest lesson09_state_observer.tests.test_closed_loop -v`

Expected: import failure for `closed_loop`.

- [ ] **Step 3: Implement simulation data flow**

At each sample: form the noisy biased encoder value; update both difference
baselines; select the state estimate used by LQR; calculate and clip torque;
advance the true plant with the clipped torque; then call the observer with the
same clipped value.  Log values before the plant advances.

```python
controller_state = np.array([q_hat_rad, selected_velocity_rad_s])
torque_unsat_nm = float(-(K @ (controller_state - reference_state)).item())
torque_cmd_nm = float(np.clip(torque_unsat_nm, -torque_limit_nm, torque_limit_nm))
observer_result = observer.update(measurement=q_measured_rad, control_input=torque_cmd_nm)
acceleration_rad_s2 = (torque_cmd_nm - damping * dq_true_rad_s) / plant_inertia
dq_true_rad_s += acceleration_rad_s2 * dt_s
q_true_rad += dq_true_rad_s * dt_s
```

Keep `q_true_rad` and `dq_true_rad_s` out of the observer constructor and
update call.

- [ ] **Step 4: Implement metrics and pass tests**

```python
metrics = {
    "q_est_rmse_rad": float(np.sqrt(np.mean(log["q_estimation_error_rad"] ** 2))),
    "dq_est_rmse_rad_s": float(np.sqrt(np.mean(log["dq_estimation_error_rad_s"] ** 2))),
    "innovation_rms_rad": float(np.sqrt(np.mean(log["innovation_rad"] ** 2))),
    "rms_torque_nm": float(np.sqrt(np.mean(log["torque_cmd_nm"] ** 2))),
    "peak_torque_nm": float(np.max(np.abs(log["torque_cmd_nm"]))),
}
```

Define convergence time as the first time after which both absolute estimation
errors remain within 2% of their initial magnitudes, with a 1e-6 floor to avoid
division by zero.

Run: `conda run -n robot-control python -m unittest lesson09_state_observer.tests.test_closed_loop -v`

Expected: `OK`.

- [ ] **Step 5: Commit closed-loop behavior**

```powershell
git add lesson09_state_observer/src/closed_loop.py lesson09_state_observer/src/metrics.py lesson09_state_observer/tests/test_closed_loop.py
git commit -m "feat: add lesson09 observer closed loop"
```

### Task 7: Experiment runner, figures, report, and guide

**Files:**
- Create: `lesson09_state_observer/src/run_experiments.py`
- Create: `lesson09_state_observer/tests/test_experiments.py`
- Create: `lesson09_state_observer/README.md`
- Create at runtime: `lesson09_state_observer/logs/*.csv`
- Create at runtime: `lesson09_state_observer/figures/*.png`
- Create at runtime: `lesson09_state_observer/reports/observer_engineering_report.md`

**Interfaces:**
- Produces `run_all(config_path: Path | str, output_root: Path | str | None = None) -> dict[str, dict[str, float | str]]`.
- Produces one log per named run and four figures: `observer_convergence.png`,
  `observer_speed_tradeoff.png`, `velocity_source_comparison.png`, and
  `observer_robustness.png`.

- [ ] **Step 1: Write failing artifact test**

```python
def test_runner_writes_experiments_figures_and_report(tmp_path):
    result = run_all(CONFIG_PATH, output_root=tmp_path)
    assert {"convergence", "speed_tradeoff", "velocity_sources", "payload", "bias", "normal", "disturbance", "stress"}.issubset(result)
    assert (tmp_path / "figures" / "observer_convergence.png").is_file()
    assert (tmp_path / "figures" / "velocity_source_comparison.png").is_file()
    assert (tmp_path / "reports" / "observer_engineering_report.md").is_file()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n robot-control python -m unittest lesson09_state_observer.tests.test_experiments -v`

Expected: import failure for `run_experiments`.

- [ ] **Step 3: Implement named experiments and CSV logs**

Run A with zero noise and mismatched initial state.  Run B at 2, 4, and 10
observer speed factors.  Run C once per velocity source.  Run D with
`plant_inertia = 1.3 * J_hat`.  Run E with `encoder_bias_rad = deg2rad(0.5)`.
Also produce acceptance runs `normal`, `disturbance`, and `stress`, where
`stress` has 10 times the configured position noise.

Use `csv.DictWriter` with all keys present in the log and write figures with
Matplotlib.  The convergence figure must contain position, velocity,
estimation-error, and innovation panels.

- [ ] **Step 4: Write the engineering report and README**

The report must list observability rank/condition/`sigma_min`, requested and
achieved observer poles, every comparison metric, and three explicit findings:

```markdown
1. Increasing observer speed reduces initial-state convergence time but can amplify encoder noise into velocity and torque.
2. The observer uses the applied torque after saturation; prediction with requested torque would violate plant-input consistency.
3. A constant encoder bias is not a state in this model, so innovation and estimated position can remain biased; use bias augmentation or a Kalman-style estimator later.
```

README must explain the data chain, each source file, commands, and the real
robot rollout sequence: sensor validation, offline observer, low-bandwidth
closed loop, disturbances, then bandwidth increase.

- [ ] **Step 5: Run focused and full lesson tests**

Run: `conda run -n robot-control python -m unittest lesson09_state_observer.tests.test_experiments -v`

Expected: `OK`.

Run: `conda run -n robot-control python -m unittest discover lesson09_state_observer/tests -v`

Expected: every Lesson 09 test reports `ok` and the final status is `OK`.

- [ ] **Step 6: Generate committed teaching artifacts and commit**

Run: `conda run -n robot-control python -m lesson09_state_observer.src.run_experiments`

```powershell
git add lesson09_state_observer
git commit -m "feat: add lesson09 observer experiments"
```

### Task 8: Integration verification and lesson handoff

**Files:**
- Verify: `lesson05_system_identification/tests/`
- Verify: `lesson06_identification_quality/tests/`
- Verify: `lesson07_state_feedback_controllability/tests/`
- Verify: `lesson08_lqr_optimal_control/tests/`
- Verify: `lesson09_state_observer/tests/`

**Interfaces:**
- Confirms that lesson 09 adds a new package without changing previous lesson
  behavior.

- [ ] **Step 1: Run regression suites**

Run each command separately:

```powershell
conda run -n robot-control python -m unittest discover lesson05_system_identification/tests -v
conda run -n robot-control python -m unittest discover lesson06_identification_quality/tests -v
conda run -n robot-control python -m unittest discover lesson07_state_feedback_controllability/tests -v
conda run -n robot-control python -m unittest discover lesson08_lqr_optimal_control/tests -v
conda run -n robot-control python -m unittest discover lesson09_state_observer/tests -v
```

Expected: all suites finish with `OK`.

- [ ] **Step 2: Check the reviewable repository state**

```powershell
git status --short
git log --oneline -8
```

Expected: Lesson 09 implementation, logs, figures, report, and guide are
committed; existing unrelated user files remain untracked and unmodified.

- [ ] **Step 3: Commit any final documentation-only correction**

```powershell
git add lesson09_state_observer/README.md lesson09_state_observer/reports/observer_engineering_report.md
git commit -m "docs: complete lesson09 observer guide"
```

