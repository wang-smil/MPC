# Lesson 05 System Identification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone single-axis robot-joint system-identification lesson that estimates inertia and viscous damping from simulated measurements and independently validates the estimated model.

**Architecture:** A true plant with Coulomb friction generates position data under torque-limited multi-sine excitation. Signal processing creates raw or filtered derivative estimates, a least-squares estimator fits the reduced `J*ddq + b*dq` model, and the runner writes logs, figures, JSON parameters, and a Markdown report for four diagnostic cases plus independent validation.

**Tech Stack:** Python 3.11, NumPy, SciPy, Matplotlib, PyYAML, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-02-lesson05-system-identification-design.md`

## Global Constraints

- Keep Lesson 05 independent of the Lesson 02 source code.
- Use radians, radians/s, radians/s², and N·m for all calculations.
- The estimator must consume `torque_applied_nm`, never an unsaturated command or `plant_true` values.
- Identification and validation excitations must differ.
- Do not treat a fixed parameter-error threshold as an industrial acceptance standard.
- Preserve the existing `unittest` execution style and run commands through `conda run -n robot-control`.

---

## Planned file structure

| File | Responsibility |
| --- | --- |
| `requirements.txt` | Declare PyYAML alongside the existing numerical dependencies. |
| `lesson05_system_identification/config/sysid.yaml` | Nominal experiment configuration and true-plant-only values. |
| `lesson05_system_identification/src/excitation.py` | Validate and evaluate deterministic multi-sine torque signals. |
| `lesson05_system_identification/src/plant.py` | Simulate the torque-limited true joint and a friction-free identified model. |
| `lesson05_system_identification/src/signal_processing.py` | Noise injection, numerical differentiation, low-pass filtering, and time slicing. |
| `lesson05_system_identification/src/estimator.py` | Least-squares `J,b` fit and residual diagnostics. |
| `lesson05_system_identification/src/metrics.py` | Parameter errors and trajectory/torque RMSE calculations. |
| `lesson05_system_identification/src/run_sysid.py` | Load config, execute cases, persist artifacts, and draw figures. |
| `lesson05_system_identification/tests/test_sysid.py` | Unit and integration tests. |
| `lesson05_system_identification/README.md` | Lesson objective, run/test commands, and interpretation notes. |

### Task 1: Configuration and excitation input

**Files:**
- Create: `lesson05_system_identification/config/sysid.yaml`
- Create: `lesson05_system_identification/src/excitation.py`
- Create: `lesson05_system_identification/tests/test_sysid.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `MultiSineExcitation(frequencies_hz: list[float], amplitudes_nm: list[float])`.
- Produces: `MultiSineExcitation.evaluate(time_s: float | np.ndarray) -> float | np.ndarray`.
- Produces: `load_config(path: Path) -> dict` in the runner, with YAML sections from the specification.

- [ ] **Step 1: Write failing tests for valid and invalid excitation**

```python
class ExcitationTest(unittest.TestCase):
    def test_multisine_is_zero_at_time_zero_and_has_expected_value(self):
        signal = MultiSineExcitation([0.5, 1.0], [0.4, 0.2])
        self.assertAlmostEqual(signal.evaluate(0.0), 0.0)
        self.assertAlmostEqual(signal.evaluate(0.25), 0.4)

    def test_mismatched_lists_are_rejected(self):
        with self.assertRaises(ValueError):
            MultiSineExcitation([0.5], [0.4, 0.2])
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_sysid.ExcitationTest -v`

Expected: FAIL because `excitation` cannot be imported.

- [ ] **Step 3: Add the YAML baseline and minimal excitation implementation**

```yaml
simulation: {dt: 0.001, duration_s: 12.0, seed: 2026}
plant_true: {inertia: 0.020, damping: 0.080, coulomb_friction: 0.10}
sensor: {position_noise_std_deg: 0.02}
actuator: {torque_limit_nm: 2.0}
excitation:
  frequencies_hz: [0.5, 1.3, 2.7]
  amplitudes_nm: [0.45, 0.30, 0.15]
identification: {discard_start_s: 1.0, lowpass_cutoff_hz: 15.0}
```

```python
class MultiSineExcitation:
    def __init__(self, frequencies_hz, amplitudes_nm):
        if not frequencies_hz or len(frequencies_hz) != len(amplitudes_nm):
            raise ValueError("frequencies_hz and amplitudes_nm must be non-empty and equal length.")
        if any(freq <= 0 for freq in frequencies_hz):
            raise ValueError("frequencies_hz must be positive.")
        self.frequencies_hz = np.asarray(frequencies_hz, dtype=float)
        self.amplitudes_nm = np.asarray(amplitudes_nm, dtype=float)

    def evaluate(self, time_s):
        time = np.asarray(time_s, dtype=float)
        return np.sum(self.amplitudes_nm * np.sin(2.0 * np.pi * self.frequencies_hz * time[..., None]), axis=-1)
```

Add `PyYAML==6.0.3` to `requirements.txt`.

- [ ] **Step 4: Run focused tests and verify they pass**

Run: `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_sysid.ExcitationTest -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add requirements.txt lesson05_system_identification/config/sysid.yaml lesson05_system_identification/src/excitation.py lesson05_system_identification/tests/test_sysid.py
git commit -m "feat: add lesson05 excitation configuration"
```

### Task 2: True plant and applied-torque logging

**Files:**
- Create: `lesson05_system_identification/src/plant.py`
- Modify: `lesson05_system_identification/tests/test_sysid.py`

**Interfaces:**
- Consumes: an applied torque command in N·m and positive `dt_s`.
- Produces: `SingleAxisPlant.step(torque_command_nm: float, dt_s: float) -> dict` containing `torque_applied_nm`, `position_true_rad`, `velocity_true_rad_s`, and `acceleration_true_rad_s2`.
- Produces: `simulate_linear_model(torque_nm: np.ndarray, time_s: np.ndarray, inertia: float, damping: float) -> tuple[np.ndarray, np.ndarray]`.

- [ ] **Step 1: Write failing plant tests**

```python
class PlantTest(unittest.TestCase):
    def test_torque_is_limited_and_acceleration_uses_applied_torque(self):
        plant = SingleAxisPlant(0.02, 0.08, 0.0, torque_limit_nm=2.0)
        sample = plant.step(5.0, 0.001)
        self.assertEqual(sample["torque_applied_nm"], 2.0)
        self.assertAlmostEqual(sample["acceleration_true_rad_s2"], 100.0)

    def test_nonpositive_dt_is_rejected(self):
        plant = SingleAxisPlant(0.02, 0.08, 0.0, torque_limit_nm=2.0)
        with self.assertRaises(ValueError):
            plant.step(0.0, 0.0)
```

- [ ] **Step 2: Run plant tests and verify they fail**

Run: `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_sysid.PlantTest -v`

Expected: FAIL because `plant` cannot be imported.

- [ ] **Step 3: Implement a semi-implicit-Euler plant**

```python
torque_applied = float(np.clip(torque_command_nm, -self.torque_limit_nm, self.torque_limit_nm))
friction = self.coulomb_friction * np.sign(self.velocity_rad_s)
acceleration = (torque_applied - self.damping * self.velocity_rad_s - friction) / self.inertia
self.velocity_rad_s += acceleration * dt_s
self.position_rad += self.velocity_rad_s * dt_s
```

Validate positive inertia, non-negative damping/friction, and positive torque limit in `__init__`.  Keep `simulate_linear_model` friction-free so it represents the identified reduced model.

- [ ] **Step 4: Run plant tests and verify they pass**

Run: `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_sysid.PlantTest -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add lesson05_system_identification/src/plant.py lesson05_system_identification/tests/test_sysid.py
git commit -m "feat: simulate lesson05 joint plant"
```

### Task 3: Signal processing and estimator

**Files:**
- Create: `lesson05_system_identification/src/signal_processing.py`
- Create: `lesson05_system_identification/src/estimator.py`
- Modify: `lesson05_system_identification/tests/test_sysid.py`

**Interfaces:**
- Produces: `estimate_derivatives(position_rad: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray]`.
- Produces: `lowpass(signal: np.ndarray, dt_s: float, cutoff_hz: float) -> np.ndarray` using a zero-phase Butterworth SOS filter.
- Produces: `discard_before(time_s, arrays, discard_start_s) -> tuple[np.ndarray, list[np.ndarray]]`.
- Produces: `identify_j_b(velocity, acceleration, torque) -> dict` with `inertia_hat`, `damping_hat`, `rank`, `condition_number`, `torque_predicted_nm`, `residual_nm`, and `torque_rmse`.

- [ ] **Step 1: Write failing estimator tests from exact data and rank-deficient data**

```python
class EstimatorTest(unittest.TestCase):
    def test_exact_data_recovers_inertia_and_damping(self):
        velocity = np.array([-2.0, -1.0, 0.5, 1.5])
        acceleration = np.array([3.0, -2.0, 1.0, 4.0])
        torque = 0.02 * acceleration + 0.08 * velocity
        result = identify_j_b(velocity, acceleration, torque)
        self.assertAlmostEqual(result["inertia_hat"], 0.02)
        self.assertAlmostEqual(result["damping_hat"], 0.08)
        self.assertAlmostEqual(result["torque_rmse"], 0.0)

    def test_rank_deficient_data_is_rejected(self):
        with self.assertRaises(ValueError):
            identify_j_b(np.ones(5), 2.0 * np.ones(5), np.ones(5))
```

- [ ] **Step 2: Run estimator tests and verify they fail**

Run: `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_sysid.EstimatorTest -v`

Expected: FAIL because `estimator` cannot be imported.

- [ ] **Step 3: Implement processing and least squares**

```python
phi = np.column_stack([acceleration, velocity])
theta_hat, _, rank, _ = np.linalg.lstsq(phi, torque, rcond=None)
if rank < 2:
    raise ValueError("Identification regression is rank deficient.")
torque_predicted = phi @ theta_hat
residual = torque - torque_predicted
```

Use `np.gradient` for differentiation.  For `lowpass`, reject cutoffs outside `(0, 0.5 / dt_s)` and apply `scipy.signal.butter(..., output="sos")` with `sosfiltfilt`; bypass filtering only when the caller explicitly requests raw derivatives.

- [ ] **Step 4: Run estimator tests and verify they pass**

Run: `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_sysid.EstimatorTest -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add lesson05_system_identification/src/signal_processing.py lesson05_system_identification/src/estimator.py lesson05_system_identification/tests/test_sysid.py
git commit -m "feat: estimate lesson05 dynamics parameters"
```

### Task 4: Metrics and independent validation

**Files:**
- Create: `lesson05_system_identification/src/metrics.py`
- Modify: `lesson05_system_identification/tests/test_sysid.py`

**Interfaces:**
- Produces: `relative_error_percent(estimate: float, truth: float) -> float`.
- Produces: `rmse(actual: np.ndarray, predicted: np.ndarray) -> float`.
- Produces: `calculate_metrics(estimate: dict, truth: dict, validation_true_position: np.ndarray, validation_model_position: np.ndarray) -> dict`.

- [ ] **Step 1: Write failing metrics test**

```python
class MetricsTest(unittest.TestCase):
    def test_metrics_report_parameter_and_validation_errors(self):
        metrics = calculate_metrics(
            {"inertia_hat": 0.021, "damping_hat": 0.076, "torque_rmse": 0.03, "condition_number": 8.0},
            {"inertia": 0.020, "damping": 0.080},
            np.array([0.0, 1.0]), np.array([0.0, 1.2]),
        )
        self.assertAlmostEqual(metrics["inertia_error_percent"], 5.0)
        self.assertAlmostEqual(metrics["damping_error_percent"], 5.0)
        self.assertAlmostEqual(metrics["validation_position_rmse_rad"], np.sqrt(0.02))
```

- [ ] **Step 2: Run metrics test and verify it fails**

Run: `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_sysid.MetricsTest -v`

Expected: FAIL because `metrics` cannot be imported.

- [ ] **Step 3: Implement metrics without a pass/fail threshold**

```python
def relative_error_percent(estimate, truth):
    if truth == 0:
        raise ValueError("truth must be non-zero for relative error.")
    return abs(estimate - truth) / abs(truth) * 100.0

def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))
```

`calculate_metrics` must pass through `torque_rmse` and `condition_number`, calculate both parameter error fields, and calculate `validation_position_rmse_rad`.

- [ ] **Step 4: Run metrics test and verify it passes**

Run: `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_sysid.MetricsTest -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add lesson05_system_identification/src/metrics.py lesson05_system_identification/tests/test_sysid.py
git commit -m "feat: add lesson05 identification metrics"
```

### Task 5: Case runner, artifacts, documentation, and acceptance run

**Files:**
- Create: `lesson05_system_identification/src/run_sysid.py`
- Create: `lesson05_system_identification/README.md`
- Create: `lesson05_system_identification/reports/identification_report.md`
- Modify: `lesson05_system_identification/tests/test_sysid.py`

**Interfaces:**
- Produces: `run_case(case_name: str, config: dict, use_filter: bool = True) -> dict`.
- Produces: `run_validation(config: dict, inertia_hat: float, damping_hat: float) -> dict` with a validation input different from identification input.
- Produces: `main() -> None` that runs normal, poor excitation, high-noise raw/filtered, and model-mismatch cases.

- [ ] **Step 1: Write a failing integration test for an artifact-producing normal case**

```python
class RunnerTest(unittest.TestCase):
    def test_normal_case_writes_applied_torque_log_and_parameters(self):
        result = run_case("normal", TEST_CONFIG)
        self.assertIn("torque_applied_nm", result["log_columns"])
        self.assertTrue(result["parameters_path"].is_file())
        self.assertTrue(result["figure_path"].is_file())
```

`TEST_CONFIG` uses `dt=0.002`, `duration_s=2.0`, and a temporary output directory to keep the test fast.

- [ ] **Step 2: Run integration test and verify it fails**

Run: `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_sysid.RunnerTest -v`

Expected: FAIL because `run_sysid` cannot be imported.

- [ ] **Step 3: Implement the runner and persisted artifacts**

For every simulated sample, append the exact twelve required CSV fields.  Fit on the time-sliced processed arrays using actual applied torque.  Write `identified_parameters.json` with estimates, rank, condition number, torque RMSE, and case name.  Create:

- a normal-case figure showing true/measured position and true/estimated velocity;
- a parameter/fit figure for normal, poor excitation, and noise cases;
- a model-mismatch residual-versus-time and residual-versus-velocity figure;
- an independent validation figure comparing true and identified-model position.

Use a validation multi-sine with frequencies `[0.8, 1.9, 3.4]` Hz so it is distinct from `[0.5, 1.3, 2.7]` Hz identification input.  The report must include a table with `J_hat`, `b_hat`, parameter errors, torque RMSE, condition number, and validation trajectory RMSE for all relevant cases, followed by a concise interpretation of excitation, differentiation noise, and model mismatch.

- [ ] **Step 4: Write the README and run the complete test suite**

README commands:

```powershell
conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_sysid -v
conda run -n robot-control python .\lesson05_system_identification\src\run_sysid.py
```

Run: `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_sysid -v`

Expected: PASS.

- [ ] **Step 5: Run the full experiment and inspect outputs**

Run: `conda run -n robot-control python .\lesson05_system_identification\src\run_sysid.py`

Expected: four named case summaries; CSV logs under `logs/`, figures under `figures/`, parameter JSON files, and `reports/identification_report.md`.

- [ ] **Step 6: Commit**

```powershell
git add lesson05_system_identification
git commit -m "feat: complete lesson05 system identification experiments"
```

## Final verification

- [ ] Run `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_sysid -v` and confirm every test passes.
- [ ] Run `conda run -n robot-control python .\lesson05_system_identification\src\run_sysid.py` and confirm each named case writes its outputs.
- [ ] Confirm no estimator function imports or reads `plant_true` configuration.
- [ ] Confirm validation frequencies differ from the identification frequencies.
- [ ] Confirm generated logs use `torque_applied_nm` for regression.
