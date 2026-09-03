# Lesson 06 Identification Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend Lesson 05 with excitation-quality, offline-filter sensitivity, friction-model, and payload-robustness experiments.

**Architecture:** Excitation strategies share one `evaluate()` interface. New identifiability, validation, and residual modules stay separate from the existing plant; `run_quality_experiments.py` composes them and writes all Lesson 06 artifacts.

**Tech Stack:** Python 3.11, NumPy, SciPy, Matplotlib, PyYAML, unittest.

**Spec:** `docs/superpowers/specs/2026-09-03-lesson06-identification-quality-design.md`

## Global Constraints

- Modify `lesson05_system_identification/`; do not create a new lesson directory.
- Use actual applied torque and processed signals for every estimator.
- Label Savitzky–Golay and zero-phase filtering as offline, non-causal processing.
- Keep `ROS_COURSE_material_2026.zip` untracked and untouched.
- Run tests with `conda run -n robot-control python -m unittest ... -v`.

## File structure

| File | Responsibility |
| --- | --- |
| `config/excitation_cases.yaml` | Four excitation definitions and shared limits. |
| `src/excitation.py` | Constant, single-sine, multi-sine, PRBS strategies. |
| `src/signal_processing.py` | Savitzky–Golay derivative estimates. |
| `src/identifiability.py` | SVD, rank, condition number. |
| `src/estimator.py` | Model B: J, b, smooth Coulomb friction. |
| `src/validation.py` | Distinct-band and payload validation helpers. |
| `src/residual_analysis.py` | Residual-vs-velocity plot. |
| `src/run_quality_experiments.py` | Runs A–D and writes report/artifacts. |
| `tests/test_identifiability.py` | Focused Lesson 06 unit/integration tests. |

---

### Task 1: Excitation strategies and fair configuration

**Files:** Create `config/excitation_cases.yaml`, `tests/test_identifiability.py`; modify `src/excitation.py`.

**Interfaces:** `ExcitationSignal.evaluate(time_s)`, `ConstantTorque(amplitude_nm)`, `SingleSine(amplitude_nm, frequency_hz, phase_rad=0.0)`, `MultiSine(frequencies_hz, amplitudes_nm, phases_rad=None)`, `PRBSExcitation(amplitude_nm, hold_time_s, seed)`.

- [ ] **Step 1: Write failing tests**

```python
class ExcitationStrategyTest(unittest.TestCase):
    def test_constant_and_single_sine_values(self):
        self.assertEqual(ConstantTorque(0.5).evaluate(2.0), 0.5)
        self.assertAlmostEqual(SingleSine(0.4, 1.0).evaluate(0.25), 0.4)

    def test_prbs_is_bounded_and_holds_each_interval(self):
        signal = PRBSExcitation(0.5, 0.1, 2026)
        values = signal.evaluate(np.array([0.00, 0.03, 0.09, 0.10]))
        self.assertTrue(np.all(np.abs(values) <= 0.5))
        self.assertEqual(values[0], values[1])
        self.assertEqual(values[1], values[2])
```

- [ ] **Step 2: Verify failure**

Run: `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_identifiability.ExcitationStrategyTest -v`

Expected: FAIL because these strategies do not exist.

- [ ] **Step 3: Implement minimal strategies**

```python
class ExcitationSignal:
    def evaluate(self, time_s):
        raise NotImplementedError

class ConstantTorque(ExcitationSignal):
    def evaluate(self, time_s):
        time = np.asarray(time_s, dtype=float)
        value = np.full_like(time, self.amplitude_nm, dtype=float)
        return float(value) if time.ndim == 0 else value
```

Make PRBS deterministic with `np.random.default_rng(seed)` and one sign per `floor(time / hold_time_s)` interval. Reject non-positive amplitude/frequency/hold time and retain `MultiSineExcitation` as a compatibility wrapper. Create YAML with `max_amplitude_nm: 0.5`, `constant`, `single_sine`, `multisine`, and `prbs`; use PRBS hold time `0.10 s`.

- [ ] **Step 4: Verify pass**

Run: `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_identifiability.ExcitationStrategyTest -v`

- [ ] **Step 5: Commit**

```powershell
git add lesson05_system_identification/config/excitation_cases.yaml lesson05_system_identification/src/excitation.py lesson05_system_identification/tests/test_identifiability.py
git commit -m "feat: add lesson06 excitation strategies"
```

### Task 2: Offline Savitzky–Golay derivatives

**Files:** Modify `src/signal_processing.py`, `tests/test_identifiability.py`.

**Interface:** `savgol_derivatives(position_rad, dt_s, window_length, polyorder=3) -> tuple[np.ndarray, np.ndarray]`.

- [ ] **Step 1: Write failing tests**

```python
class SavgolDerivativeTest(unittest.TestCase):
    def test_quadratic_signal_derivatives(self):
        time = np.linspace(0.0, 1.0, 101)
        velocity, acceleration = savgol_derivatives(2.0 * time**2, 0.01, 31)
        self.assertEqual(velocity.shape, time.shape)
        self.assertAlmostEqual(velocity[50], 2.0, places=2)
        self.assertAlmostEqual(acceleration[50], 4.0, places=2)

    def test_even_window_is_rejected(self):
        with self.assertRaises(ValueError):
            savgol_derivatives(np.zeros(20), 0.01, 10)
```

- [ ] **Step 2: Verify failure**

Run: `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_identifiability.SavgolDerivativeTest -v`

- [ ] **Step 3: Implement the offline estimator**

```python
def savgol_derivatives(position_rad, dt_s, window_length, polyorder=3):
    if dt_s <= 0 or window_length % 2 == 0 or window_length <= polyorder:
        raise ValueError("dt_s must be positive; window must be odd and exceed polyorder.")
    position = np.asarray(position_rad, dtype=float)
    if position.ndim != 1 or len(position) < window_length:
        raise ValueError("position_rad must contain at least window_length samples.")
    return (
        savgol_filter(position, window_length, polyorder, deriv=1, delta=dt_s),
        savgol_filter(position, window_length, polyorder, deriv=2, delta=dt_s),
    )
```

Use `scipy.signal.savgol_filter` and document that this centred method is offline and non-causal.

- [ ] **Step 4: Verify pass**

Run: `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_identifiability.SavgolDerivativeTest -v`

- [ ] **Step 5: Commit**

```powershell
git add lesson05_system_identification/src/signal_processing.py lesson05_system_identification/tests/test_identifiability.py
git commit -m "feat: add offline Savitzky-Golay derivatives"
```

### Task 3: Identifiability and smooth-friction Model B

**Files:** Create `src/identifiability.py`; modify `src/estimator.py`, `tests/test_identifiability.py`.

**Interfaces:** `analyse_regressor(phi) -> dict`; `identify_j_b_tau_c(velocity, acceleration, torque, friction_smoothing_rad_s) -> dict`.

- [ ] **Step 1: Write failing tests**

```python
class IdentifiabilityTest(unittest.TestCase):
    def test_independent_columns_have_rank_two(self):
        result = analyse_regressor(np.array([[1., 0.], [0., 2.], [1., 1.]]))
        self.assertEqual(result["rank"], 2)
        self.assertGreater(result["sigma_max"], result["sigma_min"])

    def test_model_b_recovers_friction(self):
        velocity = np.array([-2., -1., -0.2, 0.3, 1.2, 2.])
        acceleration = np.array([1., -2., 3., -1., 2., -3.])
        torque = 0.02 * acceleration + 0.08 * velocity + 0.10 * np.tanh(velocity / 0.05)
        result = identify_j_b_tau_c(velocity, acceleration, torque, 0.05)
        self.assertAlmostEqual(result["coulomb_friction_hat"], 0.10)
```

- [ ] **Step 2: Verify failure**

Run: `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_identifiability.IdentifiabilityTest -v`

- [ ] **Step 3: Implement SVD and Model B**

```python
def analyse_regressor(phi):
    matrix = np.asarray(phi, dtype=float)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return {
        "rank": int(np.linalg.matrix_rank(matrix)),
        "condition_number": float(np.linalg.cond(matrix)),
        "sigma_max": float(singular_values[0]),
        "sigma_min": float(singular_values[-1]),
    }
```

Build the Model B regressor with `np.column_stack([acceleration, velocity, np.tanh(velocity / smoothing)])`; reject non-positive smoothing and rank below 3. Return predicted torque, residual, RMSE, J, b, and tau_c.

- [ ] **Step 4: Verify pass**

Run: `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_identifiability.IdentifiabilityTest -v`

- [ ] **Step 5: Commit**

```powershell
git add lesson05_system_identification/src/identifiability.py lesson05_system_identification/src/estimator.py lesson05_system_identification/tests/test_identifiability.py
git commit -m "feat: analyse identifiability and fit smooth friction"
```

### Task 4: Independent validation and residual diagnostics

**Files:** Create `src/validation.py`, `src/residual_analysis.py`; modify `tests/test_identifiability.py`.

**Interfaces:** `validation_excitation() -> MultiSine`; `payload_plant_parameters(nominal, inertia_scale=1.3) -> dict`; `plot_residual_vs_velocity(velocity, residual, path) -> None`.

- [ ] **Step 1: Write failing tests**

```python
class ValidationDesignTest(unittest.TestCase):
    def test_validation_uses_distinct_high_band(self):
        self.assertEqual(validation_excitation().frequencies_hz.tolist(), [4.2, 5.1, 5.8])

    def test_payload_scales_only_inertia(self):
        payload = payload_plant_parameters({"inertia": 0.02, "damping": 0.08, "coulomb_friction": 0.1})
        self.assertAlmostEqual(payload["inertia"], 0.026)
        self.assertEqual(payload["damping"], 0.08)
```

- [ ] **Step 2: Verify failure**

Run: `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_identifiability.ValidationDesignTest -v`

- [ ] **Step 3: Implement helpers**

Return a validation multi-sine at `[4.2, 5.1, 5.8] Hz`, each amplitude at or below the YAML limit. Deep-copy nominal plant parameters, reject non-positive payload scale, and multiply only inertia. Plot residual against velocity with a zero-residual reference line, labels, grid, and `dpi=180` output.

- [ ] **Step 4: Verify pass**

Run: `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_identifiability.ValidationDesignTest -v`

- [ ] **Step 5: Commit**

```powershell
git add lesson05_system_identification/src/validation.py lesson05_system_identification/src/residual_analysis.py lesson05_system_identification/tests/test_identifiability.py
git commit -m "feat: add lesson06 validation diagnostics"
```

### Task 5: Experiments A–D and quality report

**Files:** Create `src/run_quality_experiments.py`, `reports/sysid_quality_report.md`; modify `README.md`, `tests/test_identifiability.py`.

**Interface:** `run_quality_experiments(sysid_config: dict, cases_config: dict, output_root: Path) -> dict`.

- [ ] **Step 1: Write a failing integration test**

```python
class QualityRunnerTest(unittest.TestCase):
    def test_runner_writes_required_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_quality_experiments(TEST_SYSID_CONFIG, TEST_CASES_CONFIG, Path(directory))
            self.assertEqual(set(result["excitation_cases"]), {"constant", "single_sine", "multisine", "prbs"})
            self.assertTrue((Path(directory) / "figures" / "excitation_compare.png").is_file())
            self.assertTrue((Path(directory) / "reports" / "sysid_quality_report.md").is_file())
            self.assertIn("payload_validation_rmse_rad", result)
```

`TEST_SYSID_CONFIG` uses `dt=0.002`, `duration_s=2.0`, and low noise. `TEST_CASES_CONFIG` uses four signals bounded by `0.4 N m`.

- [ ] **Step 2: Verify failure**

Run: `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_identifiability.QualityRunnerTest -v`

- [ ] **Step 3: Implement A–D**

For experiment A, simulate each signal under matched conditions, run Model A with Savitzky–Golay derivatives, analyse the regressor, and validate with a distinct band. If rank is insufficient, record `identifiable: false` and report why instead of calculating parameter error.

For B, use one fixed noisy multi-sine dataset and windows `[11, 31, 61, 101]`. For C, simulate smooth friction then compare Model A and Model B on identical training and validation data. For D, fit nominal Model B then apply exactly `1.3 * inertia` only during validation.

Write the named figures `excitation_compare.png`, `singular_values.png`, `validation_compare.png`, and `residual_vs_velocity.png`. CSV output includes `excitation_type`, `excitation_command_nm`, `q_measured_rad`, `dq_est_rad_s`, `ddq_est_rad_s2`, `torque_measured_nm`, `torque_predicted_nm`, and `torque_residual_nm`.

Write a Markdown report with four A–D tables and exactly this caveat: “Savitzky–Golay and zero-phase filtering here are offline identification processing; they are not causal online state estimators.”

- [ ] **Step 4: Verify integration and full Lesson 05 tests**

Run: `conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_identifiability -v`

Run: `conda run -n robot-control python -m unittest discover -s lesson05_system_identification/tests -v`

Expected: PASS.

- [ ] **Step 5: Run the full experiment and update README**

Run: `conda run -n robot-control python .\lesson05_system_identification\src\run_quality_experiments.py`

Expected: A–D summaries, all four figures, logs, JSON-safe summaries, and `reports/sysid_quality_report.md`. Add this command and the offline-processing caveat to `README.md`.

- [ ] **Step 6: Commit**

```powershell
git add lesson05_system_identification
git commit -m "feat: complete lesson06 identification quality experiments"
```

## Final verification

- [ ] Run full Lesson 05 test discovery with no failures.
- [ ] Confirm report covers experiments A–D and the offline-processing caveat.
- [ ] Confirm PRBS amplitude and hold interval obey YAML limits.
- [ ] Confirm validation excitation differs from identification excitation.
- [ ] Confirm payload validation uses exactly `1.3 * nominal inertia`.
