# Servo Engineering Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a smooth cubic reference, realistic simulated timing, timing/latency acceptance experiments, reproducible artifacts, and a zero-safety-trigger normal baseline to lesson 02.

**Architecture:** Keep `run_servo_test.py` as the single simulation core and extend it through small pure helpers. Add `run_timing_acceptance.py` only as an experiment orchestrator that overrides configuration, calls the core, computes metrics, and writes artifacts; it must not duplicate plant or controller equations.

**Tech Stack:** Python 3.11, NumPy, Matplotlib, PyYAML, `unittest`, CSV, Markdown.

## Global Constraints

- Default reference is a 30 degree cubic trajectory completed in 0.8 seconds.
- Acceptance simulation duration is 4.0 seconds and seed is 42.
- Jitter has 5% standard deviation and is clipped to 80% through 120% of nominal period.
- Normal acceptance thresholds are: zero safety samples, final error below 0.5 degrees, overshoot below 5%, maximum velocity below 300 degrees per second, saturation below 10%, and settling time below 1.5 seconds.
- Existing plant, friction, sensor, actuator saturation, safety behavior, and semi-implicit Euler integration remain in one simulation core.
- Generated evidence is offline numerical simulation evidence, not a hard-real-time claim.
- Do not push to GitHub during implementation.

---

### Task 1: Cubic Reference Generator

**Files:**
- Create: `lesson02_servo_control/tests/test_reference.py`
- Modify: `lesson02_servo_control/src/run_servo_test.py`

**Interfaces:**
- Produces: `cubic_reference(time_s: float, start_rad: float, target_rad: float, duration_s: float) -> tuple[float, float]`
- Returns: `(position_reference_rad, velocity_reference_rad_s)`

- [ ] **Step 1: Write the failing trajectory tests**

Create `test_reference.py`, import the `run_servo_test` module, and first obtain the new API with an explicit contract assertion:

```python
cubic_reference = getattr(run_servo_test, "cubic_reference", None)
self.assertIsNotNone(cubic_reference)
```

Then assert:

```python
start = np.deg2rad(0.0)
target = np.deg2rad(30.0)

q0, dq0 = cubic_reference(0.0, start, target, 0.8)
self.assertAlmostEqual(q0, start)
self.assertAlmostEqual(dq0, 0.0)

q_mid, dq_mid = cubic_reference(0.4, start, target, 0.8)
self.assertAlmostEqual(q_mid, np.deg2rad(15.0))
self.assertGreater(dq_mid, 0.0)

q_end, dq_end = cubic_reference(0.8, start, target, 0.8)
self.assertAlmostEqual(q_end, target)
self.assertAlmostEqual(dq_end, 0.0)

q_after, dq_after = cubic_reference(1.2, start, target, 0.8)
self.assertAlmostEqual(q_after, target)
self.assertAlmostEqual(dq_after, 0.0)
```

Add a separate test asserting `duration_s=0.0` raises `ValueError`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
conda run -n robot-control python -m unittest lesson02_servo_control.tests.test_reference -v
```

Expected: failure because `cubic_reference` does not exist.

- [ ] **Step 3: Add the minimal pure function**

Add near the top of `run_servo_test.py`:

```python
def cubic_reference(
    time_s: float,
    start_rad: float,
    target_rad: float,
    duration_s: float,
) -> tuple[float, float]:
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")

    s = float(np.clip(time_s / duration_s, 0.0, 1.0))
    delta = target_rad - start_rad
    position = start_rad + delta * (3.0 * s**2 - 2.0 * s**3)
    velocity = delta / duration_s * (6.0 * s - 6.0 * s**2)
    return float(position), float(velocity)
```

- [ ] **Step 4: Run focused and full tests and verify GREEN**

```powershell
conda run -n robot-control python -m unittest lesson02_servo_control.tests.test_reference -v
conda run -n robot-control python -m unittest discover -s .\lesson02_servo_control\tests -p "test_*.py" -v
```

Expected: trajectory tests and all existing tests pass.

- [ ] **Step 5: Commit the independently verified helper**

```powershell
git add lesson02_servo_control/tests/test_reference.py lesson02_servo_control/src/run_servo_test.py
git commit -m "feat: add cubic servo reference trajectory"
```

---

### Task 2: Reference-Aware PD Tracking and Normal Acceptance

**Files:**
- Modify: `lesson02_servo_control/config/servo.yaml`
- Modify: `lesson02_servo_control/src/run_servo_test.py`
- Create: `lesson02_servo_control/tests/test_normal_acceptance.py`
- Modify: `lesson02_servo_control/tests/test_plotting.py`

**Interfaces:**
- `simulate(config: dict, scenario: str) -> dict` additionally produces `target_velocity`.
- `data["target"]` remains a position array in radians.
- `data["target_velocity"]` is a velocity array in radians per second.

- [ ] **Step 1: Write failing integration tests**

Add tests asserting:

```python
config = run_servo_test.load_config()
data = run_servo_test.simulate(config, scenario="normal")

self.assertIn("target_velocity", data)
self.assertAlmostEqual(data["target"][0], 0.0)
self.assertAlmostEqual(data["target_velocity"][0], 0.0)
self.assertAlmostEqual(data["target"][-1], np.deg2rad(30.0))
self.assertAlmostEqual(data["target_velocity"][-1], 0.0)
self.assertEqual(int(np.sum(data["safety_fault"])), 0)
```

Compute the existing metrics plus maximum true velocity and assert every normal acceptance threshold from Global Constraints.

Update the plotting fixture to include `target_velocity` so the fixture matches the production data contract.

- [ ] **Step 2: Run the focused tests and verify RED**

```powershell
conda run -n robot-control python -m unittest lesson02_servo_control.tests.test_normal_acceptance -v
```

Expected: failure because the config lacks `reference`, output lacks `target_velocity`, and the current step command triggers safety samples.

- [ ] **Step 3: Migrate configuration and simulation**

Move the target out of `controller` and add:

```yaml
reference:
  type: cubic
  target_deg: 30.0
  move_duration_s: 0.8
```

Keep `controller.kp=18.0` and `controller.kd=1.2`. In `simulate()`:

1. Read and validate `reference.type` as either `cubic` or `step`.
2. Allocate `target_position` and `target_velocity` arrays.
3. For each cycle, calculate the selected reference.
4. Use `position_error = q_ref - q_measured[k]`.
5. Use `velocity_error = dq_ref - dq_estimated[k]`.
6. Calculate `kp * position_error + kd * velocity_error`.
7. Copy the final reference samples consistently.
8. Return the arrays as `target` and `target_velocity`.
9. Add `target_velocity` to CSV and plot it with the velocity signals.

For `step`, set position to the final target and velocity to zero for all samples. This mode exists only for comparison and regression experiments.

- [ ] **Step 4: Run focused and full tests and verify GREEN**

```powershell
conda run -n robot-control python -m unittest lesson02_servo_control.tests.test_normal_acceptance -v
conda run -n robot-control python -m unittest discover -s .\lesson02_servo_control\tests -p "test_*.py" -v
```

Expected: normal has zero safety samples and satisfies all six acceptance thresholds; existing tests pass.

- [ ] **Step 5: Commit reference tracking**

```powershell
git add lesson02_servo_control/config/servo.yaml lesson02_servo_control/src/run_servo_test.py lesson02_servo_control/tests/test_normal_acceptance.py lesson02_servo_control/tests/test_plotting.py
git commit -m "feat: track smooth reference in servo baseline"
```

---

### Task 3: Actual Period and Jitter Simulation

**Files:**
- Modify: `lesson02_servo_control/src/run_servo_test.py`
- Create: `lesson02_servo_control/tests/test_timing.py`
- Modify: `lesson02_servo_control/tests/test_simulation.py`

**Interfaces:**
- Produces: `build_time_axis(nominal_dt: float, duration: float, rng: np.random.Generator, jitter_std_ratio: float) -> tuple[np.ndarray, np.ndarray]`
- `simulate()` accepts optional `simulation.jitter_std_ratio`, defaulting to `0.0`.
- `data["actual_dt"]` has the same length as other CSV arrays; its final element repeats the last applied integration period.
- Timing statistics use `actual_dt[:-1]`, because the final element is a logging sentinel rather than another integration step.

- [ ] **Step 1: Write failing deterministic timing tests**

Import the `run_servo_test` module, assert `build_time_axis` is present with `getattr`, and then test the helper with a seeded generator:

```python
time, dt_log = build_time_axis(
    nominal_dt=0.001,
    duration=0.01,
    rng=np.random.default_rng(42),
    jitter_std_ratio=0.0,
)
self.assertAlmostEqual(time[-1], 0.01)
np.testing.assert_allclose(dt_log[:-1], 0.001)
```

For 5% jitter, assert every applied period is between `0.0008` and `0.0012`, the standard deviation is nonzero, and the final time is at least the configured duration.

Add an invalid-input test for nonpositive nominal period, nonpositive duration, and negative jitter ratio.

- [ ] **Step 2: Run the focused tests and verify RED**

```powershell
conda run -n robot-control python -m unittest lesson02_servo_control.tests.test_timing -v
```

Expected: failure because `build_time_axis` and `actual_dt` do not exist.

- [ ] **Step 3: Implement the time-axis helper and wire it into simulation**

Implement the following behavior:

1. With zero jitter, preserve the current exact endpoint and sample count.
2. With jitter, sample and clip each period, append cumulative time, and stop after the cumulative time reaches or exceeds `duration`.
3. Store the applied period at the current sample and repeat the final applied period at the array tail.
4. Use `actual_dt[k - 1]` for the measured-position difference at sample `k`, and use `actual_dt[k]` for semi-implicit Euler integration from sample `k` to `k + 1`.
5. Use the generated cumulative `time` array for scenario timing.
6. Include `actual_dt` in returned data and CSV keys.

- [ ] **Step 4: Verify GREEN and no fixed-period regression**

```powershell
conda run -n robot-control python -m unittest lesson02_servo_control.tests.test_timing lesson02_servo_control.tests.test_simulation -v
conda run -n robot-control python -m unittest discover -s .\lesson02_servo_control\tests -p "test_*.py" -v
```

Expected: fixed-period time axis still ends exactly at 4.0 seconds; jitter tests pass; all existing tests pass.

- [ ] **Step 5: Commit timing support**

```powershell
git add lesson02_servo_control/src/run_servo_test.py lesson02_servo_control/tests/test_timing.py lesson02_servo_control/tests/test_simulation.py
git commit -m "feat: simulate measured control periods and jitter"
```

---

### Task 4: Engineering and Timing Metrics

**Files:**
- Modify: `lesson02_servo_control/src/run_servo_test.py`
- Create: `lesson02_servo_control/tests/test_engineering_metrics.py`

**Interfaces:**
- Produces: `timing_metrics(dt_log: np.ndarray) -> dict[str, float]`
- Produces: `engineering_metrics(data: dict) -> dict[str, float]`

- [ ] **Step 1: Write failing metric tests with hand-calculated arrays**

Import the `run_servo_test` module and use `getattr` plus `assertIsNotNone` for each new helper so the RED result is an assertion failure rather than an import error. For timing:

```python
dt_log = np.array([0.0010, 0.0011, 0.0009])
metrics = timing_metrics(dt_log)
self.assertAlmostEqual(metrics["mean_dt_ms"], 1.0)
self.assertAlmostEqual(metrics["max_dt_ms"], 1.1)
self.assertAlmostEqual(
    metrics["jitter_std_ms"],
    np.std(dt_log) * 1000.0,
)
```

For engineering data, use small radian arrays and assert exact maximum absolute velocity and maximum absolute `target - position` in degrees.

Add tests that empty timing arrays and mismatched target/position arrays raise `ValueError`.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
conda run -n robot-control python -m unittest lesson02_servo_control.tests.test_engineering_metrics -v
```

Expected: failure because both metric helpers are absent.

- [ ] **Step 3: Add minimal metric helpers**

`timing_metrics()` returns `mean_dt_ms`, `max_dt_ms`, and `jitter_std_ms`. `engineering_metrics()` returns `max_velocity_deg_s` and `max_position_error_deg`. Both validate their required arrays before calculation.

- [ ] **Step 4: Run focused and full tests and verify GREEN**

```powershell
conda run -n robot-control python -m unittest lesson02_servo_control.tests.test_engineering_metrics -v
conda run -n robot-control python -m unittest discover -s .\lesson02_servo_control\tests -p "test_*.py" -v
```

- [ ] **Step 5: Commit metrics**

```powershell
git add lesson02_servo_control/src/run_servo_test.py lesson02_servo_control/tests/test_engineering_metrics.py
git commit -m "feat: add servo engineering timing metrics"
```

---

### Task 5: Acceptance Experiment Orchestrator and Artifacts

**Files:**
- Create: `lesson02_servo_control/config/timing_acceptance.yaml`
- Create: `lesson02_servo_control/src/run_timing_acceptance.py`
- Create: `lesson02_servo_control/tests/test_timing_acceptance.py`
- Generate: `lesson02_servo_control/logs/acceptance_normal.csv`
- Generate: `lesson02_servo_control/logs/delay_1step.csv`
- Generate: `lesson02_servo_control/logs/delay_5steps.csv`
- Generate: `lesson02_servo_control/logs/sample_5ms.csv`
- Generate: `lesson02_servo_control/logs/jitter.csv`
- Generate: `lesson02_servo_control/figures/step_vs_cubic.png`
- Generate: `lesson02_servo_control/figures/timing_comparison.png`
- Generate: `lesson02_servo_control/reports/engineering_acceptance.md`

**Interfaces:**
- Produces: `run_acceptance(base_config: dict, acceptance_config: dict) -> dict[str, dict]`
- Each result contains `data`, `metrics`, `nominal_dt_ms`, and `nominal_delay_ms`.
- Produces: `normal_passes(metrics: dict[str, float], limits: dict) -> tuple[bool, dict[str, bool]]`

- [ ] **Step 1: Write a failing artifact-contract test**

First assert that these two required project files exist:

```python
self.assertTrue((PROJECT_ROOT / "config" / "timing_acceptance.yaml").is_file())
self.assertTrue((PROJECT_ROOT / "src" / "run_timing_acceptance.py").is_file())
```

- [ ] **Step 2: Run the artifact-contract test and verify RED**

```powershell
conda run -n robot-control python -m unittest lesson02_servo_control.tests.test_timing_acceptance -v
```

Expected: assertion failure because both required files are absent.

- [ ] **Step 3: Add the configuration and importable orchestrator shell**

Create `timing_acceptance.yaml` with the five cases and six thresholds. Create `run_timing_acceptance.py` with imports, `ROOT`, and its acceptance-config path only. Re-run the artifact-contract test and confirm it passes.

- [ ] **Step 4: Add failing behavior tests**

Import the new module after placing its `src` directory on `sys.path`. Use `getattr` and `assertIsNotNone` to require `run_acceptance` and `normal_passes`, then test that the acceptance config defines exactly:

```python
{
    "acceptance_normal",
    "delay_1step",
    "delay_5steps",
    "sample_5ms",
    "jitter",
}
```

Test that each result reports the intended nominal period and delay, C and D both report 5 ms nominal delay, jitter has nonzero period standard deviation, and `normal_passes()` returns true only when all six limits pass.

- [ ] **Step 5: Run behavior tests and verify RED**

```powershell
conda run -n robot-control python -m unittest lesson02_servo_control.tests.test_timing_acceptance -v
```

Expected: assertion failure because the two required functions do not exist.

- [ ] **Step 6: Add the minimal orchestrator behavior**

In the orchestrator:

1. Deep-copy the base config for every case.
2. Apply only period, delay, jitter, and reference overrides.
3. Call `simulate(..., scenario="normal")`.
4. Merge control, engineering, timing, saturation, and safety metrics.
5. Save one CSV per named case through the existing `save_log()`.
6. Run one extra `step` comparison without treating it as an acceptance case.
7. Generate `step_vs_cubic.png` from step and cubic position/velocity.
8. Generate `timing_comparison.png` comparing position and velocity for A/B/C/D/jitter.
9. Generate Markdown containing exact result tables and the hard-real-time limitation statement.
10. Exit with a nonzero status from `main()` if `acceptance_normal` fails any threshold.

- [ ] **Step 7: Run focused and full tests and verify GREEN**

```powershell
conda run -n robot-control python -m unittest lesson02_servo_control.tests.test_timing_acceptance -v
conda run -n robot-control python -m unittest discover -s .\lesson02_servo_control\tests -p "test_*.py" -v
```

- [ ] **Step 8: Run the real acceptance experiment**

```powershell
conda run -n robot-control python .\lesson02_servo_control\src\run_timing_acceptance.py
```

Expected: five timing cases are printed, `acceptance_normal` reports PASS, and all listed CSV/PNG/Markdown artifacts exist.

- [ ] **Step 9: Commit code, config, tests, and generated evidence**

```powershell
git add lesson02_servo_control/config/timing_acceptance.yaml lesson02_servo_control/src/run_timing_acceptance.py lesson02_servo_control/tests/test_timing_acceptance.py lesson02_servo_control/logs lesson02_servo_control/figures lesson02_servo_control/reports
git commit -m "feat: add servo timing acceptance experiments"
```

---

### Task 6: Documentation and Final Verification

**Files:**
- Modify: `lesson02_servo_control/README.md`
- Modify: `README.md`

**Interfaces:**
- Documents the two executable entry points and explicitly limits all timing claims to offline simulation.

- [ ] **Step 1: Update lesson documentation**

Document:

- Why the original position step caused saturation and overspeed;
- The cubic position and velocity equations;
- The difference between motion planning parameters and PD gains;
- How to run normal/disturbance/sensor scenarios;
- How to run timing acceptance;
- The A/B/C/D/jitter experiment definitions;
- How to interpret timing metrics and safety sample counts;
- Why Python on ordinary Windows is not a hard-real-time validation platform.

Add the acceptance command to the root README.

- [ ] **Step 2: Run fresh full verification**

```powershell
conda run -n robot-control python -m unittest discover -s .\lesson02_servo_control\tests -p "test_*.py" -v
conda run -n robot-control python .\lesson02_servo_control\src\run_servo_test.py
conda run -n robot-control python .\lesson02_servo_control\src\run_timing_acceptance.py
git diff --check
git status --short
```

Expected: all tests pass, both programs exit zero, normal acceptance passes all six requirements, no whitespace errors exist, and only intended documentation/artifact changes remain.

- [ ] **Step 3: Inspect generated evidence**

Confirm that `engineering_acceptance.md` includes exact A/B/C/D/jitter values, that both PNG files open correctly, and that every acceptance CSV has `actual_dt`, `target`, and `target_velocity` columns.

- [ ] **Step 4: Commit documentation**

```powershell
git add README.md lesson02_servo_control/README.md lesson02_servo_control/reports/engineering_acceptance.md
git commit -m "docs: complete servo engineering acceptance report"
```

- [ ] **Step 5: Report local completion without pushing**

Provide the test count, exact normal acceptance metrics, generated artifact paths, commits created, and any limitations observed. Ask separately before any GitHub push.
