# Lesson 04 Digital PID and Cascade Servo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reproducible digital PID, anti-windup, and cascade-servo experiments for the single-axis joint.

**Architecture:** Lesson 04 owns a compact rotary-joint plant, stateful PID class, direct single-loop controller, and cascade controller. A single experiment runner drives all scenarios and emits CSV logs, PNG figures, and one Markdown analysis report.

**Tech Stack:** Python 3.11, NumPy, Matplotlib, PyYAML, `unittest`.

## Global Constraints

- Keep lesson02 unchanged; reuse its nominal joint values: inertia `0.020`, damping `0.080`, torque limit `3.0`, position noise `0.05 deg`, and `dt=0.001 s`.
- Digital PID uses `integral += error * dt`, derivative on measurement, output clipping, and optional integral clamp.
- Each CSV separates `torque_unsat` and `torque_cmd` and records the named controller state.
- Compare controllers under identical plant, sampling, noise, target, load, and torque-limit conditions.

---

### Task 1: Scaffold lesson04 and implement testable digital PID

**Files:**
- Create: `lesson04_digital_pid_cascade/config/servo.yaml`
- Create: `lesson04_digital_pid_cascade/src/pid.py`
- Create: `lesson04_digital_pid_cascade/tests/test_pid.py`

**Interfaces:**
- Produces: `PIDController(kp, ki, kd, output_limit, integral_limit | None)`, `reset() -> None`, and `update(error: float, measurement_rate: float, dt: float) -> tuple[float, float, bool]` returning `(torque_unsat, torque_cmd, saturated)`.

- [ ] **Step 1: Write failing PID tests**

```python
def test_integral_clamp_limits_accumulated_error(self) -> None:
    controller = PIDController(0.0, 2.0, 0.0, 10.0, 0.5)
    for _ in range(10):
        controller.update(1.0, 0.0, 0.1)
    self.assertAlmostEqual(controller.integral, 0.5)

def test_output_clipping_reports_raw_and_applied_torque(self) -> None:
    controller = PIDController(10.0, 0.0, 0.0, 3.0, None)
    raw, applied, saturated = controller.update(1.0, 0.0, 0.01)
    self.assertEqual(raw, 10.0)
    self.assertEqual(applied, 3.0)
    self.assertTrue(saturated)
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest lesson04_digital_pid_cascade.tests.test_pid -v`

Expected: FAIL because `pid.py` is missing.

- [ ] **Step 3: Implement minimal PID**

```python
def update(self, error, measurement_rate, dt):
    self.integral += error * dt
    if self.integral_limit is not None:
        self.integral = float(np.clip(self.integral, -self.integral_limit, self.integral_limit))
    torque_unsat = self.kp * error + self.ki * self.integral - self.kd * measurement_rate
    torque_cmd = float(np.clip(torque_unsat, -self.output_limit, self.output_limit))
    return torque_unsat, torque_cmd, torque_cmd != torque_unsat
```

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m unittest lesson04_digital_pid_cascade.tests.test_pid -v`

Run: `git add lesson04_digital_pid_cascade; git commit -m "feat: add lesson04 digital PID controller"`

### Task 2: Add the plant, direct-loop PD/PID, and load experiment

**Files:**
- Create: `lesson04_digital_pid_cascade/src/plant.py`
- Create: `lesson04_digital_pid_cascade/src/single_loop_pid.py`
- Create: `lesson04_digital_pid_cascade/tests/test_single_loop.py`

**Interfaces:**
- Produces: `JointPlant.step(torque_cmd, load_torque, dt) -> tuple[position_rad, velocity_rad_s]` and `run_single_loop(controller_kind, load_start_s, load_torque_nm) -> dict[str, np.ndarray]`.

- [ ] **Step 1: Write a failing load-rejection test**

```python
def test_pid_reduces_final_load_error_relative_to_pd(self) -> None:
    pd = run_single_loop("pd", 1.0, 1.5)
    pid = run_single_loop("pid", 1.0, 1.5)
    self.assertLess(abs(pid["position_error_rad"][-1]), abs(pd["position_error_rad"][-1]))
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest lesson04_digital_pid_cascade.tests.test_single_loop -v`

Expected: FAIL because `run_single_loop` is missing.

- [ ] **Step 3: Implement plant and direct loop**

Use semi-implicit Euler integration with `acceleration = (torque_cmd - load_torque - damping * velocity) / inertia`. Apply encoder noise only to the measured position, estimate measured velocity with a low-pass filtered finite difference, and log every required field.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m unittest lesson04_digital_pid_cascade.tests.test_single_loop -v`

Run: `git add lesson04_digital_pid_cascade; git commit -m "feat: compare PD and PID under load"`

### Task 3: Add cascade control and complete three artifacts

**Files:**
- Create: `lesson04_digital_pid_cascade/src/cascade_servo.py`
- Create: `lesson04_digital_pid_cascade/src/run_experiments.py`
- Create: `lesson04_digital_pid_cascade/tests/test_experiments.py`
- Create: `lesson04_digital_pid_cascade/reports/digital_pid_cascade.md`

**Interfaces:**
- Produces: `run_cascade() -> dict[str, np.ndarray]`, `run_all_experiments() -> dict[str, Path]`, and the three required figures.

- [ ] **Step 1: Write failing artifact test**

```python
def test_all_experiments_write_required_artifacts(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.object(run_experiments, "ROOT", Path(temp_dir)):
            artifacts = run_experiments.run_all_experiments()
    self.assertEqual(set(artifacts), {"pd_vs_pid_load", "anti_windup", "single_vs_cascade", "report"})
    self.assertTrue(all(path.is_file() for path in artifacts.values()))
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest lesson04_digital_pid_cascade.tests.test_experiments -v`

Expected: FAIL because `run_all_experiments` is missing.

- [ ] **Step 3: Implement cascade and experiment runner**

Implement position P with a velocity-reference limit and velocity PI using the PID class. Generate `pd_vs_pid_load.png`, `anti_windup.png`, and `single_vs_cascade.png`; write their raw simulation records as CSV. The report must answer the five course questions and state that a torque command is not an actual drive torque.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m unittest lesson04_digital_pid_cascade.tests.test_experiments -v`

Run: `git add lesson04_digital_pid_cascade; git commit -m "implement digital PID and cascade servo control"`

### Task 4: Document and fully verify

**Files:**
- Create: `lesson04_digital_pid_cascade/README.md`

- [ ] **Step 1: Document commands and observations**

Document the `robot-control` activation command, runner command, test command, three experiment meanings, anti-windup limitation, derivative-on-measurement choice, and cascade timing hierarchy.

- [ ] **Step 2: Full verification**

Run: `python -m unittest discover -s lesson04_digital_pid_cascade/tests -v; python -m unittest discover -s lesson02_servo_control/tests -v; python .\lesson04_digital_pid_cascade\src\run_experiments.py`

Expected: all lesson04 tests and 19 lesson02 regression tests pass; the three figures, report, and logs exist.

- [ ] **Step 3: Commit**

Run: `git add lesson04_digital_pid_cascade; git commit -m "docs: complete lesson04 PID cascade experiments"`

## Plan Self-Review

- Spec coverage: Tasks 1–3 implement PID, three experiments, required artifacts, logs, and report; Task 4 documents and runs regression tests.
- Placeholder scan: no unresolved work markers remain.
- Type consistency: controller, plant, runner, and test interfaces use radians, N·m, seconds, NumPy logs, and Path artifacts consistently.
