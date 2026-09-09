# Lesson 08: Discrete LQR Optimal Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible single-joint discrete LQR project that exposes the Q/R trade-off, compares LQR against pole placement, and tests LQR under payload mismatch and saturated constant load.

**Architecture:** Lesson 08 owns model construction, LQR/DARE design, saturation-aware controller, noisy plant simulation, metrics, and experiment orchestration. It uses the validated Lesson 06/07 values `J_hat=0.019762` and `b_hat=0.080257` as explicit configuration values, while all experiment output stays under the new lesson directory.

**Tech Stack:** Python 3.11, NumPy, SciPy (`cont2discrete`, `solve_discrete_are`, `place_poles`), python-control (`dlqr`), Matplotlib, PyYAML, unittest.

**Spec:** `docs/superpowers/specs/2026-09-09-lesson08-lqr-optimal-control-design.md`

## Global Constraints

- Keep all internal angles in rad, angular velocities in rad/s, and torques in N m.
- Use ZOH and `dt_s: 0.001` for all Lesson 08 controller designs and simulations.
- Use `control.dlqr` for production LQR and a DARE implementation only for independent test verification.
- Record unsaturated command, clipped command, saturation state, and state/input/total stage costs independently.
- Design the payload controller at nominal `J_hat` and do not redesign it at `1.3 * J_hat`.
- Recompute the Lesson 07 pole-placement specification at 1 ms for a fair LQR comparison; never reuse a 10 ms numeric gain.
- Do not modify Lesson 05, Lesson 06, or Lesson 07 source code.

---

### Task 1: Lesson scaffold, configuration, and discrete model

**Files:**
- Create: `lesson08_lqr_optimal_control/__init__.py`
- Create: `lesson08_lqr_optimal_control/src/__init__.py`
- Create: `lesson08_lqr_optimal_control/tests/__init__.py`
- Create: `lesson08_lqr_optimal_control/config/lqr.yaml`
- Create: `lesson08_lqr_optimal_control/src/model.py`
- Create: `lesson08_lqr_optimal_control/tests/test_model.py`

**Interfaces:**
- Produces: `load_config(path: Path) -> dict`
- Produces: `build_continuous_model(inertia: float, damping: float) -> tuple[np.ndarray, np.ndarray]`
- Produces: `discretize_zoh(A: np.ndarray, B: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: Write the failing model test**

```python
def test_identified_joint_model_and_zoh_shapes():
    A, B = build_continuous_model(0.02, 0.08)
    np.testing.assert_allclose(A, [[0.0, 1.0], [0.0, -4.0]])
    np.testing.assert_allclose(B, [[0.0], [50.0]])
    Ad, Bd = discretize_zoh(A, B, 0.001)
    assert Ad.shape == (2, 2)
    assert Bd.shape == (2, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n robot-control python -m unittest lesson08_lqr_optimal_control.tests.test_model -v`

Expected: `ModuleNotFoundError` for `lesson08_lqr_optimal_control.src.model`.

- [ ] **Step 3: Add configuration and minimal model implementation**

```python
def build_continuous_model(inertia: float, damping: float):
    if inertia <= 0.0 or damping < 0.0:
        raise ValueError("inertia must be positive and damping must be non-negative.")
    A = np.array([[0.0, 1.0], [0.0, -damping / inertia]])
    B = np.array([[0.0], [1.0 / inertia]])
    return A, B

def discretize_zoh(A: np.ndarray, B: np.ndarray, dt_s: float):
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive.")
    Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(A.shape[0]), np.zeros((A.shape[0], 1))), dt_s, method="zoh")
    return Ad, Bd
```

Write `lqr.yaml` with `dt_s: 0.001`, `duration_s: 4.0`, seed `2028`, nominal inertia/damping, 30 deg target, maximum position error `5.0 deg`, maximum velocity error `1.5 rad/s`, maximum torque `3.0 N m`, position/velocity noise `0.03`, payload scale `1.30`, and stress settings of a 1 s / 1 N m load with a 1.5 N m torque limit.

- [ ] **Step 4: Run model test to verify it passes**

Run: `conda run -n robot-control python -m unittest lesson08_lqr_optimal_control.tests.test_model -v`

Expected: model entries and 2-by-2 / 2-by-1 ZOH shapes pass.

- [ ] **Step 5: Commit**

```powershell
git add lesson08_lqr_optimal_control/config lesson08_lqr_optimal_control/src/model.py lesson08_lqr_optimal_control/tests/test_model.py lesson08_lqr_optimal_control/__init__.py lesson08_lqr_optimal_control/src/__init__.py lesson08_lqr_optimal_control/tests/__init__.py
git commit -m "feat: add lesson08 lqr model scaffold"
```

### Task 2: Bryson weights, dlqr design, and DARE cross-check

**Files:**
- Create: `lesson08_lqr_optimal_control/src/lqr_design.py`
- Create: `lesson08_lqr_optimal_control/tests/test_lqr_design.py`

**Interfaces:**
- Consumes: `Ad`, `Bd` from `model.py`.
- Produces: `build_bryson_weights(max_position_error_deg: float, max_velocity_error_rad_s: float, max_torque_nm: float, position_scale: float, velocity_scale: float, torque_scale: float) -> tuple[np.ndarray, np.ndarray]`
- Produces: `design_dlqr(Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray, R: np.ndarray) -> dict[str, np.ndarray]`
- Produces: `dlqr_from_dare(Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
- Produces: `design_pole_placement(Ad: np.ndarray, Bd: np.ndarray, damping_ratio: float, settling_time_s: float, dt_s: float) -> dict[str, np.ndarray]`

- [ ] **Step 1: Write failing Bryson and DARE-agreement tests**

```python
def test_bryson_weights_use_radians_and_input_limit():
    Q, R = build_bryson_weights(5.0, 1.5, 3.0, 1.0, 1.0, 1.0)
    np.testing.assert_allclose(Q, np.diag([1.0 / np.deg2rad(5.0) ** 2, 1.0 / 1.5 ** 2]))
    np.testing.assert_allclose(R, [[1.0 / 9.0]])

def test_dlqr_matches_independent_dare_solution():
    design = design_dlqr(Ad, Bd, Q, R)
    K_dare, P = dlqr_from_dare(Ad, Bd, Q, R)
    np.testing.assert_allclose(design["K"], K_dare, rtol=1e-6, atol=1e-8)
    assert np.all(np.abs(design["closed_loop_poles"]) < 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n robot-control python -m unittest lesson08_lqr_optimal_control.tests.test_lqr_design -v`

Expected: `ModuleNotFoundError` for `lqr_design`.

- [ ] **Step 3: Implement Bryson and LQR design functions**

```python
def build_bryson_weights(max_position_error_deg, max_velocity_error_rad_s, max_torque_nm, position_scale, velocity_scale, torque_scale):
    max_position_error_rad = np.deg2rad(max_position_error_deg)
    Q = np.diag([position_scale / max_position_error_rad**2, velocity_scale / max_velocity_error_rad_s**2])
    R = np.array([[torque_scale / max_torque_nm**2]])
    return Q, R

def dlqr_from_dare(Ad, Bd, Q, R):
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, P
```

`design_dlqr` calls `ct.dlqr`, returns `K`, `S`, and `closed_loop_poles`; it validates matrix dimensions and positive-definite scalar `R`. `design_pole_placement` maps the Lesson 07 continuous pair to 1 ms discrete poles and calls `scipy.signal.place_poles`.

- [ ] **Step 4: Run LQR design test to verify it passes**

Run: `conda run -n robot-control python -m unittest lesson08_lqr_optimal_control.tests.test_lqr_design -v`

Expected: Bryson values are unit-correct, `dlqr` and DARE gains agree, and LQR poles are inside the unit circle.

- [ ] **Step 5: Commit**

```powershell
git add lesson08_lqr_optimal_control/src/lqr_design.py lesson08_lqr_optimal_control/tests/test_lqr_design.py
git commit -m "feat: add lesson08 discrete lqr design"
```

### Task 3: LQR controller and explicit saturation record

**Files:**
- Create: `lesson08_lqr_optimal_control/src/controller.py`
- Create: `lesson08_lqr_optimal_control/tests/test_controller.py`

**Interfaces:**
- Produces: `LQRController(K: np.ndarray, torque_limit_nm: float)`
- Produces: `LQRController.update(x: np.ndarray, x_ref: np.ndarray, torque_ff: float = 0.0) -> dict[str, float | bool]`

- [ ] **Step 1: Write the failing controller test**

```python
def test_controller_records_feedforward_and_actuator_clipping():
    controller = LQRController(np.array([[10.0, 2.0]]), torque_limit_nm=3.0)
    result = controller.update(np.array([1.0, 0.0]), np.zeros(2), torque_ff=0.5)
    assert result["torque_unsat_nm"] == -9.5
    assert result["torque_cmd_nm"] == -3.0
    assert result["saturated"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n robot-control python -m unittest lesson08_lqr_optimal_control.tests.test_controller -v`

Expected: `ModuleNotFoundError` for `controller`.

- [ ] **Step 3: Implement controller**

```python
def update(self, x, x_ref, torque_ff=0.0):
    error = np.asarray(x, dtype=float) - np.asarray(x_ref, dtype=float)
    torque_unsat_nm = float(torque_ff - (self.K @ error).item())
    torque_cmd_nm = float(np.clip(torque_unsat_nm, -self.torque_limit_nm, self.torque_limit_nm))
    return {"torque_unsat_nm": torque_unsat_nm, "torque_cmd_nm": torque_cmd_nm,
            "saturated": bool(abs(torque_unsat_nm) > self.torque_limit_nm)}
```

Validate a 1-by-2 K, two-state vectors, and positive torque limit.

- [ ] **Step 4: Run controller test to verify it passes**

Run: `conda run -n robot-control python -m unittest lesson08_lqr_optimal_control.tests.test_controller -v`

Expected: the unconstrained and clipped torque values are both preserved.

- [ ] **Step 5: Commit**

```powershell
git add lesson08_lqr_optimal_control/src/controller.py lesson08_lqr_optimal_control/tests/test_controller.py
git commit -m "feat: add lesson08 lqr controller"
```

### Task 4: Noisy plant simulation and LQR cost metrics

**Files:**
- Create: `lesson08_lqr_optimal_control/src/simulator.py`
- Create: `lesson08_lqr_optimal_control/src/metrics.py`
- Create: `lesson08_lqr_optimal_control/tests/test_simulator.py`

**Interfaces:**
- Consumes: LQR controller, configuration, `Q`, and `R`.
- Produces: `simulate_closed_loop(config: dict, controller: LQRController, Q: np.ndarray, R: np.ndarray, plant_inertia: float, load_torque_nm: float = 0.0, load_start_s: float = float("inf")) -> dict[str, np.ndarray]`
- Produces: `calculate_metrics(log: dict[str, np.ndarray]) -> dict[str, float]`

- [ ] **Step 1: Write failing simulation and cost tests**

```python
def test_simulation_keeps_command_within_limit_and_logs_costs():
    log = simulate_closed_loop(config, controller, Q, R, plant_inertia=0.019762)
    assert {"torque_unsat_nm", "torque_cmd_nm", "saturated", "state_cost", "input_cost", "total_stage_cost"}.issubset(log)
    assert np.all(np.abs(log["torque_cmd_nm"]) <= 3.0)
    np.testing.assert_allclose(log["total_stage_cost"], log["state_cost"] + log["input_cost"])

def test_metrics_sums_logged_lqr_costs():
    metrics = calculate_metrics({"time_s": np.array([0.0, 0.1]), "q_ref_rad": np.ones(2), "q_rad": np.ones(2),
                                 "torque_cmd_nm": np.zeros(2), "saturated": np.zeros(2, dtype=bool),
                                 "state_cost": np.array([2.0, 3.0]), "input_cost": np.array([5.0, 7.0]),
                                 "total_stage_cost": np.array([7.0, 10.0])})
    assert metrics["state_cost_total"] == 5.0
    assert metrics["input_cost_total"] == 12.0
    assert metrics["lqr_cost_total"] == 17.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n robot-control python -m unittest lesson08_lqr_optimal_control.tests.test_simulator -v`

Expected: `ModuleNotFoundError` for `simulator` and `metrics`.

- [ ] **Step 3: Implement causal noisy simulation and metrics**

At every sample create measured `q` and `dq` by independent Gaussian noise, compute the controller output, activate `load_torque_nm` only when `time_s >= load_start_s`, and update the hidden plant through semi-implicit Euler. Log `q_ref_rad`, `dq_ref_rad_s`, true states, measured states, position/velocity errors, both torques, saturation, and all costs.

```python
error = measured_state - reference_state
state_cost = float(error @ Q @ error)
input_cost = float(controller_output["torque_unsat_nm"] ** 2 * R.item())
total_stage_cost = state_cost + input_cost
```

`calculate_metrics` computes final error, 2% settling time, overshoot, tracking RMSE, peak/RMS commanded torque, saturation ratio, and sums the three logged costs.

- [ ] **Step 4: Run simulator test to verify it passes**

Run: `conda run -n robot-control python -m unittest lesson08_lqr_optimal_control.tests.test_simulator -v`

Expected: command never crosses the configured hard limit and stage cost equals its logged components.

- [ ] **Step 5: Commit**

```powershell
git add lesson08_lqr_optimal_control/src/simulator.py lesson08_lqr_optimal_control/src/metrics.py lesson08_lqr_optimal_control/tests/test_simulator.py
git commit -m "feat: add lesson08 lqr simulation costs"
```

### Task 5: A-D experiment orchestration, artifacts, and report

**Files:**
- Create: `lesson08_lqr_optimal_control/src/run_experiments.py`
- Create: `lesson08_lqr_optimal_control/tests/test_experiments.py`
- Create: `lesson08_lqr_optimal_control/README.md`

**Interfaces:**
- Consumes: all earlier Lesson 08 modules.
- Produces: `run_all(config_path: Path, output_root: Path | None = None) -> dict`

- [ ] **Step 1: Write the failing artifact test**

```python
def test_run_all_writes_four_figures_logs_summary_and_report():
    with tempfile.TemporaryDirectory() as directory:
        result = run_all(Path("lesson08_lqr_optimal_control/config/lqr.yaml"), output_root=Path(directory))
        assert set(result["figures"]) == {"qr_tradeoff", "lqr_vs_pole_placement", "payload_mismatch", "load_saturation_stress"}
        assert all(path.is_file() for path in result["figures"].values())
        assert all(path.is_file() for path in result["logs"].values())
        assert result["summary_path"].is_file()
        assert result["report_path"].is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n robot-control python -m unittest lesson08_lqr_optimal_control.tests.test_experiments -v`

Expected: `ModuleNotFoundError` for `run_experiments`.

- [ ] **Step 3: Implement experiments and output writer**

Run the four named Bryson cases, then select Balanced as nominal LQR for the comparison, payload, and stress cases. Recompute the PP controller at 1 ms from `zeta=0.8` and `settling_time_s=0.8`. Write CSV files for every case; make figures named `qr_tradeoff.png`, `lqr_vs_pole_placement.png`, `payload_mismatch.png`, and `load_saturation_stress.png`; and write a Markdown Q/R summary table plus `lqr_report.md`.

The report explicitly states that LQR minimizes the configured Q/R objective, not every measured metric, and records stress steady-state error rather than treating it as a bug to hide.

- [ ] **Step 4: Run experiment test to verify it passes**

Run: `conda run -n robot-control python -m unittest lesson08_lqr_optimal_control.tests.test_experiments -v`

Expected: temporary output contains all four figures, all case CSVs, summary table, and report.

- [ ] **Step 5: Run real lesson experiment**

Run: `conda run -n robot-control python -m lesson08_lqr_optimal_control.src.run_experiments`

Expected: terminal prints Q/R cases, PP comparison, payload/stress metrics, and every artifact path.

- [ ] **Step 6: Commit**

```powershell
git add lesson08_lqr_optimal_control
git commit -m "feat: add lesson08 lqr experiments"
```

### Task 6: Complete engineering acceptance and usage guide

**Files:**
- Modify: `lesson08_lqr_optimal_control/README.md`

**Interfaces:**
- Consumes: generated Lesson 08 report and all Lesson 05-08 test suites.
- Produces: documented commands, model provenance, Q/R interpretation, and LQR limitations.

- [ ] **Step 1: Write README usage and interpretation**

Document commands for the focused test suites, whole Lesson 08 suite, and A-D runner. State the source parameters `J_hat=0.019762`, `b_hat=0.080257`; explain the distinction between `torque_unsat_nm` and `torque_cmd_nm`; and caution that constant load plus saturation demonstrates missing integral action and lack of hard-constraint optimization.

- [ ] **Step 2: Run the full regression suites**

Run:

```powershell
conda run -n robot-control python -m unittest discover lesson05_system_identification/tests -v
conda run -n robot-control python -m unittest discover lesson06_identification_quality/tests -v
conda run -n robot-control python -m unittest discover lesson07_state_feedback_controllability/tests -v
conda run -n robot-control python -m unittest discover lesson08_lqr_optimal_control/tests -v
```

Expected: all pre-existing and Lesson 08 tests pass without failures.

- [ ] **Step 3: Commit acceptance guide**

```powershell
git add lesson08_lqr_optimal_control/README.md
git commit -m "docs: complete lesson08 lqr guide"
```
