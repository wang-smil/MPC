# Lesson 12 Experiment A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and validate the Lesson 12 finite-horizon prediction matrices against a step-by-step rollout using the identified joint model at the MPC sample period.

**Architecture:** A thin model loader reuses Lesson 08's identified continuous model and ZOH discretization at `mpc.dt_s`. A prediction module builds `F` and `G` and exposes both matrix and recursive prediction with explicit time-major shapes. A small runner compares the two trajectories and saves the first validation figure; it does not create an optimizer.

**Tech Stack:** Python 3.11, NumPy, SciPy, Matplotlib, PyYAML, unittest.

**Spec:** `docs/superpowers/specs/2026-09-29-lesson12-linear-mpc-design.md`

## Global Constraints

- Reuse the identified joint parameters/model convention from Lessons 05–11.
- Reuse Lesson 08's continuous state-space model and ZOH discretization functions rather than copying their equations.
- The MPC prediction sample period is `10 ms`; the simulated plant and estimator may continue to run at `1 ms`. Construct `Ad` and `Bd` for the MPC period, not the plant period.
- In later closed-loop slices, use the Kalman estimate `x_hat = [q_hat, dq_hat]` as the MPC initial state, never simulator truth.
- Keep the first optimizer in non-condensed form: future states `x[:, 0:N+1]` and controls `u[:, 0:N]` are decision variables linked by dynamics equalities.
- `X = [x1, x2, ..., xN]` in time-major block order; `x0` is not included.
- `U = [u0, u1, ..., u(N-1)]` in time-major block order.
- `F = [Ad; Ad^2; ...; Ad^N]`.
- Block `(i,j)` of `G`, for one-based predicted-state row `i` and zero-based input column `j`, is `Ad^(i-1-j) @ Bd` when `j < i`, and zero otherwise.
- For `nx` states and `nu` inputs, `F` has shape `(N*nx, nx)` and `G` has shape `(N*nx, N*nu)`.
- The prediction equality must match recursive rollout with absolute tolerance `1e-10`.
- No CVXPY/OSQP optimizer or closed-loop controller is part of this first slice.
- Later state constraints must include the terminal predicted state when the requirement applies to every predicted step; an infeasible QP is a valid physical outcome that requires explicit status handling and a defined fallback.

## Review Focus

- MPC model discretization accidentally uses `plant.dt_s` instead of `mpc.dt_s`; pin with a test that `Ad` and `Bd` match the Lesson 08 ZOH model at `0.01 s`.
- State stack accidentally includes `x0` or omits `xN`; pin output shapes and compare matrix output against rollout states `[1:]`.
- MIMO input sequence is flattened in a different order from the `G` blocks; pin a two-input test with a distinct input at each horizon step.
- A zero, negative, fractional, or boolean horizon is accepted; pin clear `ValueError` behavior.
- Malformed or non-finite model, initial state, or input sequence silently propagates; pin incompatible shapes and NaN/Inf rejection.

---

### Task 1: Identified MPC model and prediction API

**Files:**
- Create: `lesson12_linear_mpc/__init__.py`
- Create: `lesson12_linear_mpc/src/__init__.py`
- Create: `lesson12_linear_mpc/config/mpc.yaml`
- Create: `lesson12_linear_mpc/src/model_loader.py`
- Create: `lesson12_linear_mpc/src/prediction.py`
- Create: `lesson12_linear_mpc/tests/__init__.py`
- Create: `lesson12_linear_mpc/tests/test_prediction_matrix.py`

**Interfaces:**
- Consumes: `lesson08_lqr_optimal_control.src.model.build_continuous_model(inertia, damping)` and `discretize_zoh(A, B, dt_s)`.
- Produces: `load_config(path: Path | str) -> dict` and `build_mpc_model(config: dict) -> dict[str, np.ndarray]` returning `A`, `B`, `Ad`, `Bd`.
- Produces: `build_prediction_matrices(Ad, Bd, horizon) -> tuple[np.ndarray, np.ndarray]`.
- Produces: `predict_states_matrix(Ad, Bd, x0, U) -> np.ndarray` with `U.shape == (N, nu)` and returned predicted `X.shape == (N, nx)`.
- Produces: `rollout_states(Ad, Bd, x0, U) -> np.ndarray` with returned states shape `(N+1, nx)`, including `x0` at row zero.

- [ ] **Step 1: Write failing tests for identified MPC discretization and prediction equality**

Add tests that load `config/mpc.yaml`, verify `Ad`/`Bd` against Lesson 08 ZOH discretization at `mpc.dt_s`, then use seeded `x0` and a seeded candidate sequence to compare `predict_states_matrix(...)` with `rollout_states(...)[1:]` at `atol=1e-10`. Assert `F.shape == (N*nx, nx)`, `G.shape == (N*nx, N*nu)`, and prediction output shapes. Include a small two-state/two-input example whose distinct input values prove time-major flattening and lower-triangular block placement. Test invalid horizon, matrix/state/input shapes, and non-finite values.

- [ ] **Step 2: Run tests to verify RED**

Run: `conda run -n robot-control python -m unittest lesson12_linear_mpc.tests.test_prediction_matrix -v`
Expected: import failure because the Lesson 12 prediction module does not exist.

- [ ] **Step 3: Implement configuration/model adapters**

Create `mpc.yaml` with identified inertia `0.019762 kg·m²`, damping `0.080257 N·m·s/rad`, plant and estimator periods `0.001 s`, MPC period `0.01 s`, validation horizon `20`, and deterministic seed `42`. `build_mpc_model` imports the Lesson 08 continuous model and ZOH discretizer and uses `config["mpc"]["dt_s"]`.

- [ ] **Step 4: Implement prediction matrix construction and independent rollout**

Build the lower block-triangular `G` from the documented formula. Require `U` as a finite two-dimensional time-major array `(N, nu)`; stack it in C order for the matrix product and reshape predicted output back to `(N, nx)`. Keep `rollout_states` as a separate recurrence so it independently checks the matrix implementation. Validate finite compatible `Ad`, `Bd`, `x0`, and `U` values and a positive integer, non-boolean horizon.

- [ ] **Step 5: Run Task 1 tests to verify GREEN**

Run: `conda run -n robot-control python -m unittest lesson12_linear_mpc.tests.test_prediction_matrix -v`
Expected: all prediction, shape, discretization, and validation tests pass; rollout and matrix predictions agree within `1e-10`.

- [ ] **Step 6: Commit prediction model**

```bash
git add lesson12_linear_mpc/config/mpc.yaml lesson12_linear_mpc/src/__init__.py lesson12_linear_mpc/src/model_loader.py lesson12_linear_mpc/src/prediction.py lesson12_linear_mpc/tests/__init__.py lesson12_linear_mpc/tests/test_prediction_matrix.py lesson12_linear_mpc/__init__.py
git commit -m "feat: add lesson12 mpc prediction model"
```

### Task 2: Experiment A runner and prediction figure

**Files:**
- Create: `lesson12_linear_mpc/src/run_prediction_validation.py`
- Create: `lesson12_linear_mpc/tests/test_prediction_validation.py`
- Create at runtime: `lesson12_linear_mpc/figures/prediction_validation.png`

**Interfaces:**
- Consumes: Task 1's `load_config`, `build_mpc_model`, `predict_states_matrix`, and `rollout_states`.
- Produces: `run_validation(config_path: Path | str, output_dir: Path | str | None = None) -> dict` containing `max_abs_error`, `X_matrix`, `X_rollout`, and `figure_path`.

- [ ] **Step 1: Write a failing runner artifact test**

Use a temporary output directory and the real Lesson 12 configuration. Assert the runner returns `max_abs_error <= 1e-10`, creates a non-empty `prediction_validation.png`, and returns matrix/rollout states with the configured horizon and state dimensions.

- [ ] **Step 2: Run the artifact test to verify RED**

Run: `conda run -n robot-control python -m unittest lesson12_linear_mpc.tests.test_prediction_validation -v`
Expected: import failure because the validation runner does not exist.

- [ ] **Step 3: Implement deterministic validation runner**

Use the configured seed to build the initial state and candidate torque sequence, call both prediction paths, calculate the maximum absolute difference, and plot rollout/matrix position and velocity over the prediction horizon. Save the figure in the requested output directory or the lesson's `figures/` directory, print a readable maximum-error result, and return the documented result mapping.

- [ ] **Step 4: Run all Lesson 12 tests and the formal validation command**

Run: `conda run -n robot-control python -m unittest discover lesson12_linear_mpc/tests -v`
Expected: all tests PASS.

Run: `conda run -n robot-control python -m lesson12_linear_mpc.src.run_prediction_validation`
Expected: a maximum absolute error no larger than `1e-10` and a generated `figures/prediction_validation.png`.

- [ ] **Step 5: Commit Experiment A runner and figure**

```bash
git add lesson12_linear_mpc/src/run_prediction_validation.py lesson12_linear_mpc/tests/test_prediction_validation.py lesson12_linear_mpc/figures/prediction_validation.png
git commit -m "feat: validate lesson12 mpc predictions"
```

### Task 3: Final Experiment A handoff

**Files:**
- Create: `lesson12_linear_mpc/README.md`

**Interfaces:**
- Consumes: completed Experiment A modules and output paths from Tasks 1–2.
- Produces: concise run instructions and an explanation of the `x0`, `U`, `X`, `F`, and `G` stacking convention.

- [ ] **Step 1: Write README from verified interfaces and generated result**

Document the repository-root command `conda run -n robot-control python -m lesson12_linear_mpc.src.run_prediction_validation`, explain that MPC prediction uses `10 ms` while plant/estimator rates are `1 ms`, and point out that this experiment validates prediction only; it does not yet solve a QP.

- [ ] **Step 2: Run final focused suite and inspect generated files**

Run: `conda run -n robot-control python -m unittest discover lesson12_linear_mpc/tests -v`
Expected: all tests PASS; config, source, tests, README, and prediction figure are present.

- [ ] **Step 3: Commit the Experiment A handoff**

```bash
git add lesson12_linear_mpc/README.md
git commit -m "docs: explain lesson12 prediction validation"
```
