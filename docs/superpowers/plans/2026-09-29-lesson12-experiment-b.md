# Lesson 12 Experiment B — Unconstrained MPC vs LQR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Keep the user-facing workflow incremental: complete and verify one task before moving to the next. Do not skip tests or weaken the stated acceptance checks.

**Goal:** Build Lesson 12 Experiment B as a reproducible, multi-rate simulation comparing unconstrained receding-horizon MPC against LQR, and demonstrate numerically that a DARE terminal cost makes their first control actions equivalent.

**Architecture:** Keep the MPC optimizer, multi-rate closed-loop simulation, experiment runner, metrics, and report separate. The controller solves the full non-condensed CVXPY problem and returns only its first action for execution. The plant and Kalman estimator run every 1 ms; the LQR/MPC controller runs every 10 ms and holds its selected torque between updates. Both methods receive identical sensor-noise samples and use the same identified/discrete plant, reference, Q/R, and DARE terminal matrix.

**Tech Stack:** Python, NumPy, CVXPY, OSQP, Matplotlib, PyYAML, `unittest`; reuse Lesson 08 LQR design and Lesson 10 Kalman estimator/model utilities.

**Design spec:** `docs/superpowers/specs/2026-09-29-lesson12-experiment-b-design.md`

## Global Constraints

- Use the Lesson 12 10 ms discrete control model and the same Q/R weights for LQR and MPC. Set the MPC terminal weight to the DARE solution (P) for that same model and Q/R.
- Run the simulated plant and Kalman estimator at 1 ms, while the controller updates every 10 ms; hold the last applied command between controller ticks.
- The feedback controller may consume only the estimated state \hat{x}; do not pass or read the true plant state inside the controller.
- Experiment B is unconstrained: do not add torque clipping, state constraints, or load disturbances. Mark clearly that this is a simulation experiment and does not establish hardware safety or performance.
- Give LQR and each MPC horizon the exact same pre-generated measurement-noise sequence, initial conditions, plant, reference, estimator, and timing.
- Use the DARE terminal matrix, record QP status and solver timing, execute only the first MPC input, and fail the scenario explicitly if the solver does not return an optimal solution. Never silently reuse a stale command.
- Preserve the existing Lesson 12 Experiment A behavior and unrelated workspace changes. Work directly in the current checkout (no worktree); do not push to GitHub.
- Add CVXPY and OSQP to the repository requirements and install them into the already-selected `robot-control` environment only as a normal implementation step after the approved plan is accepted. Verify installed versions and that OSQP solves the QP.

## Review Focus

1. **Invalid horizon or malformed/non-finite model and weight arrays** could make CVXPY fail obscurely; Task 1 tests validation and clear exceptions.
2. **Solver exception or non-optimal status** could accidentally expose an old command; Task 1 tests that `u0` is absent on failure, and Task 2 tests fail-fast simulation behavior.
3. **Off-by-one timing or a wrong initial Kalman update** could shift response and obscure the comparison; Task 2 pins controller update indices, held-command intervals, and initial measurement handling.
4. **Different stochastic inputs between methods** would invalidate the comparison; Task 2 verifies identical measurement sequences are consumed by LQR and MPC runs.
5. **Leaking plant truth or applying the full planned input sequence** would no longer represent output-feedback receding-horizon control; Task 2/3 verify the controller receives only \hat{x} and logs the executed first input, not future planned inputs.

## Task 1: Add the non-condensed CVXPY/OSQP MPC controller

**Files:**
- Modify: `requirements.txt`
- Create: `lesson12_linear_mpc/src/mpc_controller.py`
- Create: `lesson12_linear_mpc/tests/test_mpc_controller.py`

1. **Write failing tests first.** Cover:
   - A scalar/vector state and single-input model builds the expected non-condensed decision variables `X` with shape `(nx, N+1)` and `U` with shape `(nu, N)`.
   - The initial-state constraint, per-step dynamics, stage cost, and terminal cost produce a solved problem for horizons 1, 5, and 20.
   - Zero/negative/non-integer horizon, incompatible dimensions, non-symmetric or non-PSD Q/R/P, and non-finite inputs are rejected with clear `ValueError`s.
   - For several states and reference values, the first MPC action matches the 10 ms DARE LQR action within `1e-4 Nm` for N = 1, 5, and 20.
   - Solver status, objective, solve time, iteration count (when supplied), planned state/input trajectories, and the first action are returned; a non-optimal status or mocked solver exception returns no action and reports failure rather than reusing a previous one.
2. **Run the tests and confirm the expected red state.** The new controller import or implementation should be missing before implementation; record the failure as the TDD baseline.
3. **Add dependencies.** Add `cvxpy` and `osqp` to root `requirements.txt`. Install into `robot-control` using `conda run -n robot-control python -m pip install cvxpy osqp`, then verify imports, versions, and availability of the OSQP solver.
4. **Implement `LinearMPCController`.** Use CVXPY variables `X ∈ R^(nx×(N+1))`, `U ∈ R^(nu×N)`, parameterized `x0` and `x_ref`, constraints `X[:,0] == x0` and `X[:,k+1] == Ad @ X[:,k] + Bd @ U[:,k]`, stage cost Σ((X[:,k]-x_ref)'Q(X[:,k]-x_ref) + U[:,k]'R U[:,k]), and terminal cost `(X[:,N]-x_ref)'P_terminal(X[:,N]-x_ref)`. Solve with OSQP and warm start enabled. Return the complete planned trajectories and solver diagnostics but designate only `U[:,0]` as executable. On exception/non-optimal status, return no executable action.
5. **Run the focused tests and all existing Lesson 12 tests.** Confirm the MPC/LQR first-action equality across the requested horizons. Commit as `feat: add unconstrained lesson12 mpc controller`.

## Task 2: Implement and test the 1 ms plant / 10 ms output-feedback loop

**Files:**
- Modify: `lesson12_linear_mpc/config/mpc.yaml`
- Create: `lesson12_linear_mpc/src/closed_loop.py`
- Create: `lesson12_linear_mpc/tests/test_closed_loop.py`

1. **Write failing tests first.** Verify:
   - Plant and estimator update at the 1 ms grid; controller events occur at indices 0, 10, 20, … with no extra update at the 2 s endpoint.
   - The first sensor measurement at t=0 updates the initial estimate without first predicting one sample forward; subsequent estimator prediction uses the previous applied torque.
   - A controller output is held for exactly ten plant steps; the trace records requested/applied controller values and each event's QP status.
   - MPC calls receive the estimated state only; only the first input of each solved horizon is applied.
   - LQR and MPC consume identical supplied sensor noise and fail immediately with a clear error on solver failure.
2. **Run tests and confirm expected failures.**
3. **Extend `config/mpc.yaml`.** Add a named Experiment B section for 2.0 s duration, 1 ms plant/KF step, 10 ms controller step, 30-degree reference, horizons `[1, 5, 20]`, zero initial true state, sensor noise and estimator covariance settings matching the established Lesson 10/Lesson 11 configuration. Keep Experiment A defaults and behavior unchanged.
4. **Implement `simulate_closed_loop(...)`.** Reuse the existing Lesson 12 plant/discretization, Lesson 10 estimator/Kalman update, and Lesson 08 LQR routines. Construct matching 10 ms `Ad/Bd`, Q/R, DARE `P`, and LQR `K`. At each 1 ms tick, obtain the noisy position measurement, update the estimator (initial measurement update at t=0; later predict with previous applied torque and then update), and at each 10 ms control event compute either `u=-K(x_hat-x_ref)` or the MPC solution's first input. Hold that command until the next control event. Keep true state internal to plant integration and post-run metric/log generation; it must not be an argument to either controller. Provide a fail-fast error if MPC has no optimal solution.
5. **Run focused and existing Lesson 12 tests.** Confirm the exact event schedule, sample hold, shared-noise invariant, and successful repeatable simulation. Commit as `feat: add lesson12 multirate lqr mpc loop`.

## Task 3: Add the experiment runner, comparison outputs, and teaching report

**Files:**
- Create: `lesson12_linear_mpc/src/run_experiment_b.py`
- Create: `lesson12_linear_mpc/tests/test_experiment_b.py`
- Create: `lesson12_linear_mpc/reports/unconstrained_mpc_vs_lqr.md`
- Update: `lesson12_linear_mpc/README.md`
- Generate: `lesson12_linear_mpc/logs/experiment_b.csv`
- Generate: `lesson12_linear_mpc/figures/lqr_vs_mpc.png`

1. **Write failing runner/integration tests first.** Run the Experiment B entry point using a temporary output directory and verify it produces a non-empty CSV, figure, and report; required fields include true position/velocity, estimated state, reference, applied torque, controller mode, horizon, solver status, objective, solve time, and iteration count. Assert all cases solve, the first-action mismatch is within `1e-4 Nm`, and the report states unconstrained simulation-only limitations.
2. **Run the integration test and confirm expected failures.**
3. **Implement `run_experiment_b.py`.** Pre-generate one measurement-noise realization from the configured seed and pass the same array into the LQR and all MPC horizon runs. Run LQR and horizons 1/5/20 from the same initial condition. Save aligned CSV rows and a readable comparison figure with position tracking/reference, estimated velocity, and applied torque. Summarize tracking RMSE, velocity RMSE, maximum absolute LQR/MPC torque difference, solver status counts, and mean/p95/max solve time by horizon.
4. **Write the report and README instructions.** Explain the state and cost equations, non-condensed QP variables and constraints, why `X=Fx0+GU` is not required to construct the non-condensed form, why DARE terminal cost makes each horizon's first action agree with LQR in this unconstrained setup, why only `u0` is executed, the 1 ms/10 ms timing, and that real latency/safety/actuator behavior are not established. Include the actual generated metrics rather than guessed values.
5. **Run all Lesson 12 tests and execute the full experiment.** Check CSV/figure/report readability, all QP statuses, equivalence tolerance, and deterministic repeatability for a fixed seed. Commit as `feat: report lesson12 unconstrained mpc comparison`.

## Final Verification

From the repository root, run:

```powershell
conda run -n robot-control python -m unittest discover -s lesson12_linear_mpc/tests -v
conda run -n robot-control python -m lesson12_linear_mpc.src.run_experiment_b
conda run -n robot-control python -c "import cvxpy, osqp; print('cvxpy', cvxpy.__version__); print('osqp', osqp.__version__); print('solvers', cvxpy.installed_solvers())"
git status --short
```

Confirm all tests pass, every solve reports an optimal status, the horizon-to-LQR first-control mismatch is at most `1e-4 Nm`, the report and figure match the saved CSV, and unrelated existing files remain untouched. Do not push to GitHub.

## References

- CVXPY solver status/statistics and warm start: https://www.cvxpy.org/tutorial/solvers/index.html
- CVXPY installation: https://www.cvxpy.org/install/
- OSQP Python interface and solver diagnostics: https://osqp.org/docs/interfaces/python.html
- OSQP solver settings: https://osqp.org/docs/interfaces/solver_settings.html
