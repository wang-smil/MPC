# Lesson 12: Linear MPC Design

## Purpose

Teach and implement a first linear MPC for the identified one-axis robot joint. The learner should be able to explain how the discrete model predicts a future state trajectory, why the prediction can be written as `X = F x0 + G U`, how a finite-horizon quadratic program chooses inputs under constraints, and why only the first optimized input is applied before replanning from new feedback.

The implementation is intentionally incremental. The first implementation slice is Experiment A only: validate the prediction matrices against a step-by-step rollout before introducing an optimizer.

## Learning Context and Reuse

- Reuse the identified joint parameters/model convention from Lessons 05–11.
- Reuse Lesson 08's continuous state-space model and ZOH discretization functions rather than copying their equations.
- The MPC prediction sample period is `10 ms`; the simulated plant and estimator may continue to run at `1 ms`. Construct `Ad` and `Bd` for the MPC period, not the plant period.
- The state given to MPC in closed loop is the Kalman estimate `x_hat = [q_hat, dq_hat]`, never simulator truth.
- Keep the first optimizer in non-condensed form: future states `x[:, 0:N+1]` and controls `u[:, 0:N]` are decision variables linked by dynamics equalities. The `F/G` representation remains a validated prediction tool and mathematical explanation.

## First Implementation Slice: Experiment A

### Model and configuration

Create `lesson12_linear_mpc/config/mpc.yaml` with the identified joint inertia and damping, plant/estimator sample periods (`1 ms`), MPC sample period (`10 ms`), a prediction horizon for the validation experiment, and deterministic random seed.

Create `src/model_loader.py` as a thin adapter that imports Lesson 08's `build_continuous_model` and `discretize_zoh`, then returns `Ad`, `Bd` discretized at `mpc.dt_s`. Keep the numerical model derivation in Lesson 08.

### Prediction API

Create `src/prediction.py` with these responsibilities:

- `build_prediction_matrices(Ad, Bd, horizon) -> (F, G)`.
- `predict_states_matrix(Ad, Bd, x0, U) -> X`, using `X = F @ x0 + G @ U`.
- `rollout_states(Ad, Bd, x0, U) -> states`, using the recurrence `x[k+1] = Ad @ x[k] + Bd @ u[k]`.

Stacking convention:

- `X = [x1, x2, ..., xN]` in time-major block order; `x0` is not included.
- `U = [u0, u1, ..., u(N-1)]` in time-major block order.
- `F = [Ad; Ad^2; ...; Ad^N]`.
- Block `(i,j)` of `G`, for one-based predicted-state row `i` and zero-based input column `j`, is `Ad^(i-1-j) @ Bd` when `j < i`, and zero otherwise.
- For `nx` states and `nu` inputs, `F` has shape `(N*nx, nx)` and `G` has shape `(N*nx, N*nu)`.
- Public helpers validate compatible finite matrix/vector dimensions and positive integer horizon; use explicit, documented input shapes so C-order flattening agrees with the stated time-major convention.

### Experiment and acceptance

Create `tests/test_prediction_matrix.py` to compare matrix prediction and recursive rollout using the same real discrete joint model, deterministic initial state, and deterministic candidate input sequence. Require all predicted future states to match with absolute tolerance `1e-10`. Include focused shape/invalid-horizon checks.

Create `src/run_prediction_validation.py` to run the validation on the configured identified model and save `figures/prediction_validation.png`, overlaying the rollout and matrix-predicted position and velocity. Report the maximum absolute numerical mismatch in the console.

The slice is complete only when:

1. `F` and `G` have the documented dimensions and block structure.
2. `X_matrix` and `X_rollout[1:]` agree to `atol=1e-10`.
3. The validation figure and readable console result are generated.
4. The focused Lesson 12 tests pass.

No CVXPY/OSQP optimizer or closed-loop controller is part of this first slice.

## Whole-Lesson Direction (Later Incremental Slices)

The remaining work follows the supplied course material and is not part of the first implementation slice:

- Experiment B: compare unconstrained finite-horizon MPC against matching LQR behavior, using the DARE terminal penalty and several horizons.
- Experiment C: compare LQR plus actuator clipping with constrained MPC under the same torque bound.
- Experiment D: add velocity constraints and inspect torque, speed, and position to determine whether MPC brakes before the limit is reached.
- Experiment E: vary prediction horizon and compare tracking, settling, torque, velocity margin, and solve-time mean/p95/maximum and deadline misses.

Later MPC code should retain the optimizer status and solve time, apply only `u0*`, and replan from the next state estimate. It should preserve a final actuator safety clip as a guard and have an explicit fallback policy for infeasible or timed-out solves. Solver failure must not be silently ignored.

## First Optimizer Architecture (for Later Slices)

Keep the core modules small and focused:

- `prediction.py`: state prediction and matrix validation.
- `mpc_controller.py`: finite-horizon objective, state/input decision variables, constraints, solver status, and first-input result.
- `closed_loop.py`: simulated plant/sensor boundary and receding-horizon feedback.

Additional modules (`metrics.py`, `run_experiments.py`) and report/figure outputs can be added as their experiments are reached. Do not front-load nonlinear MPC, SQP, soft constraints, slack variables, recursive feasibility, terminal invariant sets, or direct sparse OSQP work into the first optimizer lesson.

## Safety and Interpretation

- Teaching constraint values are not hardware specifications. Real limits must be obtained from motor, gearbox, drive, mechanical-stop, and thermal capabilities.
- State constraints need to cover the intended prediction states, including the terminal predicted state when the requirement is “every predicted step.”
- An infeasible QP can reflect physically incompatible current state, actuator capability, and constraints. It is not automatically a software defect.
- Linear separation/prediction results do not prove safe behavior under actuator saturation, sensor faults, model mismatch, or hardware timing overruns.
