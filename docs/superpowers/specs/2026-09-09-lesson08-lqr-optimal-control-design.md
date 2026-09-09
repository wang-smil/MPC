# Lesson 08: Discrete LQR Optimal Control Design

## Goal

Build `lesson08_lqr_optimal_control` as an independent, engineering-oriented discrete LQR project. The lesson must show how LQR weights determine a state-feedback gain, what objective LQR actually minimizes, and where LQR stops being sufficient when a real actuator saturates or a constant load disturbance is present.

## Fixed model source and units

The project reuses the validated single-joint parameters from Lesson 06/07 rather than adding a duplicate parameter artifact:

- `J_hat = 0.019762 kg m^2`
- `b_hat = 0.080257 N m s/rad`

All internal angles are radians, angular velocities are rad/s, and torques are N m. The nominal discrete model is created by ZOH at `dt_s = 0.001`:

\[
x=[q,\dot q]^T,\qquad
\dot x=Ax+Bu,
\]

\[
A=\begin{bmatrix}0&1\\0&-b/J\end{bmatrix},\qquad
B=\begin{bmatrix}0\\1/J\end{bmatrix}.
\]

## Project structure

```text
lesson08_lqr_optimal_control/
├─ config/lqr.yaml
├─ src/
│  ├─ model.py
│  ├─ lqr_design.py
│  ├─ controller.py
│  ├─ simulator.py
│  ├─ metrics.py
│  └─ run_experiments.py
├─ tests/
├─ figures/
├─ logs/
├─ reports/
└─ README.md
```

Lesson 08 owns its simulator and does not modify Lesson 07. It may reuse the validated model values and the pole-placement design specification, but must not reuse a controller gain designed at a different sampling period.

## LQR design and independent verification

The production controller design calls `control.dlqr(Ad, Bd, Q, R)`. A separate test-only implementation solves the DARE with `scipy.linalg.solve_discrete_are` and forms:

\[
K=(R+B_d^TPB_d)^{-1}B_d^TPA_d.
\]

The two gains must agree within numerical tolerance. Closed-loop poles are recorded as eigenvalues of `Ad - Bd @ K`.

Bryson-normalized initial weights are:

\[
Q=\operatorname{diag}\left(\frac{s_q}{(5^\circ)^2},\frac{s_{\dot q}}{1.5^2}\right),
\qquad
R=\frac{s_u}{3.0^2}.
\]

The four weight cases are Balanced `(1,1,1)`, Position Priority `(10,1,1)`, Effort Saving `(1,1,10)`, and Velocity Priority `(1,10,1)`.

## Closed-loop data flow

At every 1 ms sample:

1. The simulator forms noisy measured position and velocity from the hidden true state.
2. The controller calculates `error = x_measured - x_ref` and `torque_unsat = torque_ff - K @ error`.
3. It clips the actuator command to the active torque limit and records `torque_cmd` and `saturated` separately.
4. The true plant advances through semi-implicit Euler:

\[
\ddot q=(\tau_{cmd}-\tau_{load}-b\dot q)/J_{real},
\quad \dot q\mathrel{+}=\ddot qdt,
\quad q\mathrel{+}=\dot qdt.
\]

5. The stage costs are logged from the controller-requested input:

\[
\ell_x=e^TQe,\qquad
\ell_u=u_{unsat}^TRu_{unsat},\qquad
\ell=\ell_x+\ell_u.
\]

Each CSV contains time, reference/true states, errors, saturated and unsaturated torques, saturation state, and the three cost columns.

## Experiments

### A. Q/R trade-off

Run all four Bryson-scaled weight cases with identical nominal plant, noise, target, and 3 N m torque limit. Report `K`, closed-loop poles, settling time, overshoot, tracking RMSE, peak/RMS torque, saturation ratio, state cost, input cost, and total cost in one summary table.

### B. Pole placement versus LQR

Compare LQR with a pole-placement controller designed from the Lesson 07 nominal specification `(zeta=0.8, settling time=0.8 s)`, recomputed at the LQR 1 ms sampling period. Both controllers receive identical plant, target, sensor noise, actuator limit, and no additional disturbance. The report must not claim that LQR is universally superior; it is optimal only for its own Q/R cost.

### C. Payload mismatch

Keep the nominal LQR gain fixed. Set `J_real = 1.3 * J_hat`, without redesign. Compare the nominal-model closed-loop pole prediction with actual trajectory metrics.

### D. Constant load and saturation stress

At 1 second, add a 1 N m load torque and reduce the actuator limit to 1.5 N m. Record the steady-state tracking error, unsaturated torque demand, applied command, saturation time, and costs. Interpret the result as two LQR limitations: no integral action for a constant unknown load, and no direct hard-constraint handling.

## Required artifacts

The runner writes one CSV per closed-loop case plus figures for Q/R comparison, LQR-vs-pole-placement, payload mismatch, and load/saturation stress. It writes `reports/lqr_report.md` including all Q/R cases, controller gains, poles, metrics, costs, and caveats.

## Verification and boundaries

Tests cover model/weight construction, `dlqr`-DARE agreement, controller saturation behavior, aligned simulator logs, cost decomposition, and required runner artifacts. The final acceptance suite runs Lesson 05, Lesson 06, Lesson 07, and Lesson 08 tests.

This remains a single-joint linear teaching plant. It does not yet include gravity, Coulomb friction, gearbox flexibility, current-loop dynamics, delays, or integral augmentation.
