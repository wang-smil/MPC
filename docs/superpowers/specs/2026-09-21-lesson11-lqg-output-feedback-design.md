# Lesson 11 LQG Output Feedback Design

## Goal

Build a modular LQG output-feedback lesson for the identified one-axis joint.  The lesson combines the existing Lesson 08 balanced LQR controller and Lesson 10 Kalman estimator without duplicating either algorithm.  It must verify the discrete separation principle, quantify the gap between ideal full-state feedback and implementable LQG, compare recursive and steady-state estimation, demonstrate estimator/controller performance coupling, and show the limit of linear theory under torque saturation.

## Reuse Boundaries

`lesson11_lqg_output_feedback` imports the identified 1 ms ZOH joint model and noise covariance construction from Lesson 10, the balanced LQR construction from Lesson 08, `DiscreteKalmanFilter` from Lesson 10, and `design_steady_state_kf` from Lesson 10.  It must not copy the state-space equations, Riccati solve, Kalman recursion, or Joseph covariance update.

The only new real-time object is `LQGController`, a composition layer rather than a replacement for the reused modules.

## LQG Controller Contract

```python
LQGController(estimator, K_controller, torque_limit)
step(measurement, previous_applied_torque, x_ref, torque_ff=0.0) -> dict
```

`step` calls the supplied estimator with scalar encoder position and scalar torque actually applied during the previous sample interval.  It computes

\[
u_{\rm request}=u_{\rm ff}-K_c(\hat x-x_{\rm ref}),
\]

clips to the actuator limit, and returns state estimate, requested torque, applied torque, saturation flag, and all estimator diagnostics.  It has no access to the plant true state.

The causal sampled-data order is:

\[
(\hat x_{k-1|k-1},P_{k-1|k-1},u_{\rm applied,k-1})
\rightarrow \text{KF predict}
\rightarrow y_k
\rightarrow \text{KF update}
\rightarrow \hat x_{k|k}
\rightarrow \text{LQR and clip}
\rightarrow u_{\rm applied,k}.
\]

`full_state_lqr` exists only as an explicitly named simulation reference baseline.  It uses `x_true` to calculate control and is never represented as deployable hardware code.

## Separation Principle Analysis

The steady-state estimator uses the Lesson 10 `control.dlqe` predictor-form gain \(L_e\).  With estimation error \(e=x-\hat x\), no saturation, and matched linear model, define:

\[
A_c=A_d-B_dK_c,
\qquad
A_e=A_d-L_eC,
\]

\[
A_{\rm aug}=
\begin{bmatrix}
A_c&B_dK_c\\
0&A_e
\end{bmatrix}.
\]

The analysis must check to numerical tolerance:

\[
\lambda(A_{\rm aug})=\lambda(A_c)\cup\lambda(A_e),
\]

and separately report whether every discrete eigenvalue has magnitude below one.  This is a linear, unsaturated-theory check only; it must not claim global nonlinear closed-loop performance.

## Modules

- `config/lqg.yaml`: sample period, shared model/noise/controller parameters, reference, initial state, load step, and reduced torque limit for the boundary experiment.
- `src/model_loader.py`: thin imports/adapters that obtain the shared 1 ms model, LQR gain, covariance matrices, and steady-state gain.
- `src/lqg_controller.py`: composition-only controller and estimator adapters for recursive KF and steady-state predictor gain.
- `src/separation_analysis.py`: matrices, poles, eigenvalue-union residual, and stability verdict.
- `src/closed_loop.py`: the exclusive owner of plant truth, encoder-noise realization, actuator behavior, and fair controller simulations.
- `src/metrics.py`: common tracking, estimate, torque, saturation, innovation, and NIS statistics, including startup and steady-state windows.
- `src/run_experiments.py`: artifact generation for experiments A–E.

## Experiments

### A: Separation principle

Use balanced \(K_c\), `dlqe` \(L_e\), and the nominal unsaturated identified model.  Save the three pole sets and their numeric union error.

### B: Full-state LQR versus LQG

Run an ideal `full_state_lqr` and recursive-KF LQG with the same true plant, target, initial state, deterministic encoder/process-noise realization, torque limit, and load step.  Compare tracking RMSE, RMS/peak torque, and LQG position/velocity estimation RMSE.  The full-state baseline may not report estimator metrics.

### C: Recursive KF versus steady-state LQG

Both controllers use identical model, \(K_c\), reference, plant, noise, and actuator.  Recursive LQG uses the Lesson 10 recursive posterior KF.  Steady-state LQG uses a fixed predictor-form gain with a documented state-time convention.  Report estimation RMSE separately over startup \([0,0.5]\) seconds and steady state \((0.5,T]\), plus tracking and control RMS.

### D: Fixed controller, varying estimator Q

Keep \(K_c\) unchanged and use an unknown load-torque step at 2 seconds.  Run assumed `Q_process` scales 0.1, 1, and 10.  Compare estimation error, tracking, control, innovation, and NIS.  The report must distinguish independent design from coupled runtime performance.

### E: Saturation boundary

Run Normal, unknown-load, and reduced-torque-limit conditions.  Compare saturation ratio, tracking, estimate errors, innovation, and NIS.  Explain that clipping is nonlinear, so the separation-principle pole calculation does not fully describe this operating regime.

## Artifacts and Tests

Generate named CSV logs, `separation_poles.png`, `full_state_vs_lqg.png`, `recursive_vs_steady_state.png`, `q_tuning_coupling.png`, `saturation_boundary.png`, and `reports/lqg_engineering_report.md`.

Tests cover: exact eigenvalue-union equality and unit-circle check; expected matrix dimensions and invalid input rejection; a causal timing spy proving `LQGController` passes previous **applied** torque into its estimator; output clipping; and runner artifact creation using a shortened test-only simulation duration.

## Hardware Transition

For an actual servo, replace only the `closed_loop.py` simulated encoder/plant boundary with hardware I/O.  Keep `LQGController.step` unchanged; pass the encoder position and logged prior applied torque/current-derived torque.  Begin with nominal actuator limits and static/dynamic R/Q calibration.  Treat saturation, long-lived innovation bias, and abnormal NIS as diagnostics that trigger safety handling or model/sensor investigation rather than as proof that the separation principle failed.
