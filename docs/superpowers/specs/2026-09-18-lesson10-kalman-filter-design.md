# Lesson 10 Kalman Filter Design

## Goal

Build a reproducible Kalman-filter lesson for the identified single-axis joint.  The lesson must estimate position and velocity from a noisy position encoder and the previously applied motor torque, drive the joint with the existing Lesson 08 balanced LQR gain, and expose estimator confidence and health diagnostics.

## Reused Engineering Assets

The nominal model remains the Lesson 06 identified inertia and damping:

\[
\dot{x}=Ax+B\tau,\qquad x=[q,\dot q]^T.
\]

Lesson 08 constructs the 1 ms ZOH matrices \(A_d,B_d\) and balanced LQR gain.  Lesson 10 must import or reproduce that public construction with the same \(\hat J,\hat b\), rather than creating independent plant parameters.  Lesson 09 supplies the comparison observer and its causal control-loop conventions.

## Estimator Contract and Timing

The discrete model is

\[
x_{k+1}=A_dx_k+B_du_{\mathrm{applied},k}+G_dw_k,\qquad
y_k=Cx_k+v_k,
\]

where \(C=[1\;0]\), \(G_d=B_d\), \(w_k\) is an equivalent unknown torque disturbance, and \(v_k\) is encoder-position noise.  In the simulation and future hardware loop, the Kalman filter receives only scalar `measurement` and scalar `applied_input`; it must never accept the plant's true state or the unclipped requested torque.

The causal order at each sample is: predict from posterior state using the prior actual input; read encoder position; correct from the innovation; compute LQR torque from the posterior estimate; saturate it; apply/log that torque; propagate the plant.  The first step uses a zero prior input.

## Recursive Kalman Filter

The filter owns posterior \(\hat x_{k|k}\) and covariance \(P_{k|k}\).  It predicts

\[
\hat x^- = A_d\hat x+B_du,\qquad
P^- = A_dPA_d^T+G_dQG_d^T.
\]

It then computes \(\nu=y-C\hat x^-\), \(S=CP^-C^T+R\), gain \(K=P^-C^TS^{-1}\), posterior state, and NIS \(\nu^TS^{-1}\nu\).  Covariance update uses Joseph form

\[
P^+=(I-KC)P^-(I-KC)^T+KRK^T,
\]

followed by explicit symmetrization.  Covariance must remain symmetric and positive semidefinite within numerical tolerance.

## Files and Responsibilities

`lesson10_kalman_filter/` contains `config/kalman.yaml`, focused `src/` modules, unit tests, generated CSV logs, figures, an engineering report, and README instructions.  `model_loader.py` builds the reused ZOH model and LQR gain; `noise_model.py` converts physical standard deviations into covariance matrices; `kalman_filter.py` implements recursive filtering; `steady_state_kf.py` computes `control.dlqe` reference values; `estimators.py` adapts raw difference, low-pass difference, Lesson 09 Luenberger, and KF to a common `step` output; `closed_loop.py` is the only true-plant owner; `metrics.py` measures estimator and controller behavior; and `run_experiments.py` generates all artifacts.

## Experiments and Acceptance

All scenarios use the same identified nominal model, encoder noise generator, actuator torque limit, LQR design, sample time, and deterministic random seed unless the scenario explicitly changes one variable.

1. Baseline starts the true joint at 20 degrees and 0.5 rad/s while the filter starts at zero, includes nominal torque disturbance and encoder noise, and plots \(q,\hat q\), \(\dot q,\hat{\dot q}\), and plus/minus three standard-deviation bands.
2. Measurement-trust sweep fixes true sensor noise and runs assumed \(R\) at 0.1x, 1x, and 10x.
3. Model-trust sweep fixes sensor noise and runs assumed \(Q\) at 0.1x, 1x, and 10x while injecting an unknown load torque at 2 seconds.
4. Estimator comparison runs raw difference, filtered difference, Lesson 09 Luenberger, and KF under identical normal conditions.
5. Formal acceptance produces Normal, Disturbance, Stress (10x encoder noise), and 30-percent payload-mismatch cases.  It records innovation and NIS as model-health diagnostics.

The runner must generate baseline confidence, R/Q tuning, estimator comparison, and robustness figures, detailed logs, and a Markdown report with a quantitative table and cautions against assuming KF is universally best.

## Tests

Tests cover physical covariance construction, covariance symmetry/PSD over many recursive updates, expected initial-error convergence in a no-noise case, filter stability and finite NIS, validation of invalid dimensions/covariances, use of applied rather than requested torque, generated artifacts, and recursive gain convergence toward the steady-state reference with an explicitly documented timing convention.

## Real Servo Transition

Start with a static encoder recording to estimate a first measurement variance \(R\).  Initialize position from the first encoder sample and velocity conservatively.  Obtain a first \(Q\) from torque-disturbance scale and tune it against innovation, NIS, residuals, saturation, and dynamic validation.  Log timestamps, position, state estimate, covariance diagonal, gain, innovation, NIS, requested torque, and applied torque.  Long-lived innovation bias must trigger investigation of encoder offset, torque calibration, load or model errors before increasing filter bandwidth.
