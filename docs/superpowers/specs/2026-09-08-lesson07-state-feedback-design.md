# Lesson 07 State Feedback and Controllability Design

## Purpose

Build a discrete-time full-state-feedback project that consumes the validated
Lesson 05 nominal estimates `J_hat = 0.019762 kg m^2` and
`b_hat = 0.080257 N m s/rad`. The project must make the chain from system
identification to controller performance explicit rather than using hidden
true parameters during controller design.

## Directory and module boundaries

```text
lesson07_state_feedback_controllability/
├─ config/controller.yaml
├─ src/
│  ├─ model.py              # continuous and ZOH-discrete J,b model
│  ├─ controllability.py    # C, rank, singular values, conditioning
│  ├─ pole_design.py        # settling-time/zeta poles and place_poles
│  ├─ state_feedback.py     # reference error, unsaturated/saturated torque
│  ├─ simulator.py          # noisy measurement, causal velocity estimate, plant
│  ├─ metrics.py            # transient, torque, saturation, tracking metrics
│  └─ run_experiments.py    # A-D orchestration, logs, figures, report
├─ figures/
├─ logs/
├─ reports/state_feedback_report.md
├─ tests/test_controllability.py
├─ tests/test_pole_placement.py
└─ README.md
```

Lesson 07 imports only the identified values from Lesson 05; it does not
rewrite the Lesson 05 plant or estimator.

## Model and controller

The design model is

\[
\dot x = Ax+Bu,\quad x=[q,\dot q]^T,
\]

\[
A=\begin{bmatrix}0&1\\0&-\hat b/\hat J\end{bmatrix},\qquad
B=\begin{bmatrix}0\\1/\hat J\end{bmatrix}.
\]

`model.py` discretizes it with zero-order hold at `dt_s` to obtain `Ad,Bd`.
`pole_design.py` maps continuous target poles from damping ratio and settling
time into the discrete plane, then checks the achieved poles of `Ad - Bd @ K`.

For a constant position reference, the controller is

\[
u_{\rm unsat}=-K(\hat x-x_{\rm ref}),\qquad
u_{\rm applied}=\operatorname{clip}(u_{\rm unsat},-u_{\max},u_{\max}).
\]

The measured position has Gaussian encoder noise. A causal finite difference
and first-order low-pass filter provide the velocity estimate used by the
controller; the report labels this as an estimate, not a true state.

## Configured safeguards and logging

`controller.yaml` includes design parameters, physical plant parameters,
target angle, torque limit, sensor-noise level, velocity-filter coefficient,
and payload inertia scale. Every simulation logs:

`time_s`, `q_ref_rad`, `q_rad`, `dq_rad_s`, `state_error_q`,
`state_error_dq`, `torque_unsat_nm`, `torque_applied_nm`, `saturated`, target
poles, and achieved closed-loop poles.

Metrics include controllability rank/condition/smallest singular value,
position and velocity feedback gains, settling time, overshoot, tracking RMSE,
peak torque, RMS torque, and saturation ratio.

## Experiments

### A — Nominal pole placement

Use `zeta=0.8`, `Ts=0.8 s`; check rank(C)=2, requested versus achieved poles,
and closed-loop tracking under torque saturation and encoder noise.

### B — Speed/effort trade-off

Hold `zeta=0.8` and regenerate poles and K for settling times `1.5`, `0.8`,
and `0.3 s`. Compare gains, transient metrics, torque, saturation, and noise
sensitivity. This does not assume that the fastest design is best.

### C — Uncontrollable unstable mode

Construct `x=[q,dq,z]` with an uncontrolled unstable mode `dz=0.5z` and
`B_bad=[0,1/J,0]^T`. Record `rank(C)<3`, identify the uncontrolled `+0.5`
mode, and reject pole placement rather than hiding the failure.

### D — Identification-to-control robustness

Design K once using `(J_hat,b_hat)`. Keep K fixed and compare the nominal
plant with `J_real=1.3*J_hat`. Report measured transient, torque/saturation,
and the gap between nominal pole prediction and the changed physical plant.

## Verification

Tests cover the controllability matrix and the nominal/uncontrollable cases;
pole mapping, placement, and achieved-pole agreement; torque saturation and
payload behaviour. `run_experiments.py` writes four figures, CSV logs, and a
Markdown report. The complete Lesson 07 suite plus existing Lesson 05/06
suites must pass before integration.
