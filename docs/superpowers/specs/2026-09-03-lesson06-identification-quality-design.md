# Lesson 06: System Identification Quality Design

## Purpose

Extend the Lesson 05 single-axis system-identification project to evaluate whether an identification dataset is informative, whether offline filtering materially changes the result, whether a friction term is required, and whether an identified model generalises to a payload change.

The work remains inside `lesson05_system_identification/`; it does not create a new lesson directory.

## Scope

The lesson adds four excitation strategies, two offline derivative-estimation strategies, two nested physical-model structures, identifiability diagnostics, independent validation, and a payload-robustness experiment.

It deliberately does not add online filtering, a state observer, hardware drivers, controller tuning, or nonlinear optimisation.

## Input-excitation architecture

`excitation.py` will define a common `ExcitationSignal` interface with `evaluate(time_s)`.  The experiment runner relies only on that interface, so the plant and data logger do not need to know which strategy generated a torque command.

The four strategies are:

- `ConstantTorque`: fixed torque.
- `SingleSine`: one sinusoid with amplitude, frequency, and optional phase.
- `MultiSine`: a sum of sinusoidal components.
- `PRBSExcitation`: deterministic piecewise-constant signed torque, limited by an amplitude and a minimum hold interval.

`config/excitation_cases.yaml` defines all case parameters.  Every comparison uses the same duration, sample period, maximum command amplitude, sensor noise, actuator torque limit, and true plant parameters.  PRBS additionally has a bounded command-change rate by construction: it cannot switch before its configured hold interval.

## Signal processing boundary

Two offline derivative-estimation methods are compared:

1. Low-pass position, numerical differentiation, then low-pass derivative estimates.
2. Savitzky–Golay local-polynomial filtering with first and second derivatives returned directly by `scipy.signal.savgol_filter`.

Both methods consume measured position and output estimated velocity and acceleration.  They are explicitly offline processing: zero-phase filtering and centred Savitzky–Golay windows can consume samples from both sides of a timestamp.  They must not be described as online state estimators.

The filter-sensitivity experiment holds the noisy data fixed and tests Savitzky–Golay windows `11`, `31`, `61`, and `101`, with polynomial order `3`.

## Models and estimator interfaces

Model A remains:

\[
\tau = J\ddot q + b\dot q.
\]

Model B adds smooth Coulomb friction, with fixed smoothing `epsilon`:

\[
\tau = J\ddot q + b\dot q + \tau_c\tanh(\dot q / \epsilon).
\]

For fixed `epsilon`, both models remain linear in their unknown parameters and use `numpy.linalg.lstsq`.

Model A regressor columns are `[acceleration, velocity]`.  Model B adds `[tanh(velocity / epsilon)]`.  Estimators receive only processed velocity, processed acceleration, and actual applied torque; they must not read true parameters.

## Identifiability and residual diagnostics

`identifiability.py` analyses a regression matrix with singular-value decomposition and returns rank, condition number, largest singular value, and smallest singular value.

Rank below the number of estimated parameters is reported as unidentifiable rather than silently converted into a trustworthy estimate.  Constant input may legitimately lead to this result.

`residual_analysis.py` creates residual-versus-time and residual-versus-velocity figures.  Structured residuals near velocity reversal are treated as evidence to investigate friction, stiction, or backlash; structured high-frequency error is treated as evidence to investigate actuator bandwidth, flexibility, sampling, or filtering.

## Experiments

### A. Excitation quality

Run constant torque, single sine, multi-sine, and PRBS under identical duration, sampling, torque amplitude, noise, and plant conditions.  Record excitation name, rank, condition number, parameter estimates/errors when identifiable, identification torque RMSE, and validation trajectory RMSE.

No strategy is predeclared the winner.  The report explains observed results using information content, constraints, and the specific plant.

### B. Filter sensitivity

Use one fixed noisy multi-sine dataset.  Estimate Model A with Savitzky–Golay windows `11`, `31`, `61`, and `101`.  Record estimated parameters, identification RMSE, condition number, and independent validation RMSE.  Strong dependence on window size is reported as a processing-sensitivity risk, not tuned away.

### C. Model structure

Generate true data with smooth Coulomb friction.  Fit Model A and Model B using the same identification dataset and evaluate each using an independent validation excitation.  Compare identification RMSE, validation RMSE, parameter estimates, and residual-versus-velocity structure.

### D. Payload disturbance

Identify Model B with the nominal true inertia.  During independent validation only, increase the true plant inertia by 30% without refitting.  Compare nominal and payload validation RMSE to demonstrate model-robustness limits.

## Outputs

Add or update these outputs:

```text
lesson05_system_identification/
├── config/excitation_cases.yaml
├── src/identifiability.py
├── src/validation.py
├── src/residual_analysis.py
├── figures/excitation_compare.png
├── figures/singular_values.png
├── figures/validation_compare.png
├── figures/residual_vs_velocity.png
├── reports/sysid_quality_report.md
└── tests/test_identifiability.py
```

CSV logs include excitation type/command, measured position, estimated velocity, estimated acceleration, actual applied torque, predicted torque, and torque residual.  Each experiment result also records parameter-estimation metadata in JSON-safe form.

The quality report contains experiment tables, plots, exact configurations, offline-processing caveat, and a conclusion based on measured evidence rather than fixed thresholds.

## Validation and safety

Tests validate input amplitudes and PRBS hold intervals, regression rank/condition diagnostics, Savitzky–Golay output shape and argument validation, Model B regressor width, distinct identification/validation excitations, and the 30% payload inertia change.

Generated excitation is bounded by the configured amplitude.  It is a simulated study; before physical deployment, torque, torque slew rate, position, velocity, thermal, resonance, and emergency-stop limits must be verified on the actual system.
