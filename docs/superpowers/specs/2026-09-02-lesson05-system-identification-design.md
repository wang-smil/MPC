# Lesson 05: Single-Axis System Identification Design

## Purpose

Build an independent, simulated system-identification workflow for a single robot joint.  The workflow estimates inertia and viscous damping from applied torque and measured position data, then checks whether the resulting model predicts an independent trajectory.

This lesson deliberately does not import the Lesson 02 plant.  It uses matching nominal physical parameters so results remain comparable, while Lesson 05 stays independently understandable and runnable.

## Scope

Create `lesson05_system_identification/` with this layout:

```text
lesson05_system_identification/
├── config/sysid.yaml
├── src/
│   ├── plant.py
│   ├── excitation.py
│   ├── signal_processing.py
│   ├── estimator.py
│   ├── metrics.py
│   └── run_sysid.py
├── tests/test_sysid.py
├── logs/
├── figures/
├── reports/identification_report.md
└── README.md
```

The lesson covers least-squares identification of inertia `J` and viscous damping `b`.  It does not add nonlinear optimisers, motor electrical dynamics, controller tuning, or a hardware interface.

## Physical model and information boundary

The simulation plant evolves the joint according to:

\[
\tau = J\ddot q + b\dot q + \tau_c\operatorname{sgn}(\dot q).
\]

`plant.py` is the only runtime component that receives the true parameters.  The nominal truth is `J = 0.020 kg m^2`, `b = 0.080 N m s/rad`, with configurable Coulomb friction.

The estimator intentionally fits the reduced linear model:

\[
\tau \approx J\ddot q + b\dot q.
\]

It receives only processed velocity, processed acceleration, and *applied* torque.  It must not read `plant_true` configuration values.  This boundary prevents simulated ground truth from leaking into the estimator.

All computation uses radians, radians per second, radians per second squared, and newton metres.  Degrees are presentation-only units.

## Components and data flow

`excitation.py` produces deterministic multi-sine torque commands for identification and a separate multi-sine sequence for validation.

`plant.py` applies the torque limit, simulates the true joint with semi-implicit Euler integration, and reports true position, velocity, acceleration, and the actual applied torque.

`signal_processing.py` adds encoder measurement noise, estimates velocity and acceleration by numerical differentiation, filters the estimated signals, and discards the configured initial segment.

`estimator.py` builds the regression matrix `Phi = [acceleration, velocity]` and solves `Phi @ [J_hat, b_hat] = torque` using `numpy.linalg.lstsq`.  It reports rank, condition number, predicted torque, residual, and torque-fit RMSE.  Rank-deficient input is an explicit error rather than a plausible-looking estimate.

`metrics.py` calculates parameter relative error, torque RMSE, validation position RMSE, and regression condition number.

`run_sysid.py` coordinates each named case, writes a CSV, writes `identified_parameters.json`, creates diagnostic figures, and renders a Markdown report.

```text
excitation -> commanded torque -> plant -> measured position
                                      |          |
                                      |          v
                                      |     differentiation/filtering
                                      v          |
                                applied torque <-+
                                      |
                                      v
                             least-squares estimator -> J_hat, b_hat
                                      |
                                      v
                           independent validation trajectory
```

## Configuration

`config/sysid.yaml` has these sections:

- `simulation`: sample period, duration, random seed.
- `plant_true`: inertia, viscous damping, Coulomb friction.
- `sensor`: position-noise standard deviation in degrees.
- `actuator`: torque limit in N m.
- `excitation`: frequency and amplitude lists for the multi-sine input.
- `identification`: start time to discard and low-pass cutoff frequency.

The configuration validator rejects non-positive time settings, negative noise or friction, non-positive torque limits, and frequency/amplitude lists with unequal lengths.

## Experiments

1. `normal`: multi-sine excitation, low measurement noise, normal sampling. Establish the baseline estimates, fit metrics, and validation trajectory error.
2. `poor_excitation`: retain data duration but use a single 0.5 Hz excitation. Compare estimates and condition number with `normal`.
3. `noise_stress`: raise encoder noise from `0.02 deg` to `0.20 deg`; compare raw differentiation with filtered differentiation.
4. `model_mismatch`: retain Coulomb friction in the true plant while leaving it out of the estimator. Plot torque residual against time and velocity, looking for repeatable structure near reversals.
5. `validation`: use an excitation sequence distinct from the identification sequence. Drive both the true plant and the `J_hat, b_hat` model with it, then compare position trajectories and calculate trajectory RMSE.

Each run records at least:

```text
timestamp_s, torque_command_nm, torque_applied_nm,
position_true_rad, position_measured_rad,
velocity_true_rad_s, velocity_est_rad_s,
acceleration_true_rad_s2, acceleration_est_rad_s2,
torque_predicted_nm, residual_nm
```

## Diagnostics and acceptance

No fixed parameter-error threshold is treated as an industrial standard.  The report presents the observed parameter errors alongside torque RMSE, condition number, and independent validation RMSE.  A small training RMSE alone is insufficient evidence of a usable model.

Tests cover:

- valid multi-sine input and invalid excitation configuration;
- plant response and actuator torque limiting;
- recovery of `J` and `b` from ideal noise-free data;
- rejection of rank-deficient regression data;
- output metric calculation and independent validation separation.

## Engineering interpretation

An unexpectedly large inertia estimate is a debugging prompt, not automatic proof that least squares failed.  Diagnose torque units and calibration, motor-side versus joint-side mapping, gear-ratio conversion, radian/degree units, timestamps, payload changes, and unmodelled friction before changing the estimator.
