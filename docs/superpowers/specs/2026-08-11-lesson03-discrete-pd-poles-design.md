# Lesson 03 Discrete PD Closed-Loop Poles Design

## Goal

Extend the existing ZOH lesson with a discrete PD state-feedback experiment
that shows how `Kp` and `Kd` change the closed-loop poles.

## Scope

Reuse the existing mass-spring-damper model and `discretize_zoh()` function.
Use the teaching model only; do not add the noise, saturation, delay, or
nonlinear friction from lesson 02 to this pole-analysis experiment.

For a zero position and velocity reference, the PD law is:

```text
u[k] = -K x[k]
K = [[Kp, Kd]]
```

The closed-loop matrix is therefore:

```text
Acl = Ad - Bd @ K
```

Discrete-time stability means every closed-loop pole has magnitude strictly
below one.

## Components

Create `lesson03_discretization/src/closed_loop_poles.py`.

- `closed_loop_matrix(sample_time_s, kp, kd)` returns `Acl`.
- `closed_loop_poles(sample_time_s, kp, kd)` returns its eigenvalues.
- `is_stable(poles)` returns whether all pole magnitudes are below one.
- `plot_closed_loop_poles(cases)` draws the unit circle and three 10 ms cases:
  `weak=(2.0, 0.2)`, `medium=(15.0, 2.0)`, and `strong=(80.0, 2.0)`.
- `sweep_kp(kp_values, kd=2.0, sample_time_s=0.01)` collects the two pole
  branches while `Kp` changes from 0 to 200.
- `plot_gain_sweep(...)` creates the gain-sweep figure.

The script entry point will create these artifacts:

```text
lesson03_discretization/figures/closed_loop_poles.png
lesson03_discretization/figures/gain_sweep.png
lesson03_discretization/reports/closed_loop_analysis.md
```

The report will include the three 10 ms cases and the fixed-gain sampling-time
comparison for `Kp=15`, `Kd=2`, at 1 ms, 10 ms, 50 ms, and 100 ms.

## Interpretation Boundaries

The figures explain the nominal linear closed-loop model only. They do not
prove the lesson 02 servo loop is safe or robust, because that loop also has
sensor noise, torque saturation, command delay, friction, timing effects, and
model uncertainty.

## Validation

Automated tests will verify:

- `Acl` exactly equals `Ad - Bd @ [[Kp, Kd]]`;
- the weak, medium, and strong case calculations expose two poles and a
  boolean stability result;
- a Kp sweep has one row per requested gain and two pole columns;
- both PNG figures and the Markdown report are written successfully.
