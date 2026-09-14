# Lesson 09: Discrete State Observer Design

## Purpose

Build an engineering-facing discrete Luenberger observer for the identified
single-joint model.  The lesson connects the validated parameters from lessons
05/06 and the balanced discrete LQR gain from lesson 08 into an observer-based
closed loop.  It must make clear both the benefit and the limits of estimating
velocity from a noisy position encoder.

## Scope and constraints

- Create `lesson09_state_observer/`; do not modify the implementations of
  earlier lessons.
- Use the identified nominal model `J_hat = 0.019762 kg m^2`,
  `b_hat = 0.080257 N m s/rad`, and `Ts = 1 ms`.
- Derive `Ad`, `Bd` using ZOH and derive the balanced LQR gain from the lesson
  08 Bryson weights.  Do not copy a hard-coded gain.
- Only a position encoder is measured.  The observer public update API accepts
  `measurement` and `control_input` only.  It has no route to `q_true` or
  `dq_true`.
- Pass the actuator-clipped `u_applied`, never `u_unsat`, to the observer.
- The simulation retains true plant states solely to calculate offline metrics
  and make teaching plots.

## Observer design

The state is `x = [q, dq]^T` and the position-only output is `y = Cx`, with
`C = [[1, 0]]`.  The observer uses the discrete form:

`x_hat[k+1] = Ad x_hat[k] + Bd u_applied[k] + Ld (y[k] - C x_hat[k])`.

Compute observability `O = [C; C Ad]` and report rank, condition number, and
smallest singular value.  Nominal rank must equal two.

Start from the lesson-08 controller poles in continuous time.  For observer
speed factor `f`, map the requested continuous poles as
`s_observer = f * s_controller`, then map to the unit disk with
`z_observer = exp(s_observer * Ts)`.  Use dual pole placement on
`(Ad.T, C.T)` to obtain `Ld`.  Verify that the achieved eigenvalues of
`Ad - Ld C` match the requested poles and lie inside the unit circle.

## Components

`model_loader.py`
: Loads the YAML configuration and produces the identified continuous and ZOH
  discrete model.  It owns the model-source values used by the observer.

`observability.py`
: Builds `O` and returns rank, condition number, and `sigma_min`.

`observer_design.py`
: Maps continuous target poles to discrete poles and returns `Ld`, requested,
  and achieved observer poles.

`observer.py`
: Provides `DiscreteObserver.update(measurement, control_input)`.  Each update
  records `y_hat`, innovation, and the corrected next estimate.

`velocity_estimators.py`
: Provides a raw backward difference and a first-order low-pass filtered
  difference for the experiment-C comparison.

`closed_loop.py`
: Simulates the true plant, noisy encoder, observer-based LQR, actuator
  clipping, and the three alternative velocity sources.  It explicitly keeps
  the observer input equal to the actual applied torque.

`metrics.py`
: Calculates estimation RMSE, convergence time, innovation RMS, tracking
  RMSE, torque RMS/peak, saturation ratio, and velocity-estimate noise.

`run_experiments.py`
: Runs all scenarios, writes CSV logs, plots, and an engineering report.

## Configuration

`observer.yaml` sets a 1 ms sample time, 4 s duration, seed 2026, 0.05 degree
position noise, 20 degree / 0.5 rad/s true initial state, zero initial
estimate, nominal torque limit 3 Nm, payload scale 1.30, high-noise scale 10,
and encoder bias 0.5 degree.  Observer speed factors are `[2, 4, 10]` and the
nominal factor is 4.  A difference-filter alpha is explicitly configured.

## Experiments and artifacts

### A. Clean convergence

Use the nominal model, zero measurement noise, normal actuator, and mismatched
initial true/estimated states.  Generate four panels: position truth versus
estimate, velocity truth versus estimate, both estimation errors, and
innovation.  The estimate must converge.

### B. Observer-speed trade-off

Use identical encoder noise, plant, and initial mismatch for factors 2, 4, and
10.  Compare convergence time, position/velocity RMSE, velocity-estimate noise
standard deviation, LQR tracking RMSE, and torque RMS.  The report must state
that a faster observer is not intrinsically better: larger `L` can inject more
encoder noise into `x_hat` and therefore torque.

### C. Difference versus observer

With position-only sensing, run raw difference, low-pass difference, and the
observer as the velocity source for the same LQR loop.  Compare velocity RMSE,
velocity-noise standard deviation, tracking RMSE, RMS torque, and peak torque.

### D. Model mismatch

Leave observer and controller at the nominal identified model but set the true
plant inertia to `1.30 * J_hat`.  Compare it to nominal behavior and report
estimation error, innovation RMS, and closed-loop tracking changes.

### E. Encoder-bias stress

Add a constant 0.5 degree offset to the position measurement.  Show its effect
on `q_hat`, innovation, and tracking.  Document that this fixed-state observer
assumes an unbiased encoder; it cannot identify an unmodelled bias state.

Formal acceptance scenarios are `normal` (nominal model + normal noise),
`disturbance` (+30% payload), and `stress` (10x encoder noise).  The bias run
is retained separately as a diagnostic limitation experiment.

Each scenario log includes time, true/measured/estimated position, true/raw/
filtered/estimated velocity, both estimation errors, predicted output,
innovation, requested and applied torque, and saturation flag.

## Tests

- Observability has rank two for the position-only nominal model; invalid
  dimensions are rejected.
- Pole mapping and dual pole placement produce stable, requested poles.
- The observer update only accepts scalar measurement and control input and
  converges in a noise-free simulation from a mismatched initial estimate.
- Saturated simulation demonstrates the observer uses applied, not requested,
  torque.
- Difference estimators have deterministic basic behavior and reject invalid
  sampling/filter arguments.
- The experiment runner creates logs, figures, and the report.

## Enterprise deployment sequence

On a real robot: validate encoder units/direction/timestamps first; run the
observer offline against recorded input-output pairs; inspect `q_hat`,
`dq_hat`, and innovation; close a low-bandwidth loop with `x_hat`; add nominal
noise, payload variation, and velocity-estimator comparisons; then increase
bandwidth.  Persistent or structured innovation is a diagnostic signal for
sensor offset, plant mismatch, periodic disturbance, or communication faults.
Production systems generally extend this fixed-gain, single-joint design with
sensor fusion, bias states, parameter scheduling, and Kalman/EKF variants.
