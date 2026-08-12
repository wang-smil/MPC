# Lesson 04 Digital PID and Cascade Servo Design

## Goal

Build an independent lesson 04 experiment that explains digital PID,
integrator windup, anti-windup, and two-loop cascade servo control using the
single-axis joint context from lesson 02.

## Scope and Reuse

Create `lesson04_digital_pid_cascade/` rather than modifying lesson 02.
Reuse the lesson 02 teaching context: a single-axis rotary joint, encoder
noise, torque saturation, and numerical simulation conventions. Keep the
lesson 04 plant and controllers local so that lesson 02 remains a reproducible
PD engineering baseline.

The first implementation includes all three course experiments:

1. PD versus PID under a constant load torque applied after 1 s.
2. Deliberate integrator windup with and without integral clamping.
3. Single-loop position PID versus a cascade controller:
   position P -> velocity reference -> velocity PI -> torque.

## Project Structure

```text
lesson04_digital_pid_cascade/
├─ config/servo.yaml
├─ src/pid.py
├─ src/plant.py
├─ src/single_loop_pid.py
├─ src/cascade_servo.py
├─ src/run_experiments.py
├─ logs/
├─ figures/
├─ reports/
├─ tests/
└─ README.md
```

`pid.py` provides a stateful digital PID controller with `reset()` and
`update(error, dt)`. It records an integral state, computes the raw
unsaturated output, applies torque clipping, and supports an explicit
integral-clamp anti-windup mode.

`plant.py` provides the joint state update, encoder noise, torque limit, and
constant external load torque. It returns measurements and true state data
needed for reproducible logs.

`single_loop_pid.py` computes direct position-to-torque PD or PID control.
`cascade_servo.py` computes position P to form a bounded velocity reference,
then velocity PI to form the torque request.

## Experiments and Artifacts

All controllers use the same nominal simulation duration, target, torque
limit, noise setting, and plant parameters for a fair comparison.

- Experiment A applies a 1.5 N·m load after 1 s and compares PD against PID.
  It records final position error, maximum post-disturbance error, recovery
  time, peak torque, saturation ratio, and integral peak. It writes
  `figures/pd_vs_pid_load.png`.
- Experiment B uses a 90 degree target, a low torque limit, and a high
  integral gain. It compares no anti-windup against integral clamping and
  writes `figures/anti_windup.png`.
- Experiment C compares direct single-loop PID against cascade position-P /
  velocity-PI control and writes `figures/single_vs_cascade.png`.

Every log separates `torque_unsat` from `torque_cmd` and includes time,
reference and measured position, velocity reference and measured velocity,
position and velocity errors, integral state, saturation flag, and load
torque.

## Engineering Boundaries

This is still a teaching simulation. A torque command is an abstraction of a
real drive chain; real torque depends on current control, PWM, motor and
voltage limits, delays, temperature, sensing, and mechanics. The cascade
architecture creates clearer time-scale separation but does not automatically
outperform a well-tuned single loop.

## Validation

Tests will verify digital PID integral updates, clipping, reset behavior, and
integral clamping. They will verify that all three experiments create logs,
figures, and an analysis report. The report will answer the five course
questions about integral action, windup, derivative noise/kick, cascade
control, and the gap between commanded and actual torque.
