# Lesson 11 — LQG Output Feedback

This lesson combines the Lesson 08 balanced discrete LQR gain with the Lesson 10 discrete Kalman filter.  The new `LQGController` is only a composition layer:

```text
encoder position + previous applied torque
          -> estimator -> state estimate -> LQR -> torque clipping
```

Run all experiments from the repository root:

```powershell
conda run -n robot-control python -m lesson11_lqg_output_feedback.src.run_experiments
```

The runner writes CSV logs, five figures, and `reports/lqg_engineering_report.md`.

`full_state_lqr` is a simulation-only reference: it reads plant truth and is never a hardware design.  The deployable variants are `recursive_lqg` and `steady_state_lqg`, both of which receive only a scalar encoder position and the prior actually applied torque.

The separation-pole result is an unsaturated linear-theory result.  Torque clipping is deliberately kept in the simulation boundary and is investigated separately in the saturation experiment.
