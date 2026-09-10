# Lesson 08: Discrete LQR Optimal Control

This lesson designs a discrete LQR controller for the identified single-joint model from Lesson 06/07:

- `J_hat = 0.019762 kg m^2`
- `b_hat = 0.080257 N m s/rad`
- sample period `dt_s = 0.001 s`

## Run the lesson

From the repository root:

```powershell
conda run -n robot-control python -m unittest discover lesson08_lqr_optimal_control/tests -v
conda run -n robot-control python -m lesson08_lqr_optimal_control.src.run_experiments
```

The runner creates CSV logs, four figures, `reports/qr_summary.md`, and `reports/lqr_report.md`.

## Controller design

The production design uses `control.dlqr(Ad, Bd, Q, R)`. The test suite independently solves the DARE and verifies that both paths return the same gain:

\[
u=-K(x-x_r).
\]

Bryson-normalized weights use acceptable position, velocity, and torque scales. `Q` expresses state-error priorities; `R` expresses input-effort priority.

## Logged torque and cost

Each log records both:

- `torque_unsat_nm`: theoretical state-feedback request;
- `torque_cmd_nm`: command after the actuator hard limit.

The plant evolves from `torque_cmd_nm`, while the LQR input cost records the controller request:

\[
\ell_x=e^TQe,\qquad
\ell_u=u_{unsat}^TRu_{unsat},\qquad
\ell=\ell_x+\ell_u.
\]

When Q/R differs between cases, absolute cost totals are not directly comparable because the objective itself has changed. Compare tracking, torque, and saturation across those cases. The LQR-vs-pole-placement experiment evaluates both controllers using the same Balanced Q/R, so that cost comparison is meaningful.

## Engineering boundaries

The stress experiment applies a constant load and reduces the torque limit. A residual position error is expected because this controller has no integral state. Standard LQR penalizes large inputs but does not impose hard input constraints inside its optimization; clipping is an external actuator safeguard. LQI or MPC are natural next steps when offset rejection or explicit constraints are required.
