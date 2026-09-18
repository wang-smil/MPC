# Lesson 10 Kalman Filter Engineering Report

## Formal Acceptance

| Scenario | q RMSE (rad) | dq RMSE (rad/s) | innovation RMS (rad) | NIS mean | control RMS (N·m) | tracking RMSE (rad) |
|---|---:|---:|---:|---:|---:|---:|
| normal | 0.000304 | 0.031929 | 0.005013 | 0.991 | 0.130161 | 0.015185 |
| disturbance | 0.001520 | 0.199205 | 0.005290 | 4.289 | 0.302917 | 0.027074 |
| stress | 0.002646 | 0.183527 | 0.010325 | 95.988 | 0.353773 | 0.016555 |
| payload_plus_30_percent | 0.000297 | 0.032406 | 0.005013 | 0.989 | 0.140449 | 0.015437 |

## R: How Much Does the Filter Trust the Encoder?

With real encoder noise held fixed, assumed `R=0.1R`, `R`, and `10R` produced position gains of 0.2067, 0.1213, and 0.0692.  Smaller R makes the estimate correct toward each encoder sample more aggressively; it can reduce lag but also carries more measurement noise into the estimated state and LQR torque.

## Q: How Much Does the Filter Trust the Model?

The 2 s unknown load step was applied with the same physical noise in each Q case.  The `0.1Q`, `Q`, and `10Q` position RMSE values are 0.004847, 0.001520, and 0.000576 rad.  Q is not a sensor-noise setting: it represents unmodelled torque, friction, payload, calibration, and discretization uncertainty.

## Estimator Comparison

| Scenario | q RMSE (rad) | dq RMSE (rad/s) | innovation RMS (rad) | NIS mean | control RMS (N·m) | tracking RMSE (rad) |
|---|---:|---:|---:|---:|---:|---:|
| raw | 0.000868 | 1.224885 | nan | nan | 2.093199 | 0.019078 |
| lpf | 0.000868 | 0.114233 | nan | nan | 0.400225 | 0.016883 |
| luenberger | 0.004033 | 0.804517 | 0.006429 | nan | 0.408603 | 0.036056 |
| kalman | 0.001520 | 0.199205 | 0.005290 | 4.289 | 0.302917 | 0.027074 |

The four methods shared the same model, initial state, deterministic encoder noise, LQR, torque limit, and load step.  Kalman filtering must not be declared universally best from a single table: its result depends on Q/R calibration, model quality, and which operational risk—noise, lag, control effort, or fault diagnosis—matters most.

## Industrial Reading

`innovation = y - C x_hat_prior` is the raw encoder/model disagreement.  `NIS = innovation² / S` normalizes it by the filter's declared uncertainty.  Persistent large NIS can indicate underestimated Q/R, payload mismatch, encoder fault, or model bias; persistent innovation bias particularly suggests offset or model bias.  On hardware, estimate R first from a static encoder recording, then tune Q against dynamic residuals, NIS, torque saturation, and an independent validation motion.  The filter must use the applied torque, never the torque request before saturation.
