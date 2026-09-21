# Lesson 11 — LQG Output Feedback Engineering Report

## Scope

The controller is a composition of the existing Lesson 08 LQR gain and Lesson 10 Kalman estimator.  The simulation alone owns plant truth; deployable LQG receives only encoder position and the previous **applied** torque.

## A. Separation principle

The `dlqe` gain is in predictor form, so the analysis uses `A_e = A_d - L_e C`.  The augmented linear, unsaturated matrix is `[[A_d-B_dK_c, B_dK_c], [0, A_d-L_eC]]`.

| union pole error | all augmented poles inside unit circle |
| ---: | :--- |
| 0.000e+00 | yes |

This confirms the eigenvalue union only for the matched, linear, unsaturated model; it is not a guarantee for a clipped actuator.

### Numerical pole evidence

Controller poles: 0.98271549+0.00000000j, 0.90508294+0.00000000j

Estimator predictor poles: 0.93412917+0.06167386j, 0.93412917-0.06167386j

Augmented LQG poles: 0.98271549+0.00000000j, 0.90508294+0.00000000j, 0.93412917+0.06167386j, 0.93412917-0.06167386j

## B–E quantitative results

| Scenario | tracking RMSE / rad | q-hat RMSE / rad | dq-hat RMSE / rad/s | control RMS / N m | peak torque / N m | saturation / % | innovation RMS / rad | mean NIS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_state | 0.0172168 | nan | nan | 0.305295 | 3 | 0.1 | nan | nan |
| recursive | 0.0270594 | 0.00152051 | 0.199152 | 0.304541 | 3 | 0.08 | 0.00528994 | 4.28912 |
| steady_state | 0.0371667 | 0.00914783 | 0.846824 | 0.446993 | 3 | 1.22 | 0.0104132 | 125.296 |
| q_scale_0.1 | 0.0384564 | 0.00484729 | 0.347754 | 0.3028 | 3 | 0.08 | 0.00721347 | 33.9833 |
| q_scale_1 | 0.0270594 | 0.00152051 | 0.199152 | 0.304541 | 3 | 0.08 | 0.00528994 | 4.28912 |
| q_scale_10 | 0.0218682 | 0.000574561 | 0.117017 | 0.313352 | 3 | 0.08 | 0.00504868 | 1.27053 |
| normal | 0.015158 | 0.000304571 | 0.031622 | 0.133841 | 3 | 0.08 | 0.00501286 | 0.990516 |
| load | 0.0270594 | 0.00152051 | 0.199152 | 0.304541 | 3 | 0.08 | 0.00528994 | 4.28912 |
| reduced_limit | 0.029322 | 0.00152051 | 0.199152 | 0.281323 | 0.4 | 15.04 | 0.00528994 | 4.28912 |

## C. Startup versus steady-state estimator error

| Estimator and window | q-hat RMSE / rad | dq-hat RMSE / rad/s |
| --- | ---: | ---: |
| recursive startup | 0.000284387 | 0.0497139 |
| recursive steady | 0.00160013 | 0.209291 |
| steady_state startup | 0.0284985 | 2.60066 |
| steady_state steady | 0.00160013 | 0.209291 |

`full_state` is a simulation-only upper/reference baseline and deliberately has no estimator statistics.  Compare `recursive` against `steady_state` over the logged startup (0–0.5 s) and steady windows; a transient difference is expected because only the recursive covariance/gain evolves.

The `q_scale_*` rows keep the LQR gain fixed while changing only the assumed process covariance.  Any changed tracking/control behavior demonstrates that independent LQR/KF design does not make runtime performance independent.

The `reduced_limit` row is deliberately nonlinear because torque clipping is active.  Interpret high saturation, innovation, and NIS as a boundary diagnostic rather than attempting to explain the response solely with `A_d-B_dK_c` and `A_d-L_eC` poles.

## Hardware transition

Replace only the simulated encoder/plant boundary with hardware I/O.  Keep `LQGController.step(measurement, previous_applied_torque, reference)` unchanged, log the actual prior actuator torque/current-derived torque, start at nominal limits, and investigate persistent innovation bias or abnormal NIS before retuning.
