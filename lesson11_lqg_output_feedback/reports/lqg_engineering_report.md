# Lesson 11 — LQG Output Feedback Engineering Report

## Scope

The controller is a composition of the existing Lesson 08 LQR gain and Lesson 10 Kalman estimator.  The simulation alone owns plant truth; deployable LQG receives only encoder position and the previous **applied** torque.

## A. Separation principle

The `dlqe` gain is in predictor form, so the analysis uses `A_e = A_d - L_e C`.  The augmented linear, unsaturated matrix is `[[A_d-B_dK_c, B_dK_c], [0, A_d-L_eC]]`.

| union pole error | all augmented poles inside unit circle |
| ---: | :--- |
| 0.000e+00 | yes |

This confirms the eigenvalue union only for the matched, linear, unsaturated model; it is not a guarantee for a clipped actuator.

## B–E quantitative results

| Scenario | tracking RMSE / rad | q-hat RMSE / rad | dq-hat RMSE / rad/s | control RMS / N m | saturation / % | mean NIS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_state | 0.0171837 | nan | nan | 0.305058 | 0.1 | nan |
| recursive | 0.0270742 | 0.00152039 | 0.199205 | 0.302917 | 0.06 | 4.28885 |
| steady_state | 0.0371336 | 0.0091476 | 0.84491 | 0.445391 | 1.22 | 125.289 |
| q_scale_0.1 | 0.0384715 | 0.00484677 | 0.3478 | 0.301195 | 0.06 | 33.9756 |
| q_scale_1 | 0.0270742 | 0.00152039 | 0.199205 | 0.302917 | 0.06 | 4.28885 |
| q_scale_10 | 0.0218755 | 0.000575708 | 0.117041 | 0.311754 | 0.06 | 1.27275 |
| normal | 0.0151853 | 0.00030444 | 0.0319288 | 0.130161 | 0.06 | 0.990755 |
| load | 0.0270742 | 0.00152039 | 0.199205 | 0.302917 | 0.06 | 4.28885 |
| reduced_limit | 0.0292764 | 0.00152041 | 0.199111 | 0.281181 | 14.68 | 4.28873 |

## C. Startup versus steady-state estimator error

| Estimator and window | q-hat RMSE / rad | dq-hat RMSE / rad/s |
| --- | ---: | ---: |
| recursive startup | 0.000285989 | 0.0538162 |
| recursive steady | 0.00159997 | 0.209235 |
| steady_state startup | 0.0284978 | 2.59448 |
| steady_state steady | 0.00159997 | 0.209235 |

`full_state` is a simulation-only upper/reference baseline and deliberately has no estimator statistics.  Compare `recursive` against `steady_state` over the logged startup (0–0.5 s) and steady windows; a transient difference is expected because only the recursive covariance/gain evolves.

The `q_scale_*` rows keep the LQR gain fixed while changing only the assumed process covariance.  Any changed tracking/control behavior demonstrates that independent LQR/KF design does not make runtime performance independent.

The `reduced_limit` row is deliberately nonlinear because torque clipping is active.  Interpret high saturation, innovation, and NIS as a boundary diagnostic rather than attempting to explain the response solely with `A_d-B_dK_c` and `A_d-L_eC` poles.

## Hardware transition

Replace only the simulated encoder/plant boundary with hardware I/O.  Keep `LQGController.step(measurement, previous_applied_torque, reference)` unchanged, log the actual prior actuator torque/current-derived torque, start at nominal limits, and investigate persistent innovation bias or abnormal NIS before retuning.
