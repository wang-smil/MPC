# Lesson 08: Discrete LQR Optimal Control Report

## A. Q/R trade-off

| Case | Kq | Kdq | Settling / s | Overshoot / % | Peak torque | RMS torque | Saturation / % | State cost | Input cost | Total cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | 32.4873 | 2.1253 | 0.254 | 0.095 | 3.000 | 0.324 | 0.925 | 2714.18 | 523.46 | 3237.64 |
| position_priority | 101.2757 | 2.6561 | 0.107 | 0.496 | 3.000 | 0.505 | 2.524 | 18046.61 | 8787.08 | 26833.70 |
| effort_saving | 10.6441 | 0.8204 | 0.250 | 0.090 | 3.000 | 0.263 | 0.475 | 2778.92 | 533.39 | 3312.31 |
| velocity_priority | 29.2940 | 5.4161 | 0.736 | 0.090 | 3.000 | 0.260 | 0.375 | 6862.61 | 188.16 | 7050.77 |

## B. LQR versus pole placement

LQR is optimal for its selected Q/R cost; it is not guaranteed to minimize every external metric more than pole placement.
Balanced LQR total cost: 3237.64. Pole-placement cost evaluated with the balanced Q/R: 6649.23.

## C. Payload mismatch

The nominal LQR gain was kept fixed while real inertia changed by +30%. Nominal tracking RMSE: 0.060386 rad; payload RMSE: 0.063025 rad.

## D. Constant load and saturation stress

A 1 N m load begins at 1 s and torque is limited to 1.5 N m. Final error: 1.761 deg; saturation: 1.77%.
The residual error illustrates that plain state-feedback LQR has no integral action for an unknown constant load. The clipped torque trace shows that an R penalty is not a hard input constraint.