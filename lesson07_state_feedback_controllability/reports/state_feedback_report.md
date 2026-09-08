# Lesson 07: State Feedback and Controllability Report

## Design model and controllability

The design model uses the Lesson 05 identified values `J_hat=0.019762 kg m^2` and `b_hat=0.080257 N m s/rad`.
The nominal controllability rank is 2 / 2; its smallest singular value is 11.784230.
The teaching failure model has rank 2 / 3, so its unstable z mode cannot be placed by torque.

## Pole placement

Nominal state-feedback gain: `K = [0.749384, 0.115369]`.
Requested discrete poles: `0.950561+0.035663j`, `0.950561-0.035663j`.
Achieved discrete poles: `0.950561+0.035663j`, `0.950561-0.035663j`.

## Experiment results

- **slow_1p5s**: final error=0.003 deg, settling=1.210 s, overshoot=0.24%, RMS torque=0.028 N m, saturation=0.00%
- **nominal_0p8s**: final error=0.014 deg, settling=0.970 s, overshoot=0.03%, RMS torque=0.062 N m, saturation=0.00%
- **aggressive_0p3s**: final error=0.026 deg, settling=0.670 s, overshoot=8.13%, RMS torque=0.331 N m, saturation=0.00%
- **payload_plus_30_percent**: final error=0.013 deg, settling=0.580 s, overshoot=0.03%, RMS torque=0.067 N m, saturation=0.00%

## Engineering interpretation

The controller operates on noisy position and a causal filtered velocity estimate, not the plant's hidden true states. CSV logs preserve `torque_unsat_nm` and `torque_applied_nm` separately, so actuator saturation is visible rather than hidden. The payload case deliberately keeps the nominal K unchanged: degraded performance is therefore a robustness result, not a redesigned-controller result.