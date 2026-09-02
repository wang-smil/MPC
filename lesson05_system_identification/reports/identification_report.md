# Lesson 05 System Identification Report

The estimator fits `torque = J * acceleration + b * velocity` using applied torque.

| Case | J_hat / kg m^2 | b_hat / N m s/rad | torque RMSE / N m | cond(Phi) |
| --- | ---: | ---: | ---: | ---: |
| normal | 0.01976 | 0.08026 | 0.02386 | 4.40 |
| poor_excitation | 0.01965 | 0.07994 | 0.02512 | 3.15 |
| noise_raw | 0.00000 | 0.05145 | 0.33281 | 512.74 |
| noise_filtered | 0.01197 | 0.07943 | 0.18507 | 5.64 |
| model_mismatch | 0.01971 | 0.11137 | 0.05481 | 4.86 |

## Independent validation

Position trajectory RMSE: 0.004970 rad.

Interpretation: condition number describes parameter sensitivity, not a standalone pass/fail result. The raw high-noise case exposes derivative noise amplification; filtering should make the fit more usable. Model-mismatch residuals can remain structured because Coulomb friction is absent from the fitted model.