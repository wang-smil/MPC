# Lesson 03 Pole Comparison Design

## Goal

Extend the ZOH discretization experiment with a pole comparison that makes the
continuous-time and discrete-time stability regions visible.

## Scope

The experiment uses the existing mass-spring-damper model and compares three
sampling periods: 1 ms, 10 ms, and 500 ms. It does not add a controller or
change the plant parameters.

## Design

`src/discretize_demo.py` will expose small functions that calculate the
continuous poles from `A` and discrete poles from the ZOH `Ad` matrix. The
existing `main()` entry point will print these values and create
`figures/pole_compare.png`.

The figure has two panels:

1. The left s-plane panel marks the continuous poles using real and imaginary
   axes.
2. The right z-plane panel draws the unit circle and marks one discrete pole
   pair for each of 1 ms, 10 ms, and 500 ms. Colours and labels identify the
   sampling period.

## Interpretation

Continuous-time stability requires every pole to have a negative real part.
Discrete-time stability requires every pole to lie strictly inside the unit
circle. With exact ZOH discretization, corresponding poles follow
`z = exp(s * Ts)`, so this stable plant remains stable for all three selected
sampling periods. The 500 ms points make the sampling-period-dependent
location easy to see; they do not indicate instability.

## Validation

Automated tests will verify that:

- the continuous poles are the expected conjugate pair;
- each discrete pole set matches `exp(continuous_pole * Ts)` for the three
  sampling periods;
- every discrete pole has magnitude below one;
- producing the plot writes `figures/pole_compare.png`.

The lesson README will include the new run command and the stability reading
rules.
