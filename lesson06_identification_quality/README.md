# Lesson 06 — Identification Quality

This lesson extends Lesson 05's single-axis identification pipeline.

- Experiment A: compare constant torque, sine, multisine, and PRBS excitation.
- Experiment B: study Savitzky–Golay derivative-window sensitivity.
- Experiment C: compare a linear model with a smooth Coulomb-friction model.
- Experiment D: validate independently under nominal and +30% payload inertia.

The physical plant, base estimator, and measurement utilities remain in
`lesson05_system_identification/src/`; this lesson imports them as shared
foundations and owns the experiment-specific code here.

Run from the repository root:

```powershell
conda run -n robot-control python -m lesson06_identification_quality.src.run_excitation_quality
conda run -n robot-control python -m lesson06_identification_quality.src.run_savgol_sensitivity
conda run -n robot-control python -m lesson06_identification_quality.src.run_friction_model_comparison
conda run -n robot-control python -m lesson06_identification_quality.src.run_friction_validation
```
