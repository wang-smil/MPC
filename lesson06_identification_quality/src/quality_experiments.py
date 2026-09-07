def validate_friction_models(duration_s: float, dt_s: float) -> dict:
    """Run independent nominal and payload validation for models A and B."""

    from .validation import validate_friction_models as _validate

    return _validate(duration_s=duration_s, dt_s=dt_s)
def compare_friction_models(
    duration_s: float,
    dt_s: float,
    position_noise_std_deg: float,
    window_length: int,
) -> dict[str, dict[str, float | int]]:
    """Fit model A and model B to the same smooth-Coulomb-friction record."""

    import numpy as np

    from lesson05_system_identification.src.estimator import identify_j_b, identify_j_b_tau_c
    from lesson05_system_identification.src.signal_processing import add_position_noise

    if duration_s <= 0.0 or dt_s <= 0.0 or position_noise_std_deg < 0.0:
        raise ValueError("duration_s, dt_s must be positive and noise cannot be negative.")
    if window_length <= 3 or window_length % 2 == 0:
        raise ValueError("window_length must be odd and greater than the polynomial order.")
    time_s = np.arange(0.0, duration_s + 0.5 * dt_s, dt_s)
    if window_length > len(time_s):
        raise ValueError("window_length must not exceed the sample count.")

    signal = default_excitation_signals()["multisine"]
    plant = SingleAxisPlant(
        inertia=0.020,
        damping=0.080,
        coulomb_friction=0.10,
        torque_limit_nm=2.0,
    )
    torque_nm = np.asarray(signal.evaluate(time_s), dtype=float)
    records = [plant.step(command, dt_s) for command in torque_nm]
    position_true_rad = np.array([record["position_true_rad"] for record in records])
    position_measured_rad = add_position_noise(
        position_true_rad,
        position_noise_std_deg,
        np.random.default_rng(2026),
    )
    velocity_est, acceleration_est = savgol_derivatives(
        position_measured_rad, dt_s=dt_s, window_length=window_length
    )
    fit_mask = time_s >= 0.1 * duration_s
    model_a = identify_j_b(
        velocity_est[fit_mask], acceleration_est[fit_mask], torque_nm[fit_mask]
    )
    model_b = identify_j_b_tau_c(
        velocity_est[fit_mask],
        acceleration_est[fit_mask],
        torque_nm[fit_mask],
        friction_smoothing_rad_s=0.02,
    )
    return {"model_a": model_a, "model_b": model_b}

def compare_savgol_windows(
    window_lengths: list[int],
    duration_s: float,
    dt_s: float,
    position_noise_std_deg: float,
) -> dict[int, dict[str, float]]:
    """Fit the same noisy multisine record with several S-G window lengths."""

    import numpy as np

    from lesson05_system_identification.src.estimator import identify_j_b
    from lesson05_system_identification.src.signal_processing import add_position_noise

    if not window_lengths or duration_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("window_lengths, duration_s and dt_s must be positive.")
    if any(window <= 3 or window % 2 == 0 for window in window_lengths):
        raise ValueError("each window length must be odd and greater than the polynomial order.")
    time_s = np.arange(0.0, duration_s + 0.5 * dt_s, dt_s)
    if max(window_lengths) > len(time_s):
        raise ValueError("each window length must not exceed the sample count.")

    signal = default_excitation_signals()["multisine"]
    plant = SingleAxisPlant(
        inertia=0.020,
        damping=0.080,
        coulomb_friction=0.0,
        torque_limit_nm=2.0,
    )
    torque_nm = np.asarray(signal.evaluate(time_s), dtype=float)
    records = [plant.step(command, dt_s) for command in torque_nm]
    position_true_rad = np.array([record["position_true_rad"] for record in records])
    position_measured_rad = add_position_noise(
        position_true_rad,
        position_noise_std_deg,
        np.random.default_rng(2026),
    )
    fit_mask = time_s >= 0.1 * duration_s
    results: dict[int, dict[str, float]] = {}
    for window_length in window_lengths:
        velocity_est, acceleration_est = savgol_derivatives(
            position_measured_rad, dt_s=dt_s, window_length=window_length
        )
        estimate = identify_j_b(
            velocity_est[fit_mask], acceleration_est[fit_mask], torque_nm[fit_mask]
        )
        results[window_length] = {
            "inertia_hat": float(estimate["inertia_hat"]),
            "damping_hat": float(estimate["damping_hat"]),
            "inertia_abs_error": abs(float(estimate["inertia_hat"]) - 0.020),
            "damping_abs_error": abs(float(estimate["damping_hat"]) - 0.080),
            "torque_rmse": float(estimate["torque_rmse"]),
        }
    return results

def compare_noisy_identification(
    duration_s: float,
    dt_s: float,
    position_noise_std_deg: float,
) -> dict[str, dict[str, float | int]]:
    """Compare fitted J and b after identical encoder noise is added."""

    import numpy as np

    from lesson05_system_identification.src.estimator import identify_j_b
    from lesson05_system_identification.src.signal_processing import add_position_noise

    if duration_s <= 0.0 or dt_s <= 0.0 or position_noise_std_deg < 0.0:
        raise ValueError("duration_s, dt_s must be positive and noise cannot be negative.")
    time_s = np.arange(0.0, duration_s + 0.5 * dt_s, dt_s)
    if len(time_s) < 31:
        raise ValueError("at least 31 samples are required for the default filter window.")

    results: dict[str, dict[str, float | int]] = {}
    for name, signal in default_excitation_signals().items():
        plant = SingleAxisPlant(
            inertia=0.020,
            damping=0.080,
            coulomb_friction=0.0,
            torque_limit_nm=2.0,
        )
        torque_nm = np.asarray(signal.evaluate(time_s), dtype=float)
        records = [plant.step(command, dt_s) for command in torque_nm]
        position_true_rad = np.array([record["position_true_rad"] for record in records])
        position_measured_rad = add_position_noise(
            position_true_rad,
            position_noise_std_deg,
            np.random.default_rng(2026),
        )
        velocity_est, acceleration_est = savgol_derivatives(
            position_measured_rad, dt_s=dt_s, window_length=31
        )
        fit_mask = time_s >= 0.1 * duration_s
        estimate = identify_j_b(
            velocity_est[fit_mask], acceleration_est[fit_mask], torque_nm[fit_mask]
        )
        results[name] = {
            "inertia_hat": estimate["inertia_hat"],
            "damping_hat": estimate["damping_hat"],
            "inertia_abs_error": abs(float(estimate["inertia_hat"]) - 0.020),
            "damping_abs_error": abs(float(estimate["damping_hat"]) - 0.080),
            "rank": estimate["rank"],
            "condition_number": estimate["condition_number"],
            "sigma_min": float(
                np.linalg.svd(
                    np.column_stack([acceleration_est[fit_mask], velocity_est[fit_mask]]), compute_uv=False
                )[-1]
            ),
            "torque_rmse": estimate["torque_rmse"],
        }
    return results

"""Small, reproducible experiments for Lesson 06 identification quality."""


import numpy as np

from lesson05_system_identification.src.excitation import ConstantTorque, ExcitationSignal, MultiSine, PRBSExcitation, SingleSine
from lesson05_system_identification.src.identifiability import analyse_regressor
from lesson05_system_identification.src.plant import SingleAxisPlant
from lesson05_system_identification.src.signal_processing import savgol_derivatives


def default_excitation_signals() -> dict[str, ExcitationSignal]:
    """Return the four bounded torque inputs compared in Experiment A."""

    return {
        "constant": ConstantTorque(amplitude_nm=0.5),
        "single_sine": SingleSine(amplitude_nm=0.5, frequency_hz=1.0),
        "multisine": MultiSine(
            frequencies_hz=[0.5, 1.3, 2.7], amplitudes_nm=[0.25, 0.18, 0.10]
        ),
        "prbs": PRBSExcitation(amplitude_nm=0.5, hold_time_s=0.1, seed=2026),
    }


def compare_excitation_quality(duration_s: float, dt_s: float) -> dict[str, dict[str, float | int]]:
    """Diagnose rank and conditioning for each excitation's reconstructed Phi.

    The derivative estimator intentionally uses only simulated encoder position,
    mirroring the measurement path used in a real joint experiment.
    """

    if duration_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("duration_s and dt_s must be positive.")
    time_s = np.arange(0.0, duration_s + 0.5 * dt_s, dt_s)
    if len(time_s) < 31:
        raise ValueError("at least 31 samples are required for the default filter window.")

    results: dict[str, dict[str, float | int]] = {}
    for name, signal in default_excitation_signals().items():
        plant = SingleAxisPlant(
            inertia=0.020,
            damping=0.080,
            coulomb_friction=0.0,
            torque_limit_nm=2.0,
        )
        torque_nm = np.asarray(signal.evaluate(time_s), dtype=float)
        records = [plant.step(command, dt_s) for command in torque_nm]
        position_rad = np.array([record["position_true_rad"] for record in records])
        velocity_est, acceleration_est = savgol_derivatives(
            position_rad, dt_s=dt_s, window_length=31
        )
        phi = np.column_stack([acceleration_est, velocity_est])
        results[name] = analyse_regressor(phi)
    return results
