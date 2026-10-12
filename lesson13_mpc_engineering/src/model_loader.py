"""Build fixed nominal prediction and estimator designs from explicit settings."""

import numpy as np

from lesson08_lqr_optimal_control.src.model import build_continuous_model, discretize_zoh
from lesson08_lqr_optimal_control.src.lqr_design import build_bryson_weights, dlqr_from_dare
from lesson10_kalman_filter.src.noise_model import build_noise_covariances
from ._validation import bounded_scalar


def build_design(config: dict) -> dict:
    inertia = bounded_scalar(config['nominal_model']['inertia'], 'inertia', positive=True)
    damping = bounded_scalar(config['nominal_model']['damping'], 'damping')
    dt = bounded_scalar(config['plant']['dt_s'], 'plant.dt_s', positive=True)
    control_dt = bounded_scalar(config['mpc']['dt_s'], 'mpc.dt_s', positive=True)
    A, B = build_continuous_model(inertia, damping)
    Ad, Bd = discretize_zoh(A, B, control_dt)
    Ad_fast, Bd_fast = discretize_zoh(A, B, dt)
    limits = config['design_limits']
    limit_values = [bounded_scalar(limits[key], key, positive=True) for key in
                    ('max_position_error_deg', 'max_velocity_error_rad_s', 'max_torque_nm')]
    Q, R = build_bryson_weights(*limit_values, 1., 1., 1.)
    K, P = dlqr_from_dare(Ad, Bd, Q, R)
    for section, names in (('measurement', ('position_noise_std_deg',)),
                           ('process_noise', ('disturbance_torque_std_nm',)),
                           ('initial_uncertainty', ('position_std_deg', 'velocity_std_rad_s'))):
        for name in names:
            bounded_scalar(config[section][name], f'{section}.{name}', positive=True)
    return {'Ad': Ad, 'Bd': Bd, 'Ad_fast': Ad_fast, 'Bd_fast': Bd_fast, 'Q': Q, 'R': R,
            'P': P, 'K': K, 'C': np.array([[1., 0.]]),
            'covariance': build_noise_covariances(config, Bd_fast)}
