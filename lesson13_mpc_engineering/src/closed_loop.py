"""Causal KF/MPC feedback around the nonideal plant; no fallback yet in phase 2."""

from numbers import Integral
from time import perf_counter

import numpy as np

from lesson10_kalman_filter.src.kalman_filter import DiscreteKalmanFilter
from ._validation import bounded_scalar, finite_scalar, integer_steps
from .actuator import TorqueActuator
from .constraint_manager import guard_command
from .model_loader import build_design
from .mpc_controller import EngineeringMPCController
from .plant import JointPlant


def simulate_closed_loop(config: dict, *, input_rate_weight: float, hard_rate: bool,
                         measurement_noise=None) -> dict:
    dt = bounded_scalar(config['plant']['dt_s'], 'plant.dt_s', positive=True)
    control_dt = bounded_scalar(config['mpc']['dt_s'], 'mpc.dt_s', positive=True)
    duration = bounded_scalar(config['experiment_ab']['duration_s'], 'duration_s', positive=True)
    count = integer_steps(duration, dt, 'duration_s')
    stride = integer_steps(control_dt, dt, 'control_dt_s')
    if min(count, stride) < 1:
        raise ValueError('Periods must contain at least one physical step.')
    if not isinstance(hard_rate, (bool, np.bool_)):
        raise ValueError('hard_rate must be boolean.')
    design = build_design(config)
    limit = bounded_scalar(config['constraints']['position_limit_deg'], 'position_limit_deg', positive=True)
    position_bound = float(np.deg2rad(limit))
    actuator = TorqueActuator(dt_s=dt, **config['actuator'])
    initial = config.get('initial_condition', {})
    state0 = [np.deg2rad(finite_scalar(initial.get('q_true_deg', 0.), 'q_true_deg')),
              finite_scalar(initial.get('dq_true_rad_s', 0.), 'dq_true_rad_s')]
    plant = JointPlant(**config['nominal_model'], **config['plant'], initial_state=state0)
    rate_delta = actuator.slew_rate_nm_s * control_dt
    controller = EngineeringMPCController(design['Ad'], design['Bd'], design['Q'], design['R'],
                                         design['P'], config['mpc']['horizon_steps'],
                                         input_rate_weight=input_rate_weight,
                                         torque_bounds=(-actuator.torque_limit_nm, actuator.torque_limit_nm),
                                         delta_bounds=(-rate_delta, rate_delta) if hard_rate else None,
                                         position_bounds=(-position_bound, position_bound))
    seed = config['experiment_ab']['seed']
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')
    if measurement_noise is None:
        std = np.deg2rad(config['measurement']['position_noise_std_deg'])
        noise = np.random.default_rng(seed).normal(0., std, count)
    else:
        raw = np.asarray(measurement_noise, dtype=object)
        if raw.shape != (count,):
            raise ValueError('measurement_noise must have one value per physical step.')
        noise = np.array([finite_scalar(item, 'measurement_noise') for item in raw])
    estimate0 = config.get('initial_estimate', {})
    x0 = [np.deg2rad(finite_scalar(estimate0.get('position_deg', 0.), 'position_deg')),
          finite_scalar(estimate0.get('velocity_rad_s', 0.), 'velocity_rad_s')]
    reference = np.array([np.deg2rad(finite_scalar(config['experiment_ab']['target_deg'], 'target_deg')), 0.])
    cov = design['covariance']
    filter_ = DiscreteKalmanFilter(design['Ad_fast'], design['Bd_fast'], design['C'],
                                  cov['Gd'], cov['Q_process'], cov['R_measurement'], x0, cov['P0'])
    fields = ('time_s', 'q_true_rad', 'dq_true_rad_s', 'q_hat_rad', 'dq_hat_rad_s',
              'q_measured_rad', 'measurement_noise_rad', 'q_ref_rad', 'torque_request_nm',
              'torque_command_nm', 'torque_actual_nm', 'delta_actual_nm', 'control_update',
              'feedback_prev_nm', 'solver_status', 'solver_iterations', 'solver_solve_time_s',
              'solver_wall_time_s', 'request_delta_nm', 'qp_first_delta_nm', 'command_delta_nm',
              'max_planned_delta_nm', 'guard_modified')
    rows, plans = [], []
    request = command = actuator.torque_actual_nm
    status, failure = 'complete', None
    state_time = 0.
    for index in range(count):
        time = index * dt
        q, velocity = plant.state
        feedback = actuator.torque_actual_nm  # Feedback from interval k-1, not the new command.
        measurement = q + noise[index]
        if index:
            filter_.predict(feedback)
        estimate = filter_.update(measurement)['x_hat']
        update = index % stride == 0
        row = {key: np.nan for key in fields}
        row.update(time_s=time, q_true_rad=q, dq_true_rad_s=velocity,
                   q_hat_rad=estimate[0], dq_hat_rad_s=estimate[1], q_measured_rad=measurement,
                   measurement_noise_rad=noise[index], q_ref_rad=reference[0],
                   torque_request_nm=request, torque_command_nm=command,
                   control_update=update, feedback_prev_nm=feedback, solver_status='')
        if abs(q) > position_bound:
            status, failure = 'safety_stop', f'Position boundary exceeded at t={time:.6f}'
            rows.append(row)
            break
        if update:
            started = perf_counter()
            result = controller.solve(estimate, reference, feedback)
            wall = perf_counter() - started
            row.update(solver_status=result['status'], solver_wall_time_s=wall,
                       solver_iterations=result['iterations'] if result['iterations'] is not None else np.nan,
                       solver_solve_time_s=result['solve_time_s'] if result['solve_time_s'] is not None else np.nan)
            if result['u0'] is None:
                status, failure = 'solver_failure', f"t={time:.6f}: {result['status']}; {result['error']}"
                # No input was applied on this failure row. Save evidence, do not use stale U.
                row.update(torque_request_nm=np.nan, torque_command_nm=np.nan)
                rows.append(row)
                break
            previous_request, previous_command = request, command
            request = float(result['u0'][0])
            command = guard_command(request, previous_command, actuator.torque_limit_nm,
                                    actuator.slew_rate_nm_s, control_dt)
            row.update(request_delta_nm=request - previous_request,
                       qp_first_delta_nm=request - feedback, command_delta_nm=command - previous_command,
                       max_planned_delta_nm=float(np.max(np.abs(result['DU']))),
                       guard_modified=abs(request - command) > 1e-5)
            plans.append({'time_s': time, 'u_prev': feedback,
                          'X': result['X'], 'U': result['U'], 'DU': result['DU']})
        output = actuator.step(command)
        row.update(torque_request_nm=request, torque_command_nm=command,
                   torque_actual_nm=output['torque_actual_nm'], delta_actual_nm=output['delta_actual_nm'])
        rows.append(row)
        next_state = plant.step(output['torque_actual_nm'])
        state_time = (index + 1) * dt
        if abs(next_state[0]) > position_bound:
            status, failure = 'safety_stop', f'Position boundary exceeded at t={time + dt:.6f}'
            break
    log = {key: np.array([row[key] for row in rows]) for key in fields}
    return {'status': status, 'log': log, 'plans': plans, 'failure_reason': failure,
            'expected_samples': count, 'final_state': plant.state,
            'final_state_time_s': state_time}
