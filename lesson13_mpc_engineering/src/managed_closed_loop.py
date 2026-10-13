"""Single-worker deadline event simulation, separate from synchronous wall-clock checks.

Injected completion times advance physical events; actual Python solve time is recorded
separately. This is not a realtime worker, cancellation API or hardware safety system.
"""

from numbers import Integral
from time import perf_counter

import numpy as np

from lesson10_kalman_filter.src.kalman_filter import DiscreteKalmanFilter
from ._validation import bounded_scalar, finite_scalar, integer_steps
from .actuator import TorqueActuator
from .constraint_manager import guard_command
from .fallback_controller import LqrFallbackController
from .model_loader import build_design
from .mpc_controller import EngineeringMPCController
from .plant import JointPlant
from .qp_solver import SolverSettings
from .solver_monitor import assess_result, SolverMonitor


def build_managed_controller(config, settings=None):
    design = build_design(config)
    settings = settings or SolverSettings(**config['solver'])
    limit = bounded_scalar(config['constraints']['position_limit_deg'], 'position_limit_deg', positive=True)
    speed = bounded_scalar(config['constraints']['velocity_limit_rad_s'], 'velocity_limit_rad_s', positive=True)
    torque = bounded_scalar(config['actuator']['torque_limit_nm'], 'torque_limit_nm', positive=True)
    delta = bounded_scalar(config['actuator']['slew_rate_nm_s'], 'slew_rate_nm_s', positive=True) * config['mpc']['dt_s']
    controller = EngineeringMPCController(design['Ad'], design['Bd'], design['Q'], design['R'], design['P'],
                                          config['mpc']['horizon_steps'], input_rate_weight=config['experiment_fg'].get('input_rate_weight', .5),
                                          torque_bounds=(-torque, torque), delta_bounds=(-delta, delta),
                                          position_bounds=(-np.deg2rad(limit), np.deg2rad(limit)),
                                          velocity_mode='soft', velocity_bounds=(-speed, speed), slack_weight=config['experiment_fg'].get('slack_weight', 1e4),
                                          solver_settings=settings)
    return controller, design


def _faults(faults, cycles, dt, timing_mode):
    if faults is None:
        return {}
    if not isinstance(faults, dict):
        raise ValueError('faults must be a dictionary.')
    validated = {}
    for cycle, value in faults.items():
        if isinstance(cycle, (bool, np.bool_)) or not isinstance(cycle, Integral) or not 0 <= cycle < cycles:
            raise ValueError('Fault cycle must be a nonnegative in-range integer.')
        if not isinstance(value, dict) or value.get('kind') not in ('delay', 'max_iter', 'error', 'nonfinite'):
            raise ValueError('Unknown fault kind.')
        value = dict(value)
        kind = value['kind']
        allowed = {'kind', 'completion_s'} if kind == 'delay' else {'kind', 'max_iter'} if kind == 'max_iter' else {'kind'}
        if set(value) - allowed:
            raise ValueError('Unexpected fault option.')
        if kind == 'delay':
            if timing_mode != 'injected' or 'completion_s' not in value:
                raise ValueError('Delay faults require injected mode and completion_s.')
            value['completion_steps'] = integer_steps(bounded_scalar(value['completion_s'], 'completion_s'), dt, 'completion_s')
        elif kind == 'max_iter':
            value['max_iter'] = SolverSettings(max_iter=value.get('max_iter', 1)).max_iter
        validated[int(cycle)] = value
    return validated


def simulate_managed_closed_loop(config: dict, *, timing_mode='injected', faults=None, measurement_noise=None) -> dict:
    if timing_mode not in ('injected', 'measured_sync'):
        raise ValueError('timing_mode must be injected or measured_sync.')
    dt = bounded_scalar(config['plant']['dt_s'], 'plant.dt_s', positive=True)
    control_dt = bounded_scalar(config['mpc']['dt_s'], 'mpc.dt_s', positive=True)
    duration = bounded_scalar(config['experiment_fg']['duration_s'], 'duration_s', positive=True)
    count, stride = integer_steps(duration, dt, 'duration_s'), integer_steps(control_dt, dt, 'control_dt_s')
    budget = bounded_scalar(config['runtime']['budget_s'], 'budget_s', positive=True)
    budget_steps = integer_steps(budget, dt, 'budget_s')
    nominal_steps = integer_steps(bounded_scalar(config['runtime']['nominal_completion_s'], 'nominal_completion_s'), dt, 'nominal_completion_s')
    if count < 1 or stride < 1 or not 0 < budget_steps < stride:
        raise ValueError('Periods must be nonempty and 0 < budget < control period.')
    cycle_count = (count - 1) // stride + 1
    fault_table = _faults(faults, cycle_count, dt, timing_mode)
    controller, design = build_managed_controller(config)
    monitor = SolverMonitor(config['runtime']['recovery_successes'])
    fallback = LqrFallbackController(design['K'])
    actuator = TorqueActuator(dt_s=dt, **config['actuator'])
    plant = JointPlant(**config['nominal_model'], **config['plant'], initial_state=[0., 0.])
    position_limit = np.deg2rad(config['constraints']['position_limit_deg'])
    reference = np.array([np.deg2rad(finite_scalar(config['experiment_fg']['target_deg'], 'target_deg')), 0.])
    seed = config['experiment_fg']['seed']
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')
    std = bounded_scalar(config['measurement']['position_noise_std_deg'], 'position_noise_std_deg', positive=True)
    if measurement_noise is None:
        noise = np.random.default_rng(seed).normal(0., np.deg2rad(std), count)
    else:
        raw = np.asarray(measurement_noise, dtype=object)
        if raw.shape != (count,):
            raise ValueError('measurement_noise must match physical samples.')
        noise = np.array([finite_scalar(item, 'measurement_noise') for item in raw])
    cov = design['covariance']
    kf = DiscreteKalmanFilter(design['Ad_fast'], design['Bd_fast'], design['C'], cov['Gd'],
                             cov['Q_process'], cov['R_measurement'], [0., 0.], cov['P0'])
    fields = ('time_s', 'q_true_rad', 'dq_true_rad_s', 'q_hat_rad', 'dq_hat_rad_s', 'q_measured_rad',
              'measurement_noise_rad', 'q_ref_rad', 'torque_request_nm', 'torque_command_nm',
              'torque_actual_nm', 'delta_actual_nm', 'control_update', 'feedback_prev_nm', 'mode',
              'selected_source', 'decision_event', 'selected_request_nm', 'command_elapsed_s',
              'command_delta_nm', 'job_id', 'good_count', 'solver_status', 'solver_iterations',
              'solver_solve_time_s', 'solver_wall_time_s', 'guard_modified')
    rows, jobs, plans, events = [], [], [], []
    pending, command, request, source, last_command_index = None, 0., 0., 'initial_hold', 0
    used_faults = set()
    status, failure, state_time = 'complete', None, 0.

    for index in range(count):
        time = index * dt
        q, speed = plant.state
        feedback = actuator.torque_actual_nm
        if index:
            kf.predict(feedback)
        measurement = q + noise[index]
        estimate = kf.update(measurement)['x_hat']
        row = dict.fromkeys(fields, np.nan)
        row.update(time_s=time, q_true_rad=q, dq_true_rad_s=speed, q_hat_rad=estimate[0], dq_hat_rad_s=estimate[1],
                   q_measured_rad=measurement, measurement_noise_rad=noise[index], q_ref_rad=reference[0],
                   control_update=index % stride == 0, feedback_prev_nm=feedback, solver_status='')
        tick_events = []
        command_before_tick = command

        def record(event, job=None, **details):
            entry = {'time_s': time, 'event': event, 'job_id': job['job_id'] if job else None,
                     'mode_before': monitor.mode, 'mode_after': monitor.mode, 'good_count': monitor.good_count,
                     'recovered': False, 'selected_source': None, 'selected_request_nm': None,
                     'command_elapsed_s': None, 'command_delta_nm': None, **details}
            events.append(entry)
            tick_events.append(event)
            return entry

        def select(event, transition=None, job=None, assessment=None):
            nonlocal request, command, source, last_command_index
            can_use_mpc = (assessment is not None and assessment['accepted'] and monitor.mode == 'mpc')
            source = 'mpc' if can_use_mpc else 'fallback'
            request = (float(job['_result']['u0'][0]) if can_use_mpc else fallback.request(estimate, reference))
            elapsed = (index - last_command_index) * dt
            old = command
            if elapsed:
                command = guard_command(request, command, actuator.torque_limit_nm, actuator.slew_rate_nm_s, elapsed)
            last_command_index = index
            entry = record(event, job, **(transition or {}), selected_source=source, selected_request_nm=request,
                           command_elapsed_s=elapsed, command_delta_nm=command - old, torque_command_nm=command,
                           guard_modified=abs(request - command) > 1e-5)
            if np.isnan(row['command_elapsed_s']):
                row['command_elapsed_s'] = elapsed
            row.update(selected_request_nm=request, job_id=entry['job_id'], guard_modified=entry['guard_modified'])
            if can_use_mpc:
                job['executed'] = True  # Selected request before the shared guard, not necessarily identical actual torque.
                if job['_plan_index'] is not None:
                    plans[job['_plan_index']]['executed'] = True

        def complete(job):
            result = job['_result']
            elapsed = job['wall_time_s'] if timing_mode == 'measured_sync' else (index - job['_start_index']) * dt
            assessment = assess_result(result, elapsed, budget, control_dt)
            job.update(completed=True, completion_time_s=time if timing_mode == 'injected' else None,
                       processed_physical_time_s=time, completion_elapsed_s=elapsed,
                       accepted=assessment['accepted'], result_category=assessment['category'],
                       deadline_missed=job['deadline_missed'] or assessment['deadline_overrun'])
            row.update(solver_status=result['status'], solver_iterations=result.get('iterations'),
                       solver_solve_time_s=result.get('solve_time_s'), solver_wall_time_s=job['wall_time_s'])
            if job['budget_missed']:
                job['discarded'] = True
                record('late_result_discarded', job, reason='Expired at the budget event; no mode/recovery update')
            else:
                job['budget_missed'] = assessment['budget_overrun']
                job['discarded'] = not assessment['accepted']
                transition = monitor.observe(assessment)
                select('result_decision', transition, job, assessment)
            if job['_plan_index'] is not None:
                plans[job['_plan_index']]['discarded'] = job['discarded']

        try:
            if abs(q) > position_limit:
                status, failure = 'safety_stop', f'Position boundary exceeded at t={time:.6f}'
                rows.append(row)
                break
            # Complete old work before budget checks or a same-tick new control trigger.
            if pending is not None:
                if index - pending['_start_index'] > stride:
                    pending['deadline_missed'] = True
                if index >= pending['_finish_index']:
                    complete(pending)
                    pending = None
            if pending is not None and not pending['budget_missed'] and index >= pending['_budget_index']:
                pending.update(budget_missed=True, budget_event_time_s=time, discarded=True)
                transition = monitor.reject('budget_timeout', 'No completed result at the budget boundary')
                select('budget_timeout', transition, pending)
                if pending['_plan_index'] is not None:
                    plans[pending['_plan_index']]['discarded'] = True
            if index % stride == 0:
                cycle = index // stride
                if pending is not None:
                    transition = monitor.reject('busy', 'Single worker still occupied by an expired job')
                    select('busy_skip', transition, pending)
                else:
                    fault = fault_table.get(cycle, {})
                    if fault:
                        used_faults.add(cycle)
                    settings = (SolverSettings(config['solver']['warm_start'], fault['max_iter'])
                                if fault.get('kind') == 'max_iter' else controller.solver_settings)
                    started = perf_counter()
                    try:
                        if fault.get('kind') == 'error':
                            raise RuntimeError('Deterministic injected solver API error')
                        result = controller.solve(estimate, reference, feedback, solver_settings=settings)
                    except Exception as exc:
                        result = {'status': 'solver_error', 'error': str(exc), 'u0': None}
                    wall = perf_counter() - started
                    if fault.get('kind') == 'nonfinite' and result.get('u0') is not None:
                        result['u0'] = np.array([np.nan])
                    latency = fault.get('completion_steps', nominal_steps) if timing_mode == 'injected' else 0
                    candidate_valid = assess_result(result, 0., budget, control_dt)['accepted']
                    plan_index = None
                    if candidate_valid:
                        plan_index = len(plans)
                        plans.append({'job_id': len(jobs), 'time_s': time, 'u_prev': feedback,
                                      **{field: result[field].copy() for field in ('X', 'U', 'DU', 'EPS')},
                                      'velocity_indices': controller.velocity_indices, 'executed': False, 'discarded': False})
                    diagnostics = {key: value for key, value in result.items() if key not in ('X', 'U', 'DU', 'EPS', 'u0')}
                    pending = {'job_id': len(jobs), 'cycle_index': cycle, 'snapshot_time_s': time, 'start_time_s': time,
                               'snapshot_estimate': estimate.copy(), 'snapshot_reference': reference.copy(), 'u_prev': feedback,
                               'fault': dict(fault), 'solver_called': fault.get('kind') != 'error',
                               'solver_settings': {'warm_start': settings.warm_start, 'max_iter': settings.max_iter},
                               'wall_time_s': wall, 'injected_completion_s': latency * dt if timing_mode == 'injected' else None,
                               'planned_completion_time_s': (index + latency) * dt if timing_mode == 'injected' else None,
                               'completion_time_s': None, 'processed_physical_time_s': None, 'completion_elapsed_s': None,
                               'budget_event_time_s': None, 'completed': False, 'accepted': False, 'executed': False,
                               'discarded': False, 'budget_missed': False, 'deadline_missed': False,
                               'candidate_valid': candidate_valid, 'diagnostics': diagnostics,
                               '_result': result, '_start_index': index, '_finish_index': index + latency,
                               '_budget_index': index + budget_steps, '_plan_index': plan_index}
                    jobs.append(pending)
                    record('submit', pending)
                    if monitor.mode == 'fallback':
                        select('fallback_while_waiting', job=pending)
                    if latency == 0:
                        complete(pending)
                        pending = None
        except (ValueError, FloatingPointError) as exc:
            status, failure = 'safety_stop', f'Invalid control state/request at t={time:.6f}: {exc}'
            row.update(torque_request_nm=np.nan, torque_command_nm=np.nan, decision_event='|'.join(tick_events))
            rows.append(row)
            break
        output = actuator.step(command)
        row.update(torque_request_nm=request, torque_command_nm=command, torque_actual_nm=output['torque_actual_nm'],
                   delta_actual_nm=output['delta_actual_nm'], mode=monitor.mode, selected_source=source,
                   decision_event='|'.join(tick_events), good_count=monitor.good_count)
        if tick_events and np.isfinite(row['command_elapsed_s']):
            row['command_delta_nm'] = command - command_before_tick
        rows.append(row)
        next_state = plant.step(output['torque_actual_nm'])
        state_time = (index + 1) * dt
        if abs(next_state[0]) > position_limit:
            status, failure = 'safety_stop', f'Position boundary exceeded at t={state_time:.6f}'
            break
    # A final physical interval can cross a pending job's deadline without another
    # sampled tick. Preserve that observed overrun, but never invent completion.
    terminal_index = int(round(state_time / dt))
    for job in jobs:
        if not job['completed'] and terminal_index - job['_start_index'] > stride:
            job['deadline_missed'] = True
    # Private computed snapshots are not an API for executing a stale plan.
    public_jobs = [{key: value for key, value in job.items() if not key.startswith('_')} for job in jobs]
    return {'status': status, 'failure_reason': failure, 'timing_mode': timing_mode,
            'log': {field: np.array([row[field] for row in rows]) for field in fields},
            'jobs': public_jobs, 'plans': plans, 'events': events,
            'final_state': plant.state, 'final_state_time_s': state_time, 'expected_samples': count,
            'unapplied_fault_cycles': sorted(set(fault_table) - used_faults)}
