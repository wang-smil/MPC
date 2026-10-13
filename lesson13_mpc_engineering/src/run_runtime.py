"""Experiment F: actual wall time. Experiment G: deterministic completion events."""

from copy import deepcopy
from numbers import Integral
from pathlib import Path
import platform
from time import perf_counter_ns

import cvxpy
import numpy as np
import osqp

from ._validation import bounded_scalar, integer_steps
from .managed_closed_loop import build_managed_controller, simulate_managed_closed_loop
from .qp_solver import SolverSettings
from .runtime_metrics import calculate_managed_metrics, calculate_runtime_metrics
from .solver_monitor import assess_result


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer.')
    return int(value)


def make_replay_sequence(config: dict, count: int) -> dict:
    count = _positive_int(count, 'replay_samples')
    time = np.arange(count) * bounded_scalar(config['mpc']['dt_s'], 'mpc.dt_s', positive=True)
    target = float(config['experiment_fg']['target_deg'])
    if not np.isfinite(target):
        raise ValueError('target_deg must be finite.')
    return {'time_s': time, 'x_hat': np.column_stack((.1 * np.sin(np.pi * time), .1 * np.pi * np.cos(np.pi * time))),
            'x_ref': np.tile([np.deg2rad(target), 0.], (count, 1)), 'u_prev': .1 * np.cos(.5 * np.pi * time)}


def benchmark_solver(config, sequence):
    count = len(sequence['time_s'])
    _positive_int(count, 'sequence length')
    for key, shape in (('time_s', (count,)), ('x_hat', (count, 2)), ('x_ref', (count, 2)), ('u_prev', (count,))):
        if np.shape(sequence[key]) != shape or not np.all(np.isfinite(sequence[key])):
            raise ValueError(f'Invalid replay {key}.')
    repetitions = _positive_int(config['experiment_fg']['repetitions'], 'repetitions')
    rows, constructions = [], []
    budget, deadline = config['runtime']['budget_s'], config['mpc']['dt_s']
    for repetition in range(repetitions):
        order = (True, False) if repetition % 2 == 0 else (False, True)
        for warm in order:
            mode = 'warm_on' if warm else 'warm_off'
            settings = SolverSettings(warm_start=warm, max_iter=config['solver']['max_iter'])
            start = perf_counter_ns()
            controller, _ = build_managed_controller(config, settings)
            construction = (perf_counter_ns() - start) * 1e-9
            constructions.append({'mode': mode, 'repetition': repetition, 'construction_time_s': construction})
            for index in range(count):
                start = perf_counter_ns()
                result = controller.solve(sequence['x_hat'][index], sequence['x_ref'][index], sequence['u_prev'][index])
                wall = (perf_counter_ns() - start) * 1e-9
                numerical = assess_result(result, 0., budget, deadline)
                row = {'mode': mode, 'repetition': repetition, 'sample_index': index, 'cold': index == 0,
                       'problem_series': f'{mode}_r{repetition}', 'construction_time_s': construction,
                       'wall_time_s': wall, 'candidate_valid': numerical['accepted'],
                       'numerical_category': numerical['category'], 'budget_missed': wall > budget,
                       'deadline_missed': wall > deadline}
                row.update({key: value for key, value in result.items() if key not in ('u0', 'X', 'U', 'DU', 'EPS')})
                rows.append(row)
    return {'rows': rows, 'constructions': constructions,
            'metrics': calculate_runtime_metrics(rows, budget, deadline)}


def _default_fault_scenarios():
    return {'g_nominal': {},
            'g_iteration_limit': {i: {'kind': 'max_iter', 'max_iter': 1} for i in range(3)},
            'g_deadline': {100: {'kind': 'delay', 'completion_s': .007},
                           101: {'kind': 'delay', 'completion_s': .012}},
            'g_error_recovery': {100: {'kind': 'error'}, 102: {'kind': 'nonfinite'}}}


def run_runtime_experiments(config: dict, output_dir: Path) -> dict:
    # Never mutate the caller's configuration or consume a different noise stream per method.
    effective = deepcopy(config)
    effective['experiment_fg'].setdefault('input_rate_weight', .5)
    effective['experiment_fg'].setdefault('slack_weight', 1e4)
    fg = effective['experiment_fg']
    dt = bounded_scalar(effective['plant']['dt_s'], 'plant.dt_s', positive=True)
    count = integer_steps(bounded_scalar(fg['duration_s'], 'duration_s', positive=True), dt, 'duration_s')
    # Validate seed as NumPy accepts booleans, but course configs must not.
    seed = fg['seed']
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')
    scenarios = deepcopy(fg.get('fault_scenarios', _default_fault_scenarios()))
    if not isinstance(scenarios, dict) or set(scenarios) != set(_default_fault_scenarios()):
        raise ValueError('fault_scenarios must explicitly contain all four G scenario names.')
    sequence = make_replay_sequence(effective, fg['replay_samples'])
    benchmark = benchmark_solver(effective, sequence)
    std = bounded_scalar(effective['measurement']['position_noise_std_deg'], 'position_noise_std_deg', positive=True)
    noise = np.random.default_rng(seed).normal(0., np.deg2rad(std), count)
    results = {'f_measured_sync': simulate_managed_closed_loop(effective, timing_mode='measured_sync', measurement_noise=noise)}
    for name, faults in scenarios.items():
        results[name] = simulate_managed_closed_loop(effective, faults=faults, measurement_noise=noise)
    summary = {'config': effective, 'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                                                'cvxpy': cvxpy.__version__, 'osqp': osqp.__version__},
               'scope': 'F/G only; measured wall time is not hard realtime; injected completion times are synthetic; comprehensive multi-seed acceptance remains pending',
               'fault_scenarios': scenarios, 'benchmark': {'metrics': benchmark['metrics'],
                                                         'constructions': benchmark['constructions']}, 'cases': {}}
    for name, result in results.items():
        summary['cases'][name] = {'status': result['status'], 'failure_reason': result['failure_reason'],
                                  'timing_mode': result['timing_mode'], 'unapplied_fault_cycles': result['unapplied_fault_cycles'],
                                  'expected_samples': result['expected_samples'],
                                  'final_state': {'time_s': result['final_state_time_s'],
                                                  'position_rad': float(result['final_state'][0]),
                                                  'velocity_rad_s': float(result['final_state'][1])},
                                  'metrics': calculate_managed_metrics(result, effective)}
    # Author artifacts after all timing runs, not in the measured loop.
    from .runtime_artifacts import export_runtime_artifacts
    return export_runtime_artifacts(summary, sequence, benchmark, results, Path(output_dir))
