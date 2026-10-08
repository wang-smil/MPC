"""Open-loop demonstration: requested torque -> guarded command -> actual torque."""

import argparse
import csv
from pathlib import Path

import numpy as np

from lesson12_linear_mpc.src.model_loader import load_config
from ._validation import bounded_scalar, integer_steps
from .actuator import TorqueActuator
from .plant import JointPlant


def simulate_demo(config: dict) -> dict[str, np.ndarray]:
    plant_settings = config['plant']
    dt = bounded_scalar(plant_settings['dt_s'], 'plant.dt_s', positive=True)
    duration = bounded_scalar(config['demo']['duration_s'], 'duration_s', positive=True)
    control_dt = bounded_scalar(config['demo']['control_dt_s'], 'control_dt_s', positive=True)
    count = integer_steps(duration, dt, 'duration_s')
    stride = integer_steps(control_dt, dt, 'control_dt_s')
    if count < 1 or stride < 1:
        raise ValueError('duration and control period must contain at least one physical step.')
    plant = JointPlant(**config['nominal_model'], **plant_settings)
    actuator = TorqueActuator(dt_s=dt, **config['actuator'])
    fields = ('time_s', 'torque_request_nm', 'torque_command_nm', 'torque_actual_nm',
              'q_rad', 'dq_rad_s', 'delta_actual_nm')
    log = {name: np.empty(count, dtype=float) for name in fields}
    request = command = 0.
    max_command_delta = actuator.slew_rate_nm_s * control_dt
    for index in range(count):
        time = index * dt
        if index % stride == 0:
            request = 0. if time < .05 or time >= .35 else (1. if time < .20 else -1.)
            # Command guard runs at the CONTROL rate; the physical actuator runs at 1 ms.
            bounded_request = float(np.clip(request, -actuator.torque_limit_nm, actuator.torque_limit_nm))
            command += float(np.clip(bounded_request - command, -max_command_delta, max_command_delta))
        q, velocity = plant.state  # State at t_k, before applying input on [t_k,t_k+dt).
        output = actuator.step(command)
        values = (time, request, command, output['torque_actual_nm'], q, velocity,
                  output['delta_actual_nm'])
        for name, value in zip(fields, values):
            log[name][index] = value
        plant.step(output['torque_actual_nm'])
    return log


def write_outputs(log: dict[str, np.ndarray], output_dir: Path,
                  dt_s: float, slew_rate_nm_s: float) -> dict[str, Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    log_dir, figure_dir = output_dir / 'logs', output_dir / 'figures'
    log_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / 'plant_actuator_demo.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(log.keys())
        writer.writerows(zip(*log.values()))

    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, layout='constrained')
    time = log['time_s']
    axes[0].step(time, log['torque_request_nm'], where='post', color='#374151',
                 linestyle='--', label='Requested torque', linewidth=1.5)
    axes[0].step(time, log['torque_command_nm'], where='post', color='#d18a00',
                 linestyle='-.', label='Guarded command', linewidth=1.6)
    axes[0].step(time, log['torque_actual_nm'], where='post', color='#2171b5',
                 label='Actual interval torque', linewidth=1.8)
    axes[0].set_ylabel('Torque / N m')
    axes[0].legend(loc='upper right', ncol=3, fontsize=9)
    axes[1].step(time, log['delta_actual_nm'] / dt_s, where='post',
                 color='#2171b5', linewidth=1.5, label='Actual torque slew')
    for boundary in (-slew_rate_nm_s, slew_rate_nm_s):
        axes[1].axhline(boundary, color='#374151', linestyle='--', linewidth=1.)
    axes[1].set_ylabel('Torque slew / N m/s')
    axes[2].plot(time, np.rad2deg(log['q_rad']), color='#2171b5', linewidth=1.6)
    axes[2].set_ylabel('Position / deg')
    axes[3].plot(time, log['dq_rad_s'], color='#2171b5', linewidth=1.6)
    axes[3].set_ylabel('Velocity / rad/s')
    axes[3].set_xlabel('Time / s')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(time[0], time[-1])
    fig.suptitle('Open-loop joint and actuator response (not position control)')
    figure_path = figure_dir / 'plant_actuator_demo.png'
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return {'log': csv_path, 'figure': figure_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    lesson_dir = Path(__file__).resolve().parents[1]
    config = load_config(args.config or lesson_dir / 'config' / 'mpc_engineering.yaml')
    log = simulate_demo(config)
    paths = write_outputs(log, lesson_dir, config['plant']['dt_s'], config['actuator']['slew_rate_nm_s'])
    print(f"Peak actual torque: {np.max(np.abs(log['torque_actual_nm'])):.6f} N m")
    print(f"Peak actual slew: {np.max(np.abs(log['delta_actual_nm'])) / config['plant']['dt_s']:.6f} N m/s")
    print(f"Peak request/actual difference: {np.max(np.abs(log['torque_request_nm'] - log['torque_actual_nm'])):.6f} N m")
    for name, path in paths.items():
        print(f'{name}: {path}')


if __name__ == '__main__':
    main()
