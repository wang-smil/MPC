import csv
from pathlib import Path
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def cubic_reference(
    time_s: float,
    start_rad: float,
    target_rad: float,
    duration_s: float,
) -> tuple[float, float]:
    """生成端点速度为零的三次多项式位置与速度参考。"""

    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")

    s = float(np.clip(time_s / duration_s, 0.0, 1.0))
    position_delta = target_rad - start_rad

    position = start_rad + position_delta * (
        3.0 * s**2 - 2.0 * s**3
    )
    velocity = position_delta / duration_s * (
        6.0 * s - 6.0 * s**2
    )

    return float(position), float(velocity)


def build_time_axis(
    nominal_dt: float,
    duration: float,
    rng: np.random.Generator,
    jitter_std_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """生成仿真时间轴及每次状态更新实际使用的周期。"""

    if nominal_dt <= 0.0:
        raise ValueError("nominal_dt must be positive")
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if jitter_std_ratio < 0.0:
        raise ValueError("jitter_std_ratio cannot be negative")

    if jitter_std_ratio == 0.0:
        step_count = int(round(duration / nominal_dt))
        time = np.arange(step_count + 1, dtype=float) * nominal_dt
        actual_dt = np.full(len(time), nominal_dt, dtype=float)
        return time, actual_dt

    time_values = [0.0]
    applied_periods = []

    while time_values[-1] < duration:
        actual_dt = nominal_dt + rng.normal(
            loc=0.0,
            scale=jitter_std_ratio * nominal_dt,
        )
        actual_dt = float(
            np.clip(
                actual_dt,
                0.8 * nominal_dt,
                1.2 * nominal_dt,
            )
        )

        applied_periods.append(actual_dt)
        time_values.append(time_values[-1] + actual_dt)

    time = np.asarray(time_values, dtype=float)
    actual_dt_log = np.asarray(
        applied_periods + [applied_periods[-1]],
        dtype=float,
    )

    return time, actual_dt_log


def load_config() -> dict:
    """读取单轴伺服系统的 YAML 参数。"""

    config_path = ROOT / "config" / "servo.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config

def calculate_metrics(
    time: np.ndarray,
    position: np.ndarray,
    target: float,
    torque: np.ndarray,
    saturated: np.ndarray,
) -> dict[str, float]:
    """根据一次仿真结果计算控制性能指标。"""

    error = target - position
    final_error = abs(error[-1])

    if abs(target) > 1e-9:
        overshoot = max(
            0.0,
            (np.max(position) - target)
            / abs(target)
            * 100.0,
        )
    else:
        overshoot = 0.0

    # 允许误差：0.5度与目标值2%中的较大者
    tolerance = max(
        np.deg2rad(0.5),
        0.02 * abs(target),
    )

    settling_time = np.nan

    for index in range(len(time)):
        remaining_error = np.abs(error[index:])

        if np.all(remaining_error <= tolerance):
            settling_time = time[index]
            break

    return {
        "final_error_deg": float(
            np.rad2deg(final_error)
        ),
        "overshoot_percent": float(overshoot),
        "settling_time_s": float(settling_time),
        "max_torque_nm": float(
            np.max(np.abs(torque))
        ),
        "rms_torque_nm": float(
            np.sqrt(np.mean(torque**2))
        ),
        "saturation_ratio_percent": float(
            np.mean(saturated) * 100.0
        ),
    }


def simulate(config: dict, scenario: str) -> dict:
    """运行一种工况下的单关节闭环仿真。"""

    sim = config["simulation"]
    plant = config["plant"]
    controller = config["controller"]
    reference = config["reference"]
    actuator = config["actuator"]
    sensor = config["sensor"]
    safety = config["safety"]

    nominal_dt = float(sim["dt"])
    duration = float(sim["duration"])
    seed = int(sim["seed"])
    jitter_std_ratio = float(sim.get("jitter_std_ratio", 0.0))
    rng = np.random.default_rng(seed)
    timing_rng = np.random.default_rng(seed + 1)

    inertia = float(plant["inertia"])
    damping = float(plant["damping"])
    coulomb_friction = float(plant["coulomb_friction"])
    friction_smoothing = float(plant["friction_smoothing"])

    kp = float(controller["kp"])
    kd = float(controller["kd"])

    reference_type = str(reference["type"]).lower()

    if reference_type not in {"cubic", "step"}:
        raise ValueError(
            "reference.type must be 'cubic' or 'step'"
        )

    target = np.deg2rad(float(reference["target_deg"]))
    move_duration = float(reference["move_duration_s"])

    if move_duration <= 0.0:
        raise ValueError("reference.move_duration_s must be positive")

    torque_limit = float(actuator["torque_limit"])
    delay_steps = max(0, int(actuator["delay_steps"]))

    noise_std = np.deg2rad(float(sensor["position_noise_std_deg"]))
    alpha = float(sensor["velocity_filter_alpha"])

    position_limit = np.deg2rad(float(safety["position_limit_deg"]))
    velocity_limit = np.deg2rad(float(safety["velocity_limit_deg_s"]))

    time, actual_dt = build_time_axis(
        nominal_dt=nominal_dt,
        duration=duration,
        rng=timing_rng,
        jitter_std_ratio=jitter_std_ratio,
    )
    count = len(time)

    q = np.zeros(count)
    dq = np.zeros(count)
    q_measured = np.zeros(count)
    dq_estimated = np.zeros(count)
    torque_command = np.zeros(count)
    torque_applied = np.zeros(count)
    load_torque = np.zeros(count)
    saturated = np.zeros(count, dtype=bool)
    safety_fault = np.zeros(count, dtype=bool)
    target_position = np.zeros(count)
    target_velocity = np.zeros(count)

    command_delay = deque(
        [0.0] * (delay_steps + 1),
        maxlen=delay_steps + 1,
    )
    for k in range(count - 1):
        current_noise_std = noise_std

        if reference_type == "cubic":
            q_ref, dq_ref = cubic_reference(
                time_s=time[k],
                start_rad=0.0,
                target_rad=target,
                duration_s=move_duration,
            )
        else:
            q_ref = target
            dq_ref = 0.0

        target_position[k] = q_ref
        target_velocity[k] = dq_ref

        # 扰动工况：1.5～2.2秒施加0.8 N·m外部负载
        if scenario == "disturbance" and 1.5 <= time[k] <= 2.2:
            load_torque[k] = 0.8

        # 传感器故障：2秒后将编码器噪声放大20倍
        if scenario == "sensor_fault" and time[k] >= 2.0:
            current_noise_std *= 20.0

        # 编码器只能测量角度，并且测量中包含噪声
        q_measured[k] = q[k] + rng.normal(
            loc=0.0,
            scale=current_noise_std,
        )

        # 用相邻两次角度测量值做差分，估算角速度
        if k > 0:
            raw_velocity = (
                q_measured[k] - q_measured[k - 1]
            ) / actual_dt[k - 1]

            # 一阶低通滤波，削弱差分放大的高频噪声
            dq_estimated[k] = (
                alpha * dq_estimated[k - 1]
                + (1.0 - alpha) * raw_velocity
            )

        # PD位置与速度跟踪控制器
        position_error = q_ref - q_measured[k]
        velocity_error = dq_ref - dq_estimated[k]

        torque_command[k] = (
            kp * position_error
            + kd * velocity_error
        )
        # 执行器转矩限幅
        torque_limited = np.clip(
            torque_command[k],
            -torque_limit,
            torque_limit,
        )

        saturated[k] = not np.isclose(
            torque_command[k],
            torque_limited,
        )

        # 模拟控制命令经过驱动器或通信产生的离散延迟
        command_delay.append(float(torque_limited))
        delayed_torque = command_delay[0]

        # 安全检查
        unsafe = (
            abs(q_measured[k]) > position_limit
            or abs(dq_estimated[k]) > velocity_limit
        )

        if unsafe:
            safety_fault[k] = True
            delayed_torque = 0.0

        torque_applied[k] = delayed_torque
        # 使用tanh平滑近似库仑摩擦
        friction = coulomb_friction * np.tanh(
            dq[k] / friction_smoothing
        )

        # 根据关节动力学计算角加速度
        ddq = (
            torque_applied[k]
            - load_torque[k]
            - damping * dq[k]
            - friction
        ) / inertia

        # 半隐式欧拉积分：先更新速度，再用新速度更新位置
        dq[k + 1] = dq[k] + ddq * actual_dt[k]
        q[k + 1] = q[k] + dq[k + 1] * actual_dt[k]
    # 循环只计算到倒数第二个采样点，
    # 因此把最后一个可用值复制到数组末尾
    q_measured[-1] = q[-1]
    dq_estimated[-1] = dq_estimated[-2]
    torque_command[-1] = torque_command[-2]
    torque_applied[-1] = torque_applied[-2]
    load_torque[-1] = load_torque[-2]
    saturated[-1] = saturated[-2]
    safety_fault[-1] = safety_fault[-2]

    if reference_type == "cubic":
        target_position[-1], target_velocity[-1] = cubic_reference(
            time_s=time[-1],
            start_rad=0.0,
            target_rad=target,
            duration_s=move_duration,
        )
    else:
        target_position[-1] = target
        target_velocity[-1] = 0.0

    data = {
        "time": time,
        "actual_dt": actual_dt,
        "position": q,
        "velocity": dq,
        "position_measured": q_measured,
        "velocity_estimated": dq_estimated,
        "torque_command": torque_command,
        "torque_applied": torque_applied,
        "load_torque": load_torque,
        "saturated": saturated,
        "safety_fault": safety_fault,
        "target": target_position,
        "target_velocity": target_velocity,
    }

    return data


def save_log(data: dict, scenario: str) -> Path:
    """把一次仿真的所有时间序列保存为 CSV。"""

    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{scenario}.csv"

    keys = [
        "time",
        "actual_dt",
        "position",
        "velocity",
        "position_measured",
        "velocity_estimated",
        "torque_command",
        "torque_applied",
        "load_torque",
        "saturated",
        "safety_fault",
        "target",
        "target_velocity",
    ]

    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(keys)
        writer.writerows(zip(*(data[key] for key in keys)))

    return log_path


def plot_result(data: dict, scenario: str) -> Path:
    """绘制位置、速度和转矩，并保存为 PNG。"""

    figure_dir = ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / f"{scenario}.png"

    time = data["time"]
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10, 9),
        sharex=True,
    )

    axes[0].plot(
        time,
        np.rad2deg(data["target"]),
        label="Target",
    )
    axes[0].plot(
        time,
        np.rad2deg(data["position"]),
        label="Position",
    )
    axes[0].plot(
        time,
        np.rad2deg(data["position_measured"]),
        linewidth=0.8,
        alpha=0.6,
        label="Encoder",
    )
    axes[0].set_ylabel("Position / deg")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        time,
        np.rad2deg(data["target_velocity"]),
        label="Target velocity",
    )
    axes[1].plot(
        time,
        np.rad2deg(data["velocity"]),
        label="True velocity",
    )
    axes[1].plot(
        time,
        np.rad2deg(data["velocity_estimated"]),
        linewidth=0.8,
        alpha=0.7,
        label="Estimated velocity",
    )
    axes[1].set_ylabel("Velocity / deg/s")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(
        time,
        data["torque_command"],
        label="Command",
    )
    axes[2].plot(
        time,
        data["torque_applied"],
        label="Applied",
    )
    axes[2].plot(
        time,
        data["load_torque"],
        label="Load",
    )
    axes[2].set_xlabel("Time / s")
    axes[2].set_ylabel("Torque / N·m")
    axes[2].grid(True)
    axes[2].legend()

    figure.suptitle(f"Single Joint Servo — {scenario}")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    return figure_path


def main() -> None:
    config = load_config()

    scenarios = [
        "normal",
        "disturbance",
        "sensor_fault",
    ]

    for scenario in scenarios:
        data = simulate(config, scenario=scenario)
        log_path = save_log(data, scenario)
        figure_path = plot_result(data, scenario)

        metrics = calculate_metrics(
            time=data["time"],
            position=data["position"],
            target=data["target"][-1],
            torque=data["torque_applied"],
            saturated=data["saturated"],
        )

        metrics["safety_fault_count"] = int(
            np.sum(data["safety_fault"])
        )

        print(f"\n===== {scenario} =====")

        for name, value in metrics.items():
            print(f"{name}: {value}")

        print(f"log: {log_path}")
        print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
