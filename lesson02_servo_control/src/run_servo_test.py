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
    actuator = config["actuator"]
    sensor = config["sensor"]
    safety = config["safety"]

    dt = float(sim["dt"])
    duration = float(sim["duration"])
    rng = np.random.default_rng(int(sim["seed"]))

    inertia = float(plant["inertia"])
    damping = float(plant["damping"])
    coulomb_friction = float(plant["coulomb_friction"])
    friction_smoothing = float(plant["friction_smoothing"])

    kp = float(controller["kp"])
    kd = float(controller["kd"])
    target = np.deg2rad(float(controller["target_deg"]))

    torque_limit = float(actuator["torque_limit"])
    delay_steps = max(0, int(actuator["delay_steps"]))

    noise_std = np.deg2rad(float(sensor["position_noise_std_deg"]))
    alpha = float(sensor["velocity_filter_alpha"])

    position_limit = np.deg2rad(float(safety["position_limit_deg"]))
    velocity_limit = np.deg2rad(float(safety["velocity_limit_deg_s"]))

    step_count = int(round(duration / dt))
    time = np.arange(step_count + 1, dtype=float) * dt
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

    command_delay = deque(
        [0.0] * (delay_steps + 1),
        maxlen=delay_steps + 1,
    )
    for k in range(count - 1):
        current_noise_std = noise_std

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
            ) / dt

            # 一阶低通滤波，削弱差分放大的高频噪声
            dq_estimated[k] = (
                alpha * dq_estimated[k - 1]
                + (1.0 - alpha) * raw_velocity
            )

        # PD位置控制器
        position_error = target - q_measured[k]

        torque_command[k] = (
            kp * position_error
            - kd * dq_estimated[k]
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
        dq[k + 1] = dq[k] + ddq * dt
        q[k + 1] = q[k] + dq[k + 1] * dt
    # 循环只计算到倒数第二个采样点，
    # 因此把最后一个可用值复制到数组末尾
    q_measured[-1] = q[-1]
    dq_estimated[-1] = dq_estimated[-2]
    torque_command[-1] = torque_command[-2]
    torque_applied[-1] = torque_applied[-2]
    load_torque[-1] = load_torque[-2]
    saturated[-1] = saturated[-2]
    safety_fault[-1] = safety_fault[-2]

    data = {
        "time": time,
        "position": q,
        "velocity": dq,
        "position_measured": q_measured,
        "velocity_estimated": dq_estimated,
        "torque_command": torque_command,
        "torque_applied": torque_applied,
        "load_torque": load_torque,
        "saturated": saturated,
        "safety_fault": safety_fault,
        "target": np.full(count, target),
    }

    return data


def save_log(data: dict, scenario: str) -> Path:
    """把一次仿真的所有时间序列保存为 CSV。"""

    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{scenario}.csv"

    keys = [
        "time",
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
            target=data["target"][0],
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
