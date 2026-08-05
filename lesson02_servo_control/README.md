# 第二课：单轴机器人关节闭环与工程验收

## 目标

将第一课的二阶状态空间基础升级为单轴机器人关节闭环仿真。系统从 0° 运动到 30°，并考虑：

- 编码器噪声；
- 角度差分速度估计；
- 速度低通滤波；
- PD 位置与速度跟踪；
- 执行器转矩限幅；
- 离散命令延迟；
- 摩擦和外部负载；
- 位置、速度安全限制；
- 采样周期与轻度周期抖动；
- CSV 日志、响应图、性能指标和自动测试。

## 核心模型

关节动力学：

```text
J·ddq = torque_applied - torque_load - b·dq - torque_friction
```

PD 跟踪控制器：

```text
torque_command = Kp·(q_ref - q_measured)
               + Kd·(dq_ref - dq_estimated)
```

执行器将转矩命令限制在 `[-torque_limit, torque_limit]`，再加入离散延迟并执行位置、速度安全检查。

## 为什么 normal 不再使用位置阶跃

位置阶跃会在零时刻把误差直接变为 30°。控制器立即提出大转矩，执行器进入饱和，关节峰值速度达到约 326.44 deg/s，超过当前 300 deg/s 安全限制。

阶跃适合压力测试，不适合作为机器人的正常运动指令。默认 `normal` 现采用 0.8 s 三次多项式轨迹：

```text
s = clip(t / T, 0, 1)

q_ref  = q0 + (qf - q0)(3s² - 2s³)
dq_ref = (qf - q0) / T · (6s - 6s²)
```

该轨迹满足起点和终点目标速度均为零。使用相同控制器后，平滑轨迹峰值速度约为 58.18 deg/s，正常工况安全触发降为 0。

## 控制循环

每个控制周期依次执行：

1. 计算当前三次位置与速度参考；
2. 注入当前独立测试工况；
3. 读取带噪声的编码器角度；
4. 使用实际采样周期对角度差分；
5. 低通滤波得到估计角速度；
6. 计算 PD 转矩命令；
7. 执行转矩限幅和离散延迟；
8. 检查位置与速度安全限制；
9. 计算摩擦和角加速度；
10. 使用实际周期进行半隐式欧拉积分；
11. 保存本周期完整数据。

## 默认测试工况

- `normal`：平滑轨迹和正常编码器噪声；
- `disturbance`：1.5～2.2 s 施加 0.8 N·m 外部负载；
- `sensor_fault`：2 s 后将编码器噪声标准差放大 20 倍。

三种故障不会同时注入，便于判断性能退化的来源。

当前默认配置结果：

| 工况 | 最终误差/deg | 超调/% | 调节时间/s | RMS转矩/N·m | 饱和/% | 安全采样点 |
|---|---:|---:|---:|---:|---:|---:|
| normal | 0.0036 | 0.0238 | 0.762 | 0.1489 | 0.0000 | 0 |
| disturbance | 0.0011 | 0.0029 | 2.354 | 0.3428 | 0.0000 | 0 |
| sensor_fault | 0.3463 | 2.2886 | 3.933 | 1.4068 | 11.2722 | 6 |

平滑轨迹解决了正常运动中的初始超速，但传感器故障仍会造成速度估计和转矩命令抖动，因此故障工况不会被正常轨迹掩盖。

## 时序验收实验

`timing_acceptance.yaml` 定义五组彼此独立的实验：

| 工况 | 名义周期 | 延迟周期 | 名义延迟 | 周期抖动 |
|---|---:|---:|---:|---:|
| acceptance_normal | 1 ms | 0 | 0 ms | 无 |
| delay_1step | 1 ms | 1 | 1 ms | 无 |
| delay_5steps | 1 ms | 5 | 5 ms | 无 |
| sample_5ms | 5 ms | 1 | 5 ms | 无 |
| jitter | 1 ms | 0 | 0 ms | 5%标准差 |

轻度抖动周期按照以下方式生成：

```text
actual_dt = nominal_dt + normal(0, 0.05·nominal_dt)
actual_dt = clip(actual_dt, 0.8·nominal_dt, 1.2·nominal_dt)
```

速度差分使用上一段实际周期，状态积分使用当前更新段实际周期。传感器噪声和周期抖动使用独立随机数流，使时序对比不被不同噪声样本混淆。

## normal 阶段性验收标准

| 指标 | 要求 | 当前结果 |
|---|---:|---:|
| 安全采样点 | 0 | 0 |
| 最终位置误差 | <0.5° | 0.0040° |
| 超调 | <5% | 0.0234% |
| 最大速度 | <300 deg/s | 58.1813 deg/s |
| 饱和比例 | <10% | 0% |
| 调节时间 | <1.5 s | 0.762 s |

当前 `acceptance_normal` 六项全部通过。这些数值仅为本教学项目的阶段性要求，不代表通用工业标准。

## 时序实验结论

- 1 ms 的 0、1、5 周期延迟在当前平滑轨迹和增益下都保持稳定；
- 5 ms 采样的调节时间增加到约 0.895 s，最大轨迹误差增加到约 0.841°；
- `delay_5steps` 与 `sample_5ms` 都有 5 ms 名义延迟，但后者传感器和控制器也只以 200 Hz 更新，二者不能视为等价；
- 5% 抖动的平均周期约 0.9998 ms，最大周期约 1.1662 ms，标准差约 0.0498 ms；
- 当前轻度抖动只造成小幅性能变化，不能据此推断系统能够承受任意抖动或长尾延迟。

## 指标定义

- `final_error_deg`：仿真结束时的真实位置误差；
- `overshoot_percent`：真实位置超过最终目标的最大比例；
- `settling_time_s`：此后始终保持在最终目标容差内的最早时间；
- `rms_torque_nm`：实际转矩的均方根；
- `saturation_ratio_percent`：命令转矩被限幅的采样点比例；
- `safety_fault_count`：超出安全限制的采样点数，不是独立事故次数；
- `mean_dt_ms`、`max_dt_ms`、`jitter_std_ms`：实际周期统计；
- `max_velocity_deg_s`：真实角速度绝对值峰值；
- `max_position_error_deg`：随时间变化参考下的全程最大跟踪误差。

## 运行

激活环境并进入仓库：

```powershell
conda activate robot-control
cd D:\MPC_learn
```

运行默认三工况：

```powershell
python .\lesson02_servo_control\src\run_servo_test.py
```

运行工程时序验收：

```powershell
python .\lesson02_servo_control\src\run_timing_acceptance.py
```

运行自动测试：

```powershell
python -m unittest discover -s .\lesson02_servo_control\tests -p "test_*.py" -v
```

## 输出

- `logs/normal.csv`、`disturbance.csv`、`sensor_fault.csv`：默认三工况日志；
- `logs/acceptance_normal.csv` 等：五组时序验收日志；
- `figures/normal.png` 等：默认三工况响应图；
- `figures/step_vs_cubic.png`：阶跃和平滑轨迹对比；
- `figures/timing_comparison.png`：采样、延迟和抖动对比；
- `reports/engineering_acceptance.md`：自动生成的工程验收报告。

## 硬实时边界

本项目完成的是控制算法和非理想时序因素的离线数值仿真验证。它不能证明：

- Windows 或 Python 控制线程能够稳定以 1 kHz 运行；
- 每个周期都能在 1 ms 截止时间内完成；
- USB、串口或 CAN 通信延迟具有确定上界；
- 操作系统调度、垃圾回收和日志写入不会产生长尾延迟。

因此目前正确结论是：

> 本项目验证了控制逻辑在设定采样、延迟和轻度抖动模型下的数值表现，但尚未完成硬实时操作系统和真实硬件平台上的确定性验证。
