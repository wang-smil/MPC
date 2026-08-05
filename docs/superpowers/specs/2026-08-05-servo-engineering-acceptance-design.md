# 单轴伺服工程验收与时序补丁设计

## 目标

在保留第二课现有动力学、传感器、执行器、安全保护、日志和指标框架的基础上：

1. 使用三次多项式参考轨迹，使 `normal` 成为零安全触发的基准工况；
2. 对采样周期、命令延迟和轻度周期抖动进行最小对比实验；
3. 生成可复核的 CSV、图片与工程验收报告；
4. 明确离线数值仿真不等于 Windows 或实机硬实时验证。

## 范围

本次修改 `lesson02_servo_control`，不实现新的控制器，不进入 MPC，不重写现有动力学模型，也不自动推送 GitHub。

现有 `run_servo_test.py` 继续作为唯一仿真核心；新验收脚本只负责组织实验、汇总数据和生成产物，不复制控制循环。

## 参考轨迹

增加：

```python
def cubic_reference(
    time_s: float,
    start_rad: float,
    target_rad: float,
    duration_s: float,
) -> tuple[float, float]:
```

归一化时间为：

```text
s = clip(time_s / duration_s, 0, 1)
```

位置和速度为：

```text
q_ref  = q0 + (qf - q0)(3s² - 2s³)
dq_ref = (qf - q0) / T · (6s - 6s²)
```

函数在 `duration_s <= 0` 时抛出 `ValueError`。轨迹在起点和终点速度均为零，超过运动时间后保持目标位置和零目标速度。

配置从控制器段中分离参考指令：

```yaml
reference:
  type: cubic
  target_deg: 30.0
  move_duration_s: 0.8
```

PD 控制律改为：

```text
torque = Kp(q_ref - q_measured) + Kd(dq_ref - dq_estimated)
```

数据日志同时保存随时间变化的 `target` 和 `target_velocity`。

## 仿真时序

仿真继续使用配置中的名义周期 `nominal_dt`，同时为每个更新周期生成 `actual_dt`：

- 无抖动：`actual_dt = nominal_dt`；
- 轻度抖动：加入标准差为名义周期 5% 的高斯扰动，再限制在名义周期的 80%～120%。

速度差分、动力学积分和时间轴推进均使用同一个 `actual_dt`，避免只改变日志而未改变实际数值计算。延迟队列仍以周期数定义，等效延迟在报告中按 `delay_steps × nominal_dt` 给出。

由于实际周期会变化，抖动实验的仿真时间轴采用逐步累积方式，并在达到配置时长后结束。无抖动工况仍保持现有等间隔时间轴和终点语义。

## 验收实验

新增独立验收配置 `config/timing_acceptance.yaml` 和脚本 `src/run_timing_acceptance.py`。脚本复用仿真核心，执行：

| 名称 | 名义周期 | 延迟周期 | 名义等效延迟 | 抖动 |
|---|---:|---:|---:|---:|
| acceptance_normal | 1 ms | 0 | 0 ms | 无 |
| delay_1step | 1 ms | 1 | 1 ms | 无 |
| delay_5steps | 1 ms | 5 | 5 ms | 无 |
| sample_5ms | 5 ms | 1 | 5 ms | 无 |
| jitter | 1 ms | 0 | 0 ms | 5%标准差 |

此外生成阶跃与三次轨迹对比数据和 `step_vs_cubic.png`，但默认 `normal` 使用三次轨迹。阶跃只作为压力测试，不作为正常工况。

每组实验分别写入命名 CSV，避免覆盖现有三工况日志。

## 指标与验收

保留现有最终误差、超调、调节时间、最大转矩、RMS 转矩和饱和比例，并增加：

- `max_velocity_deg_s`：真实角速度绝对值峰值；
- `max_position_error_deg`：全程最大真实位置误差；
- `mean_dt_ms`：实际周期均值；
- `max_dt_ms`：实际周期最大值；
- `jitter_std_ms`：实际周期标准差；
- `safety_fault_count`：安全超限采样点数。

`acceptance_normal` 必须满足：

- 安全触发采样点为 0；
- 最终位置误差小于 0.5°；
- 超调小于 5%；
- 最大速度小于 300°/s；
- 饱和比例小于 10%；
- 调节时间小于 1.5 s。

这些是教学项目阶段性验收值，不声明为通用工业标准。

## 输出产物

新增：

```text
lesson02_servo_control/
├─ config/timing_acceptance.yaml
├─ src/run_timing_acceptance.py
├─ reports/engineering_acceptance.md
├─ figures/step_vs_cubic.png
├─ figures/timing_comparison.png
└─ logs/
   ├─ acceptance_normal.csv
   ├─ delay_1step.csv
   ├─ delay_5steps.csv
   ├─ sample_5ms.csv
   └─ jitter.csv
```

报告说明阶跃超速原因、三次轨迹公式、采样与延迟对比、正常工况验收结果，以及 Python/Windows 仿真的非硬实时边界。

## 测试策略

所有行为修改采用测试驱动：先添加失败测试，再进行最小实现。

测试覆盖：

1. 三次轨迹起点、中点、终点、终点后保持及非法持续时间；
2. `simulate()` 输出参考速度和实际周期日志；
3. 无抖动周期精确等于名义周期；
4. 抖动周期保持在 80%～120% 范围且具有非零标准差；
5. 延迟队列的周期语义保持正确；
6. 时序指标计算正确；
7. 平滑轨迹 `normal` 满足全部阶段性验收条件；
8. 既有指标、绘图和时间轴测试无回归。

## 实时边界

报告和 README 明确：本项目验证的是离线数值仿真中的控制逻辑与非理想时序敏感性，不能证明 Windows 线程、Python 运行时或实机通信链路满足 1 kHz 硬实时截止时间。
