# 第二课：单轴机器人关节闭环控制

## 目标

将第一课的开环状态空间演示升级为单轴关节闭环仿真。系统从 0° 跟踪到 30°，并考虑编码器噪声、速度估计、执行器饱和、控制延迟、摩擦、外部负载和安全限制。

## 核心模型

关节动力学：

```text
J·ddq = torque_applied - torque_load - b·dq - torque_friction
```

PD 位置控制器：

```text
torque_command = Kp·(q_ref - q_measured) - Kd·dq_estimated
```

执行器将转矩命令限制在 `[-torque_limit, torque_limit]`，随后加入离散延迟并执行位置、速度安全检查。

## 控制循环

每个 1 ms 控制周期依次执行：

1. 注入当前测试工况；
2. 读取带噪声的编码器角度；
3. 差分并低通滤波得到估计角速度；
4. 计算 PD 转矩命令；
5. 执行转矩限幅、延迟和安全检查；
6. 计算角加速度并进行半隐式欧拉积分；
7. 保存本周期数据。

## 测试工况

- `normal`：正常编码器噪声；
- `disturbance`：1.5–2.2 s 施加 0.8 N·m 外部负载；
- `sensor_fault`：2 s 后将编码器噪声标准差放大 20 倍。

## 基准结果

| 工况 | 最终误差/deg | 超调/% | 调节时间/s | RMS转矩/N·m | 饱和/% | 安全触发 |
|---|---:|---:|---:|---:|---:|---:|
| normal | 0.0037 | 0.0279 | 0.338 | 0.3832 | 0.9748 | 27 |
| disturbance | 0.0011 | 0.0029 | 2.354 | 0.4922 | 0.9748 | 27 |
| sensor_fault | 0.3463 | 2.2886 | 3.933 | 1.4505 | 12.2469 | 33 |

正常工况中的安全触发来自初始位置阶跃造成的速度越限：真实峰值速度约为 332 deg/s，而配置上限为 300 deg/s。这说明“正常工况”没有注入故障，但并不自动保证控制参数满足安全约束。

## 运行

```powershell
conda activate robot-control
cd D:\MPC_learn
python .\lesson02_servo_control\src\run_servo_test.py
```

## 测试

```powershell
python -m unittest discover -s .\lesson02_servo_control\tests -p "test_*.py" -v
```

运行后，CSV 日志保存在 `logs/`，响应图保存在 `figures/`。
