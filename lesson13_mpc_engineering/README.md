# 第13课：MPC工程化

当前覆盖非理想关节/执行器、A/B输入变化量、C/D/E可行性与速度松弛量，以及F/G求解预算和fallback。综合多seed/多控制器全场景验收仍待后续，不把F/G当作综合验收。

## 实验F/G

```powershell
conda run -n robot-control python -m lesson13_mpc_engineering.src.run_experiments --experiment fg
```

看 [求解时间与降级报告](reports/runtime_fallback_report.md) 和 [验收记录](reports/phase4_verification.md)。

- F冻结200组参数，warm on/off各重复3轮，区分首次编译/冷启动和后续调用；记录真实墙钟mean/p95/p99/max，保留慢样本和失败。5 ms预算不等于10 ms控制截止。不要并行运行其他性能测试。
- G实际调用QP，但完成事件延迟是明确注入的2/7/12 ms，不冒充墙钟。5 ms无结果就选fresh LQR，迟到计划丢弃；worker仍忙就不提交新QP。连续3次有效及时结果才恢复MPC。
- `f_measured_sync`只能在Python阻塞返回后发现超时，不提供“5 ms准时切换”的物理保证。G演示该时序策略，也不是真正异步可抢占求解器。
- 每组共享同一编码器噪声。输出physical CSV、jobs JSON、events CSV、候选plans NPZ与summary，能逐个作业核对提交、过期、丢弃和恢复；`executed`表示请求被选择，之后仍经过共同保护。

核心文件：`qp_solver.py`负责本轮原始诊断；`solver_monitor.py`分类预算/结果并管理恢复；`fallback_controller.py`只用KF估计计算备用LQR；`managed_closed_loop.py`实现事件顺序和共同保护；`runtime_metrics.py`区分作业/控制周期/物理样本分母；`run_runtime.py`运行F/G，`runtime_artifacts.py`导出图与报告。

F/G的S=0.5、rho=10000、速度未来软边界±1.5 rad/s。变化率保护按每次命令事件的实际间隔计算；实际执行器仍每1 ms限速。默认故障表见生成报告；若缩短仿真，请在`experiment_fg.fault_scenarios`显式提供四组的合法周期索引，程序不会悄悄忽略越界故障。

## 实验C/D/E

```powershell
conda run -n robot-control python -m lesson13_mpc_engineering.src.run_experiments --experiment cde
```

看 [可行性实验报告](reports/feasibility_report.md)。C在纯QP中令当前速度2 rad/s却要求≤1.5，预期不可行，不画虚假的闭环恢复线；D在相同索引上加非负EPS；E比较rho=100、10000、1000000。所有输入/变化量/位置仍是硬约束。

本次交付的测试、独立审计与实现决定见 [验收记录](reports/phase3_verification.md)，以后修改代码或配置需重新验证。

`run_feasibility.py`负责C/D/E分组和产物；`feasibility_metrics.py`区分真实超速积分与QP初始slack的控制事件持有积分。预测模型未包含执行器滞后和摩擦，所以软约束成功不代表实际速度已达标。全部状态、失败原因、终态和原始计划均保留。

## 实验A/B

在项目根目录运行：

```powershell
conda run -n robot-control python -m lesson13_mpc_engineering.src.run_experiments --experiment ab
```

先看 [实验报告](reports/input_move_report.md)。A比较S=0、0.5、5；B固定S=0.5，比较是否把变化量边界放入QP。所有组使用同一编码器噪声、KF、执行器和外部保护。

核心文件：`mpc_controller.py`扩展第12课QP；`model_loader.py`复用离散模型/LQR/KF参数；`closed_loop.py`执行闭环；`constraint_manager.py`进行独立命令保护；`metrics.py`计算指标；`run_experiments.py`导出图、CSV、计划NPZ及报告。

控制每10 ms更新，物理对象/KF每1 ms更新。QP首步变化量为“本次请求－当前实际扭矩反馈”，后续为相邻规划输入之差。它不同于相邻两次控制请求之差。硬边界为±0.5 N·m；执行器实际变化率上限50 N·m/s，幅值±2 N·m。

控制器只使用带噪位置经KF得到的状态估计；真值仅供仿真推进、指标和越界监测。预测模型尚未包含摩擦、延迟和一阶滞后，不能把预测约束当作实机安全保证。本阶段未启用速度约束。

## 第一阶段开环演示

在 `D:\MPC_learn` 的终端执行：

```powershell
conda run -n robot-control python -m lesson13_mpc_engineering.src.run_plant_actuator_demo
```

程序保存 `figures/plant_actuator_demo.png` 和 `logs/plant_actuator_demo.csv`，不弹出窗口；在VSCode中打开PNG即可。

## 文件与算法

- `plant.py`：真实对象的仿真替身，计算摩擦、加速度，然后更新速度和位置。非线性模式使用半隐式欧拉；线性匹配基准可用第8课精确ZOH。
- `actuator.py`：命令延迟FIFO、一阶扭矩响应、实际幅值和变化率限制。它不是MPC，也不是电流环的完整模型。
- `run_plant_actuator_demo.py`：给定开环扭矩、进行命令保护、依次推进执行器和对象并保存数据。
- `config/mpc_engineering.yaml`：开环与A–G配置。数值是教学假设，不是已辨识舵机规格。
- `tests/`：核对物理公式、延迟边界、输入有效性与日志因果时序。

对象方程：

`acceleration = (actual_torque - load - damping*velocity - friction) / inertia`

`velocity_next = velocity + acceleration*dt`

`position_next = position + velocity_next*dt`

其中 `friction = coulomb_friction*tanh(velocity/friction_smoothing)`，相对运动方向阻碍运动。

执行器每1 ms推进一次：先取延迟后的命令，再以 `1-exp(-dt/time_constant)` 更新一阶响应；力矩每1 ms最多变化0.05 N·m。命令保护每10 ms执行一次，每次最多变化0.5 N·m。实际幅值仍为±2 N·m。

## 图怎么看

第一幅：虚线是请求扭矩，点划线是保护后的命令，实线是延迟、滞后和限速后的实际力矩。它们不应被混为同一个输入。

第二幅：实际力矩变化率，虚线为±50 N·m/s；接近边界表示执行器变化率限制正在发挥作用，不等于优化器已经规划了变化率。

第三、四幅：开环关节位置和速度。没有位置反馈控制，末尾请求归零也不会让位置自动回零；请求反向时，惯性和扭矩滞后还可能让位置短暂继续增加。

CSV一行对应区间 `[t_k,t_k+1ms)`：q/dq是区间开始的状态，actual_torque是随后推进该区间的输入。t=0.05 s首次请求+1，命令变成+0.5；2 ms延迟后t=0.052 s实际扭矩开始上升，t=0.053 s才记录到新的关节状态。最后一行t=0.499 s，覆盖区间到0.5 s，不是遗漏终点。

## 验证

```powershell
conda run -n robot-control python -m unittest discover -s lesson13_mpc_engineering/tests -t . -v
```

后续完成非理想综合验收。当前fallback、位置监测和仿真均不代表硬件安全认证或硬实时保证。
