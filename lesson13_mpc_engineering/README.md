# 第13课：MPC工程化

当前只完成第一阶段：非理想关节、执行器及开环演示。MPC变化量、软约束和fallback尚未实现；不要把此演示当成第13课完整验收。

在 `D:\MPC_learn` 的终端执行：

```powershell
conda run -n robot-control python -m lesson13_mpc_engineering.src.run_plant_actuator_demo
```

程序保存 `figures/plant_actuator_demo.png` 和 `logs/plant_actuator_demo.csv`，不弹出窗口；在VSCode中打开PNG即可。

## 文件与算法

- `plant.py`：真实对象的仿真替身，计算摩擦、加速度，然后更新速度和位置。非线性模式使用半隐式欧拉；线性匹配基准可用第8课精确ZOH。
- `actuator.py`：命令延迟FIFO、一阶扭矩响应、实际幅值和变化率限制。它不是MPC，也不是电流环的完整模型。
- `run_plant_actuator_demo.py`：给定开环扭矩、进行命令保护、依次推进执行器和对象并保存数据。
- `config/mpc_engineering.yaml`：第一阶段配置。数值是教学假设，不是已辨识舵机规格。
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

下一阶段才比较MPC的变化量惩罚和hard rate constraint，随后完成课程A–G及非理想综合验收。当前仿真不代表安全认证或硬实时保证。
