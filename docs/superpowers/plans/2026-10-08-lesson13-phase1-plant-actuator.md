# Lesson13 Phase 1: Plant and Actuator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking. Native execution in this session; no worktrees or parallel agents for implementation.

**Goal:** 建立可单独运行、经过测试的非理想关节和执行器，为后续MPC实验提供可信输入输出链。

**Execution status (2026-10-08):** Tasks 1–3 complete in e9f1996, 34b655f and d593ec4. RED/GREEN observed; 17 phase1 tests and 117 whole-repository tests passed. Independent reviewer reproduced 17/17 and reported no Critical, Important or Minor issues. Evidence and scope limitations retained in reports/plant_actuator_demo.md.

**Architecture:** 对象拥有位置/速度真值，执行器拥有延迟队列与实际力矩。两者均不知道控制器种类；演示入口用规定的扭矩序列展示实际响应，不提前实现闭环。

**Tech Stack:** Python 3.11、NumPy、SciPy（复用ZOH）、Matplotlib、PyYAML、unittest；Conda robot-control。

**Spec:** `docs/superpowers/specs/2026-10-08-lesson13-mpc-engineering-design.md`

## Global Constraints

- 直接在D:/MPC_learn工作，不创建worktree，不修改旧课代码。
- 名义J=0.019762 kg·m²，b=0.080257 N·m·s/rad；物理周期0.001 s，控制周期0.01 s。
- 实际幅值±2 N·m、实际变化率50 N·m/s；非理想执行器命令延迟0.002 s、时间常数0.005 s。
- 摩擦教学参数0.12 N·m，平滑速度0.02 rad/s；未知负载教学参数0.3 N·m。
- 单独提交自己新增文件，不触碰ROS_COURSE_material_2026.zip，不上传GitHub。
- 配置物理值只是教学假设，演示不构成实机安全或实时保证。

## Review Focus

1. 参数和输入中的NaN、inf、布尔值不得进入积分状态：Task1/2分别拒绝且状态不变。
2. 调用者修改返回数组不得改变对象内部状态：Task1检查副本隔离。
3. 2 ms延迟的边界不得提前或多迟一拍：Task2明确物理区间约定和队列测试。
4. 负方向、反向与零速度摩擦符号正确：Task1验证摩擦不主动加速对象。
5. 初始化已经超出幅值或不整数延迟步数应被拒绝，不偷偷裁剪：Task2配置测试。

## Task 1: 关节对象

**Files:** Create `lesson13_mpc_engineering/__init__.py`, `src/__init__.py`, `tests/__init__.py`, `src/plant.py`, `tests/test_plant.py`。

**Interfaces:**
- `JointPlant(inertia: float=0.019762, damping: float=0.080257, dt_s: float=0.001, coulomb_friction: float=0.0, friction_smoothing: float=0.02, initial_state: array_like=(0.0,0.0), integration: str='semi_implicit')`。
- `state -> np.ndarray`：返回形状(2,)的副本，单位rad、rad/s。
- `step(torque_actual_nm: float, load_torque_nm: float=0.0) -> np.ndarray`：推进一个物理区间，返回新状态副本。
- integration支持semi_implicit和exact_zoh；exact_zoh要求无库仑摩擦，常值负载作为净力矩处理。

- [ ] 写测试test_zero_state_remains_at_rest、test_semi_implicit_matches_one_step：J=.02、b=.08、dt=.001、u=1、初态0，期望dq=.05、q=.00005。
- [ ] 写test_load_subtracts_from_torque：u=1、load=.3、初态0，期望dq=.035、q=.000035。
- [ ] 写test_friction_opposes_both_velocity_signs：u=0、b=0、dq=±1；速度绝对值下降；dq=0时无摩擦主动运动。
- [ ] 写test_exact_zoh_matches_lesson08：使用旧课A/B和ZOH独立计算一个区间，比对新状态atol=1e-12；有非零库仑摩擦时exact_zoh抛ValueError。
- [ ] 写test_invalid_parameters_and_inputs_leave_state_unchanged：惯量≤0、dt≤0、负阻尼/摩擦、平滑速度≤0、NaN/inf/bool、错误状态形状和未知积分模式拒绝；非法step后状态不变。
- [ ] 写test_state_returns_copy：修改读取的state或step结果，对象state不变。
- [ ] 运行 `conda run -n robot-control python -m unittest lesson13_mpc_engineering.tests.test_plant -v`，确认新功能缺失导致预期失败，再实现上述API。
- [ ] 重跑测试通过后，只提交本任务新增文件：`feat: add lesson13 nonideal joint plant`。

## Task 2: 执行器

**Files:** Create `src/actuator.py`, `tests/test_actuator.py`。

**Interfaces:**
- `TorqueActuator(dt_s: float=0.001, torque_limit_nm: float=2.0, slew_rate_nm_s: float=50.0, delay_s: float=0.002, time_constant_s: float=0.005, initial_torque_nm: float=0.0)`。
- `torque_actual_nm -> float`：当前内部真实力矩，仿真对象使用；后续控制器只能读取由它生成的反馈信号。
- `step(command_nm: float) -> dict[str,float]`：键command_nm、delayed_command_nm、torque_actual_nm、delta_actual_nm；返回值作用于本次物理推进区间。

**Algorithm decisions:** 输入先验证并限幅；队列初始化为delay_steps个initial_torque。每次调用将当前命令入队并取最旧命令，因此2 ms延迟下前2个区间仍用初值，第3个区间用首条命令。响应为tau_candidate=tau+(1-exp(-dt/T))*(delayed-tau)；T=0表示即时响应。实际输出增量裁剪到±slew_rate*dt，再应用幅值限制。delay_s必须是dt整数倍；不得浮点截断。

- [ ] 写test_two_millisecond_delay_has_exact_interval_alignment：dt=.001、delay=.002、T=0、slew=1e6，连续3次command=1，实际输出为0、0、1。
- [ ] 写test_lag_matches_exponential_response：delay=0、T=.005、slew=1e6，首次输出=1-exp(-.2)。
- [ ] 写test_positive_negative_and_reversal_respect_limits：dt=.001、rate=50，连续请求±100时abs(actual)≤2，每次abs(delta_actual)≤.05+1e-12，反向不瞬间跳变。
- [ ] 写test_zero_delay_and_zero_lag：rate=1e6时命令立即生效；初始力矩非零时队列内容与初态一致。
- [ ] 写test_invalid_configuration：超幅初值、不整数步delay、负delay/T、非正dt/limit/rate及NaN/inf/bool拒绝。
- [ ] 写test_invalid_command_does_not_advance_queue：非法输入后队列、实际力矩不变化，通过后续合法输入输出比对无非法调用的参考对象。
- [ ] 运行 `conda run -n robot-control python -m unittest lesson13_mpc_engineering.tests.test_actuator -v`，确认预期失败，再实现API。
- [ ] 运行Task1+Task2测试，通过后仅提交对应文件：`feat: model torque lag delay and slew limits`。

## Task 3: 配置与可读演示

**Files:** Create `config/mpc_engineering.yaml`, `src/run_plant_actuator_demo.py`, `tests/test_plant_actuator_demo.py`, `README.md`。

**Interfaces:**
- `simulate_demo(config: dict) -> dict[str,np.ndarray]`：调用Task1/2；输出time_s、torque_request_nm、torque_command_nm、torque_actual_nm、q_rad、dq_rad_s、delta_actual_nm。
- `main() -> None`：`python -m lesson13_mpc_engineering.src.run_plant_actuator_demo`，保存CSV和四联图到logs/plant_actuator_demo.csv、figures/plant_actuator_demo.png。
- 配置保留nominal_model、plant、actuator、demo四节，后续任务再扩展控制参数，避免提前实现未测试闭环。

**Demonstration:** 时长0.5 s，1 ms物理推进。控制命令按10 ms更新：前0.05 s为0，0.05至0.20 s为+1，0.20至0.35 s为-1，之后为0。保护后命令相对上条命令每控制步变化≤.5，再进入执行器。此输入不是位置控制，位置不必回到零。四联图分别显示请求/保护命令/实际力矩、实际力矩变化率及±50界、位置、速度；单位和时间基准写明。

- [ ] 写test_demo_alignment_and_limits：500个物理区间，time=index*.001；t=.05开始请求1；command每10 ms才改变，实际delta/dt≤50，实际幅值≤2，状态均有限。
- [ ] 写test_demo_separates_request_command_and_actual：阶跃时三者存在差别，q/dq为区间开始值，actual为该区间输入；CSV不得把新区间状态贴到旧时间。
- [ ] 写test_bad_duration_and_period_are_rejected：时长非物理周期整数倍、控制周期非物理周期整数倍或非正值拒绝。
- [ ] 运行演示测试确认失败，再实现模拟和输出入口；README仅讲解本阶段已实现内容并说明后续阶段未完成。
- [ ] 运行 `conda run -n robot-control python -m unittest discover -s lesson13_mpc_engineering/tests -t . -v`，全部通过。
- [ ] 运行演示入口，读取CSV核对约束并检查图，生成数值摘要，不虚构尚未运行的结果。
- [ ] 运行 `conda run -n robot-control python -m unittest discover -v`；如默认发现遗漏，补跑各课程tests目录并报告覆盖范围和实际失败。
- [ ] 提交本阶段文件与演示：`feat: demonstrate lesson13 actuator and joint response`，给用户入口和请求→命令→实际力矩的简短讲解。

## 后续独立阶段（本计划不提前实现）

1. 变化量QP与A/B：u_prev、S和hard rate，先验证首步锚点。
2. 松弛量与C/D/E：当前状态诊断、未来约束、rho敏感性。
3. 监控/降级与F/G：墙钟统计、确定性事件延迟、恢复状态机和因果时序。
4. 公平三控制器对比、多种子非理想验收及完整工程报告。

每个阶段有独立可运行入口和测试，再进入下一阶段；不会在第一步把整课写完。
