# Lesson13 Phase 2: MPC Input Moves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Native implementation in the current session, no worktrees. Use checkbox steps; complete this phase, not subsequent C–G experiments.

**Goal:** 完成课程A/B：区分变化量惩罚、QP硬变化量约束和外部执行器保护，并在非理想关节闭环中验证。

**Architecture:** 在第12课固定结构QP之上增加u_prev参数、变化量代价与约束；复用旧QP基础求解/校验逻辑而不改动旧课接口。新增第13课闭环把KF、QP、命令保护、执行器和对象串起来，保存规划与实际之间的差异。

**Tech Stack:** Python3.11、NumPy、CVXPY+OSQP、SciPy、Matplotlib、PyYAML、unittest；robot-control环境，不增加依赖。

**Spec:** docs/superpowers/specs/2026-10-08-lesson13-mpc-engineering-design.md（本计划只实现A/B）。

## Global Constraints

- 在D:/MPC_learn直接实现并本地提交，不创建worktree，不上传，不改旧课代码，不触碰ROS_COURSE_material_2026.zip。
- 名义J=.019762、b=.080257；物理/KF周期1 ms，控制周期10 ms，N=20，目标30°，仿真4 s。
- 扭矩±2 N·m，命令变化率50 N·m/s（10 ms时每步±.5）；对象、执行器继续复用第一阶段实现。
- A仅改变S=0,.5,5，无QP硬变化量约束；B固定S=.5，对比无/有QP硬变化量约束。
- 各组都经过相同外部幅值和变化率保护，不以去掉某组保护制造算法优势。
- 主实验保留当前配置的摩擦.12、延迟2 ms、滞后5 ms；两状态预测模型不包含这些因素，必须披露失配。
- 控制器只用KF估计，u_prev来自已有区间的实际扭矩反馈，主实验暂按无噪声同步反馈假设；真值仅用于仿真及离线评价。
- 本阶段速度约束关闭，专门隔离变化量问题；±60°未来位置边界硬约束及真实越界监测启用。速度软约束C/D/E和故障降级F/G待后续阶段。
- 第12课终端P继续作为固定终端权重，不声称它是加入变化量后增广系统的最优终端权重，也不宣称与LQR严格等价。

## Review Focus

1. 首步delta错误锚定上次请求而非反馈：Task1/2使用不同请求/实际的手工例子。
2. 请求delta、QP相对反馈的首步delta、实际扭矩delta混淆：Task2/3分字段与采样率计算。
3. 失败求解残留上次可执行U：Task1无效解拒绝测试，Task2停止并保留部分日志，不能继续使用陈旧解。
4. 不同组使用不同噪声/执行器以致对比不公平：Task2/3使用预生成同一噪声序列，断言一致。
5. 时长/周期不整除、非法S/u_prev/边界参数：Task1/2按原有严格有限数值规则拒绝，不静默截断。

## Task 1: QP变化量扩展

**Files:** Create `lesson13_mpc_engineering/src/mpc_controller.py`, `tests/test_input_rate_limit.py`。

**Interfaces:** `EngineeringMPCController(Ad,Bd,Q,R,P_terminal,horizon,*,input_rate_weight: float=0.,torque_bounds=(-2.,2.),delta_bounds=None,position_bounds=None)`；本课限制nx=2、nu=1。`solve(x_hat,x_ref,u_prev: float)->dict` 返回旧课status/u0/X/U/objective/solve_time_s/iterations/error，并新增DU形状(1,N)、max_hard_violation。

**Implementation choice:** 继承第12课LinearMPCController，构造期在既有cost/constraints之上建立扩展Problem（不是循环内重建）。使用父类solve做基本有限性/动力学/幅值验证，再补验证位置和delta；返回失败必须清空u0/X/U/DU。新增u_prev Parameter；DU0=U0-u_prev，后续DU=U[k]-U[k-1]；代价加S*sum_squares(DU)，可选delta硬上下界；位置边界作用于x1..xN。S≥0，边界合法且有限，delta上下界应包含0以支持保持命令。

- [ ] 写test_zero_s_matches_lesson12：相同模型/状态/reference，S=0、无新增约束时，与旧课QP的首输入和objective在1e-5内一致。
- [ ] 写test_one_step_penalty_matches_hand_solution：Ad=I2、Bd=[[0],[1]]、Q=0、R=1、P=diag(0,1)、N=1、x0=[0,0]、xref=[0,2]、u_prev=0且无幅值约束；u=2/(2+S)，S=0为1，S=.5为.8，S=5为2/7。
- [ ] 写test_first_delta_uses_feedback：同模型xref=[0,10]、u_prev=.4、delta_bounds=(-.5,.5)，首输入应=.9，DU0=.5（容差1e-5）。再以u_prev=-.2更新同一Problem，首输入=.3，证明参数确实更新。
- [ ] 写test_all_planned_deltas_respect_bounds：N=5，所有DU满足±.5；输出DU与U及反馈独立重算一致；Problem标识跨solve不变。
- [ ] 写test_infeasible_does_not_return_stale_plan：先成功，再u_prev=10且幅值±2、delta±.5，返回不可行且u0/X/U/DU为空。
- [ ] 写test_invalid_move_parameters：负S、NaN/inf/bool、错误边界/状态/输入维度及非法u_prev拒绝；位置边界矛盾不能被当成功解。
- [ ] 运行 `conda run -n robot-control python -m unittest lesson13_mpc_engineering.tests.test_input_rate_limit -v`，确认预期失败后实现，再确认全部通过。
- [ ] 本地提交准确文件：`feat: add MPC move penalties and hard rate constraints`。

## Task 2: 估计反馈闭环与共同保护

**Files:** Create `src/model_loader.py`, `src/constraint_manager.py`, `src/closed_loop.py`, `tests/test_move_closed_loop.py`；Modify `config/mpc_engineering.yaml`（保留第一阶段配置）。

**Interfaces:**
- `build_design(config: dict)->dict`：返回Ad/Bd(10ms)、Ad_fast/Bd_fast(1ms)、Q/R/P、C、covariance。复用第8课模型/权重/DARE与第10课噪声协方差函数；显式适配配置，不复制旧课整份配置并默默覆盖。
- `guard_command(request: float,previous_command: float,torque_limit_nm: float,slew_rate_nm_s: float,elapsed_s: float)->float`：幅值与变化率保护。初始previous_command必须合法。
- `simulate_closed_loop(config: dict,*,input_rate_weight: float,hard_rate: bool,measurement_noise=None)->dict`：返回status、log、plans、failure_reason。失败保存当时状态和既有日志，不执行陈旧U；本阶段明确尚无fallback。
- `log`为同长度数组dict，物理行记录time_s、q_true_rad、dq_true_rad_s、q_hat_rad、dq_hat_rad_s、q_measured_rad、measurement_noise_rad、q_ref_rad、torque_request_nm、torque_command_nm、torque_actual_nm、delta_actual_nm、control_update、feedback_prev_nm。控制事件行另记录solver_status、solver_iterations、solver_solve_time_s、solver_wall_time_s、request_delta_nm、qp_first_delta_nm、command_delta_nm、max_planned_delta_nm、guard_modified；非事件数值为NaN，状态为空。plans每控制事件保存time、u_prev、X/U/DU，方便约束独立检查。

**Config additions:** mpc.dt_s=.01、horizon_steps=20；experiment_ab.duration_s=4、target_deg=30、seed=2026、input_rate_weights=[0,.5,5]；measurement.position_noise_std_deg=.05；constraints.position_limit_deg=60；design_limits沿用第12课max_position_error_deg=5、max_velocity_error_rad_s=1.5、max_torque_nm=3（仅权重尺度，非执行器能力）；KF过程扭矩标准差.15、初始位置不确定度5°和速度不确定度1 rad/s。所有组Q/R/P相同。

- [ ] 写test_command_guard_amplitude_and_rate：previous0/request2/dt=.01/rate50→.5，previous.5/request-2→0；dt=.02→允许1；非法previous、dt、NaN等拒绝。
- [ ] 写test_kf_uses_past_actual_torque：先用无噪声短仿真重现t0测量更新、下一步predict之前区间actual；独立调用旧KF比对日志估计。确认输入不是刚请求的新力矩。
- [ ] 写test_feedback_and_request_deltas_are_distinct：迟滞场景中请求与反馈不同，QP首步delta=request-feedback，request_delta=request-last_request，command_delta=command-last_command，物理delta_actual为1ms差分；分别独立重算。
- [ ] 写test_common_noise_is_reused：输入同一预生成噪声，不同S输出measurement_noise逐点相同，但q_measured可不同（对象运动不同）。
- [ ] 写test_failed_solve_stops_and_preserves_log：使用合法但矛盾的位置/初始配置触发失败或越界，status失败、原因非空、部分日志留存，不伪造完整4s输出。
- [ ] 写test_invalid_periods_and_noise：非整数周期/时长、噪声长度不符/非有限、非法S拒绝。
- [ ] 运行 `conda run -n robot-control python -m unittest lesson13_mpc_engineering.tests.test_move_closed_loop -v` 见预期失败后实现；先运行短闭环，再运行一次4s硬rate场景检查所有计划和实际输出界限。
- [ ] 本地提交准确文件：`feat: connect move-aware MPC to nonideal joint feedback`。

## Task 3: A/B指标、图和报告

**Files:** Create `src/metrics.py`, `src/run_experiments.py`, `tests/test_move_metrics.py`, `tests/test_experiment_ab.py`, `reports/input_move_report.md`；Modify `README.md`；Generate `figures/input_move_penalty.png`, `figures/input_rate_constraint.png`, `logs/experiment_a_*.csv`, `logs/experiment_b_*.csv`, `logs/experiment_ab_summary.json`。

**Interfaces:** `calculate_metrics(log:dict,control_dt_s:float)->dict`；`run_experiments(config:dict,output_dir:Path)->dict`；CLI `python -m lesson13_mpc_engineering.src.run_experiments --experiment ab`，不默认运行后续未实现实验。

指标：位置跟踪RMSE（全物理采样）、实际力矩RMS/峰值、请求RMSdelta与maxdelta（仅控制事件含首步相对初始0）、QP相对反馈首步delta、保护后命令delta、实际物理变化率分别统计；solver计时全事件含首轮与成功/失败计数。不将不同采样频率delta直接混用。

- [ ] 写test_metrics_use_control_events_only：手工3事件请求[1,.5,-.5]且初始0，delta=[1,-.5,-1]，RMS=sqrt(.75)、max=1；物理重复持有行不稀释控制delta。
- [ ] 写test_metrics_separate_rate_and_move：1ms实际delta=.05对应50 Nm/s，10ms命令delta=.5也对应50 Nm/s但不是相同delta量；缺失、未完成场景如实标记，NaN计时不当0。
- [ ] 写test_runner_covers_all_ab_cases：短时合法配置生成A三组、B两组，配置除S/hard_rate一致；相同seed噪声一致，summary声明采样率、模型/执行器参数、版本及失败情况。
- [ ] 写test_summary_and_csv_reconcile：从导出CSV独立重算峰值及request_delta，符合summary；B硬rate计划满足±.5（1e-5数值容差）；不对A结果硬写单调更平滑或MPC更快。
- [ ] 运行对应unittest，确认失败后实现。A画位置、请求/实际扭矩、请求delta对比；B画位置、速度、请求/实际扭矩、delta及上下界，明确曲线是规划、请求、命令还是物理输出。
- [ ] 运行完整4s A/B，保存每组配置与失败诊断；报告按公式→QP代码→曲线→实测指标解释控制平滑是否付出跟踪代价。结果不明显也保留并解释，不能挑曲线或暗改其他参数。
- [ ] 运行第一阶段演示回归及 `conda run -n robot-control python -m unittest discover -v`；新旧测试全部通过或逐项报告实际失败。渲染检查两张PNG标注/单位/布局。
- [ ] 本地提交代码/图/报告，仅当前阶段路径：`feat: compare MPC move penalties and rate-aware planning`。独立只读复核后交付运行命令和A/B实测讲解，不上传GitHub。

## 阶段边界与已知限制

本阶段的外部执行器保护可能让不同组实际扭矩看起来都很平滑；这不是实验失败，应比较规划是否被外部保护修正及跟踪差异。两状态QP仍不能准确预测执行器滞后，不以硬rate规划宣称真实状态安全。墙钟时间仅记录，5ms预算降级和真实事件延迟演示留到F/G，不能提前声称实时处理已完成。C/D/E软约束及综合多种子实验不在本次实施范围。
