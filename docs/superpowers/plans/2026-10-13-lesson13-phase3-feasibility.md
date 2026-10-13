# Lesson13 Phase 3: Feasibility and Velocity Slack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans in the current session. Implement task-by-task with RED/GREEN tests. No worktree; one final independent read-only review.

**Goal:** 完成课程C/D/E，让学习者从代码和实测记录理解硬约束不可行、速度松弛量和惩罚权重的取舍。

**Architecture:** 扩展现有EngineeringMPCController的可选速度约束，不改第8/10/12课及A/B默认行为。现有闭环负责执行和日志；新增可行性指标与独立C/D/E运行模块，由原CLI选择调用。

**Tech Stack:** robot-control / Python3.11、NumPy、CVXPY+OSQP、Matplotlib、PyYAML、unittest；不安装新依赖。

**Spec:** docs/superpowers/specs/2026-10-08-lesson13-mpc-engineering-design.md。整体设计已获批准；本计划只覆盖下一阶段C/D/E。

## Global Constraints

- 直接在D:/MPC_learn实现并本地提交；不建worktree、不上传、不触碰用户ROS_COURSE_material_2026.zip。
- J=.019762、b=.080257；物理/KF 1 ms、MPC 10 ms、N=20、目标30°、闭环4 s、种子2026。
- 扭矩±2 N·m，变化率50 N·m/s，QP变化量±.5 N·m；S固定.5。位置±60°仍为预测硬约束并独立监测实际越界。
- 保留摩擦.12 N·m、平滑速度.02 rad/s、2 ms延迟、5 ms滞后及相同外部保护；名义两状态预测模型不含这些因素。
- KF过程扭矩标准差.15、测量位置标准差.05°、初始位置/速度不确定度5°/1 rad/s；不额外施加真实随机扰动力矩。
- C/D/E速度边界±1.5 rad/s；初始真值及估计均为q=0、dq=2 rad/s。C/D专门约束x0；其余未来索引也必须一致。
- 默认约束x1..xN；只有显式include_current_velocity=True才约束x0..xN。A/B继续关闭速度约束。
- D名义rho=1e4；E使用1e2、1e4、1e6。二次惩罚rho*sum(epsilon²)，epsilon单位rad/s；不是把epsilon从实际速度中减去。
- 输入、变化量、位置不软化。有限rho不保证slack只在不可行时使用，也不保证稳态严格为零。
- 保留原Q/R/P，P只是固定终端权重，不声称增广系统最优性或实机安全/实时保证。
- 失败必须清空可执行解、保存部分日志及最终状态/时间；本阶段不实现fallback、预算切换或恢复状态机。

## Review Focus

1. x0没有被真正约束却把epsilon0当初始超限量：Task1测试索引、初值矛盾及EPS首列。
2. 父类LinearMPCController会把self.velocity_bounds当硬边界复核：新扩展用独立属性engineering_velocity_bounds，不将软边界写进父类属性；Task1软解允许受罚超限。
3. 软速度误将扭矩/位置矛盾一并解决，或失败仍返回旧EPS/U：Task1成功后硬冲突与非有限EPS拒绝测试。
4. 真实超速和预测slack混用，失败/尾部不足10ms仍按完整步积分：Task2手算真实区间及控制事件持有时长；最终状态另存，不补伪造CSV行。
5. 为展示rho效果暗改对象/噪声，或忽略大rho导致的数值失败：Task3逐点共享噪声、保存有效配置及全部状态；失败组不计成完整恢复。

## Task 1: QP硬/软速度约束

**Files:** Modify src/mpc_controller.py；Create tests/test_velocity_constraints.py（均位于lesson13_mpc_engineering）。

**Interfaces:**
- 构造器追加keyword-only参数velocity_mode='off', velocity_bounds=None, include_current_velocity=False, slack_weight=1e4；solve(x_hat,x_ref,u_prev)签名不变。
- velocity_mode仅'off'/'hard'/'soft'；enabled时bounds必须为有限递增二元组；include_current_velocity必须bool；rho有限且>0，拒绝bool/NaN/inf。off时bounds必须None且include_current_velocity=False，避免静默忽略配置。
- 属性velocity_indices为tuple：enabled时(1,...,N)，显式当前约束时(0,...,N)，off为空。engineering_velocity_bounds独立于父类velocity_bounds。
- 返回新增EPS（soft时float数组(1,len(indices))，否则None）与max_soft_violation。失败u0/X/U/DU/EPS全部None。
- hard: lower<=X[1,indices]<=upper。soft: lower-EPS<=X[1,indices]<=upper+EPS，EPS>=0，代价加rho*sum_squares(EPS)。构造时建一次Problem，不在solve内重建。

- [ ] 写并运行test_velocity_off_preserves_existing_plan：旧参数和显式off的U/X/objective一致，EPS=None；重跑现有InputMoveTest。
- [ ] 写test_current_hard_constraint_is_infeasible：toy Ad=I2,Bd=[[0],[1]],Q=I2,R=1,P=I2,N=3，x0=[0,2],ref=[0,0],u_prev=0，bounds±1.5且include_current=True；status含infeasible，可执行解为空。
- [ ] 写test_future_only_does_not_constrain_x0：同模型仅未来速度hard，扭矩±2、delta±.5；允许u0=-.5到达1.5，不因x0=2直接拒绝；indices从1开始。
- [ ] 写test_soft_current_has_real_epsilon_zero：同模型当前与未来soft，rho=1e4；成功，EPS.shape==(1,4)，EPS[0,0]>=.5-1e-5且接近.5，逐列独立检查软不等式及全部硬约束。
- [ ] 写test_softening_does_not_remove_hard_conflicts：先成功，再u_prev=10且幅值±2/delta±.5；失败并清空包括EPS的所有轨迹。位置硬冲突也不能变成功。
- [ ] 写test_invalid_velocity_options_and_solution_are_rejected：非法模式/边界/rho/bool；通过测试用例注入带非有限EPS或软边界残差>1e-5的返回值时拒绝执行（只mock数值求解边界，不mock验证器）。
- [ ] 确认新增测试RED后实现；独立复核EPS有限、shape、非负、软残差和全部原硬残差，容差1e-5；运行新旧QP测试GREEN。
- [ ] 本地提交准确文件：feat: add indexed hard and soft velocity constraints。

## Task 2: 闭环slack日志与可行性指标

**Files:** Modify src/closed_loop.py；Create src/feasibility_metrics.py、tests/test_feasibility_closed_loop.py、tests/test_feasibility_metrics.py。

**Interfaces:**
- simulate_closed_loop原参数保留，追加velocity_mode='off', velocity_bounds=None, include_current_velocity=False, slack_weight=1e4；转交QP。A/B旧调用不变。
- log新增qp_initial_slack_rad_s、max_predicted_slack_rad_s，仅成功soft控制事件有限，其他行为NaN。当前slack仅在indices确实含0时填值，不用第一未来列冒充。
- plans新增EPS及velocity_indices；off/hard的EPS为None。现有A/B导出仍只选择X/U/DU，避免object NPZ破坏旧文件。
- calculate_feasibility_metrics(result:dict, control_dt_s:float, physical_dt_s:float, velocity_limit_rad_s:float)->dict：返回基础calculate_metrics全部字段，另加max_actual_velocity_violation_rad_s、actual_overspeed_duration_s、actual_overspeed_integral_rad、max_predicted_slack_rad_s、max_initial_slack_rad_s、initial_slack_hold_integral_rad。
- 真实积分只统计实际执行过的物理区间（actual torque有限），按1ms加权；峰值另考虑已保存final_state。slack积分仅计算QP初始松弛量的控制事件持有值，权重截断到下一控制事件或final_state_time_s，不累计整个预测时域。字段名明确是计划量的持有积分，不是真实超速积分；无数据为None，不填0。

- [ ] 写test_hard_current_stops_without_applying_stale_input：真值/估计dq2、q0，零测量噪声、当前hard±1.5；第一QP失败，日志一行且actual为NaN，最终时间0。
- [ ] 写test_soft_current_runs_and_logs_epsilon：相同初值和边界soft rho1e4，短合法闭环；首QP EPS0>=.5-1e-5，保存indices含0，日志和计划EPS对应；请求/命令/实际扭矩仍受各自硬限制。
- [ ] 写test_future_slack_is_not_initial_slack：仅未来soft时qp_initial_slack为NaN，但max_predicted_slack可用；不能冒充当前超限。
- [ ] 写test_noise_and_past_input_remain_causal：不同rho共享同一预生成测量噪声；独立调用第10课KF按过去actual复现估计，沿用原时序。
- [ ] 写test_integrals_use_correct_intervals：手工15个1ms已执行区间，前10个速度2、后5个速度1；真实超速峰值.5、持续.01s、积分.005rad。t0/t.01的初始slack=.5/.25，final_time=.015，则slack持有积分=.00625rad。
- [ ] 写test_missing_and_terminal_evidence：失败无slack=>None；终态速度超界计入峰值但不增加不存在的区间；非正/非有限dt或limit拒绝。
- [ ] 先确认RED再实现；运行全部第13课测试并确认A/B入口默认行为不变。
- [ ] 本地提交：feat: log velocity slack separately from physical overspeed。

## Task 3: C/D/E实验、图、报告与交付

**Files:** Create src/run_feasibility.py、tests/test_experiment_cde.py；Modify src/run_experiments.py（仅CLI分发）、config/mpc_engineering.yaml、README.md；Generate figures/velocity_softening.png、figures/slack_penalty_comparison.png、logs/feasibility_diagnostic.json、logs/experiment_cde_summary.json、logs/experiment_{d_soft,e_rho100,e_rho10000,e_rho1e6}.csv及对应plans.npz、reports/feasibility_report.md。

**Interfaces:**
- run_feasibility_experiments(config:dict,output_dir:Path)->dict，CLI python -m lesson13_mpc_engineering.src.run_experiments --experiment cde；--experiment ab保持默认，不运行F/G。
- YAML新增constraints.velocity_limit_rad_s=1.5及experiment_cde:{duration_s:4,target_deg:30,initial_velocity_rad_s:2,seed:2026,slack_weights:[100,10000,1000000]}。
- runner对legacy闭环所读experiment_ab字段进行显式deepcopy适配duration/target/seed，并保存每组有效配置；不改用户原config或A/B配置。
- C纯QP诊断x0=[0,2],ref=[deg2rad(30),0],feedback=0，hard当前±1.5，S=.5，幅值±2、delta±.5、位置±60°；记录原始status/诊断、约束索引、已知矛盾、返回值，不伪造闭环曲线。相同输入再做soft rho1e4静态求解，记录EPS0。
- D为soft rho1e4闭环；E三组只改rho，全部include_current=True。预生成同一噪声序列用于四组；D与E名义组可以重复运行用于一致性校验，必须得到同一物理轨迹（不比较墙钟）。
- summary保存完整有效配置、版本、status、failure_reason、最终状态/时间、基础和可行性指标、csv/plans路径。NPZ保存X/U/DU/EPS和indices；无计划也保持无object的确定shape。JSON不含NaN。
- D图：位置/速度及±1.5、请求/命令/实际、QP初始与最大预测slack；全程与启动放大。C状态作为文本诊断，不画虚假hard恢复线。E图：速度、QP初始slack、位置、各组迭代/墙钟分布或汇总，显示失败截断。

- [ ] 写test_diagnostic_hard_infeasible_soft_feasible：JSON硬组预期不可行，soft成功且EPS0>=.5-1e-5，约束索引一致。
- [ ] 写test_runner_covers_cde_and_reuses_noise：短时配置覆盖D+三E；每组除rho及case标签外有效条件一致；噪声逐点相同；D/E名义物理轨迹一致；输入config未修改。
- [ ] 写test_exported_slack_and_metrics_reconcile：独立从CSV/NPZ重算软不等式、硬扭矩/delta/位置、EPS0及slack/超速指标；报告/JSON/PNG存在，版本和终态证据可读。
- [ ] 写test_failure_remains_partial_and_visible：选择确定性硬冲突输入；summary保留失败原因/无可执行计划/时间，报告不把部分RMSE称作完整4s；图不伪造恢复轨迹。
- [ ] 确认RED后实现。运行完整C/D/E；允许某rho求解数值失败并明确记录，不暗改rho/容差/其他条件制造成功；遇到失败先调查原因。
- [ ] 报告按公式→QP索引/代码→曲线→实际指标→限制讲解，说明epsilon不改变真实状态、rho不是S、不是所有硬冲突都可软化修复、有限rho不保证零slack。
- [ ] 运行conda run -n robot-control python -m unittest discover -v；复核A/B旧物理轨迹及第一阶段入口，视觉检查两PNG标注、单位、放大范围和失败标记。
- [ ] 本地提交准确本阶段路径：feat: demonstrate MPC feasibility and velocity softening。按executing-plans做一次全阶段独立只读审查，解决Critical/Important并回归；交付运行命令、图、报告和实测解释，不上传。

## 阶段边界

C硬不可行是实验预期，不是要用虚假的闭环修复隐藏它。D/E soft求解成功仍可能实际超速，或因其他硬冲突/数值问题失败；真实位置监测只用于仿真终止，不是认证停机。求解时间只是记录，F/G的预算、fallback和恢复状态机尚不实现。综合多种子及三控制器对比留到后续阶段。
