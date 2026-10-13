# Lesson13 Phase 4: Solver Runtime and Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans natively, task-by-task, RED→GREEN tests, one final independent read-only review. No worktree, no upload.

**Goal:** 完成F/G，真实测量求解性能，并用确定性事件模型验证失败/超时后的备用LQR、迟到解丢弃和连续3次成功恢复。

**Architecture:** 在第13课增加可配置QP求解适配层、结果监控状态机和独立LQR备用控制；不修改第12课。A–E默认调用保持原路径。F的真实墙钟测量与G的注入完成时刻分开；新管理闭环按事件处理命令，不把同步Python计时当硬实时调度。

**Tech Stack:** robot-control / Python3.11、CVXPY+OSQP、NumPy、Matplotlib、PyYAML、unittest；无新依赖。

**Spec:** docs/superpowers/specs/2026-10-08-lesson13-mpc-engineering-design.md（整体设计已批准；本计划仅F/G）。

## Global Constraints

- 直接在D:/MPC_learn现有main工作区实现和本地提交；不建worktree、不上传、不改旧课、不碰ROS_COURSE_material_2026.zip。
- J=.019762、b=.080257；物理/KF 1ms、控制10ms、N=20、目标30°、闭环4s、种子2026。
- 扭矩±2Nm、实际/命令变化率50Nm/s、QP变化量±.5Nm、位置±60°硬边界及独立监测。
- F/G采用速度未来x1..xN软约束±1.5rad/s、rho=1e4、S=.5、初始q/dq及估计均为0。保留摩擦.12、2ms延迟、5ms滞后、同一KF/编码器噪声；不加入真实随机扰动力矩。
- 预算5ms，控制截止周期10ms；有效结果完成时刻≤预算可接受，超过预算丢弃；超过控制周期另外计数，不能混淆两个阈值。
- opt-in F/G solver max_iter=4000，eps_abs=eps_rel=1e-8；1e-5独立验解容差。A–E原默认求解设置/归一化/行为不变。
- QP每场景只构造一次；warm on/off都复用各自Problem。F相同冻结参数序列，重复3轮，单列首次编译/冷启动与后续统计，不预设warm一定快。
- 备用律u=torque_ff-K@(x_hat-x_ref)，K来自现有名义LQR。仅用KF估计/参考，不读真值；每次选择备用时重新计算，经统一保护。
- 恢复需连续3个有效且预算内结果；任何拒绝/忙跳过重置计数。恢复过渡仍受同一保护；失败旧U不执行。
- 未建模执行器/估计误差仍存在；备用不能保证速度/位置安全，越界终止并保留部分日志、终态、时间及原因。
- 综合多种子和LQR/基础MPC/工程MPC全场景对比仍留后续阶段，不以F/G代替综合验收。

## Review Focus

1. 预算5ms和截止10ms混用、等于边界误判：Task2字面测试4/5/7/10/12ms；Task3同刻完成先于超时处理。
2. 失败/超时后执行旧计划或迟到计划：Task1清空解，Task3作业编号/到达/丢弃证据及每次fresh fallback测试。
3. 恢复计数未被失败打断、只成功一次就切回：Task2序列测试；Task3把中间一次错误插入恢复窗口。
4. 命令在不同事件时刻更新却都按10ms限速、或同时启动多个未完成QP：Task3按真实命令间隔保护，单在途作业/忙跳过测试。
5. F不同输入/不同Problem重建或隐藏冷启动，G合成时间冒充墙钟：Task4输出冻结参数、mode/repetition/冷启动标记及两种时间分列；不填缺失计时为0。

## Task 1: 可配置QP调用与原始诊断

**Files:** Create src/qp_solver.py、tests/test_qp_solver.py；Modify src/mpc_controller.py（路径均在lesson13_mpc_engineering）。

**Interfaces:**
- SolverSettings(warm_start:bool=True,max_iter:int=4000)，不可变配置，严格bool/正整数检查，拒绝bool充整数、非有限及非整数；精度固定1e-8。opt-in适配调用显式启用polishing（按本机OSQP版本使用公开参数名称），warm on/off相同，避免CVXPY缓存分支的不同polishing默认值混入比较；旧默认路径不变。
- solve_qp(problem,X_variable,U_variable,settings:SolverSettings)->dict：返回现有status/u0/X/U/objective/iterations/solve_time_s/error，并加backend_status、backend_status_val、backend_solve_time_s、setup_time_s、update_time_s、run_time_s；不可获取字段None。status保留CVXPY原始状态，backend字段来自本轮公共solver_stats.extra_stats.info；异常不得读上轮stats冒充本轮。
- EngineeringMPCController构造器追加solver_settings=None；solve追加keyword-only solver_settings=None覆盖本轮。无配置且无覆盖时仍走super().solve旧路径；配置路径设置x0/x_ref/u_prev后调用solve_qp，复用现有完整独立验解。避免复制旧课动力学验证或修改旧课接口。
- 适配层只在可接受原始状态、完整正确shape的有限X/U和有限objective时形成候选；controller继续校验初值/动力学/全部硬软边界，合格才返回u0。失败u0/X/U/DU/EPS均None。整体归一化和原单位objective不变。
- 本机CVXPY1.9.3源码已核对：solver_stats.solve_time对应OSQP run_time，不等同纯solve_time；warm_start=False会重新建立OSQP后端而Problem仍复用。报告需披露，勿用私有_solver_cache读取异常旧数据。

- [ ] 写test_settings_and_backend_options：实际toy QP在warm on/off都求出手算结果；spy真实solve调用确认warm_start/max_iter/精度确实传入；Problem跨调用不变。
- [ ] 写test_opt_in_matches_legacy_plan：相同模型/参数，默认路径与opt-in warm=True/max_iter=10000的U/X/objective一致（1e-5）；跑现有QP/软约束坏解测试。
- [ ] 写test_iteration_limit_rejects_candidate：冷启动名义QP，x0=[0,0]、目标30°、max_iter=1，观察实际返回；必须无可执行解，保留真实状态/错误，不凭配置猜原始状态。
- [ ] 写test_error_and_corrupt_candidate_do_not_reuse_previous：成功后solver边界抛异常、缺失/非有限/错误shape X/U/objective，拒绝并清空；本轮不可用stats为None。
- [ ] 写test_diagnostics_are_honest：本轮存在的OSQP info字段与返回一致；缺失info不填0。明确legacy solve_time_s是run_time兼容字段。
- [ ] 先RED后实现；conda run -n robot-control python -m unittest lesson13_mpc_engineering.tests.test_qp_solver lesson13_mpc_engineering.tests.test_input_rate_limit lesson13_mpc_engineering.tests.test_velocity_constraints -v，全部GREEN。
- [ ] 本地提交：feat: expose configurable MPC solver diagnostics。

## Task 2: 预算分类、恢复状态机与备用LQR

**Files:** Create src/solver_monitor.py、src/fallback_controller.py、tests/test_solver_monitor.py、tests/test_fallback_controller.py。

**Interfaces:**
- assess_result(result:dict,elapsed_s:float,budget_s:float,deadline_s:float)->dict：accepted、category、budget_overrun、deadline_overrun、raw_status、backend_status、reason。有限elapsed>=0、0<budget<deadline。类别accepted/accepted_inaccurate/infeasible/iteration_limit/backend_time_limit/solver_error/invalid_solution/unknown_status/budget_overrun，保留原始状态。
- 成功还要求有限u0/X/U/DU/objective、max_hard_violation≤1e-5；有EPS时EPS及max_soft_violation有限且合格。不把status=optimal当唯一接受条件。先记录原始失败原因，同时单列两个时间阈值标志。
- SolverMonitor(recovery_successes=3)；mode初始'mpc'、good_count=0。observe(assessment:dict)->dict返回mode_before/mode_after/good_count/recovered；accepted时若在fallback累加到3才恢复，否则保持fallback；拒绝重置并转fallback。reject(category:str,reason:str)->dict用于pending预算事件或busy，不伪造求解器结果。
- LqrFallbackController(K)，request(x_hat,x_ref,torque_ff=0.)->float，有限shape K=(1,2)/状态(2,)校验，计算未限幅请求；限幅和限速只在公共命令保护层做。溢出或无效估计不能发送NaN，管理闭环记录故障终止，不执行旧U。

- [ ] 写test_budget_and_deadline_are_distinct：有效结果耗时4/5ms接受，7/10ms拒绝但deadline_overrun=False，12ms拒绝且两标志True；未知状态拒绝，NaN/负耗时与非法阈值拒绝。
- [ ] 写test_bad_status_and_bad_plan_are_rejected：不可行、迭代上限、数值/接口错误、非有限u0及超残差；根据实际backend信息映射，无法判定时保留solver_error/unknown而不编造原因。
- [ ] 写test_three_consecutive_good_results_recover：failure→good→good仍fallback，第三good恢复；good→failure或busy打断，必须重新累计3次。
- [ ] 写test_fallback_uses_current_estimate：K=[[2,.5]]、x=[.2,.4]、ref=[0,0]、ff=.1，未保护请求=-.5；改x后立即改请求。非法K/状态/ff、溢出拒绝；无真值接口。
- [ ] 先RED后实现；运行两个test模块GREEN；本地提交：feat: add solver budget monitor and LQR fallback state machine。

## Task 3: 有因果时序的管理闭环与确定性作业

**Files:** Create src/managed_closed_loop.py、tests/test_managed_timing.py；现有closed_loop.py保持A–E默认行为。

**Interfaces:**
- simulate_managed_closed_loop(config:dict,*,timing_mode:str='injected',faults:dict|None=None,measurement_noise=None)->dict，返回status/failure_reason/log/jobs/plans/final_state/final_state_time_s/expected_samples。timing_mode仅'injected'或'measured_sync'。
- config新增solver:{max_iter:4000,warm_start:true}、runtime:{budget_s:.005,recovery_successes:3,nominal_completion_s:.002}、experiment_fg:{duration_s:4,target_deg:30,seed:2026,replay_samples:200,repetitions:3}。保留旧配置；FG适配为初值/估计0，位置/速度soft及其他参数如Global Constraints。
- faults以控制周期序号为键，值kind='max_iter'/'error'/'nonfinite'/'delay'及适用max_iter或completion_s。max_iter故障使用真正solver调用与本轮覆盖；error/nonfinite明确标记注入，不冒充自然OSQP故障。fault序号非负整数、延迟有限非负且与物理网格整除；未用到的故障索引记录或拒绝，不能静默丢弃。

**事件语义（必须落实）：**
- 每1ms先用过去actual做KF预测/测量更新；所有控制/备用只看当前估计。作业在控制触发时冻结estimate/reference/feedback进行求解，保存snapshot_time和job_id，绝不读取未来扰动。
- injected模式：真实solve耗时另存wall_time，但物理完成时刻仅来自明确的completion_s；正常2ms。作业未完成期间保持上次命令；到5ms无结果触发fallback，迟到结果只能丢弃，不能恢复或执行旧快照。
- 每场景一份Problem、最多一个在途作业。预算过期后不假装能中断Python/OSQP：作业仍占用worker直到注入完成。下个控制触发若仍busy，记录busy_skip，重置恢复计数并fresh fallback，不启动重叠作业。
- 同一时刻先处理既有完成事件，再处理预算事件，最后处理新控制触发；完成时间恰等预算有效，恰在新周期完成先释放worker。新作业duration=0当时处理，不能丢事件。
- 在fallback模式，每次决策重新计算备用指令；有效候选前两次仍用fresh LQR，第三次才切MPC。raw结果失败可在完成时立即转备用，不必等到预算。
- 命令保护按距上次命令更新的实际elapsed计算，不硬套10ms；初始command=0、last_command_time=0，同刻elapsed=0保持原命令。正常首次2ms完成最多变化.1Nm；5ms预算时备用最多从初始0变化.25Nm。执行器每1ms另限速.05Nm。
- measured_sync模式只记录真实阻塞完成后的预算拒绝/备用决策，物理仿真不把Python阻塞当已可靠推进；明确它只能事后发现迟到。物理时间延迟证据仅来自injected模式，不宣称这是实时线程/可抢占求解器。
- jobs保存原始状态、快照/启动/计划完成/预算触发/实际处理时刻、注入信息、接受/丢弃/忙/预算/截止标志及两类耗时。每job至多一次失败/预算计数；跨截止的未完成作业也记录deadline过期，尾部pending如实记录。
- log保留现有物理/KF/三种扭矩字段，新增mode、decision_event、selected_source、selected_request_nm、command_elapsed_s、job_id、good_count；无事件字段缺失而非假0。成功候选plans记录是否真正执行/丢弃，失败不留可执行旧解。

- [ ] 写test_late_plan_is_dropped_at_budget：7ms完成且候选请求与fallback不同；0–4ms保持，5msfresh fallback，7ms仅丢弃，无MPC应用；单列budget miss但不deadline miss。
- [ ] 写test_deadline_and_busy_skip：12ms完成跨过10ms控制触发；没有第二在途作业，busy_skip计数，迟到结果不推进恢复计数；完成后可提交新作业。
- [ ] 写test_completion_ties_and_zero_duration：5ms完成先接受；10ms完成先清理再处理新周期（该结果仍因5ms预算过期丢弃）；0ms完成不遗失，同刻命令不跳变。
- [ ] 写test_recovery_and_fresh_fallback：注入错误后连续2good不恢复，中间错误重置；第三连续good恢复。每个fallback事件selected_request独立等于-K@(当时KF-ref)，不能等于旧QP输入。
- [ ] 写test_irregular_guard_and_physical_slew：每次command_delta≤50*command_elapsed，actual每1ms≤.05、幅值≤2；拒绝非法fault/dt。独立重现KF过去actual因果顺序。
- [ ] 写test_measured_sync_does_not_claim_event_timing：测量路径真实墙钟进入monitor，日志明确模式；注入完成时间不覆盖wall_time，参数序列没有未来真值。
- [ ] 先RED后实现；运行test_managed_timing及所有课内测试GREEN；本地提交：feat: simulate causal MPC deadlines and guarded fallback。

## Task 4: F/G公平实验、指标、图和报告

**Files:** Create src/runtime_metrics.py、src/run_runtime.py、src/runtime_artifacts.py、tests/test_runtime_experiments.py；Modify src/run_experiments.py（增加fg选项）、config/mpc_engineering.yaml、README.md；Generate logs/runtime_parameter_sequence.npz、logs/runtime_benchmark.csv、logs/experiment_fg_summary.json、logs/managed_*.csv及jobs.json/plans.npz，figures/solver_runtime.png、figures/fallback_recovery.png、reports/runtime_fallback_report.md及独立验收记录。

**Interfaces:**
- make_replay_sequence(config:dict,count:int)->dict，冻结200组输入。t=i*Ts，x_hat=[.1*sin(pi*t), .1*pi*cos(pi*t)]，x_ref=[target_rad,0]，u_prev=.1*cos(.5*pi*t)。这是求解基准输入，不声称真实关节轨迹；存全部数组，warm on/off逐行使用同一数据。
- benchmark_solver(config,sequence)->dict，3轮，每轮每mode新建一个controller，随后全程复用Problem；轮次交替执行顺序降低固定先后影响。perf_counter_ns覆盖参数更新及solve，不含构造；构造耗时单列，首轮调用cold=True。保存所有状态/迭代/墙钟/可获取backend计时，不删慢样本或失败。
- calculate_runtime_metrics(rows,budget_s,deadline_s)->dict，按mode/repetition分组，cold、steady和all分别统计mean/p95/p99/max、成功率、迭代及预算/截止超时率；missing为None。管理闭环另外统计控制周期/提交job/忙跳过分母、fallback进入次数/持续时间、迟到丢弃次数及基础跟踪/扭矩/真实越界指标。
- run_runtime_experiments(config:dict,output_dir:Path)->dict；CLI python -m lesson13_mpc_engineering.src.run_experiments --experiment fg。原ab/cde及历史产物不变。
- F除冻结参数warm on/off×3，还运行一组measured_sync闭环记录真实预算拒绝；说明同步模式不是物理截止保障，不能与注入时间混成同一性能表。
- G默认四组，共享预生成位置噪声：g_nominal（2ms完成）；g_iteration_limit（周期0/1/2真正max_iter=1，之后4000）；g_deadline（周期100完成7ms、101完成12ms，其余2ms）；g_error_recovery（周期100 error、102 nonfinite，其余2ms）。4s默认长度覆盖这些索引；短测试使用显式短场景索引，不丢弃未发生故障。
- baseline也经相同命令/实际保护；每组有效配置/seed/版本/注入表与所有失败状态完整保存。F/G计时不并行跑其他性能测试；非硬实时保证。
- 报告按公式→监控/备用/事件代码→真实计时与注入图→实测指标→边界讲解。参数全部来自有效配置；合成延迟在图标题/轴及日志中明确标记。fallback不保证软速度或机械边界，未完成场景不混入成功平均。

- [ ] 写test_replay_is_identical_and_problem_is_reused：对相同sequence每mode/repetition输入逐点一致；实际spy构造/Problem标识证明每系列只建一次；cold首调用保留，既不硬写warm快也不抹去失败。
- [ ] 写test_runtime_statistics_match_hand_fixture：wall=[.004,.005,.007,.012]s，预算严格>5ms计2/4，deadline严格>10ms计1/4，mean=7ms、p95=11.25ms、max=12ms；cold首项分列后steady mean=8ms、p95=11.5ms。缺失backend不是0。
- [ ] 写test_fg_runner_covers_scenarios_and_reconciles：短合法配置/显式故障索引生成F和四G；从CSV/jobs独立重算预算/截止/忙/丢弃/恢复，噪声一致、配置未修改。所有命令与实际硬限制逐点复核，失败/终态证据可读。
- [ ] 写test_old_candidate_never_applied_after_fault：导出job_id→executed记录证明超时、max_iter失败、error和nonfinite不会执行陈旧U；恢复只在3次有效结果后发生，零次/一次成功不恢复。
- [ ] 先RED后实现，短测试全部GREEN后独立运行完整F/G，不预设timing或恢复性能。视觉检查图表全程/故障窗口和分组单位；不能把5ms预算线误标为10ms周期。
- [ ] 跑全仓conda run -n robot-control python -m unittest discover -v；A–E默认关闭新管理层时对照既有物理轨迹，新增状态/失败残差/非法值测试不退化。保存真实验证结果，完成一次只读独立复核并解决Important/Critical。
- [ ] 本地提交准确文件：feat: demonstrate MPC solver timing and fallback recovery；只交付本地结果，不上传。

## Self-review / 阶段边界

Task1提供本轮原始诊断与可配置限制；Task2只判结果和管理模式；Task3用模式决定每个事件的实际指令；Task4保存独立可复核记录。所有共享接口已写明。真实墙钟与注入完成时刻分开、预算与周期分开、每job与每control cycle分母分开。没有引入异步实机执行器、硬实时线程、直接OSQP矩阵实现或综合鲁棒性保证；这些不属于本阶段交付。
