# 第 12 课：线性 MPC 预测基础

实验 A 先完成 MPC 的“预测器”：把离散模型逐步递推得到的未来状态，与矩阵形式

\[
X = F x_0 + G U
\]

逐项对上。实验 B 在同一个模型上加入无约束二次规划，并与 LQR 对照。

## 运行实验

在仓库根目录执行：

```powershell
conda run -n robot-control python -m unittest discover lesson12_linear_mpc/tests -v
conda run -n robot-control python -m lesson12_linear_mpc.src.run_prediction_validation
```

验证器会把位置、速度的递推轨迹与矩阵预测轨迹画在一起，并打印最大绝对误差。图保存在 `lesson12_linear_mpc/figures/prediction_validation.png`。

## 从代码读懂预测

### 1. 模型与采样周期

`config/mpc.yaml` 保存辨识得到的惯量、阻尼，以及仿真各部分的采样周期。`src/model_loader.py` 复用第 8 课的连续模型和 ZOH 离散化函数，不再复制模型推导。

- 被控对象和状态估计器：`1 ms`
- 本次 MPC 预测模型：`10 ms`
- 预测步数：`N = 20`，覆盖 `0.2 s`

这里必须用 `mpc.dt_s` 离散模型。MPC 每次以 10 ms 的步长规划；实际闭环中，机器人/仿真与估计器仍可在更快的 1 ms 时钟上运行。

### 2. `x0`、`U` 和 `X`

离散模型是

\[
x_{k+1}=A_d x_k+B_d u_k,
\qquad
x_k=\begin{bmatrix}q_k\\\dot q_k\end{bmatrix}.
\]

代码中，状态行向量按时间排列：

- `x0`：当前初始状态 `[位置, 速度]`。
- `U`：候选未来输入，形状 `(N, nu)`；本例单输入时每一行是一个未来转矩 `u_k`。
- `X`：预测未来状态 `x1, ..., xN`，形状 `(N, nx)`，不包含 `x0`。
- `rollout_states` 返回 `x0, x1, ..., xN`，所以形状是 `(N+1, nx)`；比较时取 `[1:]`。

以三步、单输入为例，递推展开为：

\[
\begin{aligned}
x_1 &= A_d x_0+B_d u_0,\\
x_2 &= A_d^2 x_0+A_dB_d u_0+B_d u_1,\\
x_3 &= A_d^3 x_0+A_d^2B_d u_0+A_dB_d u_1+B_d u_2.
\end{aligned}
\]

把共同系数按块排好，就是

\[
\underbrace{\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}}_X
=
\underbrace{\begin{bmatrix}A_d\\A_d^2\\A_d^3\end{bmatrix}}_F x_0
+
\underbrace{\begin{bmatrix}
B_d&0&0\\
A_dB_d&B_d&0\\
A_d^2B_d&A_dB_d&B_d
\end{bmatrix}}_G
\underbrace{\begin{bmatrix}u_0\\u_1\\u_2\end{bmatrix}}_U.
\]

所以 `F x0` 是“如果之后不再施加新输入，当前状态自身会如何演化”的部分；`G U` 是每一步候选输入对后续状态造成的累计影响。矩阵 `G` 是下三角块结构，因为未来输入不能倒过来影响过去状态。

### 3. 两条独立的计算路径

- `prediction.py / build_prediction_matrices`：构造 `F`、`G`。
- `prediction.py / predict_states_matrix`：计算 `F @ x0 + G @ U`。输入以 C 顺序按时间展平，对应 `u0, u1, ...`。
- `prediction.py / rollout_states`：不使用 `F`、`G`，只按状态方程一步步递推。它是独立的对照路径。
- `run_prediction_validation.py`：固定随机种子生成一组候选转矩，比较两条路径并保存图。

测试还用一个手算的双输入小系统检查了 `G` 的块位置和时间优先排列，避免只在单输入例子里碰巧通过。

## 实验 A 结果与边界

本次运行得到的最大绝对预测差约为 `1.665e-16`，远小于验收阈值 `1e-10`；两条曲线在图上应基本重合。这说明在同一离散模型与输入序列下，矩阵展开和逐步递推一致。

实验 A 还没有“选择最优转矩”：这里的 `U` 是为了验算而生成的候选序列。实验 B 才加入代价函数和 QP 求解器；执行器约束留给后续实验。

## 实验 B：无约束 MPC 与 LQR

在仓库根目录执行：

```powershell
conda run -n robot-control python -m lesson12_linear_mpc.src.run_experiment_b
```

需要安装的求解依赖是 `cvxpy==1.9.3` 和 `osqp==1.1.3`，已写入根目录 `requirements.txt`。脚本会运行 LQR 和预测步数为 `1、5、20` 的 MPC，生成 `logs/experiment_b.csv`、`figures/lqr_vs_mpc.png` 和 `reports/unconstrained_mpc_vs_lqr.md`。

代码按职责分为三层：`mpc_controller.py` 建立非 Condensed QP 并返回第一个可执行转矩；`closed_loop.py` 每 1 ms 更新植物、编码器测量和 KF，每 10 ms 更新控制并保持转矩；`run_experiment_b.py` 使用同一组噪声运行四个工况并输出指标。控制器只接收 KF 后验估计 `x_hat`。

本实验的终端权重 `P` 是同一 10 ms 模型的 DARE 解。在无约束条件下，任一上述时域的首个 MPC 转矩都应与 `u=-K(x_hat-x_ref)` 一致，数值测试允许 `1e-4 N·m` 误差。图中的曲线可能重合，这正是该特定理论条件下预期的结果。转矩没有限幅，本实验只是离线仿真，不能把峰值转矩当作硬件可执行命令。
