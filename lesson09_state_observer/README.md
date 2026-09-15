# Lesson 09：离散 Luenberger Observer

本课将第 5/6 课辨识出的单关节模型与第 8 课 Balanced LQR 接起来：编码器只测位置，Observer 估计位置和速度，LQR 再使用估计状态闭环控制。

## 数据链

```text
true plant → position encoder q_measured → DiscreteObserver → x_hat
                                                       ↓
reference → LQR u = -K(x_hat - x_ref) → torque clip → u_applied
                                                       └→ plant and observer
```

Observer 的更新为：

\[
\hat x_{k+1}=A_d\hat x_k+B_du_{\mathrm{applied},k}
+L_d(q_{m,k}-C\hat x_k).
\]

注意：Observer 只接收 `measurement` 和 `control_input` 两个标量。它不知道仿真器内部的 `q_true`、`dq_true`；并且必须使用限幅后的 `u_applied`，而不是控制器请求的 `u_unsat`。

## 目录职责

- `model_loader.py`：加载已辨识的 \(\hat J,\hat b\)，构造 1 ms ZOH 模型，并重新计算 Balanced LQR 增益。
- `observability.py`：计算 \(\mathcal O=[C;CA_d]\) 的 rank、condition number 和 \(\sigma_{min}\)。
- `observer_design.py`：把 LQR 离散极点映射回连续时间，按 2×/4×/10× 放快，再设计 \(L_d\)。
- `observer.py`：预测 + innovation 校正；innovation 为 \(q_m-\hat q\)。
- `velocity_estimators.py`：原始差分与低通差分两种工程对照。
- `closed_loop.py`：真实 Plant、噪声编码器、Observer、LQR、饱和器的信号边界。
- `metrics.py`：估计、tracking、innovation 与扭矩指标。
- `run_experiments.py`：生成 CSV、图表与工程报告。

## 运行

在本仓库根目录运行：

```powershell
conda run -n robot-control python -m lesson09_state_observer.src.run_experiments
conda run -n robot-control python -m unittest discover lesson09_state_observer/tests -v
```

输出保存在 `logs/`、`figures/` 和 `reports/`。

## 实验

1. `convergence`：无噪声、初始真值与估计值不一致，验证估计误差收敛。
2. `observer_2x` / `observer_4x` / `observer_10x`：比较收敛速度与噪声进入估计速度、扭矩的权衡。
3. `velocity_raw_difference` / `velocity_filtered_difference` / `velocity_observer`：比较三种速度来源。
4. `payload_plus_30_percent`：真实惯量为 \(1.3\hat J\)，Observer 仍使用标称模型。
5. `encoder_bias`：编码器恒定偏置 0.5°，展示固定状态 Observer 的局限。
6. `normal` / `disturbance` / `stress`：正式验收工况，分别为正常噪声、+30% payload 与 10× 编码器噪声。

## 企业上机顺序

先检查编码器单位、方向和时间戳；再用记录的 `q_measured,u_applied` 离线运行 Observer，检查 \(\hat q,\hat{\dot q}\) 和 innovation；之后低带宽闭环，再逐步加入噪声、payload 和更高带宽。

若 innovation 长期有偏置，优先检查 encoder offset、模型误差、外扰与实际输入是否和 Observer 使用的输入一致。生产系统通常会进一步加入 bias state、传感器融合以及 Kalman/EKF。
