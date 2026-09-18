# Lesson 10：离散 Kalman Filter

本课在第 6 课辨识模型、第 8 课 Balanced LQR 和第 9 课固定增益 Observer 的基础上，实现位置编码器单传感器的离散 Kalman Filter。KF 每个采样周期接收当前编码器位置与上一周期的**实际执行扭矩**，输出位置、速度、协方差、Kalman 增益、innovation 和 NIS。

## 运行

```powershell
conda run -n robot-control python -m lesson10_kalman_filter.src.run_experiments
conda run -n robot-control python -m unittest discover lesson10_kalman_filter/tests -v
```

## 实验产物

- `baseline_confidence.png`：状态估计与 ±3σ 置信区间。
- `r_q_tuning.png`：错误假设 R 或 Q 时的增益、误差权衡。
- `estimator_comparison.png`：Raw Difference、LPF、Luenberger、Kalman 的公平比较。
- `robustness.png`：Normal、负载扰动、10× 编码器噪声、30% payload 下的 innovation/NIS；启动 0.1 s 被排除，避免初值尖峰遮住稳态细节。

真实上机时，先通过静止编码器记录给 R 一个起点；Q 则需要用动态残差、NIS、扭矩饱和和独立验证运动迭代校准。不要把 LQR 的代价权重 Q/R 与 Kalman Filter 的噪声协方差 Q/R 混为一谈。
