# 第 5 课：单轴关节系统辨识

本课从实际施加扭矩和带噪编码器位置中估计关节惯量 `J` 与粘性阻尼 `b`。辨识器拟合 `torque = J * acceleration + b * velocity`；真实仿真模型还可以包含库仑摩擦，因此可用于观察模型失配。

运行测试：

```powershell
conda run -n robot-control python -m unittest lesson05_system_identification.tests.test_sysid -v
```

运行全部实验：

```powershell
conda run -n robot-control python .\lesson05_system_identification\src\run_sysid.py
```

输出位于 `logs/`、`figures/` 和 `reports/`。`normal`、`poor_excitation`、`noise_raw`、`noise_filtered` 与 `model_mismatch` 分别用于建立基准、检查激励质量、检查差分噪声放大和诊断未建模摩擦。
