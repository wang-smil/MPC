# 第三课：ZOH 离散化与极点对比

## 目标

将第一课的连续质量—弹簧—阻尼状态空间模型离散化，并比较 1 ms、10 ms 与 500 ms 采样周期得到的离散矩阵和极点位置。

连续模型为：

```text
x_dot = A x + B u

A = [[0, 1], [-4, -0.8]]
B = [[0], [1]]
```

其中状态为位置和速度，输入为外力。

## 连续控制与数字控制

连续模型描述的是任意时刻的状态变化：

```text
x_dot = A x + B u
```

真实机器人、PLC 和 MCU 只能每隔一个采样周期读取传感器、计算控制量和写出命令，因此它们使用的是离散模型：

```text
x[k+1] = Ad x[k] + Bd u[k]
```

## 什么是 ZOH

ZOH（Zero-Order Hold，零阶保持）假设：一次计算出的控制输入会在整个采样周期内保持不变，直到下一次更新。它符合数字执行器“采样一次、保持一段时间”的常见工作方式。

本实验通过：

```python
scipy.signal.cont2discrete(..., method="zoh")
```

得到 `Ad` 与 `Bd`。

## 为什么采样周期属于模型

连续矩阵 `A`、`B` 固定，并不表示离散矩阵固定。采样周期改变后，状态在一次更新之间演化的时间改变，因此 `Ad`、`Bd` 也会改变。

这就是为什么同一个机械对象在 1000 Hz 机器人控制器和 100 Hz PLC 上，需要分别验证离散模型与控制器。

## 运行

```powershell
conda activate robot-control
cd D:\MPC_learn
python .\lesson03_discretization\src\discretize_demo.py
```

程序会打印 1 ms、10 ms、500 ms 下的 `Ad`、`Bd`，并保存：

```text
lesson03_discretization/Ad_Bd_output.txt
lesson03_discretization/figures/pole_compare.png
```

## 测试

```powershell
python -m unittest lesson03_discretization.tests.test_discretize_demo -v
```

测试覆盖：

- 第一课连续模型矩阵；
- 1 ms 与 10 ms 的 ZOH 离散矩阵尺寸；
- 采样周期改变时 `Ad`、`Bd` 确实改变；
- 非法采样周期；
- 文本输出文件。

## 实验 2：连续极点与离散极点

程序同时绘制 `pole_compare.png`：左图是连续系统的 s 平面极点，右图是离散系统的 z 平面极点与单位圆。

- 连续系统中，所有极点实部小于 0，系统稳定；
- 离散系统中，所有极点模长小于 1、位于单位圆内，系统稳定；
- 对精确 ZOH，有极点映射 `z = exp(s * Ts)`；
- `Ts=500 ms` 时，离散极点位置会明显变化，但仍在单位圆内。它说明采样周期会改变每一步的数字动态，并不意味着该稳定对象失稳。

## 本次结论

`Ts=1 ms`、`Ts=10 ms` 与 `Ts=500 ms` 得到的 `Ad`、`Bd` 和离散极点都不同。因此采样周期不是普通程序参数，而是离散系统模型的一部分。连续稳定性的左半平面判据，经过精确 ZOH 映射后对应离散稳定性的单位圆内判据。
