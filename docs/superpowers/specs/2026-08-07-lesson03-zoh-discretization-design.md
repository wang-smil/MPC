# 第三课实验 1：ZOH 离散化设计

## 目标

建立独立的 `lesson03_discretization` 小实验，将第一课质量—弹簧—阻尼连续状态空间模型在两个采样周期下使用零阶保持（ZOH）离散化，并保存可复核矩阵输出。

## 范围

本次只完成实验 1：比较 `Ts=0.001 s` 与 `Ts=0.01 s` 的 `Ad`、`Bd`。不绘制连续/离散极点图，不实现离散控制器，不修改第一课和第二课代码。

## 模型

沿用第一课基准参数：

```text
m = 1.0 kg
c = 0.8 N·s/m
k = 4.0 N/m
```

连续模型为：

```text
x_dot = A x + B u

A = [[0, 1], [-4, -0.8]]
B = [[0], [1]]
C = I
D = 0
```

## 实现

新增：

```text
lesson03_discretization/
├─ src/discretize_demo.py
├─ tests/test_discretize_demo.py
├─ README.md
├─ requirements.txt
└─ Ad_Bd_output.txt
```

`discretize_demo.py` 提供三个小接口：

```python
def build_continuous_model() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
def discretize_zoh(sample_time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]
def format_results(sample_times_s: tuple[float, ...]) -> str
```

`discretize_zoh()` 使用 `scipy.signal.cont2discrete` 和 `method="zoh"`。采样周期必须大于零，否则抛出 `ValueError`。

脚本入口计算 1 ms、10 ms 两组结果，将格式化文本写到 `Ad_Bd_output.txt` 并同时打印到终端。

## 验收标准

1. 两种采样周期都能生成 2×2 的 `Ad` 和 2×1 的 `Bd`；
2. 返回的离散采样周期等于输入值；
3. 1 ms 与 10 ms 的 `Ad`、`Bd` 不相同；
4. ZOH 离散模型的矩阵输出被保存到文本文件；
5. README 说明连续模型、数字模型、ZOH 和采样周期为何属于模型；
6. 自动测试覆盖矩阵尺寸、采样周期差异和非法采样周期。

## 预期学习结论

连续矩阵 `A`、`B` 固定时，离散矩阵 `Ad`、`Bd` 仍会随采样周期改变；因此采样周期是离散模型定义的一部分，而不是可忽略的程序参数。
