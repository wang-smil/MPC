import numpy as np
from scipy.signal import cont2discrete


def build_continuous_model(
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回第一课质量—弹簧—阻尼系统的连续状态空间矩阵。"""

    a = np.array(
        [
            [0.0, 1.0],
            [-4.0, -0.8],
        ]
    )
    b = np.array(
        [
            [0.0],
            [1.0],
        ]
    )
    c = np.eye(2)
    d = np.zeros((2, 1))
    return a, b, c, d


def discretize_zoh(
    sample_time_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """以零阶保持将连续状态空间模型离散化。"""

    if sample_time_s <= 0.0:
        raise ValueError("sample_time_s must be positive")

    a, b, c, d = build_continuous_model()
    ad, bd, cd, dd, dt = cont2discrete(
        (a, b, c, d),
        sample_time_s,
        method="zoh",
    )
    return ad, bd, cd, dd, float(dt)
