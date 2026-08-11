import numpy as np

from discretize_demo import discretize_zoh


def closed_loop_matrix(
    sample_time_s: float,
    kp: float,
    kd: float,
) -> np.ndarray:
    """返回零参考 PD 状态反馈后的离散闭环矩阵。"""

    ad, bd, _, _, _ = discretize_zoh(sample_time_s)
    gain = np.array([[kp, kd]])
    return ad - bd @ gain


def closed_loop_poles(
    sample_time_s: float,
    kp: float,
    kd: float,
) -> np.ndarray:
    """返回离散 PD 闭环系统的极点。"""

    return np.linalg.eigvals(closed_loop_matrix(sample_time_s, kp, kd))


def is_stable(poles: np.ndarray) -> bool:
    """判断离散极点是否全部严格位于单位圆内。"""

    return bool(np.all(np.abs(poles) < 1.0))
