from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import cont2discrete


ROOT = Path(__file__).resolve().parents[1]


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


def continuous_poles() -> np.ndarray:
    """返回连续状态矩阵 A 的极点。"""

    a, _, _, _ = build_continuous_model()
    return np.linalg.eigvals(a)


def discrete_poles(sample_time_s: float) -> np.ndarray:
    """返回指定采样周期下离散状态矩阵 Ad 的极点。"""

    ad, _, _, _, _ = discretize_zoh(sample_time_s)
    return np.linalg.eigvals(ad)


def plot_pole_comparison(sample_times_s: tuple[float, ...]) -> Path:
    """绘制连续极点和不同采样周期的离散极点。"""

    figure, (s_axes, z_axes) = plt.subplots(1, 2, figsize=(10, 4.5))

    s_poles = continuous_poles()
    s_axes.scatter(
        np.real(s_poles),
        np.imag(s_poles),
        label="Continuous poles",
    )
    s_axes.axhline(0.0, color="black", linewidth=0.8)
    s_axes.axvline(0.0, color="black", linewidth=0.8)
    s_axes.set_title("s-plane")
    s_axes.set_xlabel("Real")
    s_axes.set_ylabel("Imaginary")
    s_axes.grid(True)
    s_axes.legend()

    angle = np.linspace(0.0, 2.0 * np.pi, 400)
    z_axes.plot(
        np.cos(angle),
        np.sin(angle),
        "k--",
        label="Unit circle",
    )
    for sample_time_s in sample_times_s:
        poles = discrete_poles(sample_time_s)
        z_axes.scatter(
            np.real(poles),
            np.imag(poles),
            label=f"Ts = {sample_time_s:g} s",
        )

    z_axes.axhline(0.0, color="black", linewidth=0.8)
    z_axes.axvline(0.0, color="black", linewidth=0.8)
    z_axes.set_aspect("equal", adjustable="box")
    z_axes.set_title("z-plane")
    z_axes.set_xlabel("Real")
    z_axes.set_ylabel("Imaginary")
    z_axes.grid(True)
    z_axes.legend()

    figure.tight_layout()
    output_dir = ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "pole_compare.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)
    return figure_path


def format_results(sample_times_s: tuple[float, ...]) -> str:
    """格式化多个采样周期下的 ZOH 离散矩阵。"""

    blocks = ["Mass-Spring-Damper ZOH Discretization"]

    for sample_time_s in sample_times_s:
        ad, bd, _, _, dt = discretize_zoh(sample_time_s)
        blocks.extend(
            [
                "",
                f"Ts = {dt:g} s",
                "Ad =",
                np.array2string(
                    ad,
                    precision=8,
                    suppress_small=True,
                ),
                "Bd =",
                np.array2string(
                    bd,
                    precision=8,
                    suppress_small=True,
                ),
            ]
        )

    return "\n".join(blocks) + "\n"


def write_output(output_text: str) -> Path:
    """将离散化结果保存为课程实验文本。"""

    output_path = ROOT / "Ad_Bd_output.txt"
    output_path.write_text(output_text, encoding="utf-8")
    return output_path


def main() -> None:
    output_text = format_results((0.001, 0.01))
    output_path = write_output(output_text)

    print(output_text, end="")
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
