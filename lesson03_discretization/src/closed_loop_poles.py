from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from discretize_demo import discretize_zoh


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CASES = {
    "weak": (2.0, 0.2),
    "medium": (15.0, 2.0),
    "strong": (80.0, 2.0),
}


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


def sweep_kp(
    kp_values: np.ndarray,
    kd: float,
    sample_time_s: float,
) -> np.ndarray:
    """固定 Kd 时，记录每个 Kp 对应的两条闭环极点轨迹。"""

    return np.array(
        [
            np.sort_complex(closed_loop_poles(sample_time_s, kp, kd))
            for kp in kp_values
        ]
    )


def _draw_unit_circle(axes: plt.Axes) -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 500)
    axes.plot(
        np.cos(angle),
        np.sin(angle),
        "k--",
        label="Unit circle",
    )
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.axvline(0.0, color="black", linewidth=0.8)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("Real")
    axes.set_ylabel("Imaginary")
    axes.grid(True)


def _figure_path(filename: str) -> Path:
    output_dir = ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def plot_closed_loop_poles(
    cases: dict[str, tuple[float, float]],
) -> Path:
    """绘制三个典型 PD 增益下的 10 ms 闭环极点。"""

    figure, axes = plt.subplots(figsize=(7, 7))
    _draw_unit_circle(axes)

    for name, (kp, kd) in cases.items():
        poles = closed_loop_poles(0.01, kp, kd)
        axes.scatter(
            np.real(poles),
            np.imag(poles),
            label=f"{name}: Kp={kp:g}, Kd={kd:g}",
        )

    axes.set_title("Discrete PD Closed-Loop Poles (Ts = 10 ms)")
    axes.legend()
    figure.tight_layout()

    figure_path = _figure_path("closed_loop_poles.png")
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)
    return figure_path


def plot_gain_sweep(
    kp_values: np.ndarray,
    kd: float,
    sample_time_s: float,
) -> Path:
    """绘制固定 Kd 时，闭环极点随 Kp 移动的轨迹。"""

    history = sweep_kp(kp_values, kd, sample_time_s)
    figure, axes = plt.subplots(figsize=(7, 7))
    _draw_unit_circle(axes)

    for pole_index in range(history.shape[1]):
        axes.plot(
            np.real(history[:, pole_index]),
            np.imag(history[:, pole_index]),
            label=f"Pole {pole_index + 1}",
        )

    axes.set_title(f"Kp Sweep (Kd = {kd:g}, Ts = {sample_time_s:g} s)")
    axes.legend()
    figure.tight_layout()

    figure_path = _figure_path("gain_sweep.png")
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)
    return figure_path


def _format_poles(poles: np.ndarray) -> str:
    return np.array2string(
        poles,
        precision=6,
        suppress_small=True,
    )


def _case_summary(
    sample_time_s: float,
    kp: float,
    kd: float,
) -> str:
    poles = closed_loop_poles(sample_time_s, kp, kd)
    maximum_magnitude = float(np.max(np.abs(poles)))
    return (
        f"poles={_format_poles(poles)}, "
        f"max |z|={maximum_magnitude:.6f}, "
        f"stable={is_stable(poles)}"
    )


def _build_report_text() -> str:
    """构建离散 PD 闭环极点实验报告。"""

    lines = [
        "# 离散 PD 闭环极点分析",
        "",
        "## 1. 开环极点与闭环极点",
        "",
        "开环离散模型使用 `x[k+1] = Ad x[k] + Bd u[k]`。",
        "零参考 PD 控制写成 `u[k] = -K x[k]`，其中 `K = [[Kp, Kd]]`。",
        "因此闭环矩阵是 `Acl = Ad - Bd @ K`，应分析的是",
        "`eig(Acl)`，而不是单独的 `eig(Ad)`。",
        "",
        "## 2. 三种典型增益（Ts = 10 ms）",
        "",
        "| Case | Kp | Kd | Closed-loop result |",
        "|---|---:|---:|---|",
    ]

    for name, (kp, kd) in DEFAULT_CASES.items():
        lines.append(
            f"| {name} | {kp:g} | {kd:g} | "
            f"{_case_summary(0.01, kp, kd)} |"
        )

    lines.extend(
        [
            "",
            "## 3. Kp 增大时极点为什么移动",
            "",
            "固定 `Kd=2` 时，改变 `Kp` 会直接改变 `Bd @ K`，",
            "也就改变 `Acl` 的元素和特征值。`gain_sweep.png` 记录的是",
            "这两条闭环极点轨迹，而不是开环 Plant 的极点轨迹。",
            "增大 Kp 不能简单理解为控制一定更好：极点可能靠近或跨出单位圆。",
            "",
            "## 4. 同一组 K 对采样周期的敏感性",
            "",
            "固定 `Kp=15`、`Kd=2`：",
            "",
            "| Ts | Closed-loop result |",
            "|---:|---|",
        ]
    )

    for sample_time_s in (0.001, 0.01, 0.05, 0.1):
        lines.append(
            f"| {sample_time_s * 1000:g} ms | "
            f"{_case_summary(sample_time_s, 15.0, 2.0)} |"
        )

    lines.extend(
        [
            "",
            "同一组 K 在不同 Ts 下对应不同的 `Ad`、`Bd`，所以 `Acl` 和",
            "闭环极点也会改变。模型离散后开环稳定，不等于加入数字 PD 后",
            "闭环一定稳定。",
            "",
            "## 5. 理论极点分析的工程边界",
            "",
            "这里的结果只针对名义线性模型。它没有包含编码器噪声、速度差分与",
            "滤波、转矩饱和、命令延迟、摩擦非线性、采样抖动和参数不确定性。",
            "因此极点分析用于解释 nominal stability，不能替代 lesson02 的",
            "normal / disturbance / sensor_fault 仿真，更不能替代实机测试。",
            "",
        ]
    )
    return "\n".join(lines)


def write_analysis_report() -> Path:
    """写入闭环极点、采样周期和工程边界的实验报告。"""

    output_dir = ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "closed_loop_analysis.md"
    report_path.write_text(_build_report_text(), encoding="utf-8")
    return report_path


def main() -> None:
    """运行三个典型工况、Kp 扫描和报告生成。"""

    print("Discrete PD closed-loop pole analysis (Ts = 10 ms)")
    for name, (kp, kd) in DEFAULT_CASES.items():
        print(f"{name}: Kp={kp:g}, Kd={kd:g}")
        print(f"  {_case_summary(0.01, kp, kd)}")

    case_figure_path = plot_closed_loop_poles(DEFAULT_CASES)
    sweep_figure_path = plot_gain_sweep(
        np.linspace(0.0, 200.0, 300),
        kd=2.0,
        sample_time_s=0.01,
    )
    report_path = write_analysis_report()

    print(f"\nClosed-loop pole figure: {case_figure_path}")
    print(f"Gain sweep figure: {sweep_figure_path}")
    print(f"Analysis report: {report_path}")


if __name__ == "__main__":
    main()
