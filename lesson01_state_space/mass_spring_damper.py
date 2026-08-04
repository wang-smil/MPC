from pathlib import Path

import control as ct
import matplotlib.pyplot as plt
import numpy as np
def build_system(
    mass: float,
    damping: float,
    stiffness: float,
) -> ct.StateSpace:
    """建立质量-弹簧-阻尼系统的状态空间模型。"""

    if mass <= 0:
        raise ValueError("mass 必须大于 0。")
    if damping < 0 or stiffness < 0:
        raise ValueError("damping 和 stiffness 不能为负数。")

    A = np.array(
        [
            [0.0, 1.0],
            [-stiffness / mass, -damping / mass],
        ]
    )

    B = np.array(
        [
            [0.0],
            [1.0 / mass],
        ]
    )

    C = np.eye(2)
    D = np.zeros((2, 1))

    return ct.ss(
        A,
        B,
        C,
        D,
        inputs=["force"],
        outputs=["position", "velocity"],
        states=["position", "velocity"],
    )


def main() -> None:
    mass = 8.0
    damping = 8
    stiffness = 4.0

    system = build_system(
        mass=mass,
        damping=damping,
        stiffness=stiffness,
    )

    print("状态空间系统：")
    print(system)

    print("\nA矩阵：")
    print(system.A)

    print("\n系统极点：")
    print(ct.poles(system))
    time = np.linspace(0.0, 30.0, 1201)

    # 前1秒没有外力，从1秒开始施加1 N恒定外力
    force = np.zeros_like(time)
    force[time >= 1.0] = 1.0

    response = ct.forced_response(
        system,
        T=time,
        U=force,
        X0=[0.0, 0.0],
    )

    position = response.outputs[0]
    velocity = response.outputs[1]

    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(time, position, label="Position")
    plt.plot(time, velocity, label="Velocity")
    plt.axvline(1.0, linestyle="--", label="Force applied")

    plt.xlabel("Time / s")
    plt.ylabel("State value")
    plt.title("Mass-Spring-Damper State Response")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    figure_path = output_dir / "state_response.png"
    plt.savefig(figure_path, dpi=200)
    plt.show()

    print(f"\n图片已保存到：{figure_path}")


if __name__ == "__main__":
    main()