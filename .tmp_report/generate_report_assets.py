from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import task8_reverse as task


OUT = Path(r"E:\Code\arm_pracctice\.tmp_report\assets")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def make_formula_figure():
    fig = plt.figure(figsize=(9.2, 2.45), dpi=220, facecolor="white")
    equations = [
        r"$q_k=q_s+\frac{k}{N}(q_t-q_s),\qquad \Delta t=\frac{T}{N}$",
        r"$x_e=L_1\cos q_2+L_2\cos(q_2+q_3)+L_3\cos(q_2+q_3+q_4)$",
        r"$y_e=L_1\sin q_2+L_2\sin(q_2+q_3)+L_3\sin(q_2+q_3+q_4)$",
        r"$x_4=x_e-L_3\cos\alpha,\quad y_4=y_e-L_3\sin\alpha,\quad q_4=\alpha-q_2-q_3$",
    ]
    ys = [0.82, 0.59, 0.36, 0.13]
    for equation, y in zip(equations, ys):
        fig.text(0.5, y, equation, ha="center", va="center", fontsize=14, color="#111111")
    fig.savefig(OUT / "equations.png", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def endpoint_and_joints(pose_deg):
    q2, q3, q4 = np.deg2rad(pose_deg[1:4])
    q2 += task.JOINT2_ZERO_DIRECTION_RAD
    p0 = np.array([0.0, 0.0])
    p1 = p0 + task.LINK_1_MM * np.array([np.cos(q2), np.sin(q2)])
    p2 = p1 + task.LINK_2_MM * np.array([np.cos(q2 + q3), np.sin(q2 + q3)])
    p3 = p2 + task.LINK_3_MM * np.array(
        [np.cos(q2 + q3 + q4), np.sin(q2 + q3 + q4)]
    )
    return np.vstack((p0, p1, p2, p3))


def make_trajectory_figures():
    center, start, circle, offset = task.build_reversed_circle_trajectory(18.0)
    poses = np.vstack((start, circle))
    endpoints = np.array([endpoint_and_joints(pose)[-1] for pose in poses])

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=220, facecolor="white")
    ax.axvline(0.0, color="#8A8A8A", linestyle="--", linewidth=1.1, label="零位竖直线")
    zero_points = np.array(
        [[0.0, 0.0], [0.0, 120.0], [0.0, 240.0], [0.0, 380.0]]
    )
    ax.plot(zero_points[:, 0], zero_points[:, 1], "o--", color="#9A9A9A", linewidth=1.1)
    for idx in (0, 100, 200, 300):
        joints = endpoint_and_joints(poses[idx])
        ax.plot(joints[:, 0], joints[:, 1], "o-", color="#7DA7C8", alpha=0.34, linewidth=1.5)
    ax.plot(endpoints[:, 0], endpoints[:, 1], color="#C7473A", linewidth=2.4, label="末端目标圆")
    ax.scatter([center[0]], [center[1]], marker="x", s=70, color="#111111", label="圆心 (0, 240 mm)")
    ax.set_title("三连杆末端圆周轨迹与零位竖直线", fontsize=14, pad=10)
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    ax.axis("equal")
    ax.grid(True, color="#E5E5E5", linewidth=0.7)
    ax.legend(loc="best", frameon=False)
    ax.text(
        0.02,
        0.02,
        f"L1=120 mm  L2=120 mm  L3=140 mm\n方向偏置={offset:.1f}°  半径=18 mm",
        transform=ax.transAxes,
        fontsize=9.5,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(OUT / "endpoint_circle.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.3), dpi=220, facecolor="white")
    progress = np.linspace(0.0, 1.0, len(circle), endpoint=True)
    colors = ["#315A7D", "#C7473A", "#3E7C59"]
    for column, label, color in zip((1, 2, 3), ("Joint 2", "Joint 3", "Joint 4"), colors):
        ax.plot(progress, circle[:, column], label=label, linewidth=2.0, color=color)
    ax.axhline(90.0, color="#9A9A9A", linestyle="--", linewidth=0.9)
    ax.axhline(-90.0, color="#9A9A9A", linestyle="--", linewidth=0.9)
    ax.set_title("单圈关节角变化", fontsize=14, pad=10)
    ax.set_xlabel("圆周进度")
    ax.set_ylabel("舵机角度 / °")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-95.0, 95.0)
    ax.grid(True, color="#E5E5E5", linewidth=0.7)
    ax.legend(loc="best", frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "joint_angles.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    make_formula_figure()
    make_trajectory_figures()
    print(OUT)
