"""生成实验四报告所需的真实仿真图和数值数据。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import 实验四 as exp4


OUTPUT_DIR = Path(__file__).with_name("实验报告素材")


def robot_points(robot, q: np.ndarray) -> np.ndarray:
    transforms = robot.fkine_all(q)
    return np.asarray(transforms.t, dtype=float)


def save_trajectory_csv(
    path: Path,
    target: np.ndarray,
    actual: np.ndarray,
    q: np.ndarray,
    errors: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "sample",
                "target_x_mm",
                "target_y_mm",
                "target_z_mm",
                "actual_x_mm",
                "actual_y_mm",
                "actual_z_mm",
                "error_mm",
                "q1_deg",
                "q2_deg",
                "q3_deg",
                "q4_deg",
                "q5_deg",
            ]
        )
        q_deg = np.rad2deg(q)
        for i in range(len(target)):
            writer.writerow(
                [i + 1, *target[i], *actual[i], errors[i], *q_deg[i]]
            )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 240,
        }
    )

    robot = exp4.build_robot(exp4.dh_params_servo)
    initial_q = np.deg2rad(exp4.INITIAL_Q_DEG)
    start = np.asarray(robot.fkine(initial_q).t, dtype=float).reshape(3)
    target = exp4.generate_square_path(start)
    q, actual, errors = exp4.solve_position_ik(robot, target, initial_q)
    q_deg = np.rad2deg(q)

    joint_steps_deg = np.abs(np.diff(q_deg, axis=0))
    metrics = {
        "sample_count": int(len(target)),
        "start_position_mm": start.tolist(),
        "side_length_mm": exp4.SIDE_LENGTH_MM,
        "points_per_edge": exp4.POINTS_PER_EDGE,
        "max_position_error_mm": float(errors.max()),
        "mean_position_error_mm": float(errors.mean()),
        "closure_error_mm": float(np.linalg.norm(actual[-1] - actual[0])),
        "max_joint_step_deg": float(joint_steps_deg.max()),
        "max_joint_step_joint": int(np.unravel_index(np.argmax(joint_steps_deg), joint_steps_deg.shape)[1] + 1),
        "joint_ranges_deg": {
            f"q{i + 1}": [float(q_deg[:, i].min()), float(q_deg[:, i].max())]
            for i in range(robot.n)
        },
    }
    (OUTPUT_DIR / "实验四数值结果.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_trajectory_csv(OUTPUT_DIR / "实验四轨迹数据.csv", target, actual, q, errors)

    # 实验三：用实测 D-H 参数建立零输入姿态的骨架模型。
    q_zero = np.zeros(robot.n)
    zero_points = robot_points(robot, q_zero)
    fig = plt.figure(figsize=(10.2, 4.8), constrained_layout=True)
    ax3d = fig.add_subplot(121, projection="3d")
    ax_side = fig.add_subplot(122)
    ax3d.plot(
        zero_points[:, 0], zero_points[:, 1], zero_points[:, 2],
        "-o", color="#175CD3", lw=3.0, ms=6,
    )
    for i, point in enumerate(zero_points):
        ax3d.text(*point, f"  F{i}", fontsize=8)
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    ax3d.set_title("3D kinematic skeleton")
    ax3d.view_init(elev=24, azim=-58)
    ax3d.set_xlim(-50, 50)
    ax3d.set_ylim(-50, 50)
    ax3d.set_zlim(0, 520)
    ax3d.set_box_aspect((1, 1, 2.1))
    ax_side.plot(zero_points[:, 0], zero_points[:, 2], "-o", color="#175CD3", lw=3.0, ms=6)
    for i, point in enumerate(zero_points):
        ax_side.annotate(f"F{i}", (point[0], point[2]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax_side.set_xlabel("X (mm)")
    ax_side.set_ylabel("Z (mm)")
    ax_side.set_title("Side view")
    ax_side.set_xlim(-50, 50)
    ax_side.set_ylim(0, 520)
    ax_side.grid(True, color="#D0D5DD", alpha=0.7)
    fig.suptitle("Experiment 3: servo manipulator built from measured standard D-H parameters", fontsize=13)
    fig.savefig(OUTPUT_DIR / "图0_实验三实测DH骨架模型.png", bbox_inches="tight")
    plt.close(fig)

    # 真实末端轨迹的俯视对比。
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    ax.plot(target[:, 0], target[:, 1], "--", color="#667085", lw=2.1, label="Target")
    ax.plot(actual[:, 0], actual[:, 1], color="#C62828", lw=1.6, label="Actual")
    ax.scatter(*target[0, :2], s=52, color="#2E7D32", zorder=5, label="Start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Square trajectory: target and simulated end-effector path")
    ax.grid(True, color="#D0D5DD", alpha=0.7)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT_DIR / "图1_正方形目标与实际轨迹.png", bbox_inches="tight")
    plt.close(fig)

    # 三维机械臂构型叠加：起点和四条边末端姿态。
    fig = plt.figure(figsize=(8.2, 6.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(target[:, 0], target[:, 1], target[:, 2], "--", color="#667085", lw=1.8, label="Target path")
    ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], color="#C62828", lw=1.5, label="Actual path")
    pose_indices = [0, 69, 139, 209, 279]
    pose_colors = ["#175CD3", "#0E9384", "#7A5AF8", "#DC6803", "#344054"]
    for number, (index, color) in enumerate(zip(pose_indices, pose_colors), start=1):
        points = robot_points(robot, q[index])
        ax.plot(points[:, 0], points[:, 1], points[:, 2], "-o", color=color, lw=2.0, ms=3.8, alpha=0.83, label=f"Pose {number}")
    ax.scatter(*actual[0], color="#2E7D32", s=45)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Manipulator configurations along the square trajectory")
    ax.view_init(elev=25, azim=-58)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.savefig(OUTPUT_DIR / "图2_机械臂三维轨迹与关键构型.png", bbox_inches="tight")
    plt.close(fig)

    # 位置误差曲线。
    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    visible_errors = np.maximum(errors, 1e-12)
    ax.semilogy(np.arange(1, len(errors) + 1), visible_errors, color="#175CD3", lw=1.5)
    ax.axhline(exp4.MAX_POSITION_ERROR_MM, color="#C62828", ls="--", lw=1.2, label="1 mm limit")
    for x in [70, 140, 210]:
        ax.axvline(x, color="#D0D5DD", lw=0.9)
    ax.set_xlabel("Trajectory sample")
    ax.set_ylabel("Position error (mm, log scale)")
    ax.set_title("Numerical inverse-kinematics position error")
    ax.grid(True, color="#EAECF0", alpha=0.9)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT_DIR / "图3_逆运动学位置误差.png", bbox_inches="tight")
    plt.close(fig)

    # 各关节角随采样点变化。
    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    colors = ["#175CD3", "#0E9384", "#7A5AF8", "#DC6803", "#C62828"]
    for joint in range(robot.n):
        ax.plot(q_deg[:, joint], color=colors[joint], lw=1.4, label=f"q{joint + 1}")
    for x in [69, 139, 209]:
        ax.axvline(x, color="#D0D5DD", lw=0.9)
    ax.axhline(20.0, color="#C62828", ls="--", lw=1.0, alpha=0.7, label="q4 upper limit")
    ax.set_xlabel("Trajectory sample")
    ax.set_ylabel("Joint angle (deg)")
    ax.set_title("Joint trajectories from bounded least-squares IK")
    ax.grid(True, color="#EAECF0", alpha=0.9)
    ax.legend(ncol=3, frameon=False)
    fig.savefig(OUTPUT_DIR / "图4_关节角轨迹.png", bbox_inches="tight")
    plt.close(fig)

    # 五次时间缩放及其一、二阶导数，用于实验原理说明。
    tau = np.linspace(0.0, 1.0, 400)
    scaling = exp4.quintic_time_scaling(tau)
    velocity = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    acceleration = 60 * tau - 180 * tau**2 + 120 * tau**3
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True, constrained_layout=True)
    labels = [(scaling, "s(t)", "#175CD3"), (velocity, "ds/dt", "#0E9384"), (acceleration, "d2s/dt2", "#DC6803")]
    for ax, (series, ylabel, color) in zip(axes, labels):
        ax.plot(tau, series, color=color, lw=1.8)
        ax.axhline(0, color="#98A2B3", lw=0.8)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#EAECF0", alpha=0.9)
    axes[0].set_title("Quintic time scaling and endpoint smoothness")
    axes[-1].set_xlabel("Normalized time t")
    fig.savefig(OUTPUT_DIR / "图5_五次时间缩放.png", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
