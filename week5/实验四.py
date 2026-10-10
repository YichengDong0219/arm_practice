"""实验四：机械臂末端绘制水平正方形轨迹仿真。

直接运行：
    python 实验四.py

只进行轨迹规划与逆运动学验证（不显示动画）：
    python 实验四.py --no-animation
"""

from __future__ import annotations

import argparse
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import roboticstoolbox as rtb
from scipy.optimize import least_squares

from 实验三 import dh_params_servo


# 实验指导书给定参数。长度单位与实验三一致，均为 mm。
INITIAL_Q_DEG = np.array([0.0, -35.0, -100.0, 20.0, 0.0])
JOINT_LIMITS_DEG = np.array(
    [
        [-90.0, 90.0],
        [-90.0, 90.0],
        [-90.0, 90.0],
        [-90.0, 20.0],
        [-90.0, 90.0],
    ]
)
SIDE_LENGTH_MM = 35.0
POINTS_PER_EDGE = 70
CONTINUITY_WEIGHT = 0.01
MAX_POSITION_ERROR_MM = 1.0


def build_robot(dh_params: Sequence[dict]) -> rtb.DHRobot:
    """用实验三的标准 D-H 表建立五自由度机械臂模型。

    实验三的角度字段以度记录；Robotics Toolbox 的角度接口使用弧度，
    因而这里只在 API 边界换算，不改变原始参数表。
    """
    if len(dh_params) != 5:
        raise ValueError(f"实验四需要 5 组 D-H 参数，实际得到 {len(dh_params)} 组")

    links = []
    for index, (params, limits_deg) in enumerate(
        zip(dh_params, JOINT_LIMITS_DEG), start=1
    ):
        required = {"d", "a", "alpha"}
        missing = required.difference(params)
        if missing:
            raise KeyError(f"第 {index} 组 D-H 参数缺少字段: {sorted(missing)}")

        links.append(
            rtb.RevoluteDH(
                d=float(params["d"]),
                a=float(params["a"]),
                alpha=np.deg2rad(float(params["alpha"])),
                offset=np.deg2rad(float(params.get("offset", 0.0))),
                qlim=np.deg2rad(limits_deg),
            )
        )

    return rtb.DHRobot(links, name="Physical_Servo_Manipulator")


def quintic_time_scaling(t: np.ndarray) -> np.ndarray:
    """五次时间缩放 s(t)，其中 t 必须位于 [0, 1]。"""
    t = np.asarray(t, dtype=float)
    if np.any((t < 0.0) | (t > 1.0)):
        raise ValueError("五次时间缩放的 t 必须位于 [0, 1]")
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def generate_square_path(
    start_position: Sequence[float],
    side_length: float = SIDE_LENGTH_MM,
    points_per_edge: int = POINTS_PER_EDGE,
) -> np.ndarray:
    """以 start_position 为起点生成固定高度的闭合正方形路径。"""
    start = np.asarray(start_position, dtype=float)
    if start.shape != (3,):
        raise ValueError("start_position 必须是含 x、y、z 的三维坐标")
    if side_length <= 0.0:
        raise ValueError("正方形边长必须为正数")
    if points_per_edge < 2:
        raise ValueError("每条边至少需要 2 个采样点")

    vertices = np.array(
        [
            start,
            start + [0.0, side_length, 0.0],
            start + [side_length, side_length, 0.0],
            start + [side_length, 0.0, 0.0],
            start,
        ]
    )
    progress = quintic_time_scaling(np.linspace(0.0, 1.0, points_per_edge))

    edges = [
        vertices[i] + progress[:, None] * (vertices[i + 1] - vertices[i])
        for i in range(4)
    ]
    return np.vstack(edges)


def solve_position_ik(
    robot: rtb.DHRobot,
    target_path: np.ndarray,
    initial_q: Sequence[float],
    continuity_weight: float = CONTINUITY_WEIGHT,
    max_position_error: float = MAX_POSITION_ERROR_MM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用带边界的最小二乘法逐点求解仅约束位置的逆运动学。"""
    target_path = np.asarray(target_path, dtype=float)
    if target_path.ndim != 2 or target_path.shape[1] != 3:
        raise ValueError("target_path 的形状必须为 (N, 3)")

    lower = np.deg2rad(JOINT_LIMITS_DEG[:, 0])
    upper = np.deg2rad(JOINT_LIMITS_DEG[:, 1])
    q_previous = np.clip(np.asarray(initial_q, dtype=float), lower, upper)

    q_trajectory = np.empty((len(target_path), robot.n), dtype=float)
    actual_path = np.empty_like(target_path)
    errors = np.empty(len(target_path), dtype=float)

    for point_index, target_position in enumerate(target_path):
        q_reference = q_previous.copy()

        def residual(q: np.ndarray) -> np.ndarray:
            current_position = np.asarray(robot.fkine(q).t, dtype=float).reshape(3)
            position_error = current_position - target_position
            continuity_error = continuity_weight * (q - q_reference)
            return np.concatenate((position_error, continuity_error))

        result = least_squares(
            residual,
            q_previous,
            bounds=(lower, upper),
            method="trf",
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
            max_nfev=1000,
        )

        solved_position = np.asarray(robot.fkine(result.x).t, dtype=float).reshape(3)
        position_error = float(np.linalg.norm(solved_position - target_position))
        if (not result.success) or position_error > max_position_error:
            raise RuntimeError(
                "逆运动学求解失败："
                f"轨迹点 {point_index + 1}/{len(target_path)}，"
                f"目标坐标 {np.round(target_position, 3).tolist()} mm，"
                f"位置误差 {position_error:.6f} mm，"
                f"求解器信息：{result.message}"
            )

        q_trajectory[point_index] = result.x
        actual_path[point_index] = solved_position
        errors[point_index] = position_error
        q_previous = result.x

    return q_trajectory, actual_path, errors


def print_experiment_info(robot: rtb.DHRobot, start_position: np.ndarray) -> None:
    """按实验要求输出模型、路径及约束信息。"""
    print("\n=== 标准 D-H 参数表（长度: mm，角度: deg） ===")
    print("关节 | theta | alpha |      d |      a | offset")
    print("-----+-------+-------+--------+--------+-------")
    for i, params in enumerate(dh_params_servo, start=1):
        print(
            f"J{i:^4d} | {params.get('theta', 0):>5.1f} | "
            f"{params['alpha']:>5.1f} | {params['d']:>6.2f} | "
            f"{params['a']:>6.2f} | {params.get('offset', 0):>6.1f}"
        )

    print(f"\n模型名称: {robot.name}")
    print(f"初始关节角: {INITIAL_Q_DEG.tolist()} deg")
    print(f"正方形起点: {np.round(start_position, 3).tolist()} mm")
    print(f"正方形边长: {SIDE_LENGTH_MM:.1f} mm")
    print(f"每条边采样点数: {POINTS_PER_EDGE}（总计 {4 * POINTS_PER_EDGE} 点）")
    print("关节限位:")
    for i, (low, high) in enumerate(JOINT_LIMITS_DEG, start=1):
        print(f"  J{i}: [{low:.1f}°, {high:.1f}°]")
    print("姿态约束: 无（仅约束末端 x、y、z 位置，末端方向允许变化）")
    if np.any(
        (INITIAL_Q_DEG < JOINT_LIMITS_DEG[:, 0])
        | (INITIAL_Q_DEG > JOINT_LIMITS_DEG[:, 1])
    ):
        print(
            "说明: 指导书给定的初始 q3=-100° 超出 IK 下限 -90°；"
            "起点仍由该初始姿态计算，首个逆解从限位内的 -90° 开始搜索。"
        )


def animate_trajectory(
    robot: rtb.DHRobot,
    q_trajectory: np.ndarray,
    target_path: np.ndarray,
    actual_path: np.ndarray,
    interval: float = 0.03,
) -> None:
    """动态显示机械臂、目标正方形和已绘制的实际轨迹。"""
    all_points = np.vstack((target_path, actual_path, np.zeros((1, 3))))
    margin = 60.0
    limits = [
        float(all_points[:, 0].min() - margin),
        float(all_points[:, 0].max() + margin),
        float(all_points[:, 1].min() - margin),
        float(all_points[:, 1].max() + margin),
        float(min(0.0, all_points[:, 2].min() - margin)),
        float(all_points[:, 2].max() + margin),
    ]

    env = robot.plot(
        q_trajectory[0],
        backend="pyplot",
        block=False,
        limits=limits,
        jointaxes=True,
        shadow=True,
    )
    ax = env.ax
    ax.plot(
        target_path[:, 0],
        target_path[:, 1],
        target_path[:, 2],
        "--",
        color="0.45",
        linewidth=1.5,
        label="目标正方形",
    )
    actual_line, = ax.plot([], [], [], "r-", linewidth=2.0, label="实际末端轨迹")
    ax.scatter(
        target_path[0, 0],
        target_path[0, 1],
        target_path[0, 2],
        color="green",
        s=45,
        label="起点",
    )
    ax.legend(loc="best")
    ax.set_title("实验四：机械臂末端绘制正方形轨迹")

    for i, q in enumerate(q_trajectory):
        robot.q = q
        actual_line.set_data(actual_path[: i + 1, 0], actual_path[: i + 1, 1])
        actual_line.set_3d_properties(actual_path[: i + 1, 2])
        env.step(interval)

    plt.show(block=True)


def run_experiment(show_animation: bool = True, interval: float = 0.03) -> None:
    robot = build_robot(dh_params_servo)
    initial_q = np.deg2rad(INITIAL_Q_DEG)
    start_position = np.asarray(robot.fkine(initial_q).t, dtype=float).reshape(3)
    target_path = generate_square_path(start_position)

    print_experiment_info(robot, start_position)
    print("\n正在逐点求解逆运动学……")
    q_trajectory, actual_path, errors = solve_position_ik(
        robot, target_path, initial_q
    )

    max_error_index = int(np.argmax(errors))
    max_joint_step_deg = float(
        np.max(np.abs(np.rad2deg(np.diff(q_trajectory, axis=0))))
    )
    closure_error = float(np.linalg.norm(actual_path[-1] - actual_path[0]))
    print(f"逆运动学求解完成，共 {len(q_trajectory)} 个轨迹点。")
    print(
        f"最大位置误差: {errors[max_error_index]:.6f} mm "
        f"（第 {max_error_index + 1} 点）"
    )
    print(f"相邻采样点最大关节变化: {max_joint_step_deg:.6f}°")
    print(f"实际轨迹闭合误差: {closure_error:.6f} mm")

    if show_animation:
        animate_trajectory(robot, q_trajectory, target_path, actual_path, interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="只完成轨迹规划和逆运动学验证，不打开三维动画窗口",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.03,
        help="动画相邻帧间隔（秒，默认 0.03）",
    )
    args = parser.parse_args()
    if args.interval < 0.0:
        parser.error("--interval 不能为负数")
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    run_experiment(show_animation=not cli_args.no_animation, interval=cli_args.interval)
