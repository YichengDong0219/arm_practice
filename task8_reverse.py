"""实验八反向版：使用平面 3R 逆运动学驱动末端执行器连续画圆。"""

import argparse
import time

import numpy as np

import task8 as base


# 反向模型使用 L1=12 cm、L2=12 cm、L3=14 cm，代码内部统一采用 mm。
LINK_1_MM = 120.0
LINK_2_MM = 120.0
LINK_3_MM = 140.0
ORIENTATION_SEARCH_STEP_DEG = 0.5
JOINT2_ZERO_DIRECTION_RAD = np.pi / 2.0

# 零位时 Joint 2、3、4 轴心位于竖直线上，圆心取 Joint 4 零位轴心。
CIRCLE_CENTER_MM = np.array([0.0, LINK_1_MM + LINK_2_MM], dtype=np.float64)


def inverse_kinematics_tool_reversed(x_mm, y_mm, tool_angle_rad):
    """给定末端位置和第三连杆方向，求反向肘部的平面 3R 逆解。"""
    joint4_x_mm = x_mm - LINK_3_MM * np.cos(tool_angle_rad)
    joint4_y_mm = y_mm - LINK_3_MM * np.sin(tool_angle_rad)

    cosine_q3 = (
        joint4_x_mm * joint4_x_mm
        + joint4_y_mm * joint4_y_mm
        - LINK_1_MM * LINK_1_MM
        - LINK_2_MM * LINK_2_MM
    ) / (2.0 * LINK_1_MM * LINK_2_MM)

    reachability_tolerance = 1e-12
    if cosine_q3 < -1.0 - reachability_tolerance or cosine_q3 > 1.0 + reachability_tolerance:
        raise ValueError(
            f"末端目标点不可达: ({x_mm:.3f}, {y_mm:.3f}) mm, "
            f"cos(q3)={cosine_q3:.6f}"
        )

    cosine_q3 = float(np.clip(cosine_q3, -1.0, 1.0))
    # 使用负角肘部分支，使 Joint 4 能在无机械限位的反向区域运动。
    q3_rad = -float(np.arccos(cosine_q3))
    q2_model_rad = float(
        np.arctan2(joint4_y_mm, joint4_x_mm)
        - np.arctan2(
            LINK_2_MM * np.sin(q3_rad),
            LINK_1_MM + LINK_2_MM * np.cos(q3_rad),
        )
    )
    q4_rad = float(
        (tool_angle_rad - q2_model_rad - q3_rad + np.pi)
        % (2.0 * np.pi)
        - np.pi
    )
    # 实物零位沿 +Y 轴；解析 IK 的首关节角以 +X 轴为零，因此减去 90°。
    q2_servo_rad = q2_model_rad - JOINT2_ZERO_DIRECTION_RAD

    pose_rad = np.zeros(6, dtype=np.float64)
    pose_rad[1:4] = [q2_servo_rad, q3_rad, q4_rad]
    return np.rad2deg(pose_rad)


def calculate_reversed_circle_center():
    """返回零位关节竖直线上的 Joint 4 轴心坐标。"""
    return CIRCLE_CENTER_MM.copy()


def build_reversed_circle_trajectory(radius_mm):
    """搜索无固定末端姿态约束的一周连续顺时针 3R 轨迹。"""
    center = calculate_reversed_circle_center()
    best_result = None

    orientation_offsets_deg = np.arange(
        -180.0,
        180.0 + ORIENTATION_SEARCH_STEP_DEG / 2.0,
        ORIENTATION_SEARCH_STEP_DEG,
    )
    for orientation_offset_deg in orientation_offsets_deg:
        orientation_offset_rad = np.deg2rad(orientation_offset_deg)
        poses_deg = np.empty((base.CIRCLE_STEPS + 1, 6), dtype=np.float64)
        candidate_valid = True

        for index in range(base.CIRCLE_STEPS + 1):
            theta = -2.0 * np.pi * index / base.CIRCLE_STEPS
            x_mm = center[0] + radius_mm * np.cos(theta)
            y_mm = center[1] + radius_mm * np.sin(theta)

            # 末端方向随目标点相对基座的方位变化，不再保持水平或竖直。
            tool_angle_rad = np.arctan2(y_mm, x_mm) + orientation_offset_rad
            try:
                pose_deg = inverse_kinematics_tool_reversed(
                    x_mm, y_mm, tool_angle_rad
                )
            except ValueError:
                candidate_valid = False
                break
            if (
                np.max(np.abs(pose_deg)) > base.ANGLE_LIMIT_DEG
                or pose_deg[3] > base.ANGLE_TOLERANCE_DEG
            ):
                candidate_valid = False
                break
            poses_deg[index] = pose_deg

        if not candidate_valid:
            continue

        # 首尾目标坐标相同，强制使用完全相同的关节解消除浮点闭环误差。
        poses_deg[-1] = poses_deg[0]
        max_abs_angle_deg = float(np.max(np.abs(poses_deg[:, 1:4])))
        max_frame_delta_deg = float(
            np.max(np.abs(np.diff(poses_deg[:, 1:4], axis=0)))
        )
        # 同时偏好远离限位且帧间变化小的末端方向策略。
        score = max_abs_angle_deg + 10.0 * max_frame_delta_deg
        if best_result is None or score < best_result[0]:
            best_result = (
                score,
                orientation_offset_deg,
                poses_deg[0].copy(),
                poses_deg[1:].copy(),
            )

    if best_result is None:
        raise ValueError(
            f"半径 {radius_mm:.3f} mm 在 Joint 2、3、4 的 "
            f"±{base.ANGLE_LIMIT_DEG:.1f}° 范围内没有完整连续解，请减小半径"
        )

    _, orientation_offset_deg, start_pose_deg, circle_poses_deg = best_result
    return center, start_pose_deg, circle_poses_deg, orientation_offset_deg


def validate_reversed_trajectory(
    center, start_pose_deg, circle_poses_deg, radius_mm
):
    """打开串口前验证末端圆度、闭环性、限位和通信帧。"""
    if circle_poses_deg.shape != (base.CIRCLE_STEPS, 6):
        raise ValueError("反向圆周轨迹维度错误")

    all_poses = np.vstack((np.zeros(6), start_pose_deg, circle_poses_deg))
    if not np.all(np.isfinite(all_poses)):
        raise ValueError("反向轨迹中包含 NaN 或无穷大")
    if np.max(np.abs(all_poses)) > base.ANGLE_LIMIT_DEG + base.ANGLE_TOLERANCE_DEG:
        raise ValueError(f"反向轨迹超过 ±{base.ANGLE_LIMIT_DEG:.1f}° 软件限位")
    if not np.allclose(
        all_poses[:, [0, 4, 5]], 0.0, atol=base.ANGLE_TOLERANCE_DEG
    ):
        raise ValueError("Joint 1、5、6 必须全程保持 0°")
    if not np.all(circle_poses_deg[:, 2] < 0.0):
        raise ValueError("反向 IK 未能保持负角肘部构型")
    if np.any(circle_poses_deg[:, 3] > base.ANGLE_TOLERANCE_DEG):
        raise ValueError("Joint 4 进入了存在机械限位的正角方向")
    if not np.allclose(
        circle_poses_deg[-1], start_pose_deg, atol=base.ANGLE_TOLERANCE_DEG
    ):
        raise ValueError("反向圆周轨迹首尾未闭合")

    q2_rad = (
        np.deg2rad(circle_poses_deg[:, 1]) + JOINT2_ZERO_DIRECTION_RAD
    )
    q3_rad = np.deg2rad(circle_poses_deg[:, 2])
    q4_rad = np.deg2rad(circle_poses_deg[:, 3])
    x_mm = (
        LINK_1_MM * np.cos(q2_rad)
        + LINK_2_MM * np.cos(q2_rad + q3_rad)
        + LINK_3_MM * np.cos(q2_rad + q3_rad + q4_rad)
    )
    y_mm = (
        LINK_1_MM * np.sin(q2_rad)
        + LINK_2_MM * np.sin(q2_rad + q3_rad)
        + LINK_3_MM * np.sin(q2_rad + q3_rad + q4_rad)
    )
    radii_mm = np.hypot(x_mm - center[0], y_mm - center[1])
    radius_error_mm = float(np.max(np.abs(radii_mm - radius_mm)))
    if radius_error_mm > base.POSITION_TOLERANCE_MM:
        raise ValueError(
            f"反向正运动学回算的最大圆度误差过大: {radius_error_mm:.9f} mm"
        )

    closed_poses = np.vstack((start_pose_deg, circle_poses_deg))
    max_frame_delta_deg = float(
        np.max(np.abs(np.diff(closed_poses[:, 1:4], axis=0)))
    )

    for pose_deg in all_poses:
        frame = base.pack_frame(np.deg2rad(pose_deg))
        base.validate_frame(frame)

    return {
        "radius_error_mm": radius_error_mm,
        "max_frame_delta_deg": max_frame_delta_deg,
        "joint_min_deg": np.min(circle_poses_deg[:, 1:4], axis=0),
        "joint_max_deg": np.max(circle_poses_deg[:, 1:4], axis=0),
    }


def print_reversed_summary(
    center,
    start_pose_deg,
    validation,
    radius_mm,
    cycles,
    orientation_offset_deg,
):
    """输出反向轨迹的实机运动前检查结果。"""
    joint_min = validation["joint_min_deg"]
    joint_max = validation["joint_max_deg"]
    print("三连杆末端轨迹预检通过：")
    print(f"  末端圆心: ({center[0]:.3f}, {center[1]:.3f}) mm")
    print(f"  半径: {radius_mm:.3f} mm")
    print(f"  圆起点姿态 J2/J3/J4: {start_pose_deg[1:4].round(3).tolist()}°")
    print(
        "  末端固定姿态约束已取消，"
        f"自动选择的方向偏置为 {orientation_offset_deg:.1f}°"
    )
    print("  Joint 4 已强制使用不受机械限位的负角旋转方向")
    for index, joint_number in enumerate((2, 3, 4)):
        print(
            f"  Joint {joint_number}: "
            f"{joint_min[index]:.3f}° ～ {joint_max[index]:.3f}°"
        )
    print(f"  最大相邻帧角度变化: {validation['max_frame_delta_deg']:.3f}°")
    print(f"  正运动学最大圆度误差: {validation['radius_error_mm']:.9f} mm")
    print(
        f"  连续顺时针运行: {cycles} 圈，"
        f"{cycles * base.CIRCLE_DURATION_S:.1f} 秒，"
        f"{cycles * base.CIRCLE_STEPS} 个圆周控制帧"
    )


def positive_float(value):
    """供 argparse 使用的有限正浮点数解析器。"""
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("半径必须是数值") from error
    if not np.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("半径必须是大于 0 的有限数值")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-r",
        "--radius",
        type=positive_float,
        default=base.CIRCLE_RADIUS_MM,
        metavar="MM",
        help=f"圆周半径（mm），默认 {base.CIRCLE_RADIUS_MM:g}",
    )
    parser.add_argument(
        "-n",
        "--cycles",
        type=base.positive_integer,
        default=1,
        help="连续画圆圈数，必须为正整数，默认 1",
    )
    args = parser.parse_args(argv)

    try:
        (
            center,
            circle_start_pose,
            circle_poses,
            orientation_offset_deg,
        ) = build_reversed_circle_trajectory(args.radius)
        validation = validate_reversed_trajectory(
            center, circle_start_pose, circle_poses, args.radius
        )
    except ValueError as error:
        print(f"反向轨迹预检失败，拒绝打开串口: {error}")
        return

    print_reversed_summary(
        center,
        circle_start_pose,
        validation,
        args.radius,
        args.cycles,
        orientation_offset_deg,
    )
    print("请确认机械臂真实姿态位于全零位附近，且反向运动范围内没有障碍物。")

    ser = base.serial.Serial()
    ser.port = base.SERIAL_PORT
    ser.baudrate = base.BAUDRATE
    ser.timeout = base.SERIAL_TIMEOUT
    ser.dtr = False
    ser.rts = False

    try:
        ser.open()
        print(f"成功连接控制板串口: {base.SERIAL_PORT}")
        print("等待 STM32 初始化就绪 (1.5 秒)...")
        time.sleep(1.5)

        home_pose = np.zeros(6, dtype=np.float64)
        print("平滑移动到末端圆周起点...")
        current_pose = base.move_interpolated(
            ser,
            home_pose,
            circle_start_pose,
            duration_s=base.MOVE_DURATION_S,
            steps=base.MOVE_STEPS,
        )
        time.sleep(base.HOLD_DURATION_S)

        print(f"末端执行器开始连续顺时针画圆，共 {args.cycles} 圈...")
        base.send_circle_trajectory(ser, circle_poses, args.cycles)
        current_pose = circle_poses[-1].copy()
        print("末端圆周运动完成。")
        time.sleep(base.HOLD_DURATION_S)

        print("平滑复位至全零位...")
        base.move_interpolated(
            ser,
            current_pose,
            home_pose,
            duration_s=base.MOVE_DURATION_S,
            steps=base.MOVE_STEPS,
        )
        print("复位完成。")
    except KeyboardInterrupt:
        print("\n检测到手动中断，立即停止发送；不会在异常状态下自动复位。")
    except Exception as error:
        print(f"实物运行异常，立即停止发送且不自动复位: {error}")
    finally:
        if ser.is_open:
            ser.close()
            print("物理串口通信链路已释放。")


if __name__ == "__main__":
    main()
