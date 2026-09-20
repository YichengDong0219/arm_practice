"""实验八：通过平面 2R 逆运动学驱动 Joint 4 轴心连续画圆。"""

import argparse
import time

import numpy as np
import serial


# 串口与控制协议参数
SERIAL_PORT = "COM5"
BAUDRATE = 115200
SERIAL_TIMEOUT = 1

# 平面机械臂模型：Joint 2 -> Joint 3 -> Joint 4
LINK_1_MM = 120.0
LINK_2_MM = 120.0
REFERENCE_JOINTS_DEG = np.array([30.0, -45.0, 15.0], dtype=np.float64)

# 圆周轨迹参数
CIRCLE_RADIUS_MM = 18.0
CIRCLE_DURATION_S = 8.0
SAMPLE_RATE_HZ = 50.0
CIRCLE_STEPS = int(round(CIRCLE_DURATION_S * SAMPLE_RATE_HZ))

# 过渡和安全参数
MOVE_DURATION_S = 2.0
MOVE_STEPS = 40
HOLD_DURATION_S = 0.5
ANGLE_LIMIT_DEG = 90.0
POSITION_TOLERANCE_MM = 1e-6
ANGLE_TOLERANCE_DEG = 1e-9


def pack_frame(q_rad):
    """将 6 轴弧度数组打包为 STM32 使用的 16 字节控制帧。"""
    q_rad = np.asarray(q_rad, dtype=np.float64)
    if q_rad.shape != (6,):
        raise ValueError("关节角数组必须包含 6 个元素")
    if not np.all(np.isfinite(q_rad)):
        raise ValueError("关节角数组中不能包含 NaN 或无穷大")

    tx = bytearray(16)
    tx[0] = 0xAA
    for i in range(6):
        value = int(q_rad[i] * 1000.0)
        value = max(min(value, 32767), -32768)
        tx[1 + i * 2] = (value >> 8) & 0xFF
        tx[2 + i * 2] = value & 0xFF

    tx[13] = 0x01
    check = 0
    for byte in tx[:14]:
        check ^= byte
    tx[14] = check
    tx[15] = 0xBB
    return tx


def validate_frame(frame):
    """检查单个控制帧的长度、固定字节和 XOR 校验和。"""
    if len(frame) != 16:
        raise ValueError("控制帧长度必须为 16 字节")
    if frame[0] != 0xAA or frame[13] != 0x01 or frame[15] != 0xBB:
        raise ValueError("控制帧的帧头、模式位或帧尾错误")

    check = 0
    for byte in frame[:14]:
        check ^= byte
    if frame[14] != check:
        raise ValueError("控制帧 XOR 校验和错误")


def forward_kinematics_joint4(q2_rad, q3_rad):
    """计算 Joint 4 轴心相对 Joint 2 轴心的平面坐标，单位为 mm。"""
    x = LINK_1_MM * np.cos(q2_rad) + LINK_2_MM * np.cos(q2_rad + q3_rad)
    y = LINK_1_MM * np.sin(q2_rad) + LINK_2_MM * np.sin(q2_rad + q3_rad)
    return np.array([x, y], dtype=np.float64)


def inverse_kinematics_joint4(x_mm, y_mm):
    """求 Joint 4 轴心位置的负角肘部 2R 解析逆解，返回 6 轴角度数组。"""
    cosine_q3 = (
        x_mm * x_mm
        + y_mm * y_mm
        - LINK_1_MM * LINK_1_MM
        - LINK_2_MM * LINK_2_MM
    ) / (2.0 * LINK_1_MM * LINK_2_MM)

    reachability_tolerance = 1e-12
    if cosine_q3 < -1.0 - reachability_tolerance or cosine_q3 > 1.0 + reachability_tolerance:
        raise ValueError(
            f"目标点不可达: ({x_mm:.3f}, {y_mm:.3f}) mm, "
            f"cos(q3)={cosine_q3:.6f}"
        )

    cosine_q3 = float(np.clip(cosine_q3, -1.0, 1.0))
    q3_rad = -float(np.arccos(cosine_q3))
    q2_rad = float(
        np.arctan2(y_mm, x_mm)
        - np.arctan2(
            LINK_2_MM * np.sin(q3_rad),
            LINK_1_MM + LINK_2_MM * np.cos(q3_rad),
        )
    )

    # 姿态补偿：使 q2 + q3 + q4 恒为 0，夹爪方向近似保持水平。
    q4_rad = -(q2_rad + q3_rad)
    pose_rad = np.zeros(6, dtype=np.float64)
    pose_rad[1:4] = [q2_rad, q3_rad, q4_rad]
    return np.rad2deg(pose_rad)


def calculate_circle_center():
    """用实验七参考姿态计算本实验圆心。"""
    q2_rad, q3_rad = np.deg2rad(REFERENCE_JOINTS_DEG[:2])
    return forward_kinematics_joint4(q2_rad, q3_rad)


def build_circle_trajectory():
    """生成圆起点以及随后 400 个逆时针圆周目标姿态。"""
    center = calculate_circle_center()
    start_x = center[0] + CIRCLE_RADIUS_MM
    start_y = center[1]
    start_pose_deg = inverse_kinematics_joint4(start_x, start_y)

    circle_poses_deg = np.empty((CIRCLE_STEPS, 6), dtype=np.float64)
    for index in range(1, CIRCLE_STEPS + 1):
        theta = 2.0 * np.pi * index / CIRCLE_STEPS
        x_mm = center[0] + CIRCLE_RADIUS_MM * np.cos(theta)
        y_mm = center[1] + CIRCLE_RADIUS_MM * np.sin(theta)
        circle_poses_deg[index - 1] = inverse_kinematics_joint4(x_mm, y_mm)

    return center, start_pose_deg, circle_poses_deg


def validate_trajectory(center, start_pose_deg, circle_poses_deg):
    """在打开串口前检查轨迹的安全性、连续性、闭环性和圆度。"""
    if circle_poses_deg.shape != (CIRCLE_STEPS, 6):
        raise ValueError("圆周轨迹维度错误")

    all_poses = np.vstack((np.zeros(6), start_pose_deg, circle_poses_deg))
    if not np.all(np.isfinite(all_poses)):
        raise ValueError("轨迹中包含 NaN 或无穷大")
    if np.max(np.abs(all_poses)) > ANGLE_LIMIT_DEG + ANGLE_TOLERANCE_DEG:
        raise ValueError(f"轨迹超过 ±{ANGLE_LIMIT_DEG:.1f}° 软件限位")
    if not np.allclose(all_poses[:, [0, 4, 5]], 0.0, atol=ANGLE_TOLERANCE_DEG):
        raise ValueError("Joint 1、5、6 必须全程保持 0°")
    if not np.all(circle_poses_deg[:, 2] < 0.0):
        raise ValueError("IK 轨迹未能保持负角肘部构型")
    if not np.allclose(
        all_poses[:, 1] + all_poses[:, 2] + all_poses[:, 3],
        0.0,
        atol=ANGLE_TOLERANCE_DEG,
    ):
        raise ValueError("Joint 4 姿态补偿关系不成立")
    if not np.allclose(
        circle_poses_deg[-1], start_pose_deg, atol=ANGLE_TOLERANCE_DEG
    ):
        raise ValueError("圆周轨迹首尾未闭合")

    q2_rad = np.deg2rad(circle_poses_deg[:, 1])
    q3_rad = np.deg2rad(circle_poses_deg[:, 2])
    x_mm = LINK_1_MM * np.cos(q2_rad) + LINK_2_MM * np.cos(q2_rad + q3_rad)
    y_mm = LINK_1_MM * np.sin(q2_rad) + LINK_2_MM * np.sin(q2_rad + q3_rad)
    radii_mm = np.hypot(x_mm - center[0], y_mm - center[1])
    radius_error_mm = float(np.max(np.abs(radii_mm - CIRCLE_RADIUS_MM)))
    if radius_error_mm > POSITION_TOLERANCE_MM:
        raise ValueError(f"正运动学回算的最大圆度误差过大: {radius_error_mm:.9f} mm")

    closed_poses = np.vstack((start_pose_deg, circle_poses_deg))
    max_frame_delta_deg = float(np.max(np.abs(np.diff(closed_poses[:, 1:4], axis=0))))

    for pose_deg in all_poses:
        frame = pack_frame(np.deg2rad(pose_deg))
        validate_frame(frame)

    return {
        "radius_error_mm": radius_error_mm,
        "max_frame_delta_deg": max_frame_delta_deg,
        "joint_min_deg": np.min(circle_poses_deg[:, 1:4], axis=0),
        "joint_max_deg": np.max(circle_poses_deg[:, 1:4], axis=0),
    }


def sleep_until(deadline):
    """等待至绝对时间点；若串口处理已经超时则立即继续。"""
    remaining = deadline - time.perf_counter()
    if remaining > 0.0:
        time.sleep(remaining)


def move_interpolated(ser, start_pose_deg, target_pose_deg, duration_s, steps):
    """在关节空间中平滑移动到指定姿态。"""
    start_pose_deg = np.asarray(start_pose_deg, dtype=np.float64)
    target_pose_deg = np.asarray(target_pose_deg, dtype=np.float64)
    period_s = duration_s / steps
    start_time = time.perf_counter()

    for index in range(1, steps + 1):
        ratio = index / steps
        pose_deg = start_pose_deg + ratio * (target_pose_deg - start_pose_deg)
        ser.write(pack_frame(np.deg2rad(pose_deg)))
        ser.flush()
        sleep_until(start_time + index * period_s)

    return target_pose_deg.copy()


def send_circle_trajectory(ser, circle_poses_deg, cycles):
    """以绝对时间基准连续发送指定圈数的圆轨迹。"""
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1:
        raise ValueError("圆周运动圈数必须是大于或等于 1 的整数")

    period_s = 1.0 / SAMPLE_RATE_HZ
    start_time = time.perf_counter()
    frame_index = 0

    for _ in range(cycles):
        for pose_deg in circle_poses_deg:
            frame_index += 1
            ser.write(pack_frame(np.deg2rad(pose_deg)))
            ser.flush()
            sleep_until(start_time + frame_index * period_s)


def print_trajectory_summary(center, start_pose_deg, validation, cycles):
    """输出实机运动前的轨迹校验结果。"""
    joint_min = validation["joint_min_deg"]
    joint_max = validation["joint_max_deg"]
    print("轨迹预检通过：")
    print(f"  圆心: ({center[0]:.3f}, {center[1]:.3f}) mm")
    print(f"  半径: {CIRCLE_RADIUS_MM:.3f} mm")
    print(f"  圆起点姿态 J2/J3/J4: {start_pose_deg[1:4].round(3).tolist()}°")
    for index, joint_number in enumerate((2, 3, 4)):
        print(
            f"  Joint {joint_number}: "
            f"{joint_min[index]:.3f}° ～ {joint_max[index]:.3f}°"
        )
    print(f"  最大相邻帧角度变化: {validation['max_frame_delta_deg']:.3f}°")
    print(f"  正运动学最大圆度误差: {validation['radius_error_mm']:.9f} mm")
    print(
        f"  连续运行: {cycles} 圈，{cycles * CIRCLE_DURATION_S:.1f} 秒，"
        f"{cycles * CIRCLE_STEPS} 个圆周控制帧"
    )


def positive_integer(value):
    """供 argparse 使用的正整数解析器。"""
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("圈数必须是整数") from error
    if result < 1:
        raise argparse.ArgumentTypeError("圈数必须大于或等于 1")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n",
        "--cycles",
        type=positive_integer,
        default=1,
        help="连续画圆圈数，必须为正整数，默认 1",
    )
    args = parser.parse_args(argv)

    try:
        center, circle_start_pose, circle_poses = build_circle_trajectory()
        validation = validate_trajectory(center, circle_start_pose, circle_poses)
    except ValueError as error:
        print(f"轨迹预检失败，拒绝打开串口: {error}")
        return

    print_trajectory_summary(center, circle_start_pose, validation, args.cycles)
    print("请确认机械臂真实姿态位于全零位附近，且运动范围内没有障碍物。")

    ser = serial.Serial()
    ser.port = SERIAL_PORT
    ser.baudrate = BAUDRATE
    ser.timeout = SERIAL_TIMEOUT
    ser.dtr = False
    ser.rts = False

    try:
        ser.open()
        print(f"成功连接控制板串口: {SERIAL_PORT}")
        print("等待 STM32 初始化就绪 (1.5 秒)...")
        time.sleep(1.5)

        home_pose = np.zeros(6, dtype=np.float64)
        print("平滑移动到圆周起点...")
        current_pose = move_interpolated(
            ser,
            home_pose,
            circle_start_pose,
            duration_s=MOVE_DURATION_S,
            steps=MOVE_STEPS,
        )
        time.sleep(HOLD_DURATION_S)

        print(f"Joint 4 轴心开始连续逆时针画圆，共 {args.cycles} 圈...")
        send_circle_trajectory(ser, circle_poses, args.cycles)
        current_pose = circle_poses[-1].copy()
        print("圆周运动完成。")
        time.sleep(HOLD_DURATION_S)

        print("平滑复位至全零位...")
        move_interpolated(
            ser,
            current_pose,
            home_pose,
            duration_s=MOVE_DURATION_S,
            steps=MOVE_STEPS,
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
