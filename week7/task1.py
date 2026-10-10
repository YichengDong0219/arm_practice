"""第七周实验 1.2：关节空间 MoveJ 轨迹规划。"""

import argparse
import time

import matplotlib
import numpy as np
import roboticstoolbox as rtb
import serial
import spatialmath as sm

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt


# 模型角到控制侧角、控制侧角到舵机角各乘一次 -1，最终发送角与模型角一致。
SERVO_DIRECTION_MAP = np.full(6, -1.0)

# 实机确认的关节指令范围（与 teleop/position_ik.py、week6 一致），单位 deg。
JOINT_LIMITS_DEG = np.array(
    [
        [-90.0, 90.0],    # J1 底座偏航
        [-90.0, 90.0],    # J2 大臂俯仰
        [-90.0, 90.0],    # J3 小臂俯仰
        [-90.0, 25.0],    # J4 腕部俯仰
        [-180.0, 180.0],  # J5 腕部横滚（锁零）
    ]
)

# 基坐标系中的桌面高度（week6 实机标定值），用于检查连杆是否扫到桌面。
TABLE_Z = 52.0
TABLE_CLEARANCE = 5.0

# 逆解种子：指导书要求的“肘部向上预弯曲构型”。Target A 在关节限位内
# 只存在 J2>0、肘部折叠的一族解；若用 q_prepare 作种子，LM 会收敛到
# J2≈-122°、J4≈+96° 的越限解。
Q_IK_SEED_DEG = [0.0, 45.0, -80.0, -80.0, 0.0]


def check_joint_limits(q_trajectory, label="轨迹"):
    """逐帧检查五轴角度是否落在实机限位内，越限则拒绝下发。"""
    q_deg = np.rad2deg(np.atleast_2d(q_trajectory))
    low = q_deg < JOINT_LIMITS_DEG[:, 0] - 1e-6
    high = q_deg > JOINT_LIMITS_DEG[:, 1] + 1e-6
    if np.any(low | high):
        frame, joint = np.argwhere(low | high)[0]
        raise RuntimeError(
            f"{label}第 {frame + 1} 帧 J{joint + 1}={q_deg[frame, joint]:.2f}° "
            f"超出限位 {JOINT_LIMITS_DEG[joint].tolist()}，已拒绝下发。"
        )


def check_table_clearance(robot, q_trajectory):
    """检查各关节坐标原点与末端是否始终高于桌面。返回最低点高度。"""
    lowest = min(
        frame.t[2]
        for q in q_trajectory
        for frame in robot.fkine_all(q)[2:]
    )
    if lowest < TABLE_Z + TABLE_CLEARANCE:
        raise RuntimeError(
            f"连杆最低点 z={lowest:.1f} mm 低于桌面安全高度 "
            f"{TABLE_Z + TABLE_CLEARANCE:.1f} mm。"
        )
    return lowest


def pack_frame(q_rad):
    """将 6 个关节弧度值打包为 STM32 使用的 16 字节控制帧。"""
    q_rad = np.asarray(q_rad, dtype=np.float64)
    if q_rad.shape != (6,):
        raise ValueError(f"q_rad 必须包含 6 个关节角，实际形状为 {q_rad.shape}")

    tx = bytearray(16)
    tx[0] = 0xAA

    for i, q_i in enumerate(q_rad):
        value = int(q_i * 1000)
        value = max(min(value, 32767), -32768)
        tx[1 + i * 2] = (value >> 8) & 0xFF
        tx[2 + i * 2] = value & 0xFF

    tx[13] = 0x01
    checksum = 0
    for byte in tx[:14]:
        checksum ^= byte
    tx[14] = checksum
    tx[15] = 0xBB
    return tx


def init_serial(port="COM3", baudrate=115200):
    """打开控制板串口并发送六轴零位；连接失败时返回 ``None``。"""
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baudrate
    ser.timeout = 1
    ser.dtr = False
    ser.rts = False

    try:
        ser.open()
    except Exception as exc:
        print(f"未检测到物理串口 {port}（{exc}），仅运行仿真。")
        return None

    try:
        print(f"已连接控制板串口：{port}")
        time.sleep(1.0)
        ser.write(pack_frame(np.zeros(6, dtype=np.float64)))
        ser.flush()
        time.sleep(1.5)
        print("已发送六轴零位目标；当前协议不提供实际到位反馈。")
        return ser
    except Exception:
        ser.close()
        raise


def build_robot_model(dh_params, name="Physical_Servo_Arm"):
    """根据标准 D-H 参数建立五关节机械臂模型。"""
    links = [
        rtb.RevoluteDH(
            d=params["d"],
            a=params["a"],
            alpha=params["alpha"],
            offset=params.get("offset", 0.0),
        )
        for params in dh_params
    ]
    return rtb.DHRobot(links, name=name)


def animate_robot_motion(robot, q_trajectory, dt=0.04, target_point=None):
    """播放 MoveJ 动画，同时绘制规划末端路径和实际播放轨迹。"""
    q_trajectory = np.asarray(q_trajectory, dtype=np.float64)
    tool_path = np.asarray([robot.fkine(q).t for q in q_trajectory])

    env = robot.plot(
        q_trajectory[0],
        block=False,
        limits=[-150, 450, -250, 250, 0, 500],
        jointaxes=True,
        shadow=True,
        dt=dt,
    )
    ax = env.ax
    ax.plot(
        *tool_path.T,
        "--",
        color="0.55",
        linewidth=1.4,
        label="Planned TCP path",
    )
    if target_point is not None:
        ax.scatter(*target_point, color="crimson", s=35, label="Target A")

    (traced,) = ax.plot(
        [], [], [], color="darkorange", linewidth=2.5, label="TCP trace"
    )
    ax.legend(loc="upper left")
    ax.set_title("MoveJ - end-effector path")

    for index, q in enumerate(q_trajectory):
        robot.q = q
        traced.set_data(tool_path[: index + 1, 0], tool_path[: index + 1, 1])
        traced.set_3d_properties(tool_path[: index + 1, 2])
        env.step(dt)

    plt.pause(1.0)


def execute_joint_trajectory(ser, q_traj_rad, duration=3.0):
    """按给定总时长向 STM32 等间隔发送五轴轨迹。"""
    q_traj_rad = np.asarray(q_traj_rad, dtype=np.float64)
    if q_traj_rad.ndim != 2 or q_traj_rad.shape[1] != 5:
        raise ValueError(
            "q_traj_rad 必须是 N x 5 的关节轨迹，"
            f"实际形状为 {q_traj_rad.shape}"
        )
    if duration <= 0:
        raise ValueError("duration 必须大于 0")
    if ser is None or not ser.is_open:
        print("无物理硬件连接，跳过硬件轨迹下发。")
        return

    sample_period = duration / len(q_traj_rad)
    for q in q_traj_rad:
        q_model = np.append(q, 0.0)  # 第 6 轴夹爪保持零位。
        q_control = q_model * SERVO_DIRECTION_MAP
        q_servo = q_control * SERVO_DIRECTION_MAP
        ser.write(pack_frame(q_servo))
        ser.flush()
        time.sleep(sample_period)


def run_movej_experiment(show_animation=True, mode="auto", port="COM3"):
    """完成零位、预备位、Target A 与回零之间的 MoveJ 实验流程。"""
    if mode not in {"auto", "simulate", "hardware"}:
        raise ValueError(f"不支持的运行模式：{mode}")

    # 与前序实验相同的实测五轴标准 D-H 参数，长度单位为 mm。
    dh_params_servo = [
        {"d": 108.5, "a": 0.0, "alpha": np.pi / 2, "offset": 0.0},
        {"d": 0.0, "a": 105.0, "alpha": 0.0, "offset": np.pi / 2},
        {"d": 0.0, "a": 97.5, "alpha": 0.0, "offset": 0.0},
        {"d": 0.0, "a": 28.22, "alpha": np.pi / 2, "offset": np.pi / 2},
        {"d": 172.70, "a": 0.0, "alpha": 0.0, "offset": 0.0},
    ]

    robot = build_robot_model(dh_params_servo)
    q_zero = np.zeros(5, dtype=np.float64)
    q_prepare = np.deg2rad([0.0, 0.0, 20.0, 10.0, 0.0])
    target_a = np.array([120.0, 80.0, 220.0], dtype=np.float64)

    target_pose = sm.SE3(*target_a)
    ik_solution = robot.ikine_LM(
        target_pose,
        q0=np.deg2rad(Q_IK_SEED_DEG),
        mask=[1, 1, 1, 0, 0, 0],
    )
    if not ik_solution.success:
        reason = getattr(ik_solution, "reason", "未知原因")
        raise RuntimeError(f"Target A 逆运动学求解失败：{reason}")

    q_target = np.asarray(ik_solution.q, dtype=np.float64)
    q_target[4] = 0.0  # 第 5 轴不影响末端位置，锁零。
    check_joint_limits(q_target, label="Target A 逆解")
    reached_point = robot.fkine(q_target).t
    position_error = np.linalg.norm(reached_point - target_a)

    print("\n========== 第七周实验 1.2：MoveJ ==========")
    print(f"Target A：{target_a} mm")
    print(f"逆解关节角：{np.round(np.rad2deg(q_target), 3)} deg")
    print(f"正解末端坐标：{np.round(reached_point, 3)} mm")
    print(f"位置误差：{position_error:.6f} mm")

    # 三段关节空间五次多项式轨迹：进入预备位、移动至目标、回到零位。
    traj_prepare = rtb.jtraj(q_zero, q_prepare, 40).q
    traj_target = rtb.jtraj(q_prepare, q_target, 60).q
    traj_return = rtb.jtraj(q_target, q_zero, 60).q
    complete_trajectory = np.vstack(
        [traj_prepare, traj_target[1:], traj_return[1:]]
    )
    check_joint_limits(complete_trajectory, label="完整轨迹")
    lowest_z = check_table_clearance(robot, complete_trajectory)
    print(f"整条轨迹连杆最低点：{lowest_z:.1f} mm（桌面 {TABLE_Z:.1f} mm）")

    # 按指导书顺序：先完成规划与仿真，再连接串口并下发全零目标。
    if show_animation:
        animate_robot_motion(
            robot,
            complete_trajectory,
            dt=0.04,
            target_point=target_a,
        )

    ser = None
    try:
        if mode != "simulate":
            ser = init_serial(port=port)
            if mode == "hardware" and ser is None:
                raise RuntimeError(f"hardware 模式无法打开串口 {port}")

        execute_joint_trajectory(ser, traj_prepare, duration=2.0)
        execute_joint_trajectory(ser, traj_target, duration=3.0)
        if ser is not None:
            time.sleep(1.5)
        execute_joint_trajectory(ser, traj_return, duration=3.0)
    finally:
        if ser is not None and ser.is_open:
            ser.close()
            print("串口已关闭。")

    return {
        "robot": robot,
        "q_target": q_target,
        "target": target_a,
        "reached": reached_point,
        "position_error": position_error,
        "trajectory": complete_trajectory,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第七周实验一：关节空间 MoveJ")
    parser.add_argument(
        "--mode",
        choices=("auto", "simulate", "hardware"),
        default="auto",
    )
    parser.add_argument(
        "--port",
        default="COM3",
        help="auto/hardware 模式使用的串口，默认 COM3",
    )
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()
    run_movej_experiment(
        show_animation=not args.no_animation,
        mode=args.mode,
        port=args.port,
    )
