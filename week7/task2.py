"""第七周实验 1.3：笛卡尔空间 MoveL / MoveC 轨迹规划。"""

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

# 当前机械臂基坐标系中的实测/标定桌面高度，单位为 mm。
TABLE_Z = 52.0
# D-H 末端到夹爪最低点的 z 向偏移；尚未单独标定时先按 0 mm 处理。
# 若夹爪最低点比 D-H 末端低，应把该值改成相应的负数。
TOOL_LOWEST_POINT_OFFSET_Z = 0.0
# 规划时要求夹爪最低点至少高出桌面 5 mm。
TABLE_CLEARANCE = 5.0

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
# 笛卡尔路径相邻两点允许的最大关节角变化；超过即认为逆解跳了分支。
MAX_JOINT_STEP_DEG = 5.0


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


def pack_frame(q_rad):
    """将 6 个关节弧度值打包为 STM32 使用的 16 字节控制帧。"""
    q_rad = np.asarray(q_rad, dtype=np.float64)
    if q_rad.shape != (6,):
        raise ValueError(f"q_rad 必须包含 6 个关节角，实际形状为 {q_rad.shape}")

    tx = bytearray(16)
    tx[0] = 0xAA
    for index, q_i in enumerate(q_rad):
        value = int(q_i * 1000)
        value = max(min(value, 32767), -32768)
        tx[1 + index * 2] = (value >> 8) & 0xFF
        tx[2 + index * 2] = value & 0xFF

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


def animate_robot_motion(
    robot,
    q_trajectory,
    dt=0.04,
    reference_points=None,
    via_point=None,
    motion="movel",
    path_start_index=0,
):
    """播放完整轨迹，并同时显示笛卡尔参考路径和末端实际轨迹。"""
    q_trajectory = np.asarray(q_trajectory, dtype=np.float64)
    reference_points = np.asarray(reference_points, dtype=np.float64)
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
    if hasattr(env, "timer"):
        env.timer.set_visible(False)

    table_x, table_y = np.meshgrid(
        np.linspace(-150.0, 450.0, 2),
        np.linspace(-250.0, 250.0, 2),
    )
    ax.plot_surface(
        table_x,
        table_y,
        np.full_like(table_x, TABLE_Z),
        color="sandybrown",
        alpha=0.18,
        shade=False,
    )
    ax.text(-140.0, -240.0, TABLE_Z + 4.0, f"Table z={TABLE_Z:.0f} mm")

    ax.plot(
        *tool_path.T,
        "--",
        color="0.65",
        linewidth=1.2,
        label="Planned TCP path",
    )
    ax.plot(
        *reference_points.T,
        "--",
        color="royalblue",
        linewidth=2.8,
        zorder=15,
        label=f"{motion.upper()} reference",
    )
    ax.scatter(*reference_points[0], color="green", s=35, label="Path start")
    ax.scatter(*reference_points[-1], color="crimson", s=35, label="Path goal")
    if via_point is not None:
        ax.scatter(*via_point, color="purple", s=35, label="Arc via point")

    (traced,) = ax.plot(
        [], [], [], color="darkorange", linewidth=2.2, label="TCP trace"
    )
    ax.legend(loc="upper left")
    ax.set_title(f"{motion.upper()} - end-effector path")

    plane_axes = (0, 1) if motion == "movec" else (1, 2)
    horizontal_axis, vertical_axis = plane_axes
    plane_name = "X-Y" if motion == "movec" else "Y-Z"
    detail = env.fig.add_axes([0.64, 0.69, 0.30, 0.26])
    detail.plot(
        reference_points[:, horizontal_axis],
        reference_points[:, vertical_axis],
        "--",
        color="royalblue",
        linewidth=3.2,
    )
    detail.scatter(
        reference_points[0, horizontal_axis],
        reference_points[0, vertical_axis],
        color="green",
        s=25,
        zorder=5,
    )
    detail.scatter(
        reference_points[-1, horizontal_axis],
        reference_points[-1, vertical_axis],
        color="crimson",
        s=25,
        zorder=5,
    )
    if via_point is not None:
        detail.scatter(
            via_point[horizontal_axis],
            via_point[vertical_axis],
            color="purple",
            s=25,
            zorder=5,
        )
    (detail_trace,) = detail.plot([], [], color="darkorange", linewidth=1.5)
    low = reference_points[:, plane_axes].min(axis=0) - 10.0
    high = reference_points[:, plane_axes].max(axis=0) + 10.0
    detail.set_xlim(low[0], high[0])
    detail.set_ylim(low[1], high[1])
    detail.set_xlabel(f"{plane_name[0]} (mm)", fontsize=8)
    detail.set_ylabel(f"{plane_name[-1]} (mm)", fontsize=8)
    detail.set_title(f"{motion.upper()} TCP path ({plane_name})", fontsize=9)
    detail.tick_params(labelsize=7)
    detail.grid(True, alpha=0.3)

    for index, q in enumerate(q_trajectory):
        robot.q = q
        traced.set_data(tool_path[: index + 1, 0], tool_path[: index + 1, 1])
        traced.set_3d_properties(tool_path[: index + 1, 2])
        if index >= path_start_index:
            path_end = min(index + 1, path_start_index + len(reference_points))
            detail_trace.set_data(
                tool_path[path_start_index:path_end, horizontal_axis],
                tool_path[path_start_index:path_end, vertical_axis],
            )
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
    if len(q_traj_rad) == 0:
        raise ValueError("q_traj_rad 不能为空")
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


def validate_table_clearance(robot, q_trajectory):
    """检查完整轨迹中的夹爪最低点是否保持在桌面安全高度以上。"""
    tool_path = np.asarray([robot.fkine(q).t for q in q_trajectory])
    lowest_point_z = tool_path[:, 2] + TOOL_LOWEST_POINT_OFFSET_Z
    minimum_index = int(np.argmin(lowest_point_z))
    minimum_z = float(lowest_point_z[minimum_index])
    required_z = TABLE_Z + TABLE_CLEARANCE

    if minimum_z < required_z:
        raise RuntimeError(
            "轨迹桌面净空检查失败："
            f"第 {minimum_index + 1} 帧夹爪最低点 z={minimum_z:.3f} mm，"
            f"要求不低于 {required_z:.3f} mm "
            f"（TABLE_Z={TABLE_Z:.3f} mm）。"
        )
    return minimum_z


def plan_cartesian_arc(p0, p1, p2, steps=60):
    """根据三个不共线空间点生成经过 ``p1`` 的五次时间参数化圆弧。"""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    if p0.shape != (3,) or p1.shape != (3,) or p2.shape != (3,):
        raise ValueError("p0、p1、p2 都必须是长度为 3 的空间坐标")
    if steps < 2:
        raise ValueError("steps 必须至少为 2")

    v1 = p1 - p0
    v2 = p2 - p0
    normal = np.cross(v1, v2)
    normal_length = np.linalg.norm(normal)
    if normal_length < 1e-6:
        raise ValueError("三个路径点共线，无法确定唯一圆弧")
    normal /= normal_length

    # 利用三角形重心坐标形式计算外接圆圆心。
    a = np.linalg.norm(p1 - p2)
    b = np.linalg.norm(p0 - p2)
    c = np.linalg.norm(p0 - p1)
    alpha = a**2 * (b**2 + c**2 - a**2)
    beta = b**2 * (a**2 + c**2 - b**2)
    gamma = c**2 * (a**2 + b**2 - c**2)
    denominator = alpha + beta + gamma
    if abs(denominator) < 1e-8:
        raise ValueError("三个路径点构成退化三角形")

    center = (alpha * p0 + beta * p1 + gamma * p2) / denominator
    radius = np.linalg.norm(p0 - center)
    u_axis = (p0 - center) / radius
    v_axis = np.cross(normal, u_axis)

    via_angle = np.arctan2(
        np.dot(p1 - center, v_axis), np.dot(p1 - center, u_axis)
    ) % (2 * np.pi)
    goal_angle = np.arctan2(
        np.dot(p2 - center, v_axis), np.dot(p2 - center, u_axis)
    ) % (2 * np.pi)
    if goal_angle < via_angle:
        goal_angle += 2 * np.pi

    tau = np.linspace(0.0, 1.0, steps)
    blend = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    angles = blend * goal_angle
    return (
        center
        + radius * np.cos(angles)[:, None] * u_axis
        + radius * np.sin(angles)[:, None] * v_axis
    )


def solve_cartesian_path(robot, cartesian_points, q_seed):
    """逐点求解位置逆运动学，并以前一时刻解保持分支连续。"""
    cartesian_points = np.asarray(cartesian_points, dtype=np.float64)
    q_previous = np.asarray(q_seed, dtype=np.float64).copy()
    q_trajectory = np.empty((len(cartesian_points), robot.n), dtype=np.float64)

    for index, point in enumerate(cartesian_points):
        solution = robot.ikine_LM(
            sm.SE3(*point),
            q0=q_previous,
            mask=[1, 1, 1, 0, 0, 0],
        )
        if not solution.success:
            reason = getattr(solution, "reason", "未知原因")
            raise RuntimeError(
                f"第 {index + 1}/{len(cartesian_points)} 个路径点逆解失败："
                f"{np.round(point, 3)} mm；{reason}"
            )

        q_current = np.asarray(solution.q, dtype=np.float64)
        q_current[4] = 0.0  # 第 5 轴不影响末端位置，锁零以避免随机漂移。
        q_trajectory[index] = q_current
        q_previous = q_current

    return q_trajectory


def run_movel_experiment(show_animation=True, mode="auto", motion="movel", port="COM3"):
    """完成展开、MoveL/MoveC 笛卡尔作业及安全回零流程。"""
    if mode not in {"auto", "simulate", "hardware"}:
        raise ValueError(f"不支持的运行模式：{mode}")
    if motion not in {"movel", "movec"}:
        raise ValueError(f"不支持的运动类型：{motion}")

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
    # 指导书给定的种子。原先的 [0,-45,-80,80,0] 会让整条路径收敛到
    # J3≈-96°、J4≈+145° 的越限分支（J4 实机上限 25°）。
    q_seed = np.deg2rad([0.0, 45.0, -80.0, -80.0, 0.0])
    p_start = np.array([120.0, 0.0, 260.0], dtype=np.float64)
    p_via = np.array([160.0, 40.0, 260.0], dtype=np.float64)
    p_goal = (
        np.array([120.0, 80.0, 220.0], dtype=np.float64)
        if motion == "movel"
        else np.array([120.0, 80.0, 260.0], dtype=np.float64)
    )
    path_steps = 60

    if motion == "movel":
        poses = rtb.ctraj(sm.SE3(*p_start), sm.SE3(*p_goal), path_steps)
        cartesian_points = np.asarray([pose.t for pose in poses])
        via_point = None
    else:
        cartesian_points = plan_cartesian_arc(
            p_start, p_via, p_goal, steps=path_steps
        )
        via_point = p_via

    q_path = solve_cartesian_path(robot, cartesian_points, q_seed)
    q_enter = rtb.jtraj(q_zero, q_path[0], 40).q
    q_return = rtb.jtraj(q_path[-1], q_zero, 60).q
    complete_trajectory = np.vstack([q_enter, q_path[1:], q_return[1:]])

    minimum_tool_z = validate_table_clearance(robot, complete_trajectory)
    check_joint_limits(complete_trajectory, label="完整轨迹")

    actual_path = np.asarray([robot.fkine(q).t for q in q_path])
    point_errors = np.linalg.norm(actual_path - cartesian_points, axis=1)
    max_joint_step = np.max(np.abs(np.diff(np.rad2deg(q_path), axis=0)))
    if max_joint_step > MAX_JOINT_STEP_DEG:
        raise RuntimeError(
            f"笛卡尔路径相邻关节角最大变化 {max_joint_step:.2f}° 超过 "
            f"{MAX_JOINT_STEP_DEG:.1f}°，逆解可能跳分支，已拒绝下发。"
        )

    print(f"\n========== 第七周实验 1.3：{motion.upper()} ==========")
    print(f"路径起点：{p_start} mm")
    if via_point is not None:
        print(f"圆弧过渡点：{via_point} mm")
    print(f"路径终点：{p_goal} mm")
    print(f"路径点数：{len(cartesian_points)}")
    print(f"最大位置逆解误差：{point_errors.max():.6f} mm")
    print(f"最大相邻关节角变化：{max_joint_step:.6f} deg")
    print(f"桌面高度：{TABLE_Z:.3f} mm")
    print(f"整条轨迹夹爪最低点：{minimum_tool_z:.3f} mm")
    print(f"最小桌面净空：{minimum_tool_z - TABLE_Z:.3f} mm")

    # 按指导书先完成整条路径规划和仿真，再尝试连接实机。
    if show_animation:
        animate_robot_motion(
            robot,
            complete_trajectory,
            dt=0.04,
            reference_points=cartesian_points,
            via_point=via_point,
            motion=motion,
            path_start_index=len(q_enter) - 1,
        )

    ser = None
    try:
        if mode != "simulate":
            ser = init_serial(port=port)
            if mode == "hardware" and ser is None:
                raise RuntimeError(f"hardware 模式无法打开串口 {port}")

        execute_joint_trajectory(ser, q_enter, duration=2.0)
        execute_joint_trajectory(ser, q_path, duration=4.0)
        if ser is not None:
            time.sleep(1.5)
        execute_joint_trajectory(ser, q_return, duration=3.0)
    finally:
        if ser is not None and ser.is_open:
            ser.close()
            print("串口已关闭。")

    return {
        "robot": robot,
        "motion": motion,
        "cartesian_points": cartesian_points,
        "actual_path": actual_path,
        "q_path": q_path,
        "trajectory": complete_trajectory,
        "position_errors": point_errors,
        "minimum_tool_z": minimum_tool_z,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="第七周实验二：笛卡尔 MoveL / MoveC"
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "simulate", "hardware"),
        default="auto",
    )
    parser.add_argument(
        "--motion",
        choices=("movel", "movec", "both"),
        default="movel",
    )
    parser.add_argument(
        "--port",
        default="COM3",
        help="auto/hardware 模式使用的串口，默认 COM3",
    )
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()

    if args.motion == "both":
        if args.mode != "simulate":
            parser.error(
                "连续预览 movel 和 movec 请显式指定 --mode simulate；"
                "auto/hardware 一次只执行一条路径"
            )
        for motion_name in ("movel", "movec"):
            run_movel_experiment(
                show_animation=not args.no_animation,
                mode=args.mode,
                motion=motion_name,
                port=args.port,
            )
    else:
        run_movel_experiment(
            show_animation=not args.no_animation,
            mode=args.mode,
            motion=args.motion,
            port=args.port,
        )
