"""Keyboard teleoperation for the robot arm.

Two control modes are available.

CARTESIAN mode:
    A/D : base turn, J1 only
    W/S : radial forward/backward
    Q/E : vertical up/down
    O/C/N : gripper open/close/reference
    H : toggle HORIZONTAL/FREE orientation mode

JOINT mode:
    J1 : A/D  -> +/-
    J2 : W/S  -> +/-
    J3 : Q/E  -> +/-
    J4 : R/F  -> +/-
    J5 : T/G  -> +/-
    J6 : O/C  -> +/-
    N  : J6 returns toward 0 deg

ESC performs a normal smooth return to all-zero pose.

Displayed joint values are commanded joint angles, not encoder feedback.
"""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import sys
import time

import numpy as np
import serial

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teleop import hardware as base
from teleop import position_ik
from teleop.trajectory_io import TrajectoryRecorder, default_recording_path


CONTROL_RATE_HZ = 50.0
DEFAULT_SPEED_MM_S = 60.0
DEFAULT_J1_SPEED_DEG_S = 20.0
DEFAULT_JOINT_SPEED_DEG_S = 20.0
ORIENTATION_ALIGN_DURATION_S = 0.8

PREP_POSE_DEG = np.array(
    [0.0, 0.0, -60.0, -30.0, 0.0, 0.0],
    dtype=np.float64,
)

GRIPPER_OPEN_DEG = 50.0
GRIPPER_CLOSED_DEG = -40.0
GRIPPER_NEUTRAL_DEG = 0.0
GRIPPER_SPEED_DEG_S = 60.0

# Conservative protection used only in direct JOINT debug mode.
DIRECT_JOINT_LIMIT_DEG = 90.0

HOME_DURATION_S = 2.5

VK_ESCAPE = 0x1B
VK_P = ord("P")
VK_H = ord("H")
VK_W = ord("W")
VK_A = ord("A")
VK_S = ord("S")
VK_D = ord("D")
VK_Q = ord("Q")
VK_E = ord("E")
VK_R = ord("R")
VK_F = ord("F")
VK_T = ord("T")
VK_G = ord("G")
VK_O = ord("O")
VK_C = ord("C")
VK_N = ord("N")

VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28

CONTROL_KEYS = (
    VK_ESCAPE,
    VK_P,
    VK_H,
    VK_W,
    VK_A,
    VK_S,
    VK_D,
    VK_Q,
    VK_E,
    VK_R,
    VK_F,
    VK_T,
    VK_G,
    VK_O,
    VK_C,
    VK_N,
    VK_LEFT,
    VK_UP,
    VK_RIGHT,
    VK_DOWN,
)

_user32 = ctypes.windll.user32


def key_down(vk: int) -> bool:
    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)


def wait_key_release(vk: int) -> None:
    while key_down(vk):
        time.sleep(0.01)


def wait_all_control_keys_released(stable_s: float = 0.15) -> None:
    released_since = None

    while True:
        pressed = any(key_down(vk) for vk in CONTROL_KEYS)
        now = time.perf_counter()

        if pressed:
            released_since = None
        elif released_since is None:
            released_since = now
        elif now - released_since >= stable_s:
            return

        time.sleep(0.01)


def safe_shutdown_prompt() -> None:
    print()
    print("所有控制已停止。正在等待控制键释放...")
    wait_all_control_keys_released()
    print("控制键已全部释放。")
    print("按 Enter 返回 PowerShell。")

    try:
        input()
    except EOFError:
        pass


def radial_vertical_command() -> np.ndarray:
    command = np.zeros(2, dtype=np.float64)

    if key_down(VK_W) or key_down(VK_UP):
        command[0] += 1.0
    if key_down(VK_S) or key_down(VK_DOWN):
        command[0] -= 1.0

    if key_down(VK_Q):
        command[1] += 1.0
    if key_down(VK_E):
        command[1] -= 1.0

    norm = float(np.linalg.norm(command))
    if norm > 1.0:
        command /= norm

    return command


def turn_command() -> float:
    value = 0.0

    if key_down(VK_A) or key_down(VK_LEFT):
        value += 1.0
    if key_down(VK_D) or key_down(VK_RIGHT):
        value -= 1.0

    return value


def joint_keyboard_command() -> np.ndarray:
    """Return +/- velocity commands for J1..J6 in JOINT mode."""
    command = np.zeros(6, dtype=np.float64)

    # J1
    if key_down(VK_A):
        command[0] += 1.0
    if key_down(VK_D):
        command[0] -= 1.0

    # J2
    if key_down(VK_W):
        command[1] += 1.0
    if key_down(VK_S):
        command[1] -= 1.0

    # J3
    if key_down(VK_Q):
        command[2] += 1.0
    if key_down(VK_E):
        command[2] -= 1.0

    # J4
    if key_down(VK_R):
        command[3] += 1.0
    if key_down(VK_F):
        command[3] -= 1.0

    # J5
    if key_down(VK_T):
        command[4] += 1.0
    if key_down(VK_G):
        command[4] -= 1.0

    # J6
    if key_down(VK_O):
        command[5] += 1.0
    if key_down(VK_C):
        command[5] -= 1.0

    return command


def apply_base_turn(
    pose_deg,
    world_turn_sign: float,
    *,
    dt_s: float,
    j1_sign: float,
    speed_deg_s: float,
) -> np.ndarray:
    pose = np.asarray(pose_deg, dtype=np.float64).copy()

    if world_turn_sign != 0.0:
        world_delta_deg = (
            world_turn_sign
            * speed_deg_s
            * dt_s
        )
        pose[0] += world_delta_deg / float(j1_sign)

    return pose


def apply_joint_debug_limits(pose_deg) -> np.ndarray:
    """Conservative software limits for direct joint debugging."""
    pose = np.asarray(pose_deg, dtype=np.float64).copy()

    pose[0:3] = np.clip(
        pose[0:3],
        -DIRECT_JOINT_LIMIT_DEG,
        DIRECT_JOINT_LIMIT_DEG,
    )

    # Physical J4 command range.
    pose[3] = np.clip(
        pose[3],
        position_ik.JOINT4_MIN_DEG,
        position_ik.JOINT4_MAX_DEG,
    )

    pose[4] = np.clip(
        pose[4],
        -DIRECT_JOINT_LIMIT_DEG,
        DIRECT_JOINT_LIMIT_DEG,
    )

    pose[5] = np.clip(
        pose[5],
        GRIPPER_CLOSED_DEG,
        GRIPPER_OPEN_DEG,
    )

    return pose


def open_serial(port: str, baudrate: int):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baudrate
    ser.timeout = base.SERIAL_TIMEOUT
    ser.dtr = False
    ser.rts = False
    ser.open()
    return ser


def send_pose(ser, pose, *, live: bool) -> None:
    if live:
        ser.write(
            base.pack_frame(
                np.deg2rad(pose)
            )
        )
        ser.flush()


def record_pose(
    recorder,
    pose,
    *,
    j1_sign: float,
) -> None:
    if recorder is not None:
        recorder.record(
            position_ik.forward_xyz(
                pose,
                j1_sign=j1_sign,
            ),
            pose,
        )


def move_pose_timed(
    ser,
    start_pose_deg,
    target_pose_deg,
    *,
    duration_s: float,
    recorder=None,
    j1_sign: float = 1.0,
    live: bool = True,
) -> np.ndarray:
    start = np.asarray(
        start_pose_deg,
        dtype=np.float64,
    )
    target = np.asarray(
        target_pose_deg,
        dtype=np.float64,
    )

    steps = max(
        1,
        int(round(duration_s * CONTROL_RATE_HZ)),
    )
    period_s = duration_s / steps
    start_time = time.perf_counter()

    for index in range(1, steps + 1):
        ratio = index / steps
        pose = start + ratio * (target - start)

        send_pose(
            ser,
            pose,
            live=live,
        )

        record_pose(
            recorder,
            pose,
            j1_sign=j1_sign,
        )

        base.sleep_until(
            start_time + index * period_s
        )

    return target.copy()


def plan_orientation_alignment(
    start_pose,
    *,
    duration_s: float = ORIENTATION_ALIGN_DURATION_S,
):
    """Plan FREE -> HORIZONTAL while holding the current r,z."""
    start = np.asarray(
        start_pose,
        dtype=np.float64,
    ).copy()

    fixed_rz = position_ik.forward_radial_z(
        start
    )

    start_sum = position_ik.horizontal_sum_deg(
        start
    )
    target_sum = (
        position_ik.HORIZONTAL_JOINT_SUM_DEG
    )

    steps = max(
        1,
        int(round(duration_s * CONTROL_RATE_HZ)),
    )

    poses = []
    previous = start.copy()

    for index in range(1, steps + 1):
        ratio = index / steps

        requested_sum = (
            start_sum
            + ratio
            * (target_sum - start_sum)
        )

        previous = (
            position_ik.solve_radial_z_with_joint_sum(
                fixed_rz,
                requested_sum,
                previous,
            )
        )

        poses.append(previous.copy())

    return poses


def execute_pose_sequence(
    ser,
    poses,
    *,
    duration_s: float,
    recorder=None,
    j1_sign: float,
    live: bool,
) -> np.ndarray:
    if not poses:
        raise ValueError("poses must not be empty")

    period_s = duration_s / len(poses)
    start_time = time.perf_counter()

    for index, pose in enumerate(
        poses,
        start=1,
    ):
        send_pose(
            ser,
            pose,
            live=live,
        )

        record_pose(
            recorder,
            pose,
            j1_sign=j1_sign,
        )

        base.sleep_until(
            start_time + index * period_s
        )

    return np.asarray(
        poses[-1],
        dtype=np.float64,
    ).copy()


def approach_scalar(
    current: float,
    target: float,
    max_step: float,
) -> float:
    delta = target - current

    if abs(delta) <= max_step:
        return float(target)

    return float(
        current
        + np.sign(delta) * max_step
    )


def print_control_help(args) -> None:
    print()

    if args.control_mode == "cartesian":
        print("=== CARTESIAN 末端遥操作 ===")
        print("A / ← : J1 左转 / 逆时针")
        print("D / → : J1 右转 / 顺时针")
        print("W / ↑ : 径向前进")
        print("S / ↓ : 径向后退")
        print("Q     : 上升")
        print("E     : 下降")
        print("O     : 夹爪打开")
        print("C     : 夹爪闭合")
        print("N     : 夹爪回 0°")
        print("H     : HORIZONTAL / FREE 切换")

    else:
        print("=== JOINT 关节调试 ===")
        print("J1: A / D = + / -")
        print("J2: W / S = + / -")
        print("J3: Q / E = + / -")
        print("J4: R / F = + / -")
        print("J5: T / G = + / -")
        print("J6: O / C = + / -")
        print("N : J6 缓慢回到 0°")
        print(
            "直接关节调试保护: "
            f"J1-J5 ±{DIRECT_JOINT_LIMIT_DEG:g}°, "
            f"J6 [{GRIPPER_CLOSED_DEG:g}, "
            f"{GRIPPER_OPEN_DEG:g}]°"
        )

    print("ESC   : 正常结束并平滑回全零位")
    print()


def print_status(
    *,
    args,
    current_pose,
    orientation_mode,
) -> None:
    current_rz = position_ik.forward_radial_z(
        current_pose
    )

    theta_world_deg = (
        args.j1_sign
        * float(current_pose[0])
    )

    joint_sum = (
        position_ik.horizontal_sum_deg(
            current_pose
        )
    )

    live_label = (
        "LIVE"
        if args.live
        else "DRY"
    )

    if args.control_mode == "cartesian":
        extra = (
            f"Orient={orientation_mode.upper():10s} "
            f"r={current_rz[0]:7.1f}mm "
            f"theta={theta_world_deg:+7.1f}° "
            f"z={current_rz[1]:7.1f}mm "
            f"J234sum={joint_sum:+7.2f}° "
        )
    else:
        extra = "Control=JOINT "

    print(
        f"\r[{live_label}] "
        f"{extra}"
        f"J1={current_pose[0]:+7.2f}° "
        f"J2={current_pose[1]:+7.2f}° "
        f"J3={current_pose[2]:+7.2f}° "
        f"J4={current_pose[3]:+7.2f}° "
        f"J5={current_pose[4]:+7.2f}° "
        f"J6={current_pose[5]:+7.2f}°",
        end="",
        flush=True,
    )


def run_teleop(args) -> None:
    ser = None
    recorder = None
    current_pose = None
    normal_stop = False

    orientation_mode = args.orientation_mode
    gripper_target_deg = GRIPPER_NEUTRAL_DEG

    print_control_help(args)

    print(
        f"控制频率: {CONTROL_RATE_HZ:.1f} Hz"
    )

    if args.control_mode == "cartesian":
        print(
            f"直线速度: {args.speed:.1f} mm/s"
        )
        print(
            f"J1 转速: {args.j1_speed:.1f} deg/s"
        )
        print(
            f"初始姿态模式: "
            f"{orientation_mode.upper()}"
        )
    else:
        print(
            f"关节旋转速度: "
            f"{args.joint_speed:.1f} deg/s"
        )

    if args.live:
        ser = open_serial(
            args.port,
            args.baudrate,
        )
        print(
            f"已连接 {args.port}，"
            "等待 STM32 1.5 秒..."
        )
        time.sleep(1.5)
    else:
        print("[DRY RUN] 不打开串口。")

    try:
        print("按 P 开始；按 ESC 退出。")

        while True:
            if key_down(VK_ESCAPE):
                normal_stop = True
                return

            if key_down(VK_P):
                break

            time.sleep(0.01)

        wait_key_release(VK_P)

        home_pose = np.zeros(
            6,
            dtype=np.float64,
        )

        if args.control_mode == "cartesian":
            current_pose = PREP_POSE_DEG.copy()

            if args.live:
                print("平滑进入准备姿态...")

                current_pose = move_pose_timed(
                    ser,
                    home_pose,
                    current_pose,
                    duration_s=2.0,
                    live=True,
                    recorder=None,
                    j1_sign=args.j1_sign,
                )

        else:
            # Direct debugging starts at zero; pressing P itself
            # does not command a preparation movement.
            current_pose = home_pose.copy()
            print(
                "JOINT 模式从全零命令姿态开始。"
            )

        if not args.no_record:
            record_path = (
                Path(args.record_file)
                if args.record_file
                else default_recording_path()
            )

            recorder = TrajectoryRecorder(
                record_path,
                metadata={
                    "source": (
                        "teleop/"
                        "keyboard_position_ik.py"
                    ),
                    "mode": (
                        "live"
                        if args.live
                        else "dry-run"
                    ),
                    "control_mode": (
                        args.control_mode
                    ),
                    "orientation_mode_initial": (
                        orientation_mode
                    ),
                    "control_rate_hz": (
                        CONTROL_RATE_HZ
                    ),
                    "linear_speed_mm_s": (
                        float(args.speed)
                    ),
                    "j1_speed_deg_s": (
                        float(args.j1_speed)
                    ),
                    "joint_speed_deg_s": (
                        float(args.joint_speed)
                    ),
                    "j1_sign": (
                        float(args.j1_sign)
                    ),
                },
            )

            print(
                f"轨迹文件: "
                f"{recorder.csv_path}"
            )

        period_s = (
            1.0 / CONTROL_RATE_HZ
        )

        gripper_step_deg = (
            GRIPPER_SPEED_DEG_S
            * period_s
        )

        next_tick = (
            time.perf_counter()
        )
        last_status = 0.0
        last_error = 0.0

        while True:
            if key_down(VK_ESCAPE):
                print(
                    "\nESC：停止交互控制，"
                    "准备回零。"
                )
                normal_stop = True
                break

            if args.control_mode == "cartesian":
                if key_down(VK_H):
                    wait_key_release(VK_H)

                    if (
                        orientation_mode
                        == "horizontal"
                    ):
                        orientation_mode = "free"
                        print(
                            "\n姿态模式 -> FREE"
                        )

                    else:
                        print(
                            "\n尝试切换 -> "
                            "HORIZONTAL..."
                        )

                        try:
                            planned = (
                                plan_orientation_alignment(
                                    current_pose,
                                    duration_s=(
                                        ORIENTATION_ALIGN_DURATION_S
                                    ),
                                )
                            )

                        except position_ik.IKError as error:
                            print(
                                "当前位置无法保持 "
                                "r,z 转为水平；"
                                "继续保持 FREE。"
                            )
                            print(
                                f"原因: {error}"
                            )

                        else:
                            current_pose = (
                                execute_pose_sequence(
                                    ser,
                                    planned,
                                    duration_s=(
                                        ORIENTATION_ALIGN_DURATION_S
                                    ),
                                    recorder=recorder,
                                    j1_sign=(
                                        args.j1_sign
                                    ),
                                    live=bool(
                                        args.live
                                        and ser is not None
                                        and ser.is_open
                                    ),
                                )
                            )

                            orientation_mode = (
                                "horizontal"
                            )

                            print(
                                "姿态模式 -> "
                                "HORIZONTAL"
                            )

                    next_tick = (
                        time.perf_counter()
                    )
                    continue

                current_pose = apply_base_turn(
                    current_pose,
                    turn_command(),
                    dt_s=period_s,
                    j1_sign=args.j1_sign,
                    speed_deg_s=(
                        args.j1_speed
                    ),
                )

                rz_command = (
                    radial_vertical_command()
                )

                if np.any(rz_command):
                    current_rz = (
                        position_ik.forward_radial_z(
                            current_pose
                        )
                    )

                    candidate_rz = (
                        current_rz
                        + rz_command
                        * args.speed
                        * period_s
                    )

                    try:
                        if (
                            orientation_mode
                            == "horizontal"
                        ):
                            candidate_pose = (
                                position_ik
                                .solve_horizontal_radial_z(
                                    candidate_rz,
                                    current_pose,
                                )
                            )

                        else:
                            candidate_pose = (
                                position_ik
                                .solve_radial_z(
                                    candidate_rz,
                                    current_pose,
                                )
                            )

                    except position_ik.IKError as error:
                        now = (
                            time.perf_counter()
                        )

                        if (
                            now - last_error
                            > 0.5
                        ):
                            print(
                                f"\nIK 不可达: "
                                f"{error}"
                            )
                            last_error = now

                    else:
                        current_pose = (
                            candidate_pose
                        )

                if key_down(VK_O):
                    gripper_target_deg = (
                        GRIPPER_OPEN_DEG
                    )

                elif key_down(VK_C):
                    gripper_target_deg = (
                        GRIPPER_CLOSED_DEG
                    )

                elif key_down(VK_N):
                    gripper_target_deg = (
                        GRIPPER_NEUTRAL_DEG
                    )

                current_pose[5] = (
                    approach_scalar(
                        float(
                            current_pose[5]
                        ),
                        float(
                            gripper_target_deg
                        ),
                        gripper_step_deg,
                    )
                )

                if (
                    orientation_mode
                    == "horizontal"
                ):
                    joint_sum = (
                        position_ik
                        .horizontal_sum_deg(
                            current_pose
                        )
                    )

                    error_deg = (
                        joint_sum
                        - position_ik
                        .HORIZONTAL_JOINT_SUM_DEG
                    )

                    if abs(error_deg) > 1e-6:
                        raise RuntimeError(
                            "HORIZONTAL 约束被破坏，"
                            "拒绝继续发送。"
                        )

            else:
                joint_sign = (
                    joint_keyboard_command()
                )

                current_pose = (
                    current_pose
                    + joint_sign
                    * args.joint_speed
                    * period_s
                )

                if key_down(VK_N):
                    current_pose[5] = (
                        approach_scalar(
                            float(
                                current_pose[5]
                            ),
                            0.0,
                            args.joint_speed
                            * period_s,
                        )
                    )

                current_pose = (
                    apply_joint_debug_limits(
                        current_pose
                    )
                )

            send_pose(
                ser,
                current_pose,
                live=bool(args.live),
            )

            record_pose(
                recorder,
                current_pose,
                j1_sign=args.j1_sign,
            )

            now = time.perf_counter()

            if (
                now - last_status
                > 0.20
            ):
                print_status(
                    args=args,
                    current_pose=current_pose,
                    orientation_mode=(
                        orientation_mode
                    ),
                )

                last_status = now

            next_tick += period_s
            base.sleep_until(next_tick)

    except KeyboardInterrupt:
        print(
            "\nCtrl+C：人工中断，"
            "不自动回零。"
        )

    finally:
        if (
            normal_stop
            and current_pose is not None
        ):
            print(
                "\n正常结束："
                "平滑回全零位..."
            )

            try:
                current_pose = move_pose_timed(
                    ser,
                    current_pose,
                    np.zeros(
                        6,
                        dtype=np.float64,
                    ),
                    duration_s=HOME_DURATION_S,
                    recorder=recorder,
                    j1_sign=args.j1_sign,
                    live=bool(
                        args.live
                        and ser is not None
                        and ser.is_open
                    ),
                )

                print(
                    "已回到全零目标姿态。"
                )

            except Exception as error:
                print(
                    "回零过程中出现异常，"
                    f"停止继续发送: {error}"
                )

        if recorder is not None:
            recorder.close()

            print(
                f"轨迹已保存: "
                f"{recorder.csv_path}"
            )

            print(
                f"元数据已保存: "
                f"{recorder.json_path}"
            )

        if (
            ser is not None
            and ser.is_open
        ):
            ser.close()
            print("串口已关闭。")

        safe_shutdown_prompt()


def positive_float(text: str) -> float:
    value = float(text)

    if (
        not np.isfinite(value)
        or value <= 0.0
    ):
        raise argparse.ArgumentTypeError(
            "必须为正有限数值"
        )

    return value


def j1_sign_value(text: str) -> float:
    value = float(text)

    if value not in (-1.0, 1.0):
        raise argparse.ArgumentTypeError(
            "只能为 1 或 -1"
        )

    return value


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--live",
        action="store_true",
    )

    parser.add_argument(
        "--port",
        default=base.SERIAL_PORT,
    )

    parser.add_argument(
        "--baudrate",
        type=int,
        default=base.BAUDRATE,
    )

    parser.add_argument(
        "--control-mode",
        choices=("cartesian", "joint"),
        default="cartesian",
        help=(
            "cartesian=末端遥操作；"
            "joint=直接关节调试"
        ),
    )

    parser.add_argument(
        "--speed",
        type=positive_float,
        default=DEFAULT_SPEED_MM_S,
    )

    parser.add_argument(
        "--j1-speed",
        type=positive_float,
        default=DEFAULT_J1_SPEED_DEG_S,
    )

    parser.add_argument(
        "--joint-speed",
        type=positive_float,
        default=DEFAULT_JOINT_SPEED_DEG_S,
    )

    parser.add_argument(
        "--orientation-mode",
        choices=("horizontal", "free"),
        default="horizontal",
        help=(
            "CARTESIAN 模式的启动姿态；"
            "运行中可按 H 切换"
        ),
    )

    parser.add_argument(
        "--j1-sign",
        type=j1_sign_value,
        default=1.0,
    )

    parser.add_argument(
        "--record-file",
    )

    parser.add_argument(
        "--no-record",
        action="store_true",
    )

    args = parser.parse_args(argv)
    run_teleop(args)


if __name__ == "__main__":
    main()
