"""Cylindrical teleoperation with runtime-switchable end-effector orientation.

Motion:
    A/D : base turn, J1 only
    W/S : radial forward/backward
    Q/E : vertical up/down
    O/C/N : gripper open/close/neutral
    H : toggle HORIZONTAL <-> FREE orientation mode
    ESC : normal stop and return to all-zero pose

Modes:
    HORIZONTAL:
        J2 + J3 + J4 = -90 deg exactly.
    FREE:
        only r,z are controlled; J2/J3/J4 are otherwise unconstrained.

Switching FREE -> HORIZONTAL is planned first and then executed smoothly while
holding the current r,z target. If that r,z cannot support a horizontal tool,
the switch is refused and the robot stays in FREE mode.
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
ORIENTATION_ALIGN_DURATION_S = 0.8

PREP_POSE_DEG = np.array(
    [0.0, 0.0, -60.0, -30.0, 0.0, 0.0],
    dtype=np.float64,
)

GRIPPER_OPEN_DEG = 50.0
GRIPPER_CLOSED_DEG = -40.0
GRIPPER_NEUTRAL_DEG = 0.0
GRIPPER_SPEED_DEG_S = 60.0
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
VK_O = ord("O")
VK_C = ord("C")
VK_N = ord("N")
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28

CONTROL_KEYS = (
    VK_ESCAPE, VK_P, VK_H, VK_W, VK_A, VK_S, VK_D, VK_Q, VK_E,
    VK_O, VK_C, VK_N, VK_LEFT, VK_UP, VK_RIGHT, VK_DOWN,
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
        world_delta_deg = world_turn_sign * speed_deg_s * dt_s
        pose[0] += world_delta_deg / float(j1_sign)
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
        ser.write(base.pack_frame(np.deg2rad(pose)))
        ser.flush()


def record_pose(recorder, pose, *, j1_sign: float) -> None:
    if recorder is not None:
        recorder.record(
            position_ik.forward_xyz(pose, j1_sign=j1_sign),
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
    start = np.asarray(start_pose_deg, dtype=np.float64)
    target = np.asarray(target_pose_deg, dtype=np.float64)
    steps = max(1, int(round(duration_s * CONTROL_RATE_HZ)))
    period_s = duration_s / steps
    t0 = time.perf_counter()

    for index in range(1, steps + 1):
        ratio = index / steps
        pose = start + ratio * (target - start)
        send_pose(ser, pose, live=live)
        record_pose(recorder, pose, j1_sign=j1_sign)
        base.sleep_until(t0 + index * period_s)

    return target.copy()


def plan_orientation_alignment(
    start_pose,
    *,
    duration_s: float = ORIENTATION_ALIGN_DURATION_S,
):
    """Precompute FREE -> HORIZONTAL poses at constant r,z.

    Planning the complete path before any frame is sent means that if an
    intermediate fixed-pitch configuration is unreachable, the real robot
    does not start a partial transition.
    """
    start = np.asarray(start_pose, dtype=np.float64).copy()
    fixed_rz = position_ik.forward_radial_z(start)
    start_sum = position_ik.horizontal_sum_deg(start)
    target_sum = position_ik.HORIZONTAL_JOINT_SUM_DEG

    steps = max(1, int(round(duration_s * CONTROL_RATE_HZ)))
    poses = []
    previous = start.copy()

    for index in range(1, steps + 1):
        ratio = index / steps
        requested_sum = (
            start_sum
            + ratio * (target_sum - start_sum)
        )
        previous = position_ik.solve_radial_z_with_joint_sum(
            fixed_rz,
            requested_sum,
            previous,
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
    t0 = time.perf_counter()

    for index, pose in enumerate(poses, start=1):
        send_pose(ser, pose, live=live)
        record_pose(recorder, pose, j1_sign=j1_sign)
        base.sleep_until(t0 + index * period_s)

    return np.asarray(poses[-1], dtype=np.float64).copy()


def approach_scalar(current: float, target: float, max_step: float) -> float:
    delta = target - current
    if abs(delta) <= max_step:
        return float(target)
    return float(current + np.sign(delta) * max_step)


def run_teleop(args) -> None:
    ser = None
    recorder = None
    current_pose = None
    normal_stop = False
    orientation_mode = args.orientation_mode
    gripper_target_deg = GRIPPER_NEUTRAL_DEG

    print("圆柱坐标遥操作")
    print("  A / ← : 左转 / 俯视逆时针，只控制 J1")
    print("  D / → : 右转 / 俯视顺时针，只控制 J1")
    print("  W / ↑ : 前进 / 径向伸出")
    print("  S / ↓ : 后退 / 径向缩回")
    print("  Q/E    : 竖直上升 / 下降")
    print("  O/C/N  : 夹爪打开 / 闭合 / 参考位")
    print("  H      : HORIZONTAL / FREE 姿态模式切换")
    print("  ESC    : 正常结束并平滑回全零位")
    print()
    print(f"初始姿态模式: {orientation_mode.upper()}")
    print(f"直线速度: {args.speed:.1f} mm/s")
    print(f"J1 转速: {args.j1_speed:.1f} deg/s")

    if args.live:
        ser = open_serial(args.port, args.baudrate)
        print(f"已连接 {args.port}，等待 STM32 1.5 秒...")
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

        # We use one common, reachable preparation pose for both modes.
        # FREE mode simply stops enforcing the -90 deg sum after startup.
        current_pose = PREP_POSE_DEG.copy()

        if args.live:
            print("平滑进入准备姿态...")
            current_pose = move_pose_timed(
                ser,
                np.zeros(6, dtype=np.float64),
                current_pose,
                duration_s=2.0,
                live=True,
                recorder=None,
                j1_sign=args.j1_sign,
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
                    "source": "teleop/keyboard_position_ik.py",
                    "mode": "live" if args.live else "dry-run",
                    "control_space": "cylindrical_r_theta_z",
                    "orientation_mode_initial": orientation_mode,
                    "horizontal_constraint": "J2+J3+J4=-90deg",
                    "control_rate_hz": CONTROL_RATE_HZ,
                    "linear_speed_mm_s": float(args.speed),
                    "j1_speed_deg_s": float(args.j1_speed),
                    "j1_sign": float(args.j1_sign),
                },
            )
            print(f"轨迹文件: {recorder.csv_path}")

        period_s = 1.0 / CONTROL_RATE_HZ
        gripper_step_deg = GRIPPER_SPEED_DEG_S * period_s
        next_tick = time.perf_counter()
        last_status = 0.0
        last_error = 0.0

        while True:
            if key_down(VK_ESCAPE):
                print("\nESC：停止交互控制，准备回零。")
                normal_stop = True
                break

            if key_down(VK_H):
                wait_key_release(VK_H)

                if orientation_mode == "horizontal":
                    orientation_mode = "free"
                    print(
                        "\n姿态模式 -> FREE：解除 J2+J3+J4=-90° 约束。"
                    )
                else:
                    print(
                        "\n尝试切换 -> HORIZONTAL："
                        "保持当前 r,z，平滑调整末端姿态..."
                    )
                    try:
                        planned = plan_orientation_alignment(
                            current_pose,
                            duration_s=ORIENTATION_ALIGN_DURATION_S,
                        )
                    except position_ik.IKError as error:
                        print(
                            "当前位置无法在保持 r,z 的同时转为水平；"
                            "继续保持 FREE。"
                        )
                        print(f"原因: {error}")
                    else:
                        current_pose = execute_pose_sequence(
                            ser,
                            planned,
                            duration_s=ORIENTATION_ALIGN_DURATION_S,
                            recorder=recorder,
                            j1_sign=args.j1_sign,
                            live=bool(
                                args.live
                                and ser is not None
                                and ser.is_open
                            ),
                        )
                        orientation_mode = "horizontal"
                        print(
                            "姿态模式 -> HORIZONTAL："
                            "J2+J3+J4=-90°。"
                        )

                next_tick = time.perf_counter()
                continue

            current_pose = apply_base_turn(
                current_pose,
                turn_command(),
                dt_s=period_s,
                j1_sign=args.j1_sign,
                speed_deg_s=args.j1_speed,
            )

            rz_command = radial_vertical_command()
            if np.any(rz_command):
                current_rz = position_ik.forward_radial_z(current_pose)
                candidate_rz = (
                    current_rz
                    + rz_command * args.speed * period_s
                )

                try:
                    if orientation_mode == "horizontal":
                        candidate_pose = (
                            position_ik.solve_horizontal_radial_z(
                                candidate_rz,
                                current_pose,
                            )
                        )
                    else:
                        candidate_pose = position_ik.solve_radial_z(
                            candidate_rz,
                            current_pose,
                        )
                except position_ik.IKError as error:
                    now = time.perf_counter()
                    if now - last_error > 0.5:
                        print(f"\nIK 不可达: {error}")
                        last_error = now
                else:
                    current_pose = candidate_pose

            if key_down(VK_O):
                gripper_target_deg = GRIPPER_OPEN_DEG
            elif key_down(VK_C):
                gripper_target_deg = GRIPPER_CLOSED_DEG
            elif key_down(VK_N):
                gripper_target_deg = GRIPPER_NEUTRAL_DEG

            current_pose[5] = approach_scalar(
                float(current_pose[5]),
                float(gripper_target_deg),
                gripper_step_deg,
            )

            joint_sum = position_ik.horizontal_sum_deg(current_pose)

            if orientation_mode == "horizontal":
                error_deg = (
                    joint_sum
                    - position_ik.HORIZONTAL_JOINT_SUM_DEG
                )
                if abs(error_deg) > 1e-6:
                    raise RuntimeError(
                        "HORIZONTAL 模式约束被破坏，拒绝继续发送: "
                        f"J2+J3+J4={joint_sum:.9f}°"
                    )

            current_xyz = position_ik.forward_xyz(
                current_pose,
                j1_sign=args.j1_sign,
            )
            current_rz = position_ik.forward_radial_z(current_pose)
            theta_world_deg = args.j1_sign * float(current_pose[0])

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
            if now - last_status > 0.20:
                live_label = "LIVE" if args.live else "DRY"
                orient_label = (
                    "HORIZONTAL"
                    if orientation_mode == "horizontal"
                    else "FREE"
                )
                print(
                    f"\r[{live_label}] "
                    f"Orient={orient_label:10s}  "
                    f"r={current_rz[0]:7.1f} mm  "
                    f"theta={theta_world_deg:+7.1f}°  "
                    f"z={current_rz[1]:7.1f} mm  "
                    f"J234sum={joint_sum:+7.2f}°  "
                    f"CmdJ1-6={np.round(current_pose, 2).tolist()}°  "
                    f"GripTarget={gripper_target_deg:+.0f}°",
                    end="",
                    flush=True,
                )
                last_status = now

            next_tick += period_s
            base.sleep_until(next_tick)

    except KeyboardInterrupt:
        print("\nCtrl+C：人工中断，不自动回零。")
    finally:
        if normal_stop and current_pose is not None:
            print("\n正常结束：平滑回全零位...")
            try:
                current_pose = move_pose_timed(
                    ser,
                    current_pose,
                    np.zeros(6, dtype=np.float64),
                    duration_s=HOME_DURATION_S,
                    recorder=recorder,
                    j1_sign=args.j1_sign,
                    live=bool(
                        args.live
                        and ser is not None
                        and ser.is_open
                    ),
                )
                print("已回到全零目标姿态。")
            except Exception as error:
                print(f"回零过程中出现异常，停止继续发送: {error}")

        if recorder is not None:
            recorder.close()
            print(f"轨迹已保存: {recorder.csv_path}")
            print(f"元数据已保存: {recorder.json_path}")

        if ser is not None and ser.is_open:
            ser.close()
            print("串口已关闭。")

        safe_shutdown_prompt()


def positive_float(text: str) -> float:
    value = float(text)
    if not np.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("必须为正有限数值")
    return value


def j1_sign_value(text: str) -> float:
    value = float(text)
    if value not in (-1.0, 1.0):
        raise argparse.ArgumentTypeError("只能为 1 或 -1")
    return value


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--port", default=base.SERIAL_PORT)
    parser.add_argument("--baudrate", type=int, default=base.BAUDRATE)
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
        "--orientation-mode",
        choices=("horizontal", "free"),
        default="horizontal",
        help="启动姿态模式；运行中可按 H 随时切换",
    )
    parser.add_argument("--j1-sign", type=j1_sign_value, default=1.0)
    parser.add_argument("--record-file")
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args(argv)
    run_teleop(args)


if __name__ == "__main__":
    main()
