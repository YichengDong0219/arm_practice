"""Replay a recorded teleoperation joint-command trajectory.

The CSV stores commanded J1..J6 targets and timestamps. Replay intentionally
does NOT run IK again; it sends the recorded joint commands directly.

Dry-run is the default and validates the file instantly.
Use --live to actually open the STM32 serial port.

Live replay assumes the physical arm begins near the all-zero pose. After the
user presses P, the arm is interpolated to the first recorded joint pose, then
the recorded timed command sequence starts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import ctypes

import numpy as np
import serial

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teleop import hardware as base
from teleop.trajectory_io import (
    load_metadata,
    load_trajectory,
)


VK_ESCAPE = 0x1B
VK_P = ord("P")
_user32 = ctypes.windll.user32


def key_down(vk: int) -> bool:
    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)


def wait_for_start() -> bool:
    print("按 P 开始 replay；按 ESC 取消。")
    while True:
        if key_down(VK_ESCAPE):
            return False
        if key_down(VK_P):
            while key_down(VK_P):
                time.sleep(0.01)
            return True
        time.sleep(0.01)


def open_serial(port: str, baudrate: int):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baudrate
    ser.timeout = base.SERIAL_TIMEOUT
    ser.dtr = False
    ser.rts = False
    ser.open()
    return ser


def positive_float(text: str) -> float:
    value = float(text)
    if not np.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(
            "必须是正有限数值"
        )
    return value


def replay_live(
    data,
    *,
    port: str,
    baudrate: int,
    speed_scale: float,
) -> None:
    ser = open_serial(port, baudrate)
    try:
        print(f"已连接 {port}，等待 STM32 1.5 秒...")
        time.sleep(1.5)

        if not wait_for_start():
            print("已取消 replay。")
            return

        home = np.zeros(6, dtype=np.float64)
        first_pose = data.joints_deg[0]

        print("平滑移动到记录轨迹首帧姿态...")
        base.move_interpolated(
            ser,
            home,
            first_pose,
            duration_s=2.0,
            steps=100,
        )

        print(
            f"开始 replay：{data.frame_count} 帧，"
            f"原始时长 {data.duration_s:.3f} s，"
            f"速度倍率 {speed_scale:g}x"
        )

        start = time.perf_counter()

        for index in range(data.frame_count):
            if key_down(VK_ESCAPE):
                print("\nESC：replay 中止。")
                return

            deadline = (
                start
                + float(data.t_s[index]) / speed_scale
            )
            base.sleep_until(deadline)

            ser.write(
                base.pack_frame(
                    np.deg2rad(
                        data.joints_deg[index]
                    )
                )
            )
            ser.flush()

            if (
                index == data.frame_count - 1
                or index % 25 == 0
            ):
                print(
                    f"\rframe {index + 1}/{data.frame_count}  "
                    f"t={data.t_s[index]:.2f}s  "
                    f"J1-6="
                    f"{np.round(data.joints_deg[index], 2).tolist()}°",
                    end="",
                    flush=True,
                )

        print("\nReplay 完成。")

    finally:
        if ser.is_open:
            ser.close()
            print("串口已关闭。")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trajectory",
        help="记录得到的 trajectory_*.csv",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="真正向机械臂发送 replay；省略时只校验文件",
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
        "--speed-scale",
        type=positive_float,
        default=1.0,
        help=(
            "时间播放倍率；1=原速，0.5=半速，2=两倍速"
        ),
    )
    args = parser.parse_args(argv)

    data = load_trajectory(args.trajectory)
    metadata = load_metadata(args.trajectory)

    print("轨迹校验通过：")
    print(f"  文件: {Path(args.trajectory).resolve()}")
    print(f"  帧数: {data.frame_count}")
    print(f"  时长: {data.duration_s:.3f} s")
    print(
        "  首帧 J1-6: "
        f"{np.round(data.joints_deg[0], 3).tolist()}°"
    )
    print(
        "  末帧 J1-6: "
        f"{np.round(data.joints_deg[-1], 3).tolist()}°"
    )

    if metadata:
        print(
            "  记录语义: "
            f"{metadata.get('angle_semantics', 'unknown')}"
        )
        print(
            "  原记录模式: "
            f"{metadata.get('mode', 'unknown')}"
        )

    if not args.live:
        print(
            "[DRY RUN] 未打开串口。"
            "加 --live 后才会执行真实 replay。"
        )
        return

    replay_live(
        data,
        port=args.port,
        baudrate=args.baudrate,
        speed_scale=args.speed_scale,
    )


if __name__ == "__main__":
    main()
