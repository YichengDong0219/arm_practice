"""关节空间插值、串口连接和无硬件预演工具。"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .protocol import pack_frame


JOINT_LIMIT_DEG = 90.0


def validate_pose(pose_deg: Iterable[float]) -> tuple[float, ...]:
    pose = tuple(float(value) for value in pose_deg)
    if len(pose) != 6:
        raise ValueError("姿态必须包含 [J1, J2, J3, J4, J5, J6] 六个角度")
    if not all(math.isfinite(value) for value in pose):
        raise ValueError("姿态不能包含 NaN 或无穷大")
    if any(abs(value) > JOINT_LIMIT_DEG for value in pose):
        raise ValueError(f"姿态超过 ±{JOINT_LIMIT_DEG:g}° 软件安全限位: {pose}")
    return pose


@dataclass(frozen=True)
class MotionStep:
    name: str
    target_deg: Sequence[float]
    duration_s: float
    steps: int
    hold_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_deg", validate_pose(self.target_deg))
        if self.duration_s <= 0:
            raise ValueError("动作时间必须大于 0")
        if self.steps <= 0:
            raise ValueError("插值步数必须为正整数")
        if self.hold_s < 0:
            raise ValueError("保持时间不能为负数")


class DryRunSerial:
    """只记录控制帧、不访问硬件的串口替身。"""

    def __init__(self) -> None:
        self.frames: list[bytes] = []
        self.is_open = True

    def write(self, data: bytes | bytearray) -> int:
        frame = bytes(data)
        self.frames.append(frame)
        return len(frame)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.is_open = False


def interpolate_pose(
    start_deg: Sequence[float], target_deg: Sequence[float], steps: int
) -> list[tuple[float, ...]]:
    start = validate_pose(start_deg)
    target = validate_pose(target_deg)
    if steps <= 0:
        raise ValueError("插值步数必须为正整数")
    return [
        tuple(start[i] + (target[i] - start[i]) * step / steps for i in range(6))
        for step in range(1, steps + 1)
    ]


def _sleep_until(deadline: float) -> None:
    remaining = deadline - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def move_joint_interpolated(
    serial_link,
    current_deg: Sequence[float],
    target_deg: Sequence[float],
    *,
    duration_s: float,
    steps: int,
    realtime: bool,
) -> tuple[float, ...]:
    """用绝对截止时间调度逐帧发送，减少周期误差累积。"""
    if duration_s <= 0:
        raise ValueError("动作时间必须大于 0")
    poses = interpolate_pose(current_deg, target_deg, steps)
    start_time = time.perf_counter()
    period = duration_s / steps

    for index, pose_deg in enumerate(poses, start=1):
        pose_rad = tuple(math.radians(value) for value in pose_deg)
        serial_link.write(pack_frame(pose_rad))
        serial_link.flush()
        if realtime:
            _sleep_until(start_time + index * period)
    return validate_pose(target_deg)


def run_sequence(
    serial_link,
    steps: Sequence[MotionStep],
    *,
    initial_pose_deg: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    realtime: bool,
) -> tuple[float, ...]:
    current = validate_pose(initial_pose_deg)
    for index, step in enumerate(steps, start=1):
        print(f"[{index}/{len(steps)}] {step.name}: {tuple(step.target_deg)}")
        current = move_joint_interpolated(
            serial_link,
            current,
            step.target_deg,
            duration_s=step.duration_s,
            steps=step.steps,
            realtime=realtime,
        )
        if realtime and step.hold_s:
            time.sleep(step.hold_s)
    return current


def open_serial(port: str, *, baudrate: int = 115200, timeout: float = 1.0):
    """延迟导入 pyserial；在打开端口前显式关闭 DTR/RTS。"""
    try:
        import serial
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "未安装 pyserial；请先激活 ra_class 或执行 "
            "python -m pip install -r week4/requirements.txt"
        ) from exc

    link = serial.Serial()
    link.port = port
    link.baudrate = baudrate
    link.timeout = timeout
    link.dtr = False
    link.rts = False
    link.open()
    return link

