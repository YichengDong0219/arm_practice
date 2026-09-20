"""STM32 六轴机械臂的 16 字节下行协议。"""

from __future__ import annotations

import math
from collections.abc import Iterable


FRAME_LENGTH = 16
JOINT_COUNT = 6
FRAME_HEADER = 0xAA
CONTROL_MODE = 0x01
FRAME_TAIL = 0xBB
RADIAN_SCALE = 1000.0


def _six_finite_values(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != JOINT_COUNT:
        raise ValueError("关节角数组必须包含 6 个元素")
    if not all(math.isfinite(value) for value in result):
        raise ValueError("关节角数组不能包含 NaN 或无穷大")
    return result


def pack_frame(q_rad: Iterable[float]) -> bytes:
    """将 6 个弧度角打包为 STM32 使用的固定 16 字节控制帧。"""
    values = _six_finite_values(q_rad)
    frame = bytearray(FRAME_LENGTH)
    frame[0] = FRAME_HEADER

    for index, angle_rad in enumerate(values):
        encoded = int(angle_rad * RADIAN_SCALE)
        encoded = max(min(encoded, 32767), -32768)
        frame[1 + index * 2] = (encoded >> 8) & 0xFF
        frame[2 + index * 2] = encoded & 0xFF

    frame[13] = CONTROL_MODE
    checksum = 0
    for value in frame[:14]:
        checksum ^= value
    frame[14] = checksum
    frame[15] = FRAME_TAIL
    return bytes(frame)


def validate_frame(frame: bytes | bytearray) -> None:
    """验证帧长度、固定字段及 XOR 校验和。"""
    if len(frame) != FRAME_LENGTH:
        raise ValueError("控制帧长度必须为 16 字节")
    if frame[0] != FRAME_HEADER:
        raise ValueError("帧头错误")
    if frame[13] != CONTROL_MODE:
        raise ValueError("控制模式位错误")
    if frame[15] != FRAME_TAIL:
        raise ValueError("帧尾错误")

    checksum = 0
    for value in frame[:14]:
        checksum ^= value
    if frame[14] != checksum:
        raise ValueError("XOR 校验和错误")


def decode_frame(frame: bytes | bytearray) -> tuple[float, ...]:
    """将有效控制帧解码为 6 个弧度角，供测试和记录使用。"""
    validate_frame(frame)
    values = []
    for index in range(JOINT_COUNT):
        raw = (frame[1 + index * 2] << 8) | frame[2 + index * 2]
        if raw >= 0x8000:
            raw -= 0x10000
        values.append(raw / RADIAN_SCALE)
    return tuple(values)

