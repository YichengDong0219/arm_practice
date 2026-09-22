"""Low-level STM32 protocol used by teleoperation."""

from __future__ import annotations

import time
import numpy as np

SERIAL_PORT = "COM5"
BAUDRATE = 115200
SERIAL_TIMEOUT = 1


def pack_frame(q_rad):
    q_rad = np.asarray(q_rad, dtype=np.float64)
    if q_rad.shape != (6,):
        raise ValueError("joint angle array must contain 6 values")
    if not np.all(np.isfinite(q_rad)):
        raise ValueError("joint angle array contains non-finite values")

    tx = bytearray(16)
    tx[0] = 0xAA

    for i in range(6):
        value = int(q_rad[i] * 1000.0)
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


def sleep_until(deadline):
    remaining = deadline - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def move_interpolated(ser, start_pose_deg, target_pose_deg, duration_s, steps):
    start = np.asarray(start_pose_deg, dtype=np.float64)
    target = np.asarray(target_pose_deg, dtype=np.float64)

    period_s = duration_s / steps
    start_time = time.perf_counter()

    for index in range(1, steps + 1):
        ratio = index / steps
        pose_deg = start + ratio * (target - start)
        ser.write(pack_frame(np.deg2rad(pose_deg)))
        ser.flush()
        sleep_until(start_time + index * period_s)

    return target.copy()
