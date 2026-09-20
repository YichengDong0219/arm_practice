"""实验 1.10：六轴机械臂抓取与搬运。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 兼容直接运行本文件：python week4/experiments/experiment_10_pick_and_place.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from week4.common.motion import DryRunSerial, MotionStep, open_serial, run_sequence


def build_sequence() -> list[MotionStep]:
    return [
        MotionStep("到达抓取位", [30, -30, -80, -15, 0, 0], 2.0, 40, 2.0),
        MotionStep("J6 闭合夹爪", [30, -30, -80, -15, 0, -40], 2.0, 40, 2.0),
        MotionStep("保持夹持并搬运", [0, -10, -60, -15, 0, -40], 2.0, 40, 5.0),
        MotionStep("全轴复位并释放", [0, 0, 0, 0, 0, 0], 2.0, 40),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM5", help="实际串口号；不要照抄指导书 COM6")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正打开串口并驱动机械臂；省略时只进行无硬件预演",
    )
    args = parser.parse_args(argv)
    sequence = build_sequence()

    if not args.execute:
        link = DryRunSerial()
        final_pose = run_sequence(link, sequence, realtime=False)
        print(f"\n预演完成：共生成 {len(link.frames)} 帧，最终姿态 {final_pose}")
        print("指导书姿态尚未按你的机械零位和物体位置标定，当前未打开串口。")
        return 0

    link = open_serial(args.port)
    try:
        print(f"已连接 {args.port}，等待 STM32 初始化 1.5 秒……")
        time.sleep(1.5)
        run_sequence(link, sequence, realtime=True)
        print("实验 1.10 抓取搬运序列完成。")
    except KeyboardInterrupt:
        print("用户中断：停止发送，不在未知状态下强制复位。")
        return 130
    finally:
        if link.is_open:
            link.close()
            print("串口已释放。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
