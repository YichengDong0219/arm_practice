"""实验 1.9：六轴机械臂多关节联动与夹爪控制实机验证。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 兼容直接运行本文件：python week4/experiments/experiment_09_multijoint_gripper.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from week4.common.motion import DryRunSerial, MotionStep, open_serial, run_sequence


def build_sequence() -> list[MotionStep]:
    return [
        MotionStep("J1 底座旋转 +45°", [45, 0, 0, 0, 0, 0], 1.5, 30, 1.0),
        MotionStep("保持 J1，J3 抬升 +30°", [45, 0, 30, 0, 0, 0], 1.5, 30, 1.0),
        MotionStep("J6 夹爪张开 +50°", [45, 0, 30, 0, 0, 50], 1.0, 20, 1.0),
        MotionStep("J6 夹爪闭合回零", [45, 0, 30, 0, 0, 0], 1.0, 20, 1.0),
        MotionStep("J1 至 J5 协同联动", [30, -20, 35, -15, 20, 0], 2.0, 40, 2.0),
        MotionStep("全轴复位", [0, 0, 0, 0, 0, 0], 2.0, 40),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM5", help="实际串口号，例如 COM5")
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
        print("未打开串口。确认零位、供电和活动空间后加 --execute。")
        return 0

    link = open_serial(args.port)
    try:
        print(f"已连接 {args.port}，等待 STM32 初始化 1.5 秒……")
        time.sleep(1.5)
        run_sequence(link, sequence, realtime=True)
        print("实验 1.9 动作序列完成。")
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
