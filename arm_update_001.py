#!/usr/bin/env python3
"""Apply arm_practice ChatGPT update 001.

Run this file from anywhere inside your local arm_practice Git repository.
It creates/switches to feature/keyboard-teleop, writes the first teleop
prototype, runs a syntax check, stages the files, and commits them.
"""

from pathlib import Path
import subprocess
import sys

BRANCH = "feature/keyboard-teleop"
COMMIT_MESSAGE = "feat: add keyboard XY teleoperation prototype"

FILES = {
    "teleop/keyboard_xy.py": '"""Keyboard teleoperation prototype for the planar J2-J3-J4 chain.\n\nWindows only for now. It reuses the existing STM32 16-byte joint command\nprotocol from task8.py and the planar 3R kinematics from task8_reverse.py.\n\nControls after preparation:\n    W / S : +Y / -Y\n    A / D : -X / +X\n    ESC   : stop sending and exit\n\nStartup is intentionally conservative:\n    1. Put the physical arm near the all-zero pose.\n    2. Run this program.\n    3. Press P once to move to a non-singular teleop preparation pose.\n    4. Use WASD to move the end effector continuously.\n\nThe program never auto-returns home after an exception or ESC.\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport ctypes\nfrom pathlib import Path\nimport sys\nimport time\n\nimport numpy as np\nimport serial\n\nROOT = Path(__file__).resolve().parents[1]\nif str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))\n\nimport task8 as base\nimport task8_reverse as kin\n\n\nCONTROL_RATE_HZ = 50.0\nDEFAULT_SPEED_MM_S = 20.0\nMAX_JOINT_STEP_DEG = 3.0\nPREP_POSE_DEG = np.array([0.0, 0.0, -30.0, -30.0, 0.0, 0.0], dtype=np.float64)\n\n# Windows virtual-key codes.\nVK_ESCAPE = 0x1B\nVK_P = ord("P")\nVK_W = ord("W")\nVK_A = ord("A")\nVK_S = ord("S")\nVK_D = ord("D")\n\n_user32 = ctypes.windll.user32\n\n\ndef key_down(vk: int) -> bool:\n    """Return True while a keyboard key is physically held down."""\n    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)\n\n\ndef wait_key_release(vk: int) -> None:\n    """Wait for one key to be released so a press is not reused immediately."""\n    while key_down(vk):\n        time.sleep(0.01)\n\n\ndef forward_tool_pose_xy(pose_deg: np.ndarray) -> tuple[np.ndarray, float]:\n    """Return planar tool-tip XY [mm] and tool direction [rad]."""\n    pose_deg = np.asarray(pose_deg, dtype=np.float64)\n    q2_model = np.deg2rad(pose_deg[1]) + kin.JOINT2_ZERO_DIRECTION_RAD\n    q3 = np.deg2rad(pose_deg[2])\n    q4 = np.deg2rad(pose_deg[3])\n\n    x_mm = (\n        kin.LINK_1_MM * np.cos(q2_model)\n        + kin.LINK_2_MM * np.cos(q2_model + q3)\n        + kin.LINK_3_MM * np.cos(q2_model + q3 + q4)\n    )\n    y_mm = (\n        kin.LINK_1_MM * np.sin(q2_model)\n        + kin.LINK_2_MM * np.sin(q2_model + q3)\n        + kin.LINK_3_MM * np.sin(q2_model + q3 + q4)\n    )\n    tool_angle = q2_model + q3 + q4\n    return np.array([x_mm, y_mm], dtype=np.float64), float(tool_angle)\n\n\ndef pose_is_safe(pose_deg: np.ndarray) -> bool:\n    """Apply the software limits already used by the experiment code."""\n    pose_deg = np.asarray(pose_deg, dtype=np.float64)\n    if pose_deg.shape != (6,) or not np.all(np.isfinite(pose_deg)):\n        return False\n    if np.max(np.abs(pose_deg)) > base.ANGLE_LIMIT_DEG + base.ANGLE_TOLERANCE_DEG:\n        return False\n    if not np.allclose(pose_deg[[0, 4, 5]], 0.0, atol=base.ANGLE_TOLERANCE_DEG):\n        return False\n    # Existing hardware notes in task8_reverse.py keep Joint 4 out of\n    # its mechanically limited positive-angle region.\n    if pose_deg[3] > base.ANGLE_TOLERANCE_DEG:\n        return False\n    return True\n\n\ndef orientation_offsets_deg():\n    """Try nearby tool directions first, then a sparse global fallback."""\n    local = [0.0]\n    for value in range(1, 31):\n        local.extend((-float(value), float(value)))\n\n    global_fallback = []\n    for value in range(33, 181, 3):\n        global_fallback.extend((-float(value), float(value)))\n\n    return local + global_fallback\n\n\nORIENTATION_OFFSETS_DEG = orientation_offsets_deg()\n\n\ndef solve_continuous_ik(\n    target_xy_mm: np.ndarray,\n    previous_pose_deg: np.ndarray,\n    previous_tool_angle_rad: float,\n) -> tuple[np.ndarray, float] | None:\n    """Find a nearby valid 3R IK solution for one small Cartesian step.\n\n    The solver prefers the previous tool direction and the smallest joint-space\n    change. This reduces branch jumps during interactive control.\n    """\n    best = None\n\n    for offset_deg in ORIENTATION_OFFSETS_DEG:\n        tool_angle = previous_tool_angle_rad + np.deg2rad(offset_deg)\n        try:\n            pose_deg = kin.inverse_kinematics_tool_reversed(\n                float(target_xy_mm[0]),\n                float(target_xy_mm[1]),\n                float(tool_angle),\n            )\n        except ValueError:\n            continue\n\n        if not pose_is_safe(pose_deg):\n            continue\n\n        joint_delta = np.abs(pose_deg - previous_pose_deg)\n        max_delta = float(np.max(joint_delta))\n        if max_delta > MAX_JOINT_STEP_DEG:\n            continue\n\n        # Prefer continuity first; changing the tool direction is a secondary cost.\n        score = max_delta + 0.01 * abs(offset_deg)\n        if best is None or score < best[0]:\n            best = (score, pose_deg, float(tool_angle))\n\n    if best is None:\n        return None\n    return best[1].copy(), best[2]\n\n\ndef keyboard_direction() -> np.ndarray:\n    """Return normalized planar direction requested by WASD."""\n    direction = np.zeros(2, dtype=np.float64)\n\n    if key_down(VK_A):\n        direction[0] -= 1.0\n    if key_down(VK_D):\n        direction[0] += 1.0\n    if key_down(VK_S):\n        direction[1] -= 1.0\n    if key_down(VK_W):\n        direction[1] += 1.0\n\n    norm = float(np.linalg.norm(direction))\n    if norm > 1.0:\n        direction /= norm\n    return direction\n\n\ndef open_serial(port: str, baudrate: int):\n    ser = serial.Serial()\n    ser.port = port\n    ser.baudrate = baudrate\n    ser.timeout = base.SERIAL_TIMEOUT\n    ser.dtr = False\n    ser.rts = False\n    ser.open()\n    return ser\n\n\ndef run_teleop(args) -> None:\n    live = args.live\n    ser = None\n\n    print("二维键盘遥操作原型")\n    print("  P     : 从全零位平滑进入遥操作准备姿态")\n    print("  W / S : 末端 +Y / -Y")\n    print("  A / D : 末端 -X / +X")\n    print("  ESC   : 立即停止发送并退出")\n    print(f"  控制频率: {CONTROL_RATE_HZ:.1f} Hz")\n    print(f"  平移速度: {args.speed:.1f} mm/s")\n    print("  退出或异常时不会自动复位。")\n\n    if live:\n        print()\n        print("请确认：机械臂当前真实姿态位于全零位附近，运动范围内无人员和障碍物。")\n        ser = open_serial(args.port, args.baudrate)\n        print(f"已连接串口 {args.port}，等待 STM32 初始化 1.5 秒...")\n        time.sleep(1.5)\n    else:\n        print()\n        print("[DRY RUN] 不会打开串口，也不会驱动机械臂。")\n\n    try:\n        print()\n        print("准备完成。按 P 进入非奇异遥操作姿态；按 ESC 退出。")\n        while True:\n            if key_down(VK_ESCAPE):\n                return\n            if key_down(VK_P):\n                break\n            time.sleep(0.01)\n\n        wait_key_release(VK_P)\n\n        home_pose = np.zeros(6, dtype=np.float64)\n        current_pose = PREP_POSE_DEG.copy()\n\n        if live:\n            print("正在平滑进入遥操作准备姿态...")\n            base.move_interpolated(\n                ser,\n                home_pose,\n                PREP_POSE_DEG,\n                duration_s=2.0,\n                steps=100,\n            )\n        else:\n            print(f"[DRY RUN] 准备姿态 J2/J3/J4 = {PREP_POSE_DEG[1:4].tolist()}°")\n\n        target_xy, tool_angle = forward_tool_pose_xy(current_pose)\n        print(\n            "遥操作已启用。当前末端平面坐标 "\n            f"X={target_xy[0]:.1f} mm, Y={target_xy[1]:.1f} mm"\n        )\n\n        period_s = 1.0 / CONTROL_RATE_HZ\n        next_tick = time.perf_counter()\n        last_status_time = 0.0\n        last_block_time = 0.0\n\n        while True:\n            if key_down(VK_ESCAPE):\n                print("\\nESC：停止发送。")\n                break\n\n            direction = keyboard_direction()\n            moving = bool(np.any(direction))\n            candidate_pose = current_pose\n            candidate_tool_angle = tool_angle\n            candidate_xy = target_xy\n\n            if moving:\n                candidate_xy = target_xy + direction * args.speed * period_s\n                result = solve_continuous_ik(\n                    candidate_xy,\n                    current_pose,\n                    tool_angle,\n                )\n\n                if result is not None:\n                    candidate_pose, candidate_tool_angle = result\n                else:\n                    now = time.perf_counter()\n                    if now - last_block_time > 0.5:\n                        print(\n                            "\\n目标被安全约束拦截：不可达、超过关节限位，"\n                            "或单帧关节变化过大。"\n                        )\n                        last_block_time = now\n                    candidate_xy = target_xy\n\n            if live:\n                ser.write(base.pack_frame(np.deg2rad(candidate_pose)))\n                ser.flush()\n\n            if moving and not np.array_equal(candidate_xy, target_xy):\n                current_pose = candidate_pose.copy()\n                tool_angle = candidate_tool_angle\n                target_xy = candidate_xy.copy()\n\n            now = time.perf_counter()\n            if now - last_status_time > 0.20:\n                mode = "LIVE" if live else "DRY"\n                print(\n                    f"\\r[{mode}] X={target_xy[0]:8.2f} mm  "\n                    f"Y={target_xy[1]:8.2f} mm  "\n                    f"J2/J3/J4={np.round(current_pose[1:4], 2).tolist()}°",\n                    end="",\n                    flush=True,\n                )\n                last_status_time = now\n\n            next_tick += period_s\n            base.sleep_until(next_tick)\n\n    except KeyboardInterrupt:\n        print("\\nCtrl+C：停止发送。")\n    finally:\n        if ser is not None and ser.is_open:\n            ser.close()\n            print("\\n串口已关闭。")\n\n\ndef positive_float(text: str) -> float:\n    value = float(text)\n    if not np.isfinite(value) or value <= 0.0:\n        raise argparse.ArgumentTypeError("必须是大于 0 的有限数值")\n    return value\n\n\ndef positive_int(text: str) -> int:\n    value = int(text)\n    if value <= 0:\n        raise argparse.ArgumentTypeError("必须是正整数")\n    return value\n\n\ndef main(argv=None) -> None:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument(\n        "--live",\n        action="store_true",\n        help="真正打开串口驱动机械臂；不加此参数时默认为 dry-run",\n    )\n    parser.add_argument("--port", default=base.SERIAL_PORT, help="串口，例如 COM5")\n    parser.add_argument(\n        "--baudrate",\n        type=positive_int,\n        default=base.BAUDRATE,\n        help=f"串口波特率，默认 {base.BAUDRATE}",\n    )\n    parser.add_argument(\n        "--speed",\n        type=positive_float,\n        default=DEFAULT_SPEED_MM_S,\n        metavar="MM_S",\n        help=f"末端平移速度 mm/s，默认 {DEFAULT_SPEED_MM_S:g}",\n    )\n    args = parser.parse_args(argv)\n    run_teleop(args)\n\n\nif __name__ == "__main__":\n    main()\n',
    "teleop/README.md": '# Keyboard XY Teleoperation Prototype\n\n这是基于当前 `task8.py` / `task8_reverse.py` 的第一版实时键盘遥操作原型。\n\n## 当前范围\n\n这一版**不修改 STM32 固件**，继续使用现有 16 字节关节角控制协议。\n\n数据流：\n\n```text\nWASD\n  ↓\n平面末端速度指令\n  ↓ 50 Hz\nXY 小位移积分\n  ↓\n连续 3R IK（J2/J3/J4）\n  ↓\n关节限位 + 单帧变化保护\n  ↓\n现有 pack_frame()\n  ↓\nSTM32\n```\n\n当前固定 `J1=J5=J6=0°`，因此还是二维原型。它的目标是先验证“键盘连续输入 → 在线 IK → 实物实时跟随”这条链路，再扩展到完整 6DoF。\n\n## 控制键\n\n- `P`：从全零位进入非奇异遥操作准备姿态\n- `W / S`：末端 `+Y / -Y`\n- `A / D`：末端 `-X / +X`\n- `ESC`：立即停止发送并退出\n\n松开 WASD 后，目标位置停止变化。\n\n## 先做 dry-run\n\n在仓库根目录执行：\n\n```powershell\npython teleop/keyboard_xy.py\n```\n\n按 `P` 后可以测试键盘、IK 和软件限位，但不会打开串口。\n\n## 实机运行\n\n确认机械臂真实姿态在全零位附近，且运动范围内没有人员或障碍物，再运行：\n\n```powershell\npython teleop/keyboard_xy.py --live --port COM5\n```\n\n如果串口不是 COM5，替换成实际端口。\n\n可以降低速度：\n\n```powershell\npython teleop/keyboard_xy.py --live --port COM5 --speed 10\n```\n\n## 安全策略\n\n- 默认不驱动实机，必须显式加 `--live`\n- 启动后不会立即运动，必须手动按 `P`\n- 保留 `±90°` 软件关节限位\n- 保留 Joint 4 不进入正角机械限位区\n- 单帧关节变化超过阈值时拒绝该 Cartesian 步进\n- 不可达点直接拦截\n- `ESC` / `Ctrl+C` / 异常退出时立即停止继续发送\n- **退出时不会自动复位**\n\n## 下一步\n\n实机验证这一版后，再迭代：\n\n1. 根据真实机械臂坐标方向修正 WASD 映射；\n2. 调整准备姿态、速度、关节变化阈值；\n3. 加入 J1，将二维平面扩展为 XYZ；\n4. 建完整 6DoF FK/Jacobian；\n5. 根据需要增加 STM32 → PC 的真实关节反馈。\n',
    "teleop/__init__.py": '',
}


def run(*args, cwd=None, check=True):
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
    )


def main():
    try:
        root_result = run("git", "rev-parse", "--show-toplevel")
    except Exception:
        print("错误：当前目录不在 arm_practice 的 Git 仓库内。")
        print("请先 cd 到你的 arm_practice 仓库，再运行本脚本。")
        sys.exit(1)

    root = Path(root_result.stdout.strip())

    repo_name = root.name
    if repo_name != "arm_practice":
        print(f"警告：检测到仓库目录名为 {repo_name!r}，不是 'arm_practice'。")
        answer = input("仍要继续吗？输入 YES：").strip()
        if answer != "YES":
            sys.exit(1)

    status = run("git", "status", "--porcelain", cwd=root).stdout.strip()
    if status:
        print("错误：当前仓库存在未提交修改。为避免覆盖你的工作，本更新器拒绝继续。")
        print(status)
        print("请先提交或 stash，再重新运行。")
        sys.exit(1)

    branch_exists = (
        run(
            "git",
            "show-ref",
            "--verify",
            "--quiet",
            f"refs/heads/{BRANCH}",
            cwd=root,
            check=False,
        ).returncode
        == 0
    )

    if branch_exists:
        run("git", "switch", BRANCH, cwd=root)
    else:
        # Base the new work on the latest remote main so ChatGPT and local state agree.
        fetch = run("git", "fetch", "origin", "main", cwd=root, check=False)
        if fetch.returncode == 0:
            run("git", "switch", "-c", BRANCH, "origin/main", cwd=root)
        else:
            print("警告：无法 fetch origin/main，将从当前本地提交创建分支。")
            run("git", "switch", "-c", BRANCH, cwd=root)

    for relative_path, content in FILES.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        print(f"写入 {relative_path}")

    try:
        run(
            sys.executable,
            "-m",
            "py_compile",
            str(root / "teleop/keyboard_xy.py"),
            cwd=root,
        )
    except subprocess.CalledProcessError as exc:
        print("语法检查失败：")
        print(exc.stderr)
        sys.exit(1)

    run("git", "add", "teleop/keyboard_xy.py", "teleop/README.md", "teleop/__init__.py", cwd=root)

    staged = run("git", "diff", "--cached", "--quiet", cwd=root, check=False)
    if staged.returncode == 0:
        print("没有检测到新的文件变化，无需提交。")
    else:
        try:
            commit = run("git", "commit", "-m", COMMIT_MESSAGE, cwd=root)
            print(commit.stdout.strip())
        except subprocess.CalledProcessError as exc:
            print("文件已经写入并暂存，但 git commit 失败。")
            print("常见原因是本机尚未配置 git user.name / user.email。")
            print(exc.stderr)
            sys.exit(1)

    print()
    print("正在推送开发分支到 GitHub...")
    push = run("git", "push", "-u", "origin", BRANCH, cwd=root, check=False)
    if push.returncode != 0:
        print("警告：本地更新和 commit 已完成，但自动 push 失败。")
        print(push.stderr.strip())
        print(f"你可以稍后手动执行：git push -u origin {BRANCH}")
    else:
        print("远端分支已同步。")

    print()
    print("Update 001 完成。")
    print(f"当前分支：{BRANCH}")
    print()
    print("先进行无实机测试：")
    print("  python teleop/keyboard_xy.py")
    print()
    print("确认 dry-run 正常后，再连接机械臂：")
    print("  python teleop/keyboard_xy.py --live --port COM5")


if __name__ == "__main__":
    main()
