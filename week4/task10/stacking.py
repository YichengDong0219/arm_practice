"""实验10扩展：固定供料、固定堆叠点的三层物块堆叠。"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from week4.common.motion import (
    DryRunSerial,
    MotionStep,
    interpolate_pose,
    open_serial,
    run_sequence,
    validate_pose,
)
from week4.common.protocol import pack_frame, validate_frame


LINK_1_MM = 120.0
LINK_2_MM = 120.0
LINK_3_MM = 140.0
JOINT_2_ZERO_OFFSET_DEG = 90.0
JOINT_LIMIT_DEG = 90.0

DEFAULT_BLOCK_HEIGHT_MM = 30.0
DEFAULT_COUNT = 3
DEFAULT_CLEARANCE_MM = 45.0
STACK_INWARD_OFFSET_MM = 25.0
PICKUP_OUTWARD_OFFSET_MM = 40.0
HORIZONTAL_TOOL_ANGLE_DEG = 0.0
VERTICAL_SEGMENTS = 10

GRIPPER_OPEN_DEG = 0.0
GRIPPER_CLOSED_DEG = -40.0
HOME_POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
PICKUP_POSE = (30.0, -30.0, -80.0, -15.0, 0.0, 0.0)
# 以下姿态直接来自 main.py 当前未注释的第一层放置动作。
BASE_PLACE_POSE = (0.0, -90.0, 0.0, -15.0, 0.0, -40.0)

MODEL_XY_TOLERANCE_MM = 6.0
MODEL_HEIGHT_TOLERANCE_MM = 6.0
MIN_HOVER_RISE_MM = 5.0
MAX_LAYER_JOINT_CHANGE_DEG = 70.0
MAX_INTERPOLATION_STEP_DEG = 3.0

def _wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _with_gripper(pose, gripper_deg: float) -> tuple[float, ...]:
    result = list(validate_pose(pose))
    result[5] = float(gripper_deg)
    return validate_pose(result)


def forward_tool_pose(pose_deg) -> dict[str, float]:
    """按J1+平面3R模型回算末端世界坐标和工具俯仰角。"""
    pose = validate_pose(pose_deg)
    yaw = math.radians(pose[0])
    q2 = math.radians(pose[1] + JOINT_2_ZERO_OFFSET_DEG)
    q3 = math.radians(pose[2])
    q4 = math.radians(pose[3])
    alpha = q2 + q3 + q4

    radial = (
        LINK_1_MM * math.cos(q2)
        + LINK_2_MM * math.cos(q2 + q3)
        + LINK_3_MM * math.cos(alpha)
    )
    z_mm = (
        LINK_1_MM * math.sin(q2)
        + LINK_2_MM * math.sin(q2 + q3)
        + LINK_3_MM * math.sin(alpha)
    )
    return {
        "x_mm": radial * math.cos(yaw),
        "y_mm": radial * math.sin(yaw),
        "radial_mm": radial,
        "z_mm": z_mm,
        "yaw_deg": pose[0],
        "tool_angle_deg": math.degrees(alpha),
    }


def _ik_candidates(
    radial_mm: float,
    z_mm: float,
    tool_angle_deg: float,
    *,
    joint1_deg: float,
    joint5_deg: float,
    gripper_deg: float,
) -> list[tuple[float, ...]]:
    alpha = math.radians(tool_angle_deg)
    wrist_r = radial_mm - LINK_3_MM * math.cos(alpha)
    wrist_z = z_mm - LINK_3_MM * math.sin(alpha)
    cosine_q3 = (
        wrist_r * wrist_r
        + wrist_z * wrist_z
        - LINK_1_MM * LINK_1_MM
        - LINK_2_MM * LINK_2_MM
    ) / (2.0 * LINK_1_MM * LINK_2_MM)
    if cosine_q3 < -1.0 - 1e-9 or cosine_q3 > 1.0 + 1e-9:
        return []
    cosine_q3 = max(-1.0, min(1.0, cosine_q3))

    candidates = []
    for elbow_sign in (-1.0, 1.0):
        q3 = elbow_sign * math.acos(cosine_q3)
        q2 = math.atan2(wrist_z, wrist_r) - math.atan2(
            LINK_2_MM * math.sin(q3),
            LINK_1_MM + LINK_2_MM * math.cos(q3),
        )
        q4 = math.radians(tool_angle_deg) - q2 - q3
        pose = (
            float(joint1_deg),
            math.degrees(q2) - JOINT_2_ZERO_OFFSET_DEG,
            math.degrees(q3),
            _wrap_deg(math.degrees(q4)),
            float(joint5_deg),
            float(gripper_deg),
        )
        if all(math.isfinite(value) and abs(value) <= JOINT_LIMIT_DEG for value in pose):
            candidates.append(pose)
    return candidates


def _candidate_score(candidate, previous_pose) -> float:
    previous = validate_pose(previous_pose)
    motion_cost = sum((candidate[index] - previous[index]) ** 2 for index in (1, 2, 3))
    joint_cost = 0.2 * max(abs(candidate[index]) for index in (1, 2, 3))
    singularity_cost = 5.0 * max(0.0, 15.0 - abs(candidate[2]))
    elbow_cost = 1000.0 if candidate[2] > 0.0 else 0.0
    return motion_cost + joint_cost + singularity_cost + elbow_cost


def solve_fixed_orientation(
    radial_mm: float,
    z_mm: float,
    tool_angle_deg: float,
    previous_pose,
    *,
    joint1_deg: float,
    joint5_deg: float,
    gripper_deg: float,
) -> tuple[float, ...]:
    candidates = _ik_candidates(
        radial_mm,
        z_mm,
        tool_angle_deg,
        joint1_deg=joint1_deg,
        joint5_deg=joint5_deg,
        gripper_deg=gripper_deg,
    )
    if not candidates:
        raise ValueError(
            f"目标不可达: radial={radial_mm:.3f} mm, z={z_mm:.3f} mm, "
            f"tool={tool_angle_deg:.3f}°"
        )
    return min(candidates, key=lambda pose: _candidate_score(pose, previous_pose))


def solve_free_orientation(
    radial_mm: float,
    z_mm: float,
    previous_pose,
    *,
    joint1_deg: float,
    joint5_deg: float,
    gripper_deg: float,
) -> tuple[float, ...]:
    previous_tool_angle = forward_tool_pose(previous_pose)["tool_angle_deg"]
    best_pose = None
    best_score = math.inf
    for half_degree in range(-180, 181):
        tool_angle_deg = half_degree * 0.5
        for candidate in _ik_candidates(
            radial_mm,
            z_mm,
            tool_angle_deg,
            joint1_deg=joint1_deg,
            joint5_deg=joint5_deg,
            gripper_deg=gripper_deg,
        ):
            orientation_cost = 2.0 * abs(_wrap_deg(tool_angle_deg - previous_tool_angle))
            score = _candidate_score(candidate, previous_pose) + orientation_cost
            if score < best_score:
                best_pose = candidate
                best_score = score
    if best_pose is None:
        raise ValueError(f"目标不可达: radial={radial_mm:.3f} mm, z={z_mm:.3f} mm")
    return best_pose


def _choose_stack_release_poses(
    radial_mm: float,
    base_z_mm: float,
    block_height_mm: float,
    count: int,
    reference_pose,
) -> tuple[float, list[tuple[float, ...]]]:
    reference = validate_pose(reference_pose)
    reference_alpha = forward_tool_pose(reference)["tool_angle_deg"]
    best = None

    for half_degree in range(-180, 181):
        alpha_deg = half_degree * 0.5
        poses = []
        previous = reference
        feasible = True
        for layer in range(count):
            try:
                pose = solve_fixed_orientation(
                    radial_mm,
                    base_z_mm + layer * block_height_mm,
                    alpha_deg,
                    previous,
                    joint1_deg=reference[0],
                    joint5_deg=reference[4],
                    gripper_deg=GRIPPER_CLOSED_DEG,
                )
            except ValueError:
                feasible = False
                break
            poses.append(pose)
            previous = pose
        if not feasible:
            continue

        maximum_angle = max(abs(value) for pose in poses for value in pose[1:4])
        transition = sum(
            max(abs(poses[i][joint] - poses[i - 1][joint]) for joint in (1, 2, 3))
            for i in range(1, len(poses))
        )
        singularity = sum(max(0.0, 15.0 - abs(pose[2])) for pose in poses)
        score = (
            maximum_angle
            + 0.1 * transition
            + 0.2 * abs(_wrap_deg(alpha_deg - reference_alpha))
            + 2.0 * singularity
        )
        if best is None or score < best[0]:
            best = (score, alpha_deg, poses)

    if best is None:
        raise ValueError("无法为全部堆叠层找到连续、限位内的IK解；请移动堆叠点")
    return best[1], best[2]


def generate_stacking_plan(
    block_height_mm: float,
    count: int,
    clearance_mm: float = DEFAULT_CLEARANCE_MM,
) -> dict:
    if block_height_mm <= 0.0:
        raise ValueError("物块高度必须大于0")
    if count <= 0:
        raise ValueError("堆叠数量必须为正整数")
    if clearance_mm <= 0.0:
        raise ValueError("安全高度必须大于0")

    pickup_source_model = forward_tool_pose(PICKUP_POSE)
    pickup_radial = pickup_source_model["radial_mm"] + PICKUP_OUTWARD_OFFSET_MM
    pickup_horizontal = solve_fixed_orientation(
        pickup_radial,
        pickup_source_model["z_mm"],
        HORIZONTAL_TOOL_ANGLE_DEG,
        PICKUP_POSE,
        joint1_deg=PICKUP_POSE[0],
        joint5_deg=PICKUP_POSE[4],
        gripper_deg=GRIPPER_OPEN_DEG,
    )
    pickup_hover = solve_fixed_orientation(
        pickup_radial,
        pickup_source_model["z_mm"] + clearance_mm,
        HORIZONTAL_TOOL_ANGLE_DEG,
        pickup_horizontal,
        joint1_deg=PICKUP_POSE[0],
        joint5_deg=PICKUP_POSE[4],
        gripper_deg=GRIPPER_OPEN_DEG,
    )

    place_model = forward_tool_pose(BASE_PLACE_POSE)
    # 原点位在夹爪严格水平时会令 J2 约为 -91.36°。向基座微调 5 mm，
    # 为水平放置和高位等待留出 ±90° 限位余量。
    stack_radial = place_model["radial_mm"] - STACK_INWARD_OFFSET_MM

    # 三层保持相同 x、y，仅增加 z；所有点都令 q2+q3+q4=0°，
    # J4 因而只负责抵消 J2/J3 的姿态变化，使夹爪保持水平。
    release_poses = []
    previous_release = BASE_PLACE_POSE
    for layer in range(count):
        previous_release = solve_fixed_orientation(
            stack_radial,
            place_model["z_mm"] + layer * block_height_mm,
            HORIZONTAL_TOOL_ANGLE_DEG,
            previous_release,
            joint1_deg=BASE_PLACE_POSE[0],
            joint5_deg=BASE_PLACE_POSE[4],
            gripper_deg=GRIPPER_CLOSED_DEG,
        )
        release_poses.append(previous_release)

    hover_poses = []
    for layer, release_pose in enumerate(release_poses):
        hover_poses.append(
            solve_fixed_orientation(
                stack_radial,
                place_model["z_mm"] + layer * block_height_mm + clearance_mm,
                HORIZONTAL_TOOL_ANGLE_DEG,
                release_pose,
                joint1_deg=release_pose[0],
                joint5_deg=release_pose[4],
                gripper_deg=GRIPPER_CLOSED_DEG,
            )
        )

    safe_transfer = _with_gripper(pickup_hover, GRIPPER_CLOSED_DEG)

    return {
        "version": 1,
        "block_height_mm": float(block_height_mm),
        "count": int(count),
        "clearance_mm": float(clearance_mm),
        "gripper_open_deg": GRIPPER_OPEN_DEG,
        "gripper_closed_deg": GRIPPER_CLOSED_DEG,
        "home_pose": list(HOME_POSE),
        "pickup_pose": list(pickup_horizontal),
        "pickup_hover_pose": list(pickup_hover),
        "safe_transfer_pose": list(safe_transfer),
        "stack_release_poses": [list(pose) for pose in release_poses],
        "stack_hover_poses": [list(pose) for pose in hover_poses],
        "model": {
            "link_lengths_mm": [LINK_1_MM, LINK_2_MM, LINK_3_MM],
            "joint2_zero_offset_deg": JOINT_2_ZERO_OFFSET_DEG,
            "stack_inward_offset_mm": STACK_INWARD_OFFSET_MM,
            "pickup_outward_offset_mm": PICKUP_OUTWARD_OFFSET_MM,
            "stack_radial_mm": stack_radial,
            "stack_base_z_mm": place_model["z_mm"],
            "stack_tool_angle_deg": HORIZONTAL_TOOL_ANGLE_DEG,
        },
    }


def validate_plan(calibration: dict, block_height_mm: float, count: int) -> None:
    required = {
        "version",
        "block_height_mm",
        "count",
        "home_pose",
        "pickup_pose",
        "pickup_hover_pose",
        "safe_transfer_pose",
        "stack_release_poses",
        "stack_hover_poses",
        "gripper_open_deg",
        "gripper_closed_deg",
    }
    missing = sorted(required - calibration.keys())
    if missing:
        raise ValueError(f"轨迹计划缺少字段: {', '.join(missing)}")
    if not math.isclose(float(calibration["block_height_mm"]), block_height_mm, abs_tol=1e-6):
        raise ValueError("请求的物块高度与轨迹计划不一致")
    if count > int(calibration["count"]):
        raise ValueError("请求层数超过轨迹计划覆盖范围")

    fixed_poses = [
        calibration["home_pose"],
        calibration["pickup_pose"],
        calibration["pickup_hover_pose"],
        calibration["safe_transfer_pose"],
    ]
    releases = calibration["stack_release_poses"][:count]
    hovers = calibration["stack_hover_poses"][:count]
    if len(releases) != count or len(hovers) != count:
        raise ValueError("轨迹计划中的放置位或撤离位数量不足")
    for pose in fixed_poses + releases + hovers:
        validate_pose(pose)

    pickup_model = forward_tool_pose(calibration["pickup_pose"])
    pickup_hover_model = forward_tool_pose(calibration["pickup_hover_pose"])
    pickup_horizontal_error = math.hypot(
        pickup_model["x_mm"] - pickup_hover_model["x_mm"],
        pickup_model["y_mm"] - pickup_hover_model["y_mm"],
    )
    if pickup_horizontal_error > MODEL_XY_TOLERANCE_MM:
        raise ValueError("抓取点与抓取高位不在同一竖直线上")
    if pickup_hover_model["z_mm"] < pickup_model["z_mm"] + MIN_HOVER_RISE_MM:
        raise ValueError("抓取高位高度不足")
    for label, pose in (
        ("抓取点", calibration["pickup_pose"]),
        ("抓取高位", calibration["pickup_hover_pose"]),
    ):
        if abs(sum(float(value) for value in pose[1:4]) + 90.0) > 1e-6:
            raise ValueError(f"{label}未满足 J2+J3+J4=-90°")

    release_models = [forward_tool_pose(pose) for pose in releases]
    base = release_models[0]
    for layer, model in enumerate(release_models):
        horizontal_error = math.hypot(model["x_mm"] - base["x_mm"], model["y_mm"] - base["y_mm"])
        expected_z = base["z_mm"] + layer * block_height_mm
        if horizontal_error > MODEL_XY_TOLERANCE_MM:
            raise ValueError(f"第{layer + 1}层模型水平偏差过大: {horizontal_error:.3f} mm")
        if abs(model["z_mm"] - expected_z) > MODEL_HEIGHT_TOLERANCE_MM:
            raise ValueError(f"第{layer + 1}层模型高度偏差过大")
        hover_model = forward_tool_pose(hovers[layer])
        if hover_model["z_mm"] < model["z_mm"] + MIN_HOVER_RISE_MM:
            raise ValueError(f"第{layer + 1}层撤离位高度不足")
        if abs(sum(float(value) for value in releases[layer][1:4]) + 90.0) > 1e-6:
            raise ValueError(f"第{layer + 1}层放置位未保持夹爪水平")
        if abs(sum(float(value) for value in hovers[layer][1:4]) + 90.0) > 1e-6:
            raise ValueError(f"第{layer + 1}层高位未保持夹爪水平")
        if layer:
            change = max(abs(releases[layer][j] - releases[layer - 1][j]) for j in (1, 2, 3))
            if change > MAX_LAYER_JOINT_CHANGE_DEG:
                raise ValueError(f"第{layer + 1}层与上一层关节变化过大")


def _build_vertical_steps(
    start_pose,
    end_pose,
    *,
    gripper_deg: float,
    label: str,
    duration_s: float = 1.5,
) -> list[MotionStep]:
    """以笛卡尔高度点逼近竖直运动，并在每一点保持夹爪水平。"""
    start = _with_gripper(start_pose, gripper_deg)
    end_model = forward_tool_pose(end_pose)
    start_model = forward_tool_pose(start)
    horizontal_error = math.hypot(
        end_model["x_mm"] - start_model["x_mm"],
        end_model["y_mm"] - start_model["y_mm"],
    )
    if horizontal_error > 1e-6:
        raise ValueError(f"{label}的起止点不在同一竖直线上")

    result = []
    previous = start
    for segment in range(1, VERTICAL_SEGMENTS + 1):
        ratio = segment / VERTICAL_SEGMENTS
        z_mm = start_model["z_mm"] + ratio * (end_model["z_mm"] - start_model["z_mm"])
        target = solve_fixed_orientation(
            start_model["radial_mm"],
            z_mm,
            HORIZONTAL_TOOL_ANGLE_DEG,
            previous,
            joint1_deg=start[0],
            joint5_deg=start[4],
            gripper_deg=gripper_deg,
        )
        result.append(
            MotionStep(
                f"{label} {segment}/{VERTICAL_SEGMENTS}",
                target,
                duration_s / VERTICAL_SEGMENTS,
                3,
            )
        )
        previous = target
    return result


def build_cycle_steps(calibration: dict, layer: int) -> list[MotionStep]:
    opened = float(calibration["gripper_open_deg"])
    closed = float(calibration["gripper_closed_deg"])
    pickup = calibration["pickup_pose"]
    stack_hover = calibration["stack_hover_poses"][layer]
    stack_release = calibration["stack_release_poses"][layer]
    home = calibration["home_pose"]

    steps = [
        MotionStep("动作1：水平夹爪移动到固定取料点", _with_gripper(pickup, opened), 2.0, 40),
        MotionStep("动作2：夹住物块", _with_gripper(pickup, closed), 1.0, 20, 0.7),
        MotionStep(f"动作3：移动到第{layer + 1}层正上方", _with_gripper(stack_hover, closed), 2.0, 40),
    ]
    steps.extend(
        _build_vertical_steps(
            stack_hover,
            stack_release,
            gripper_deg=closed,
            label=f"动作4：水平夹持并竖直下降到第{layer + 1}层",
        )
    )
    steps.extend([
        MotionStep("动作5：松开物块", _with_gripper(stack_release, opened), 1.0, 20, 0.7),
    ])
    steps.extend(
        _build_vertical_steps(
            stack_release,
            stack_hover,
            gripper_deg=opened,
            label="动作6：保持水平并竖直抬起",
        )
    )
    steps.extend([
        MotionStep("动作7：返回全零位", _with_gripper(home, opened), 2.5, 50),
    ])
    return steps


def preflight(calibration: dict, block_height_mm: float, count: int, start_layer: int) -> None:
    validate_plan(calibration, block_height_mm, count)
    if start_layer < 0 or start_layer >= count:
        raise ValueError("--start-layer 必须满足 0 <= start-layer < count")

    current = validate_pose(calibration["home_pose"])
    for layer in range(start_layer, count):
        for step in build_cycle_steps(calibration, layer):
            target = validate_pose(step.target_deg)
            maximum_increment = max(abs(target[j] - current[j]) / step.steps for j in range(6))
            if maximum_increment > MAX_INTERPOLATION_STEP_DEG:
                raise ValueError(f"动作“{step.name}”单帧关节变化过大")
            for pose in interpolate_pose(current, target, step.steps):
                frame = pack_frame(math.radians(value) for value in pose)
                validate_frame(frame)
            current = target


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("必须是大于0的有限数值")
    return result


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM5", help="实际串口号，例如 COM5")
    parser.add_argument("--block-height-mm", type=_positive_float, default=DEFAULT_BLOCK_HEIGHT_MM)
    parser.add_argument("--count", type=_positive_int, default=DEFAULT_COUNT)
    parser.add_argument("--start-layer", type=int, default=0, help="从0开始的恢复层号")
    parser.add_argument("--clearance-mm", type=_positive_float, default=DEFAULT_CLEARANCE_MM)
    parser.add_argument("--execute", action="store_true", help="允许连接串口并实际运动")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    nominal = generate_stacking_plan(args.block_height_mm, args.count, args.clearance_mm)

    # 本模式明确忽略旧标定结果：所有放置点均由 main.py 的有效角度和
    # 120/120/140 mm 模型现场计算。
    calibration = nominal
    print("轨迹来源: main.py 有效角度 + 运动学计算（忽略标定文件）")
    for layer, pose in enumerate(calibration["stack_release_poses"]):
        model = forward_tool_pose(pose)
        print(
            f"第{layer + 1}层: xyz=({model['x_mm']:.3f}, {model['y_mm']:.3f}, "
            f"{model['z_mm']:.3f}) mm, J1~J4={tuple(round(value, 3) for value in pose[:4])}"
        )

    preflight(calibration, args.block_height_mm, args.count, args.start_layer)

    if not args.execute:
        link = DryRunSerial()
        current = validate_pose(calibration["home_pose"])
        for layer in range(args.start_layer, args.count):
            print(f"\n=== 预演第 {layer + 1}/{args.count} 块 ===")
            current = run_sequence(
                link,
                build_cycle_steps(calibration, layer),
                initial_pose_deg=current,
                realtime=False,
            )
        print(f"\n预演完成：共生成 {len(link.frames)} 帧；未打开串口。")
        return 0

    link = open_serial(args.port)
    try:
        print(f"已连接 {args.port}，等待 STM32 初始化 1.5 秒……")
        time.sleep(1.5)
        current = validate_pose(calibration["home_pose"])
        for layer in range(args.start_layer, args.count):
            answer = input(
                f"\n请放入第 {layer + 1} 块物料；按回车开始，输入 q 结束: "
            ).strip().lower()
            if answer == "q":
                print("已在新一轮开始前停止。")
                break
            current = run_sequence(
                link,
                build_cycle_steps(calibration, layer),
                initial_pose_deg=current,
                realtime=True,
            )
            print(f"第 {layer + 1} 块已释放并完成撤离。")
        print("堆叠流程正常结束，机械臂位于等待零位。")
    except KeyboardInterrupt:
        print("用户中断：立即停止发送，不在未知姿态下强制复位。")
        return 130
    finally:
        if link.is_open:
            link.close()
            print("串口已释放。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
