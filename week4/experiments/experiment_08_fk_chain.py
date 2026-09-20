"""实验 1.8：多链式机构局部旋转传递与末端位姿解算。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 兼容直接运行本文件：python week4/experiments/experiment_08_fk_chain.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from week4.common.transforms import assert_rigid_transform, build_transform, rot_x, rot_z


def build_initial_chain() -> dict[str, np.ndarray]:
    return {
        "T_0_A": build_transform(np.eye(3), [0.0, 0.0, 1.0]),
        "T_A_B": build_transform(rot_x(90.0), [0.0, 0.0, 0.5]),
        "T_B_C": build_transform(np.eye(3), [1.5, 0.0, 0.0]),
        "T_C_D": build_transform(np.eye(3), [1.2, 0.0, 0.0]),
        "T_D_E": build_transform(np.eye(3), [0.8, 0.0, 0.0]),
    }


def solve_forward_kinematics(
    base_rotation_deg: float = 45.0, wrist_rotation_deg: float = -30.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    chain = build_initial_chain()
    t_0_a_new = chain["T_0_A"] @ build_transform(rot_z(base_rotation_deg), [0, 0, 0])
    t_c_d_new = chain["T_C_D"] @ build_transform(rot_z(wrist_rotation_deg), [0, 0, 0])
    t_0_e = (
        t_0_a_new
        @ chain["T_A_B"]
        @ chain["T_B_C"]
        @ t_c_d_new
        @ chain["T_D_E"]
    )
    assert_rigid_transform(t_0_e)
    return t_0_e, t_0_e[:3, :3].copy(), t_0_e[:3, 3].copy(), chain


def play_optional_animation(chain: dict[str, np.ndarray]) -> None:
    try:
        from animator_3d import play_3d_step_animation
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "选做动画缺少 animator_3d.py；该模块未包含在当前工程资产中"
        ) from exc

    frames = [
        {"T": chain["T_0_A"], "label": "{A}"},
        {"T": chain["T_A_B"], "label": "{B}"},
        {"T": chain["T_B_C"], "label": "{C}"},
        {"T": chain["T_C_D"], "label": "{D}"},
        {"T": chain["T_D_E"], "label": "{E}"},
    ]
    play_3d_step_animation(
        reference_frames=frames,
        translation_vec_start=[0.0, 0.0, 0.0],
        rotate_frame=["A", "D"],
        rotate_R=[[rot_z(45.0)], [rot_z(-30.0)]],
        frame_rot_timing="before",
        xlim=(-1, 4),
        ylim=(-1, 4),
        zlim=(0, 4),
        frames_per_step=30,
        fps=30,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animate", action="store_true", help="运行指导书选做动画")
    args = parser.parse_args(argv)

    transform, rotation, position, chain = solve_forward_kinematics()
    np.set_printoptions(precision=4, suppress=True)
    print("末端工具坐标系 {E} 相对于世界坐标系 {0} 的总齐次变换矩阵 T_0_E:")
    print(transform)
    print("\n最终姿态旋转矩阵 R_0_E:")
    print(rotation)
    print("\n末端原点空间位置 [X, Y, Z]:")
    print(position)

    if args.animate:
        play_optional_animation(chain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
