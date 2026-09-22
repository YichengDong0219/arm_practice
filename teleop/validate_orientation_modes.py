"""Validate switchable FREE/HORIZONTAL orientation kinematics."""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teleop import position_ik
from teleop.keyboard_position_ik import PREP_POSE_DEG, plan_orientation_alignment


def main():
    start = PREP_POSE_DEG.copy()
    rz0 = position_ik.forward_radial_z(start)

    # Horizontal mode remains exact.
    h = position_ik.solve_horizontal_radial_z(
        rz0 + np.array([10.0, 10.0]),
        start,
    )
    assert abs(position_ik.horizontal_sum_deg(h) + 90.0) < 1e-9

    # FREE mode reaches r,z without enforcing the sum.
    free = position_ik.solve_radial_z(
        position_ik.forward_radial_z(h) + np.array([-10.0, -5.0]),
        h,
    )
    free_target = position_ik.forward_radial_z(h) + np.array([-10.0, -5.0])
    assert np.linalg.norm(
        position_ik.forward_radial_z(free) - free_target
    ) < 1e-5
    assert free[0] == h[0]

    # Generic fixed-sum IK exactly preserves r,z for several tool pitches.
    fixed_rz = position_ik.forward_radial_z(start)
    previous = start.copy()
    for requested_sum in (-85.0, -80.0, -85.0, -90.0):
        previous = position_ik.solve_radial_z_with_joint_sum(
            fixed_rz,
            requested_sum,
            previous,
        )
        assert abs(
            position_ik.horizontal_sum_deg(previous) - requested_sum
        ) < 1e-9
        assert np.linalg.norm(
            position_ik.forward_radial_z(previous) - fixed_rz
        ) < 1e-5

    # Runtime alignment planner ends exactly horizontal and holds r,z.
    planned = plan_orientation_alignment(free)
    assert planned
    final_pose = planned[-1]
    assert abs(position_ik.horizontal_sum_deg(final_pose) + 90.0) < 1e-9
    assert np.linalg.norm(
        position_ik.forward_radial_z(final_pose)
        - position_ik.forward_radial_z(free)
    ) < 1e-5

    print("PASS: FREE/HORIZONTAL mode kinematics")
    print("PASS: FREE -> HORIZONTAL alignment preserves r,z")


if __name__ == "__main__":
    main()
