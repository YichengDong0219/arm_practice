"""Bounded kinematics for the teleoperation runtime.

Physical command ranges confirmed on the real robot:
    J1: [-90,  90] deg
    J2: [-90,  90] deg
    J3: [-90,  90] deg
    J4: [-90,  25] deg

J5 range is currently unspecified.
J6 is handled by gripper control separately.

All IK solvers keep their internal state inside the physical J1-J4 space.
"""

from __future__ import annotations

import math
import numpy as np


LINK_1_MM = 120.0
LINK_2_MM = 120.0
LINK_3_MM = 140.0
JOINT2_ZERO_DIRECTION_RAD = np.pi / 2.0

DLS_DAMPING = 2.0
IK_POSITION_TOLERANCE_MM = 1e-6
IK_MAX_ITERATIONS = 100
MAX_INTERNAL_STEP_DEG = 5.0

HORIZONTAL_JOINT_SUM_DEG = -90.0
HORIZONTAL_TOLERANCE_DEG = 1e-9

JOINT1_MIN_DEG = -90.0
JOINT1_MAX_DEG = 90.0
JOINT2_MIN_DEG = -90.0
JOINT2_MAX_DEG = 90.0
JOINT3_MIN_DEG = -90.0
JOINT3_MAX_DEG = 90.0
JOINT4_MIN_DEG = -90.0
JOINT4_MAX_DEG = 25.0

IK_LOWER_DEG = np.array(
    [JOINT2_MIN_DEG, JOINT3_MIN_DEG, JOINT4_MIN_DEG],
    dtype=np.float64,
)
IK_UPPER_DEG = np.array(
    [JOINT2_MAX_DEG, JOINT3_MAX_DEG, JOINT4_MAX_DEG],
    dtype=np.float64,
)

XYZ_LOWER_DEG = np.array(
    [JOINT1_MIN_DEG, JOINT2_MIN_DEG, JOINT3_MIN_DEG, JOINT4_MIN_DEG],
    dtype=np.float64,
)
XYZ_UPPER_DEG = np.array(
    [JOINT1_MAX_DEG, JOINT2_MAX_DEG, JOINT3_MAX_DEG, JOINT4_MAX_DEG],
    dtype=np.float64,
)


class IKError(RuntimeError):
    """Raised when a target has no valid constrained IK solution."""


def _inside(value: float, low: float, high: float, tol: float = 1e-9) -> bool:
    return low - tol <= float(value) <= high + tol


def joint1_in_limits(value: float) -> bool:
    return _inside(value, JOINT1_MIN_DEG, JOINT1_MAX_DEG)


def joint2_in_limits(value: float) -> bool:
    return _inside(value, JOINT2_MIN_DEG, JOINT2_MAX_DEG)


def joint3_in_limits(value: float) -> bool:
    return _inside(value, JOINT3_MIN_DEG, JOINT3_MAX_DEG)


def joint4_in_limits(value: float) -> bool:
    return _inside(value, JOINT4_MIN_DEG, JOINT4_MAX_DEG)


def ik_joints_in_limits(joints_deg) -> bool:
    q = np.asarray(joints_deg, dtype=np.float64)
    return (
        joint2_in_limits(q[1])
        and joint3_in_limits(q[2])
        and joint4_in_limits(q[3])
    )


def arm_joints_in_limits(joints_deg) -> bool:
    q = np.asarray(joints_deg, dtype=np.float64)
    return joint1_in_limits(q[0]) and ik_joints_in_limits(q)


def horizontal_sum_deg(joints_deg) -> float:
    q = np.asarray(joints_deg, dtype=np.float64)
    if q.shape != (6,):
        raise ValueError("joints_deg must contain exactly 6 values")
    return float(np.sum(q[1:4]))


def is_horizontal_pose(
    joints_deg,
    tolerance_deg: float = HORIZONTAL_TOLERANCE_DEG,
) -> bool:
    return abs(
        horizontal_sum_deg(joints_deg) - HORIZONTAL_JOINT_SUM_DEG
    ) <= tolerance_deg


def forward_radial_z(joints_deg) -> np.ndarray:
    q = np.asarray(joints_deg, dtype=np.float64)
    if q.shape != (6,):
        raise ValueError("joints_deg must contain exactly 6 values")

    q2 = np.deg2rad(q[1]) + JOINT2_ZERO_DIRECTION_RAD
    q3 = np.deg2rad(q[2])
    q4 = np.deg2rad(q[3])

    q23 = q2 + q3
    q234 = q23 + q4

    r = (
        LINK_1_MM * np.cos(q2)
        + LINK_2_MM * np.cos(q23)
        + LINK_3_MM * np.cos(q234)
    )
    z = (
        LINK_1_MM * np.sin(q2)
        + LINK_2_MM * np.sin(q23)
        + LINK_3_MM * np.sin(q234)
    )

    return np.array([r, z], dtype=np.float64)


def radial_z_jacobian(joints_deg) -> np.ndarray:
    q = np.asarray(joints_deg, dtype=np.float64)
    if q.shape != (6,):
        raise ValueError("joints_deg must contain exactly 6 values")

    q2 = np.deg2rad(q[1]) + JOINT2_ZERO_DIRECTION_RAD
    q3 = np.deg2rad(q[2])
    q4 = np.deg2rad(q[3])

    q23 = q2 + q3
    q234 = q23 + q4

    r = (
        LINK_1_MM * np.cos(q2)
        + LINK_2_MM * np.cos(q23)
        + LINK_3_MM * np.cos(q234)
    )
    z = (
        LINK_1_MM * np.sin(q2)
        + LINK_2_MM * np.sin(q23)
        + LINK_3_MM * np.sin(q234)
    )

    return np.array(
        [
            [
                -z,
                -LINK_2_MM * np.sin(q23)
                - LINK_3_MM * np.sin(q234),
                -LINK_3_MM * np.sin(q234),
            ],
            [
                r,
                LINK_2_MM * np.cos(q23)
                + LINK_3_MM * np.cos(q234),
                LINK_3_MM * np.cos(q234),
            ],
        ],
        dtype=np.float64,
    )


def forward_xyz(joints_deg, j1_sign: float = 1.0) -> np.ndarray:
    q = np.asarray(joints_deg, dtype=np.float64)
    if q.shape != (6,):
        raise ValueError("joints_deg must contain exactly 6 values")

    r, z = forward_radial_z(q)
    q1_world = float(j1_sign) * np.deg2rad(q[0])

    return np.array(
        [
            r * np.cos(q1_world),
            r * np.sin(q1_world),
            z,
        ],
        dtype=np.float64,
    )


def position_jacobian(joints_deg, j1_sign: float = 1.0) -> np.ndarray:
    q = np.asarray(joints_deg, dtype=np.float64)
    r, _ = forward_radial_z(q)
    rz_j = radial_z_jacobian(q)

    sign = float(j1_sign)
    q1_world = sign * np.deg2rad(q[0])
    c1 = np.cos(q1_world)
    s1 = np.sin(q1_world)

    x = r * c1
    y = r * s1
    dr = rz_j[0]
    dz = rz_j[1]

    return np.array(
        [
            [-sign * y, c1 * dr[0], c1 * dr[1], c1 * dr[2]],
            [ sign * x, s1 * dr[0], s1 * dr[1], s1 * dr[2]],
            [0.0, dz[0], dz[1], dz[2]],
        ],
        dtype=np.float64,
    )


def _dls_step(jacobian: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
    system = (
        jacobian @ jacobian.T
        + float(damping) ** 2
        * np.eye(jacobian.shape[0], dtype=np.float64)
    )
    return jacobian.T @ np.linalg.solve(system, error)


def _limit_step(delta_q_rad: np.ndarray) -> np.ndarray:
    maximum = float(np.max(np.abs(delta_q_rad)))
    limit = np.deg2rad(MAX_INTERNAL_STEP_DEG)

    if maximum > limit:
        return delta_q_rad * (limit / maximum)

    return delta_q_rad


def solve_radial_z_with_joint_sum(
    target_radial_z_mm,
    target_joint_sum_deg: float,
    previous_joints_deg,
    *,
    tolerance_mm: float = IK_POSITION_TOLERANCE_MM,
) -> np.ndarray:
    """Analytic r/z + fixed pitch IK, filtered by J2/J3/J4 limits."""
    target = np.asarray(target_radial_z_mm, dtype=np.float64)
    previous = np.asarray(previous_joints_deg, dtype=np.float64)

    if target.shape != (2,):
        raise ValueError("target_radial_z_mm must contain exactly 2 values")
    if previous.shape != (6,):
        raise ValueError("previous_joints_deg must contain exactly 6 values")
    if not ik_joints_in_limits(previous):
        raise IKError(
            f"current J2-J4 pose is outside physical limits: "
            f"{previous[1:4].tolist()}"
        )

    target_r = float(target[0])
    target_z = float(target[1])

    alpha = math.radians(float(target_joint_sum_deg) + 90.0)

    wrist_r = target_r - LINK_3_MM * math.cos(alpha)
    wrist_z = target_z - LINK_3_MM * math.sin(alpha)

    l1 = LINK_1_MM
    l2 = LINK_2_MM

    cos_q3 = (
        wrist_r * wrist_r
        + wrist_z * wrist_z
        - l1 * l1
        - l2 * l2
    ) / (2.0 * l1 * l2)

    if cos_q3 < -1.0 - 1e-10 or cos_q3 > 1.0 + 1e-10:
        raise IKError(
            f"target is geometrically unreachable: "
            f"r={target_r:.3f}, z={target_z:.3f}"
        )

    cos_q3 = float(np.clip(cos_q3, -1.0, 1.0))
    q3_abs = math.acos(cos_q3)

    geometric = []
    valid = []

    for q3_model in (-q3_abs, q3_abs):
        q2_model = math.atan2(wrist_z, wrist_r) - math.atan2(
            l2 * math.sin(q3_model),
            l1 + l2 * math.cos(q3_model),
        )

        q2_deg = math.degrees(q2_model) - 90.0
        q3_deg = math.degrees(q3_model)
        q4_deg = float(target_joint_sum_deg) - q2_deg - q3_deg

        candidate = previous.copy()
        candidate[1] = q2_deg
        candidate[2] = q3_deg
        candidate[3] = q4_deg

        error = float(
            np.linalg.norm(forward_radial_z(candidate) - target)
        )

        if error <= max(tolerance_mm * 10.0, 1e-7):
            geometric.append(candidate)
            if ik_joints_in_limits(candidate):
                valid.append(candidate)

    if not valid:
        if geometric:
            descriptions = [
                np.round(c[1:4], 3).tolist()
                for c in geometric
            ]
            raise IKError(
                "target has mathematical IK solutions, but all violate "
                f"physical J2-J4 limits; candidates={descriptions}"
            )

        raise IKError("no valid fixed-pitch IK candidate")

    return min(
        valid,
        key=lambda c: float(np.sum((c[1:4] - previous[1:4]) ** 2)),
    )


def solve_horizontal_radial_z(
    target_radial_z_mm,
    previous_joints_deg,
    *,
    tolerance_mm: float = IK_POSITION_TOLERANCE_MM,
) -> np.ndarray:
    return solve_radial_z_with_joint_sum(
        target_radial_z_mm,
        HORIZONTAL_JOINT_SUM_DEG,
        previous_joints_deg,
        tolerance_mm=tolerance_mm,
    )


def solve_radial_z(
    target_radial_z_mm,
    previous_joints_deg,
    *,
    damping: float = DLS_DAMPING,
    tolerance_mm: float = IK_POSITION_TOLERANCE_MM,
    max_iterations: int = IK_MAX_ITERATIONS,
) -> np.ndarray:
    """FREE-mode projected DLS with hard J2/J3/J4 bounds."""
    target = np.asarray(target_radial_z_mm, dtype=np.float64)
    previous = np.asarray(previous_joints_deg, dtype=np.float64)

    if target.shape != (2,):
        raise ValueError("target_radial_z_mm must contain exactly 2 values")
    if previous.shape != (6,):
        raise ValueError("previous_joints_deg must contain exactly 6 values")
    if not ik_joints_in_limits(previous):
        raise IKError(
            f"current J2-J4 pose is outside physical limits: "
            f"{previous[1:4].tolist()}"
        )

    result = previous.copy()
    last_error = None
    stagnant = 0

    for _ in range(max_iterations):
        error = target - forward_radial_z(result)
        norm = float(np.linalg.norm(error))

        if norm <= tolerance_mm:
            return result

        dq = _dls_step(
            radial_z_jacobian(result),
            error,
            damping,
        )
        dq = _limit_step(dq)

        result[1:4] = np.clip(
            result[1:4] + np.rad2deg(dq),
            IK_LOWER_DEG,
            IK_UPPER_DEG,
        )

        if last_error is not None and abs(last_error - norm) < 1e-10:
            stagnant += 1
        else:
            stagnant = 0

        last_error = norm

        if stagnant >= 12:
            break

    final_error = float(
        np.linalg.norm(target - forward_radial_z(result))
    )

    active = []
    names = ("J2", "J3", "J4")

    for name, value, low, high in zip(
        names,
        result[1:4],
        IK_LOWER_DEG,
        IK_UPPER_DEG,
    ):
        if abs(value - low) < 1e-6:
            active.append(f"{name}=MIN({low:.1f}°)")
        elif abs(value - high) < 1e-6:
            active.append(f"{name}=MAX({high:.1f}°)")

    suffix = f"; active limits={active}" if active else ""

    raise IKError(
        "target is not reachable inside physical J2-J4 limits; "
        f"remaining error={final_error:.6f} mm{suffix}"
    )


def solve_position(
    target_xyz_mm,
    previous_joints_deg,
    *,
    j1_sign: float = 1.0,
    damping: float = DLS_DAMPING,
    tolerance_mm: float = IK_POSITION_TOLERANCE_MM,
    max_iterations: int = IK_MAX_ITERATIONS,
) -> np.ndarray:
    """Bounded XYZ DLS with J1-J4 hard limits."""
    target = np.asarray(target_xyz_mm, dtype=np.float64)
    previous = np.asarray(previous_joints_deg, dtype=np.float64)

    if target.shape != (3,):
        raise ValueError("target_xyz_mm must contain exactly 3 values")
    if previous.shape != (6,):
        raise ValueError("previous_joints_deg must contain exactly 6 values")
    if not arm_joints_in_limits(previous):
        raise IKError(
            f"current J1-J4 pose is outside physical limits: "
            f"{previous[:4].tolist()}"
        )

    result = previous.copy()

    for _ in range(max_iterations):
        error = target - forward_xyz(result, j1_sign=j1_sign)

        if float(np.linalg.norm(error)) <= tolerance_mm:
            return result

        dq = _dls_step(
            position_jacobian(result, j1_sign=j1_sign),
            error,
            damping,
        )
        dq = _limit_step(dq)

        result[:4] = np.clip(
            result[:4] + np.rad2deg(dq),
            XYZ_LOWER_DEG,
            XYZ_UPPER_DEG,
        )

    final_error = float(
        np.linalg.norm(
            target - forward_xyz(result, j1_sign=j1_sign)
        )
    )

    raise IKError(
        "XYZ target is not reachable inside physical J1-J4 limits; "
        f"remaining error={final_error:.6f} mm"
    )
