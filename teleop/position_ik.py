"""Kinematics for cylindrical teleoperation with switchable tool orientation.

The operator can use either:
- FREE: only radial distance r and height z are constrained; J2/J3/J4 are redundant.
- HORIZONTAL: r, z and J2+J3+J4=-90 deg are constrained.

The generic fixed-sum analytic solver also allows a smooth FREE -> HORIZONTAL
transition at constant r,z by gradually changing the requested joint sum.
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
IK_MAX_ITERATIONS = 80
MAX_INTERNAL_STEP_DEG = 5.0

HORIZONTAL_JOINT_SUM_DEG = -90.0
HORIZONTAL_TOLERANCE_DEG = 1e-9


class IKError(RuntimeError):
    """Raised when numerical or analytic IK fails."""


def horizontal_sum_deg(joints_deg) -> float:
    joints = np.asarray(joints_deg, dtype=np.float64)
    if joints.shape != (6,):
        raise ValueError("joints_deg must contain exactly 6 values")
    return float(np.sum(joints[1:4]))


def is_horizontal_pose(
    joints_deg,
    tolerance_deg: float = HORIZONTAL_TOLERANCE_DEG,
) -> bool:
    return abs(
        horizontal_sum_deg(joints_deg) - HORIZONTAL_JOINT_SUM_DEG
    ) <= tolerance_deg


def forward_radial_z(joints_deg) -> np.ndarray:
    joints = np.asarray(joints_deg, dtype=np.float64)
    if joints.shape != (6,):
        raise ValueError("joints_deg must contain exactly 6 values")

    q2 = np.deg2rad(joints[1]) + JOINT2_ZERO_DIRECTION_RAD
    q3 = np.deg2rad(joints[2])
    q4 = np.deg2rad(joints[3])

    q23 = q2 + q3
    q234 = q23 + q4

    radial = (
        LINK_1_MM * np.cos(q2)
        + LINK_2_MM * np.cos(q23)
        + LINK_3_MM * np.cos(q234)
    )
    z_mm = (
        LINK_1_MM * np.sin(q2)
        + LINK_2_MM * np.sin(q23)
        + LINK_3_MM * np.sin(q234)
    )

    return np.array([radial, z_mm], dtype=np.float64)


def radial_z_jacobian(joints_deg) -> np.ndarray:
    joints = np.asarray(joints_deg, dtype=np.float64)
    if joints.shape != (6,):
        raise ValueError("joints_deg must contain exactly 6 values")

    q2 = np.deg2rad(joints[1]) + JOINT2_ZERO_DIRECTION_RAD
    q3 = np.deg2rad(joints[2])
    q4 = np.deg2rad(joints[3])

    q23 = q2 + q3
    q234 = q23 + q4

    radial = (
        LINK_1_MM * np.cos(q2)
        + LINK_2_MM * np.cos(q23)
        + LINK_3_MM * np.cos(q234)
    )
    z_mm = (
        LINK_1_MM * np.sin(q2)
        + LINK_2_MM * np.sin(q23)
        + LINK_3_MM * np.sin(q234)
    )

    return np.array(
        [
            [
                -z_mm,
                -LINK_2_MM * np.sin(q23)
                - LINK_3_MM * np.sin(q234),
                -LINK_3_MM * np.sin(q234),
            ],
            [
                radial,
                LINK_2_MM * np.cos(q23)
                + LINK_3_MM * np.cos(q234),
                LINK_3_MM * np.cos(q234),
            ],
        ],
        dtype=np.float64,
    )


def forward_xyz(joints_deg, j1_sign: float = 1.0) -> np.ndarray:
    joints = np.asarray(joints_deg, dtype=np.float64)
    if joints.shape != (6,):
        raise ValueError("joints_deg must contain exactly 6 values")

    radial, z_mm = forward_radial_z(joints)
    q1_world = float(j1_sign) * np.deg2rad(joints[0])

    return np.array(
        [
            radial * np.cos(q1_world),
            radial * np.sin(q1_world),
            z_mm,
        ],
        dtype=np.float64,
    )


def position_jacobian(joints_deg, j1_sign: float = 1.0) -> np.ndarray:
    joints = np.asarray(joints_deg, dtype=np.float64)
    radial, _ = forward_radial_z(joints)
    rz_j = radial_z_jacobian(joints)

    sign = float(j1_sign)
    q1_world = sign * np.deg2rad(joints[0])
    c1 = np.cos(q1_world)
    s1 = np.sin(q1_world)

    x_mm = radial * c1
    y_mm = radial * s1
    dr = rz_j[0]
    dz = rz_j[1]

    return np.array(
        [
            [-sign * y_mm, c1 * dr[0], c1 * dr[1], c1 * dr[2]],
            [ sign * x_mm, s1 * dr[0], s1 * dr[1], s1 * dr[2]],
            [0.0,           dz[0],      dz[1],      dz[2]],
        ],
        dtype=np.float64,
    )


def solve_radial_z_with_joint_sum(
    target_radial_z_mm,
    target_joint_sum_deg: float,
    previous_joints_deg,
    *,
    tolerance_mm: float = IK_POSITION_TOLERANCE_MM,
) -> np.ndarray:
    """Solve r,z and an exact J2+J3+J4 servo-angle sum.

    For a requested servo sum S:
        tool model pitch alpha = S + 90 deg

    Link 3 therefore contributes:
        [L3*cos(alpha), L3*sin(alpha)]

    Removing that vector leaves a standard two-link wrist target for J2/J3.
    Both elbow branches are generated and the candidate nearest to the
    previous J2/J3/J4 command is selected.
    """
    target = np.asarray(target_radial_z_mm, dtype=np.float64)
    previous = np.asarray(previous_joints_deg, dtype=np.float64)

    if target.shape != (2,):
        raise ValueError("target_radial_z_mm must contain exactly 2 values")
    if previous.shape != (6,):
        raise ValueError("previous_joints_deg must contain exactly 6 values")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(previous)):
        raise ValueError("IK input must contain only finite values")
    if not np.isfinite(target_joint_sum_deg):
        raise ValueError("target_joint_sum_deg must be finite")

    target_r = float(target[0])
    target_z = float(target[1])
    alpha = math.radians(float(target_joint_sum_deg) + 90.0)

    wrist_r = target_r - LINK_3_MM * math.cos(alpha)
    wrist_z = target_z - LINK_3_MM * math.sin(alpha)

    l1 = float(LINK_1_MM)
    l2 = float(LINK_2_MM)

    cosine_q3 = (
        wrist_r * wrist_r
        + wrist_z * wrist_z
        - l1 * l1
        - l2 * l2
    ) / (2.0 * l1 * l2)

    reach_tolerance = 1e-10
    if cosine_q3 < -1.0 - reach_tolerance or cosine_q3 > 1.0 + reach_tolerance:
        raise IKError(
            f"fixed-pitch target unreachable: r={target_r:.3f} mm, "
            f"z={target_z:.3f} mm, sum={target_joint_sum_deg:.3f} deg"
        )

    cosine_q3 = max(-1.0, min(1.0, cosine_q3))
    q3_abs = math.acos(cosine_q3)

    candidates = []
    for q3_model in (-q3_abs, q3_abs):
        q2_model = math.atan2(wrist_z, wrist_r) - math.atan2(
            l2 * math.sin(q3_model),
            l1 + l2 * math.cos(q3_model),
        )

        q2_servo_deg = math.degrees(q2_model) - 90.0
        q3_servo_deg = math.degrees(q3_model)
        q4_servo_deg = (
            float(target_joint_sum_deg)
            - q2_servo_deg
            - q3_servo_deg
        )

        candidate = previous.copy()
        candidate[1] = q2_servo_deg
        candidate[2] = q3_servo_deg
        candidate[3] = q4_servo_deg

        position_error = float(
            np.linalg.norm(forward_radial_z(candidate) - target)
        )
        if position_error <= max(tolerance_mm * 10.0, 1e-7):
            candidates.append(candidate)

    if not candidates:
        raise IKError(
            f"no fixed-pitch IK candidate for r={target_r:.3f} mm, "
            f"z={target_z:.3f} mm, sum={target_joint_sum_deg:.3f} deg"
        )

    best = min(
        candidates,
        key=lambda candidate: float(
            np.sum((candidate[1:4] - previous[1:4]) ** 2)
        ),
    )
    best[3] = float(target_joint_sum_deg) - best[1] - best[2]
    return best


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


def _dls_step(jacobian: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
    system = (
        jacobian @ jacobian.T
        + float(damping) ** 2
        * np.eye(jacobian.shape[0], dtype=np.float64)
    )
    return jacobian.T @ np.linalg.solve(system, error)


def _limit_internal_step(delta_q_rad: np.ndarray) -> np.ndarray:
    max_abs = float(np.max(np.abs(delta_q_rad)))
    max_step_rad = np.deg2rad(MAX_INTERNAL_STEP_DEG)
    if max_abs > max_step_rad:
        delta_q_rad = delta_q_rad * (max_step_rad / max_abs)
    return delta_q_rad


def solve_radial_z(
    target_radial_z_mm,
    previous_joints_deg,
    *,
    damping: float = DLS_DAMPING,
    tolerance_mm: float = IK_POSITION_TOLERANCE_MM,
    max_iterations: int = IK_MAX_ITERATIONS,
) -> np.ndarray:
    """FREE-mode r/z solver. J1/J5/J6 are held exactly."""
    target = np.asarray(target_radial_z_mm, dtype=np.float64)
    previous = np.asarray(previous_joints_deg, dtype=np.float64)

    if target.shape != (2,):
        raise ValueError("target_radial_z_mm must contain exactly 2 values")
    if previous.shape != (6,):
        raise ValueError("previous_joints_deg must contain exactly 6 values")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(previous)):
        raise ValueError("IK input must contain only finite values")
    if damping <= 0.0:
        raise ValueError("damping must be positive")

    result = previous.copy()

    for _ in range(max_iterations):
        error = target - forward_radial_z(result)
        if float(np.linalg.norm(error)) <= tolerance_mm:
            return result

        delta_q_rad = _dls_step(
            radial_z_jacobian(result),
            error,
            damping,
        )
        result[1:4] += np.rad2deg(
            _limit_internal_step(delta_q_rad)
        )

    final_error = float(np.linalg.norm(target - forward_radial_z(result)))
    raise IKError(
        f"radial-z IK did not converge after {max_iterations} iterations; "
        f"remaining error={final_error:.6f} mm"
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
    """General legacy XYZ redundant position IK."""
    target = np.asarray(target_xyz_mm, dtype=np.float64)
    previous = np.asarray(previous_joints_deg, dtype=np.float64)

    if target.shape != (3,):
        raise ValueError("target_xyz_mm must contain exactly 3 values")
    if previous.shape != (6,):
        raise ValueError("previous_joints_deg must contain exactly 6 values")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(previous)):
        raise ValueError("IK input must contain only finite values")
    if damping <= 0.0:
        raise ValueError("damping must be positive")

    result = previous.copy()

    for _ in range(max_iterations):
        error = target - forward_xyz(result, j1_sign=j1_sign)
        if float(np.linalg.norm(error)) <= tolerance_mm:
            return result
        delta_q_rad = _dls_step(
            position_jacobian(result, j1_sign=j1_sign),
            error,
            damping,
        )
        result[:4] += np.rad2deg(
            _limit_internal_step(delta_q_rad)
        )

    final_error = float(
        np.linalg.norm(target - forward_xyz(result, j1_sign=j1_sign))
    )
    raise IKError(
        f"position IK did not converge after {max_iterations} iterations; "
        f"remaining error={final_error:.6f} mm"
    )
