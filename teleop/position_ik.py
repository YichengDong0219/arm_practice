"""Constrained kinematics for cylindrical teleoperation.

Two orientation modes are supported:

FREE
    Control radial distance r and height z. J2/J3/J4 are redundant, but J4 is
    physically constrained to [-90, +25] deg.

HORIZONTAL
    Control r and z while enforcing:
        J2 + J3 + J4 = -90 deg
    The same physical J4 limit is enforced when selecting analytic IK branches.

Important:
Joint limits are enforced INSIDE IK. They are not merely clamped immediately
before transmission, because doing that would make the software FK state
diverge from the real commanded robot state.
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

# Confirmed physical command range for Joint 4.
JOINT4_MIN_DEG = -90.0
JOINT4_MAX_DEG = 25.0
JOINT_LIMIT_TOLERANCE_DEG = 1e-9


class IKError(RuntimeError):
    """Raised when a target has no valid constrained IK solution."""


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
        horizontal_sum_deg(joints_deg)
        - HORIZONTAL_JOINT_SUM_DEG
    ) <= tolerance_deg


def joint4_in_limits(j4_deg: float) -> bool:
    value = float(j4_deg)
    return (
        JOINT4_MIN_DEG - JOINT_LIMIT_TOLERANCE_DEG
        <= value
        <= JOINT4_MAX_DEG + JOINT_LIMIT_TOLERANCE_DEG
    )


def _require_valid_previous_joint4(previous) -> None:
    if not joint4_in_limits(previous[3]):
        raise IKError(
            "current software pose is already outside the physical J4 range: "
            f"J4={previous[3]:.3f} deg, "
            f"allowed=[{JOINT4_MIN_DEG:.1f}, {JOINT4_MAX_DEG:.1f}] deg"
        )


def forward_radial_z(joints_deg) -> np.ndarray:
    joints = np.asarray(joints_deg, dtype=np.float64)

    if joints.shape != (6,):
        raise ValueError("joints_deg must contain exactly 6 values")

    q2 = (
        np.deg2rad(joints[1])
        + JOINT2_ZERO_DIRECTION_RAD
    )
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
        [radial, z_mm],
        dtype=np.float64,
    )


def radial_z_jacobian(joints_deg) -> np.ndarray:
    joints = np.asarray(joints_deg, dtype=np.float64)

    if joints.shape != (6,):
        raise ValueError("joints_deg must contain exactly 6 values")

    q2 = (
        np.deg2rad(joints[1])
        + JOINT2_ZERO_DIRECTION_RAD
    )
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


def forward_xyz(
    joints_deg,
    j1_sign: float = 1.0,
) -> np.ndarray:
    joints = np.asarray(
        joints_deg,
        dtype=np.float64,
    )

    if joints.shape != (6,):
        raise ValueError(
            "joints_deg must contain exactly 6 values"
        )

    radial, z_mm = forward_radial_z(joints)

    q1_world = (
        float(j1_sign)
        * np.deg2rad(joints[0])
    )

    return np.array(
        [
            radial * np.cos(q1_world),
            radial * np.sin(q1_world),
            z_mm,
        ],
        dtype=np.float64,
    )


def position_jacobian(
    joints_deg,
    j1_sign: float = 1.0,
) -> np.ndarray:
    joints = np.asarray(
        joints_deg,
        dtype=np.float64,
    )

    radial, _ = forward_radial_z(joints)
    rz_j = radial_z_jacobian(joints)

    sign = float(j1_sign)

    q1_world = (
        sign * np.deg2rad(joints[0])
    )

    c1 = np.cos(q1_world)
    s1 = np.sin(q1_world)

    x_mm = radial * c1
    y_mm = radial * s1

    dr = rz_j[0]
    dz = rz_j[1]

    return np.array(
        [
            [
                -sign * y_mm,
                c1 * dr[0],
                c1 * dr[1],
                c1 * dr[2],
            ],
            [
                sign * x_mm,
                s1 * dr[0],
                s1 * dr[1],
                s1 * dr[2],
            ],
            [
                0.0,
                dz[0],
                dz[1],
                dz[2],
            ],
        ],
        dtype=np.float64,
    )


def _dls_step(
    jacobian: np.ndarray,
    error: np.ndarray,
    damping: float,
) -> np.ndarray:
    system = (
        jacobian @ jacobian.T
        + float(damping) ** 2
        * np.eye(
            jacobian.shape[0],
            dtype=np.float64,
        )
    )

    return (
        jacobian.T
        @ np.linalg.solve(
            system,
            error,
        )
    )


def _limit_internal_step(
    delta_q_rad: np.ndarray,
) -> np.ndarray:
    max_abs = float(
        np.max(
            np.abs(delta_q_rad)
        )
    )

    max_step_rad = np.deg2rad(
        MAX_INTERNAL_STEP_DEG
    )

    if max_abs > max_step_rad:
        delta_q_rad = (
            delta_q_rad
            * (max_step_rad / max_abs)
        )

    return delta_q_rad


def solve_radial_z_with_joint_sum(
    target_radial_z_mm,
    target_joint_sum_deg: float,
    previous_joints_deg,
    *,
    tolerance_mm: float = IK_POSITION_TOLERANCE_MM,
) -> np.ndarray:
    """Solve r,z with an exact J2+J3+J4 sum and physical J4 limits."""
    target = np.asarray(
        target_radial_z_mm,
        dtype=np.float64,
    )

    previous = np.asarray(
        previous_joints_deg,
        dtype=np.float64,
    )

    if target.shape != (2,):
        raise ValueError(
            "target_radial_z_mm must contain exactly 2 values"
        )

    if previous.shape != (6,):
        raise ValueError(
            "previous_joints_deg must contain exactly 6 values"
        )

    if (
        not np.all(np.isfinite(target))
        or not np.all(np.isfinite(previous))
    ):
        raise ValueError(
            "IK input must contain only finite values"
        )

    if not np.isfinite(target_joint_sum_deg):
        raise ValueError(
            "target_joint_sum_deg must be finite"
        )

    _require_valid_previous_joint4(previous)

    target_r = float(target[0])
    target_z = float(target[1])

    alpha = math.radians(
        float(target_joint_sum_deg)
        + 90.0
    )

    wrist_r = (
        target_r
        - LINK_3_MM * math.cos(alpha)
    )

    wrist_z = (
        target_z
        - LINK_3_MM * math.sin(alpha)
    )

    l1 = float(LINK_1_MM)
    l2 = float(LINK_2_MM)

    cosine_q3 = (
        wrist_r * wrist_r
        + wrist_z * wrist_z
        - l1 * l1
        - l2 * l2
    ) / (2.0 * l1 * l2)

    reach_tolerance = 1e-10

    if (
        cosine_q3 < -1.0 - reach_tolerance
        or cosine_q3 > 1.0 + reach_tolerance
    ):
        raise IKError(
            "target is geometrically unreachable: "
            f"r={target_r:.3f} mm, "
            f"z={target_z:.3f} mm, "
            f"sum={target_joint_sum_deg:.3f} deg"
        )

    cosine_q3 = max(
        -1.0,
        min(1.0, cosine_q3),
    )

    q3_abs = math.acos(
        cosine_q3
    )

    valid_candidates = []
    geometric_candidates = 0

    for q3_model in (
        -q3_abs,
        q3_abs,
    ):
        q2_model = (
            math.atan2(
                wrist_z,
                wrist_r,
            )
            - math.atan2(
                l2 * math.sin(q3_model),
                l1
                + l2 * math.cos(q3_model),
            )
        )

        q2_servo_deg = (
            math.degrees(q2_model)
            - 90.0
        )

        q3_servo_deg = math.degrees(
            q3_model
        )

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
            np.linalg.norm(
                forward_radial_z(candidate)
                - target
            )
        )

        if (
            position_error
            <= max(
                tolerance_mm * 10.0,
                1e-7,
            )
        ):
            geometric_candidates += 1

            if joint4_in_limits(
                q4_servo_deg
            ):
                valid_candidates.append(
                    candidate
                )

    if not valid_candidates:
        if geometric_candidates:
            raise IKError(
                "target has geometric IK solutions, "
                "but all violate the physical J4 range "
                f"[{JOINT4_MIN_DEG:.1f}, {JOINT4_MAX_DEG:.1f}] deg"
            )

        raise IKError(
            "no valid fixed-pitch IK candidate"
        )

    best = min(
        valid_candidates,
        key=lambda candidate: float(
            np.sum(
                (
                    candidate[1:4]
                    - previous[1:4]
                ) ** 2
            )
        ),
    )

    best[3] = (
        float(target_joint_sum_deg)
        - best[1]
        - best[2]
    )

    if not joint4_in_limits(best[3]):
        raise IKError(
            "internal error: selected J4 exceeds physical limit"
        )

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


def _constrained_rz_iteration(
    result,
    error,
    *,
    damping: float,
) -> np.ndarray:
    """One active-set DLS step for q2/q3/q4 with a hard q4 bound."""
    jacobian = radial_z_jacobian(
        result
    )

    delta = _dls_step(
        jacobian,
        error,
        damping,
    )

    delta = _limit_internal_step(
        delta
    )

    delta_deg = np.rad2deg(delta)

    proposed_q4 = (
        result[3]
        + delta_deg[2]
    )

    if (
        JOINT4_MIN_DEG
        <= proposed_q4
        <= JOINT4_MAX_DEG
    ):
        next_result = result.copy()
        next_result[1:4] += delta_deg
        return next_result

    # q4 wants to leave its physical interval. Move q4 only up to the
    # relevant boundary, then use q2/q3 as the active free variables.
    if proposed_q4 > JOINT4_MAX_DEG:
        q4_bound = JOINT4_MAX_DEG
    else:
        q4_bound = JOINT4_MIN_DEG

    fixed_delta_q4_rad = np.deg2rad(
        q4_bound - result[3]
    )

    residual = (
        error
        - jacobian[:, 2]
        * fixed_delta_q4_rad
    )

    free_jacobian = (
        jacobian[:, :2]
    )

    free_delta = _dls_step(
        free_jacobian,
        residual,
        damping,
    )

    combined = np.array(
        [
            free_delta[0],
            free_delta[1],
            fixed_delta_q4_rad,
        ],
        dtype=np.float64,
    )

    combined = _limit_internal_step(
        combined
    )

    next_result = result.copy()
    next_result[1:4] += np.rad2deg(
        combined
    )

    # Numerically enforce the hard boundary.
    next_result[3] = float(
        np.clip(
            next_result[3],
            JOINT4_MIN_DEG,
            JOINT4_MAX_DEG,
        )
    )

    return next_result


def solve_radial_z(
    target_radial_z_mm,
    previous_joints_deg,
    *,
    damping: float = DLS_DAMPING,
    tolerance_mm: float = IK_POSITION_TOLERANCE_MM,
    max_iterations: int = IK_MAX_ITERATIONS,
) -> np.ndarray:
    """FREE-mode constrained r/z IK.

    J1/J5/J6 are preserved.
    J4 is hard constrained to [-90, +25] deg.
    When J4 hits a boundary, q2/q3 remain active and continue trying to
    satisfy the Cartesian target.
    """
    target = np.asarray(
        target_radial_z_mm,
        dtype=np.float64,
    )

    previous = np.asarray(
        previous_joints_deg,
        dtype=np.float64,
    )

    if target.shape != (2,):
        raise ValueError(
            "target_radial_z_mm must contain exactly 2 values"
        )

    if previous.shape != (6,):
        raise ValueError(
            "previous_joints_deg must contain exactly 6 values"
        )

    if (
        not np.all(np.isfinite(target))
        or not np.all(np.isfinite(previous))
    ):
        raise ValueError(
            "IK input must contain only finite values"
        )

    if damping <= 0.0:
        raise ValueError(
            "damping must be positive"
        )

    _require_valid_previous_joint4(
        previous
    )

    result = previous.copy()

    for _ in range(max_iterations):
        error = (
            target
            - forward_radial_z(result)
        )

        if (
            float(np.linalg.norm(error))
            <= tolerance_mm
        ):
            if not joint4_in_limits(
                result[3]
            ):
                raise IKError(
                    "IK converged numerically but J4 violates its limit"
                )

            return result

        result = _constrained_rz_iteration(
            result,
            error,
            damping=damping,
        )

    final_error = float(
        np.linalg.norm(
            target
            - forward_radial_z(result)
        )
    )

    raise IKError(
        "constrained FREE IK did not converge; "
        f"remaining error={final_error:.6f} mm, "
        f"J4={result[3]:.3f} deg, "
        f"allowed=[{JOINT4_MIN_DEG:.1f}, {JOINT4_MAX_DEG:.1f}] deg"
    )


def _constrained_xyz_iteration(
    result,
    error,
    *,
    j1_sign: float,
    damping: float,
) -> np.ndarray:
    """One active-set DLS step for legacy XYZ IK with a hard J4 bound."""
    jacobian = position_jacobian(
        result,
        j1_sign=j1_sign,
    )

    delta = _dls_step(
        jacobian,
        error,
        damping,
    )

    delta = _limit_internal_step(
        delta
    )

    delta_deg = np.rad2deg(
        delta
    )

    proposed_q4 = (
        result[3]
        + delta_deg[3]
    )

    if (
        JOINT4_MIN_DEG
        <= proposed_q4
        <= JOINT4_MAX_DEG
    ):
        next_result = result.copy()
        next_result[:4] += delta_deg
        return next_result

    if proposed_q4 > JOINT4_MAX_DEG:
        q4_bound = JOINT4_MAX_DEG
    else:
        q4_bound = JOINT4_MIN_DEG

    fixed_delta_q4_rad = np.deg2rad(
        q4_bound - result[3]
    )

    residual = (
        error
        - jacobian[:, 3]
        * fixed_delta_q4_rad
    )

    free_jacobian = (
        jacobian[:, :3]
    )

    free_delta = _dls_step(
        free_jacobian,
        residual,
        damping,
    )

    combined = np.array(
        [
            free_delta[0],
            free_delta[1],
            free_delta[2],
            fixed_delta_q4_rad,
        ],
        dtype=np.float64,
    )

    combined = _limit_internal_step(
        combined
    )

    next_result = result.copy()

    next_result[:4] += np.rad2deg(
        combined
    )

    next_result[3] = float(
        np.clip(
            next_result[3],
            JOINT4_MIN_DEG,
            JOINT4_MAX_DEG,
        )
    )

    return next_result


def solve_position(
    target_xyz_mm,
    previous_joints_deg,
    *,
    j1_sign: float = 1.0,
    damping: float = DLS_DAMPING,
    tolerance_mm: float = IK_POSITION_TOLERANCE_MM,
    max_iterations: int = IK_MAX_ITERATIONS,
) -> np.ndarray:
    """Legacy XYZ IK with the same physical J4 constraint."""
    target = np.asarray(
        target_xyz_mm,
        dtype=np.float64,
    )

    previous = np.asarray(
        previous_joints_deg,
        dtype=np.float64,
    )

    if target.shape != (3,):
        raise ValueError(
            "target_xyz_mm must contain exactly 3 values"
        )

    if previous.shape != (6,):
        raise ValueError(
            "previous_joints_deg must contain exactly 6 values"
        )

    if (
        not np.all(np.isfinite(target))
        or not np.all(np.isfinite(previous))
    ):
        raise ValueError(
            "IK input must contain only finite values"
        )

    if damping <= 0.0:
        raise ValueError(
            "damping must be positive"
        )

    _require_valid_previous_joint4(
        previous
    )

    result = previous.copy()

    for _ in range(max_iterations):
        error = (
            target
            - forward_xyz(
                result,
                j1_sign=j1_sign,
            )
        )

        if (
            float(np.linalg.norm(error))
            <= tolerance_mm
        ):
            return result

        result = _constrained_xyz_iteration(
            result,
            error,
            j1_sign=j1_sign,
            damping=damping,
        )

    final_error = float(
        np.linalg.norm(
            target
            - forward_xyz(
                result,
                j1_sign=j1_sign,
            )
        )
    )

    raise IKError(
        "constrained XYZ IK did not converge; "
        f"remaining error={final_error:.6f} mm, "
        f"J4={result[3]:.3f} deg"
    )
