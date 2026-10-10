import time
import serial
import numpy as np
import spatialmath as sm
import roboticstoolbox as rtb

from scipy.optimize import least_squares


# ============================================================
# 实验三：多层码垛 + 逐层回收
#
# 核心策略：
#
# 1. 大范围搬运：
#       position-only IK
#       不限制夹爪姿态
#
# 2. 接近物块最后 APPROACH_HEIGHT：
#       要求夹爪接近水平
#
# 3. 水平不是严格 -90°
#       而是允许 ± LEVEL_TOL_DEG
#
# 4. 最后若干毫米：
#       Cartesian 垂直下降 / 提升
# ============================================================


# ============================================================
# 0. 串口
# ============================================================

SERIAL_PORT = "COM5"
BAUDRATE = 115200


# ============================================================
# 1. 木块 / 桌面
# ============================================================

BLOCK_SIZE = 30.0

# TABLE_Z 表示木块实际放置平面在当前 D-H 基坐标系中的 z 坐标。
# 当前采用实机测量与调试后确认的标定值 52.0 mm；若机械臂或桌面
# 的安装位置改变，必须重新测量，不能仅由连杆 D-H 尺寸推算。
TABLE_Z = 52.0


# ============================================================
# 2. PICK
# ============================================================

PICK_X = 200.0
PICK_Y = 100.0

# 第一块木块中心
PICK_Z = (
    TABLE_Z
    + BLOCK_SIZE / 2.0
)

PICK_POINT = np.array(
    [
        PICK_X,
        PICK_Y,
        PICK_Z
    ],
    dtype=np.float64
)


# 大范围安全点
PICK_SAFE_CLEARANCE = 60.0

PICK_SAFE_POINT = np.array(
    [
        PICK_X,
        PICK_Y,
        PICK_Z + PICK_SAFE_CLEARANCE
    ],
    dtype=np.float64
)


# ============================================================
# 3. STACK
# ============================================================

STACK_X = 180.0
STACK_Y = -40.0

STACK_SAFE_CLEARANCE = 60.0


# ============================================================
# 4. 最后接近阶段
# ============================================================

# 从目标上方 15 mm 开始要求接近水平
APPROACH_HEIGHT = 15.0

# 最后 15 mm 分几个 Cartesian 小段
VERTICAL_STEPS = 5


# ============================================================
# 5. 夹爪水平约束
# ============================================================

# 对你当前机械臂：
#
# q2 + q3 + q4 = -90°
#
# 对应水平夹爪
LEVEL_TARGET_DEG = -90.0


# 不严格锁死水平
#
# 允许：
# -100° <= q2+q3+q4 <= -80°
LEVEL_TOL_DEG = 10.0


# 为了充分利用 ±10° 工作空间，
# IK 会依次尝试这些姿态
LEVEL_TARGET_CANDIDATES = [
    -90.0,
    -85.0,
    -95.0,
    -80.0,
    -100.0,
]


# 姿态误差在 least_squares 中的权重
#
# 单位可以理解为：
# 1° 姿态误差大约相当于多少 mm 的位置残差
LEVEL_WEIGHT = 3.0


# ============================================================
# 6. 层数
# ============================================================

# 第一次仍然建议 1
TARGET_LAYERS = 1

# 只是软件保护值
MAX_DESIGNED_LAYERS = 12


# ============================================================
# 7. 夹爪 J6
# ============================================================

GRIP_OPEN_DEG = 0.0
GRIP_CLOSE_DEG = -50.0


# ============================================================
# 8. 实机
# ============================================================

ENABLE_HARDWARE = True

WAIT_FOR_HUMAN = True


# ============================================================
# 9. 前五轴方向
# ============================================================

SERVO_DIRECTION_MAP = np.array(
    [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
    ],
    dtype=np.float64
)


# ============================================================
# 10. 机械限位
# ============================================================

JOINT_LIMITS_DEG = np.array(
    [
        [-90.0, 90.0],      # J1
        [-90.0, 90.0],      # J2
        [-90.0, 90.0],      # J3
        [-90.0, 25.0],      # J4
        [-180.0, 180.0],    # J5
    ],
    dtype=np.float64
)


# ============================================================
# 11. D-H 参数
# ============================================================

DH_PARAMS = [

    {
        "d": 108.5,
        "a": 0.0,
        "alpha": np.pi / 2.0,
        "offset": 0.0
    },

    {
        "d": 0.0,
        "a": 105.0,
        "alpha": 0.0,
        "offset": np.pi / 2.0
    },

    {
        "d": 0.0,
        "a": 97.5,
        "alpha": 0.0,
        "offset": 0.0
    },

    {
        "d": 0.0,
        "a": 28.22,
        "alpha": np.pi / 2.0,
        "offset": np.pi / 2.0
    },

    {
        "d": 172.70,
        "a": 0.0,
        "alpha": 0.0,
        "offset": 0.0
    }
]


# ============================================================
# 12. D-H
# ============================================================

def calc_dh_matrix(
    theta,
    alpha,
    d,
    a
):

    ct = np.cos(theta)
    st = np.sin(theta)

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array(
        [
            [
                ct,
                -st * ca,
                st * sa,
                a * ct
            ],

            [
                st,
                ct * ca,
                -ct * sa,
                a * st
            ],

            [
                0.0,
                sa,
                ca,
                d
            ],

            [
                0.0,
                0.0,
                0.0,
                1.0
            ]
        ],
        dtype=np.float64
    )


# ============================================================
# 13. FK
# ============================================================

def fk_matrix(q_deg):

    T = np.eye(
        4,
        dtype=np.float64
    )

    for i, p in enumerate(DH_PARAMS):

        theta = (
            np.deg2rad(
                q_deg[i]
            )
            + p["offset"]
        )

        T = (
            T
            @ calc_dh_matrix(
                theta,
                p["alpha"],
                p["d"],
                p["a"]
            )
        )

    return T


def fk_position(q_deg):

    return fk_matrix(
        q_deg
    )[:3, 3]


# ============================================================
# 14. Robotics Toolbox 模型
# ============================================================

def build_robot_model():

    links = []

    for p in DH_PARAMS:

        links.append(
            rtb.RevoluteDH(
                d=p["d"],
                a=p["a"],
                alpha=p["alpha"],
                offset=p["offset"]
            )
        )

    return rtb.DHRobot(
        links,
        name="Week6_Stacking_Arm"
    )


# ============================================================
# 15. XYZ -> SE3
# ============================================================

def xyz_to_se3(xyz):

    x, y, z = xyz

    T = np.array(
        [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0]
        ],
        dtype=np.float64
    )

    return sm.SE3(T)


# ============================================================
# 16. 工具姿态指标
# ============================================================

def gripper_pitch_sum(q_deg):
    """
    当前实验定义：

        q2 + q3 + q4

    接近 -90° 时，
    夹爪近似水平。
    """

    return (
        q_deg[1]
        + q_deg[2]
        + q_deg[3]
    )


def level_error(q_deg):

    return abs(
        gripper_pitch_sum(q_deg)
        - LEVEL_TARGET_DEG
    )


# ============================================================
# 17. 角度归一化
# ============================================================

def wrap_angle_deg(q_deg):

    q_deg = np.asarray(
        q_deg,
        dtype=np.float64
    )

    return (
        q_deg + 180.0
    ) % 360.0 - 180.0


# ============================================================
# 18. 限位
# ============================================================

def joint_limits_ok(q_deg):

    for i in range(5):

        if not (
            JOINT_LIMITS_DEG[i, 0]
            <= q_deg[i]
            <= JOINT_LIMITS_DEG[i, 1]
        ):

            return False

    return True


# ============================================================
# 19. 自由姿态 IK
#
# 用于：
# PICK_SAFE
# STACK_SAFE
# 大范围搬运
# ============================================================

def solve_free_ik(
    robot,
    xyz,
    q_previous_deg
):

    xyz = np.asarray(
        xyz,
        dtype=np.float64
    )

    q_previous_deg = np.asarray(
        q_previous_deg,
        dtype=np.float64
    )

    target = xyz_to_se3(
        xyz
    )

    mask = [
        1, 1, 1,
        0, 0, 0
    ]


    q1_guess = np.rad2deg(
        np.arctan2(
            xyz[1],
            xyz[0]
        )
    )


    seed_list = [

        q_previous_deg,

        np.array(
            [
                q1_guess,
                -20,
                -80,
                -75,
                0
            ],
            dtype=np.float64
        ),

        np.array(
            [
                q1_guess,
                10,
                -80,
                -80,
                0
            ],
            dtype=np.float64
        ),

        np.array(
            [
                q1_guess,
                40,
                -80,
                -80,
                0
            ],
            dtype=np.float64
        ),

        np.array(
            [
                q1_guess,
                -40,
                -40,
                -40,
                0
            ],
            dtype=np.float64
        ),

        np.array(
            [
                q1_guess,
                0,
                -50,
                -50,
                0
            ],
            dtype=np.float64
        ),

        np.array(
            [
                q1_guess,
                -60,
                30,
                -50,
                0
            ],
            dtype=np.float64
        ),
    ]


    candidates = []


    for seed in seed_list:

        try:

            result = robot.ikine_LM(
                target,
                q0=np.deg2rad(
                    seed
                ),
                mask=mask
            )

        except Exception:

            continue


        if not result.success:

            continue


        q_deg = np.rad2deg(
            np.asarray(
                result.q
            )
        )


        q_deg = wrap_angle_deg(
            q_deg
        )


        # J5 当前固定
        q_deg[4] = 0.0


        if not joint_limits_ok(
            q_deg
        ):

            continue


        error = np.linalg.norm(
            fk_position(q_deg)
            - xyz
        )


        if error > 1.0:

            continue


        continuity = np.linalg.norm(
            q_deg[:4]
            - q_previous_deg[:4]
        )


        candidates.append(
            (
                continuity,
                q_deg,
                error
            )
        )


    if not candidates:

        return None


    candidates.sort(
        key=lambda x: x[0]
    )


    return candidates[0][1]


# ============================================================
# 20. 近水平 IK
#
# 用于：
#
# PICK_APPROACH
# PICK
# STACK_APPROACH
# STACK
#
# 允许水平 ±10°
# ============================================================

def solve_level_ik(
    xyz,
    q_previous_deg
):

    xyz = np.asarray(
        xyz,
        dtype=np.float64
    )

    q_previous_deg = np.asarray(
        q_previous_deg,
        dtype=np.float64
    )


    # 优化变量：
    #
    # [q1, q2, q3, q4]
    #
    # J5 = 0

    lower = np.array(
        [
            -90.0,
            -90.0,
            -90.0,
            -90.0
        ],
        dtype=np.float64
    )

    upper = np.array(
        [
            90.0,
            90.0,
            90.0,
            25.0
        ],
        dtype=np.float64
    )


    q1_guess = np.rad2deg(
        np.arctan2(
            xyz[1],
            xyz[0]
        )
    )


    candidates = []


    # ========================================================
    # 尝试 -90、-85、-95、-80、-100
    #
    # 即允许使用 ±10° 姿态空间
    # ========================================================

    for pitch_target in LEVEL_TARGET_CANDIDATES:


        # ----------------------------------------------------
        # 根据目标 pitch 自动构造若干初值
        # ----------------------------------------------------

        seed_list = []


        # 上一姿态
        seed_previous = (
            q_previous_deg[:4].copy()
        )

        seed_previous[3] = (
            pitch_target
            - seed_previous[1]
            - seed_previous[2]
        )

        seed_list.append(
            seed_previous
        )


        # 一些常用构型
        base_pairs = [

            (-20.0, -40.0),

            (0.0, -60.0),

            (20.0, -70.0),

            (40.0, -80.0),

            (-40.0, -20.0),

            (-60.0, 20.0),

            (60.0, -70.0),

        ]


        for q2_seed, q3_seed in base_pairs:

            q4_seed = (
                pitch_target
                - q2_seed
                - q3_seed
            )


            seed_list.append(
                np.array(
                    [
                        q1_guess,
                        q2_seed,
                        q3_seed,
                        q4_seed
                    ],
                    dtype=np.float64
                )
            )


        # ====================================================
        # 残差函数
        # ====================================================

        def residual(q4_deg):

            q_deg = np.array(
                [
                    q4_deg[0],
                    q4_deg[1],
                    q4_deg[2],
                    q4_deg[3],
                    0.0
                ],
                dtype=np.float64
            )


            xyz_now = fk_position(
                q_deg
            )


            pos_error = (
                xyz_now
                - xyz
            )


            pitch_now = (
                q_deg[1]
                + q_deg[2]
                + q_deg[3]
            )


            pitch_error = (
                pitch_now
                - pitch_target
            )


            return np.array(
                [
                    pos_error[0],
                    pos_error[1],
                    pos_error[2],
                    pitch_error * LEVEL_WEIGHT
                ],
                dtype=np.float64
            )


        # ====================================================
        # 多初值求解
        # ====================================================

        for seed in seed_list:


            seed = np.clip(
                seed,
                lower + 1e-4,
                upper - 1e-4
            )


            try:

                result = least_squares(
                    residual,
                    seed,
                    bounds=(
                        lower,
                        upper
                    ),
                    max_nfev=3000,
                    xtol=1e-10,
                    ftol=1e-10,
                    gtol=1e-10
                )

            except Exception:

                continue


            q_deg = np.array(
                [
                    result.x[0],
                    result.x[1],
                    result.x[2],
                    result.x[3],
                    0.0
                ],
                dtype=np.float64
            )


            if not joint_limits_ok(
                q_deg
            ):

                continue


            xyz_error = np.linalg.norm(
                fk_position(q_deg)
                - xyz
            )


            if xyz_error > 1.0:

                continue


            pitch_now = gripper_pitch_sum(
                q_deg
            )


            # ------------------------------------------------
            # 必须落在总体 ±10° 范围
            # ------------------------------------------------

            if abs(
                pitch_now
                - LEVEL_TARGET_DEG
            ) > (
                LEVEL_TOL_DEG + 0.5
            ):

                continue


            # ------------------------------------------------
            # 当前子目标误差不能太大
            # ------------------------------------------------

            if abs(
                pitch_now
                - pitch_target
            ) > 1.0:

                continue


            continuity = np.linalg.norm(
                q_deg[:4]
                - q_previous_deg[:4]
            )


            # 更水平略微优先，
            # 但连续性仍然是主要指标
            level_penalty = abs(
                pitch_now
                - LEVEL_TARGET_DEG
            )


            score = (
                continuity
                + 0.3 * level_penalty
            )


            candidates.append(
                (
                    score,
                    q_deg,
                    xyz_error,
                    pitch_now
                )
            )


    if not candidates:

        return None


    candidates.sort(
        key=lambda x: x[0]
    )


    return candidates[0][1]


# ============================================================
# 21. 通用 IK 接口
# ============================================================

def solve_ik(
    robot,
    xyz,
    q_previous_deg,
    mode="free"
):

    if mode == "free":

        return solve_free_ik(
            robot,
            xyz,
            q_previous_deg
        )


    if mode == "level":

        return solve_level_ik(
            xyz,
            q_previous_deg
        )


    raise ValueError(
        f"未知 IK mode: {mode}"
    )


# ============================================================
# 22. STACK
# ============================================================

def stack_point(layer):

    z = (
        TABLE_Z
        + (
            layer - 0.5
        ) * BLOCK_SIZE
    )


    return np.array(
        [
            STACK_X,
            STACK_Y,
            z
        ],
        dtype=np.float64
    )


def stack_approach_point(layer):

    p = stack_point(
        layer
    ).copy()

    p[2] += APPROACH_HEIGHT

    return p


def stack_safe_point(layer):

    p = stack_point(
        layer
    ).copy()

    p[2] += STACK_SAFE_CLEARANCE

    return p


# ============================================================
# 23. PICK approach
# ============================================================

def pick_approach_point():

    p = PICK_POINT.copy()

    p[2] += APPROACH_HEIGHT

    return p


# ============================================================
# 24. Cartesian 竖直路径
# ============================================================

def generate_vertical_points(
    start_xyz,
    end_xyz,
    steps=VERTICAL_STEPS
):

    start_xyz = np.asarray(
        start_xyz,
        dtype=np.float64
    )

    end_xyz = np.asarray(
        end_xyz,
        dtype=np.float64
    )


    points = []


    for i in range(
        1,
        steps + 1
    ):

        ratio = (
            i / steps
        )


        xyz = (
            start_xyz
            + ratio
            * (
                end_xyz
                - start_xyz
            )
        )


        points.append(
            xyz
        )


    return points


# ============================================================
# 25. 基础关节路径检查
# ============================================================

def check_joint_path(
    q_start_deg,
    q_end_deg,
    samples=30
):

    for i in range(
        samples + 1
    ):

        ratio = (
            i / samples
        )


        q = (
            q_start_deg
            + ratio
            * (
                q_end_deg
                - q_start_deg
            )
        )


        if not joint_limits_ok(q):

            return (
                False,
                "插值过程中出现关节超限"
            )


        xyz = fk_position(
            q
        )


        if xyz[2] < (
            TABLE_Z - 3.0
        ):

            return (
                False,
                (
                    "轨迹进入桌面以下："
                    f"z={xyz[2]:.2f}"
                )
            )


    return (
        True,
        "OK"
    )


# ============================================================
# 26. 生成运动计划
# ============================================================

def build_motion_plan(
    robot,
    layers
):

    plan = []


    q_current = np.zeros(
        5,
        dtype=np.float64
    )


    # ========================================================
    # 添加一个运动点
    # ========================================================

    def add_move(
        name,
        xyz,
        mode
    ):

        nonlocal q_current


        q_target = solve_ik(
            robot,
            xyz,
            q_current,
            mode=mode
        )


        if q_target is None:

            raise RuntimeError(
                "\nIK 求解失败"
                f"\n动作：{name}"
                f"\n模式：{mode}"
                f"\nXYZ：{xyz}"
            )


        ok, message = check_joint_path(
            q_current,
            q_target
        )


        if not ok:

            raise RuntimeError(
                "\n路径检查失败"
                f"\n动作：{name}"
                f"\n原因：{message}"
            )


        error = np.linalg.norm(
            fk_position(q_target)
            - xyz
        )


        plan.append(
            {
                "type": "move",
                "name": name,
                "xyz": xyz.copy(),
                "q": q_target.copy(),
                "mode": mode,
                "error": error,
                "pitch": gripper_pitch_sum(
                    q_target
                )
            }
        )


        q_current = (
            q_target.copy()
        )


    def add_grip(
        name,
        angle
    ):

        plan.append(
            {
                "type": "grip",
                "name": name,
                "angle_deg": float(
                    angle
                )
            }
        )


    def add_wait(
        name,
        message
    ):

        plan.append(
            {
                "type": "wait",
                "name": name,
                "message": message
            }
        )


    # ========================================================
    # 码垛
    # ========================================================

    for layer in range(
        1,
        layers + 1
    ):


        add_wait(
            f"准备第 {layer} 块",
            (
                f"请将第 {layer} 块木块放到 PICK 工位，"
                "然后按 Enter。"
            )
        )


        # ----------------------------------------------------
        # 大范围移动到 PICK_SAFE
        # 姿态自由
        # ----------------------------------------------------

        add_move(
            f"L{layer} -> PICK_SAFE",
            PICK_SAFE_POINT,
            mode="free"
        )


        # ----------------------------------------------------
        # 进入最后 15 mm 上方
        #
        # 从这里开始要求接近水平
        # ----------------------------------------------------

        add_move(
            f"L{layer} -> PICK_APPROACH",
            pick_approach_point(),
            mode="level"
        )


        # ----------------------------------------------------
        # 垂直下降 15 mm
        # ----------------------------------------------------

        down_points = generate_vertical_points(
            pick_approach_point(),
            PICK_POINT
        )


        for i, xyz in enumerate(
            down_points,
            start=1
        ):

            add_move(
                (
                    f"L{layer} "
                    f"PICK_VERTICAL_DOWN "
                    f"{i}/{VERTICAL_STEPS}"
                ),
                xyz,
                mode="level"
            )


        # ----------------------------------------------------
        # 抓
        # ----------------------------------------------------

        add_grip(
            f"L{layer} 夹紧",
            GRIP_CLOSE_DEG
        )


        # ----------------------------------------------------
        # 垂直提升
        # ----------------------------------------------------

        up_points = generate_vertical_points(
            PICK_POINT,
            pick_approach_point()
        )


        for i, xyz in enumerate(
            up_points,
            start=1
        ):

            add_move(
                (
                    f"L{layer} "
                    f"PICK_VERTICAL_UP "
                    f"{i}/{VERTICAL_STEPS}"
                ),
                xyz,
                mode="level"
            )


        # ----------------------------------------------------
        # 离开 PICK
        #
        # 重新释放姿态
        # ----------------------------------------------------

        add_move(
            f"L{layer} -> PICK_SAFE after grasp",
            PICK_SAFE_POINT,
            mode="free"
        )


        # ----------------------------------------------------
        # 搬运到 STACK_SAFE
        #
        # 姿态自由
        # ----------------------------------------------------

        add_move(
            f"L{layer} -> STACK_SAFE",
            stack_safe_point(
                layer
            ),
            mode="free"
        )


        # ----------------------------------------------------
        # 进入 STACK_APPROACH
        #
        # 开始恢复近水平
        # ----------------------------------------------------

        add_move(
            f"L{layer} -> STACK_APPROACH",
            stack_approach_point(
                layer
            ),
            mode="level"
        )


        # ----------------------------------------------------
        # 最后 15 mm 垂直下降
        # ----------------------------------------------------

        down_points = generate_vertical_points(
            stack_approach_point(
                layer
            ),
            stack_point(
                layer
            )
        )


        for i, xyz in enumerate(
            down_points,
            start=1
        ):

            add_move(
                (
                    f"L{layer} "
                    f"STACK_VERTICAL_DOWN "
                    f"{i}/{VERTICAL_STEPS}"
                ),
                xyz,
                mode="level"
            )


        # ----------------------------------------------------
        # 放下
        # ----------------------------------------------------

        add_grip(
            f"L{layer} 松开",
            GRIP_OPEN_DEG
        )


        # ----------------------------------------------------
        # 垂直离开木块
        # ----------------------------------------------------

        up_points = generate_vertical_points(
            stack_point(
                layer
            ),
            stack_approach_point(
                layer
            )
        )


        for i, xyz in enumerate(
            up_points,
            start=1
        ):

            add_move(
                (
                    f"L{layer} "
                    f"STACK_VERTICAL_UP "
                    f"{i}/{VERTICAL_STEPS}"
                ),
                xyz,
                mode="level"
            )


        # ----------------------------------------------------
        # 离开塔顶以后重新自由姿态
        # ----------------------------------------------------

        add_move(
            f"L{layer} -> STACK_SAFE after place",
            stack_safe_point(
                layer
            ),
            mode="free"
        )


    # ========================================================
    # 码垛完成
    # ========================================================

    add_wait(
        "码垛完成",
        (
            f"{layers} 层码垛完成。"
            "确认稳定后按 Enter 开始回收。"
        )
    )


    # ========================================================
    # 回收
    # ========================================================

    for layer in range(
        layers,
        0,
        -1
    ):


        # ----------------------------------------------------
        # 自由姿态到安全点
        # ----------------------------------------------------

        add_move(
            f"回收 L{layer} -> STACK_SAFE",
            stack_safe_point(
                layer
            ),
            mode="free"
        )


        # ----------------------------------------------------
        # 进入近水平 approach
        # ----------------------------------------------------

        add_move(
            f"回收 L{layer} -> STACK_APPROACH",
            stack_approach_point(
                layer
            ),
            mode="level"
        )


        # ----------------------------------------------------
        # 垂直下降
        # ----------------------------------------------------

        down_points = generate_vertical_points(
            stack_approach_point(
                layer
            ),
            stack_point(
                layer
            )
        )


        for i, xyz in enumerate(
            down_points,
            start=1
        ):

            add_move(
                (
                    f"回收 L{layer} "
                    f"VERTICAL_DOWN "
                    f"{i}/{VERTICAL_STEPS}"
                ),
                xyz,
                mode="level"
            )


        # ----------------------------------------------------
        # 夹紧
        # ----------------------------------------------------

        add_grip(
            f"回收 L{layer} 夹紧",
            GRIP_CLOSE_DEG
        )


        # ----------------------------------------------------
        # 垂直提起
        # ----------------------------------------------------

        up_points = generate_vertical_points(
            stack_point(
                layer
            ),
            stack_approach_point(
                layer
            )
        )


        for i, xyz in enumerate(
            up_points,
            start=1
        ):

            add_move(
                (
                    f"回收 L{layer} "
                    f"VERTICAL_UP "
                    f"{i}/{VERTICAL_STEPS}"
                ),
                xyz,
                mode="level"
            )


        # ----------------------------------------------------
        # 自由搬运
        # ----------------------------------------------------

        add_move(
            f"回收 L{layer} -> STACK_SAFE",
            stack_safe_point(
                layer
            ),
            mode="free"
        )


        add_move(
            f"回收 L{layer} -> PICK_SAFE",
            PICK_SAFE_POINT,
            mode="free"
        )


        # ----------------------------------------------------
        # PICK APPROACH
        # ----------------------------------------------------

        add_move(
            f"回收 L{layer} -> PICK_APPROACH",
            pick_approach_point(),
            mode="level"
        )


        # ----------------------------------------------------
        # 垂直下降
        # ----------------------------------------------------

        down_points = generate_vertical_points(
            pick_approach_point(),
            PICK_POINT
        )


        for i, xyz in enumerate(
            down_points,
            start=1
        ):

            add_move(
                (
                    f"回收 L{layer} "
                    f"PICK_DOWN "
                    f"{i}/{VERTICAL_STEPS}"
                ),
                xyz,
                mode="level"
            )


        # ----------------------------------------------------
        # 松开
        # ----------------------------------------------------

        add_grip(
            f"回收 L{layer} 松开",
            GRIP_OPEN_DEG
        )


        # ----------------------------------------------------
        # 垂直退出
        # ----------------------------------------------------

        up_points = generate_vertical_points(
            PICK_POINT,
            pick_approach_point()
        )


        for i, xyz in enumerate(
            up_points,
            start=1
        ):

            add_move(
                (
                    f"回收 L{layer} "
                    f"PICK_UP "
                    f"{i}/{VERTICAL_STEPS}"
                ),
                xyz,
                mode="level"
            )


        add_move(
            f"回收 L{layer} -> PICK_SAFE",
            PICK_SAFE_POINT,
            mode="free"
        )


        add_wait(
            f"移走第 {layer} 块",
            (
                "请人工取走 PICK 工位上的木块，"
                "然后按 Enter。"
            )
        )


    # ========================================================
    # HOME
    # ========================================================

    plan.append(
        {
            "type": "home",
            "name": "返回零位",
            "q": np.zeros(
                5,
                dtype=np.float64
            )
        }
    )


    return plan


# ============================================================
# 27. 打印规划
# ============================================================

def print_plan(plan):

    print(
        "\n============================================"
    )

    print(
        "完整离线规划"
    )

    print(
        "============================================"
    )


    move_index = 0


    for step in plan:


        if step["type"] == "move":

            move_index += 1


            print(
                f"\n[{move_index:03d}] "
                f"{step['name']}"
            )


            print(
                f"模式 = {step['mode']}"
            )


            print(
                "XYZ =",
                np.round(
                    step["xyz"],
                    2
                )
            )


            print(
                "J1~J5 =",
                np.round(
                    step["q"],
                    2
                )
            )


            print(
                f"pitch(q2+q3+q4) = "
                f"{step['pitch']:.2f}°"
            )


            if step["mode"] == "level":

                print(
                    f"水平误差 = "
                    f"{abs(step['pitch'] - LEVEL_TARGET_DEG):.2f}°"
                )


            print(
                f"FK error = "
                f"{step['error']:.4f} mm"
            )


        elif step["type"] == "grip":

            print(
                f"\n[夹爪] "
                f"{step['name']} -> "
                f"J6={step['angle_deg']:.1f}°"
            )


# ============================================================
# 28. 六轴串口帧
# ============================================================

def pack_frame(q6_rad):

    q6_rad = np.asarray(
        q6_rad,
        dtype=np.float64
    )


    if len(q6_rad) != 6:

        raise ValueError(
            "必须传入 J1~J6 六轴"
        )


    tx = bytearray(16)


    tx[0] = 0xAA


    for i in range(6):

        val = int(
            q6_rad[i]
            * 1000
        )


        val = max(
            min(
                val,
                32767
            ),
            -32768
        )


        tx[
            1 + 2 * i
        ] = (
            val >> 8
        ) & 0xFF


        tx[
            2 + 2 * i
        ] = (
            val
        ) & 0xFF


    tx[13] = 0x01


    check = 0

    for b in tx[:14]:

        check ^= b


    tx[14] = check

    tx[15] = 0xBB


    return tx


# ============================================================
# 29. 构造六轴命令
# ============================================================

def build_six_axis_command(
    q5_deg,
    gripper_deg
):

    q5_deg = np.asarray(
        q5_deg,
        dtype=np.float64
    )


    q5_rad = np.deg2rad(
        q5_deg
    )


    q5_rad *= (
        SERVO_DIRECTION_MAP
    )


    q6_rad = np.zeros(
        6,
        dtype=np.float64
    )


    q6_rad[:5] = q5_rad


    q6_rad[5] = np.deg2rad(
        gripper_deg
    )


    return q6_rad


# ============================================================
# 30. 串口
# ============================================================

def init_serial():

    ser = serial.Serial()


    ser.port = SERIAL_PORT
    ser.baudrate = BAUDRATE
    ser.timeout = 1

    ser.dtr = False
    ser.rts = False


    try:

        ser.open()


        print(
            f"\n[硬件] 已连接 {SERIAL_PORT}"
        )


        time.sleep(1.0)


        q_zero = np.zeros(
            5,
            dtype=np.float64
        )


        q6_rad = build_six_axis_command(
            q_zero,
            GRIP_OPEN_DEG
        )


        ser.write(
            pack_frame(
                q6_rad
            )
        )


        ser.flush()


        time.sleep(1.5)


        return ser


    except Exception as e:

        print(
            f"[错误] 串口初始化失败：{e}"
        )

        return None


# ============================================================
# 31. 前五轴运动
# ============================================================

def move_joint_interpolated(
    ser,
    q_start_deg,
    q_target_deg,
    gripper_deg,
    duration=1.5,
    steps=30
):

    q_start_deg = np.asarray(
        q_start_deg,
        dtype=np.float64
    )

    q_target_deg = np.asarray(
        q_target_deg,
        dtype=np.float64
    )


    dt = duration / steps


    for i in range(
        1,
        steps + 1
    ):

        ratio = i / steps


        q_deg = (
            q_start_deg
            + ratio
            * (
                q_target_deg
                - q_start_deg
            )
        )


        q6_rad = build_six_axis_command(
            q_deg,
            gripper_deg
        )


        ser.write(
            pack_frame(
                q6_rad
            )
        )


        ser.flush()


        time.sleep(dt)


# ============================================================
# 32. 夹爪
# ============================================================

def set_gripper(
    ser,
    q_current_deg,
    current_deg,
    target_deg,
    duration=0.8,
    steps=20
):

    dt = duration / steps


    print(
        f"[夹爪] "
        f"{current_deg:.1f}° "
        f"-> {target_deg:.1f}°"
    )


    for i in range(
        1,
        steps + 1
    ):

        ratio = i / steps


        grip_deg = (
            current_deg
            + ratio
            * (
                target_deg
                - current_deg
            )
        )


        q6_rad = build_six_axis_command(
            q_current_deg,
            grip_deg
        )


        ser.write(
            pack_frame(
                q6_rad
            )
        )


        ser.flush()


        time.sleep(dt)


    time.sleep(0.4)


    return float(
        target_deg
    )


# ============================================================
# 33. 执行
# ============================================================

def execute_plan(
    ser,
    plan
):

    q_current = np.zeros(
        5,
        dtype=np.float64
    )

    gripper_deg = (
        GRIP_OPEN_DEG
    )


    for index, step in enumerate(
        plan,
        start=1
    ):


        print(
            "\n--------------------------------------------"
        )

        print(
            f"Step {index}/{len(plan)}: "
            f"{step['name']}"
        )


        # ----------------------------------------------------
        # 等待
        # ----------------------------------------------------

        if step["type"] == "wait":

            if WAIT_FOR_HUMAN:

                input(
                    "\n"
                    + step["message"]
                    + "\n"
                )

            continue


        # ----------------------------------------------------
        # 运动
        # ----------------------------------------------------

        if step["type"] == "move":

            q_target = (
                step["q"]
            )


            print(
                "XYZ =",
                np.round(
                    step["xyz"],
                    2
                )
            )


            print(
                "q =",
                np.round(
                    q_target,
                    2
                )
            )


            print(
                f"IK mode = "
                f"{step['mode']}"
            )


            print(
                f"pitch = "
                f"{step['pitch']:.2f}°"
            )


            # Cartesian 小步已经由规划阶段拆开，
            # 这里每两个规划点之间仍采用关节插值
            move_joint_interpolated(
                ser,
                q_current,
                q_target,
                gripper_deg,
                duration=1.2
                if step["mode"] == "level"
                else 1.8,
                steps=30
            )


            q_current = (
                q_target.copy()
            )


            continue


        # ----------------------------------------------------
        # 夹爪
        # ----------------------------------------------------

        if step["type"] == "grip":

            gripper_deg = set_gripper(
                ser,
                q_current,
                gripper_deg,
                step["angle_deg"]
            )


            continue


        # ----------------------------------------------------
        # HOME
        # ----------------------------------------------------

        if step["type"] == "home":

            if abs(
                gripper_deg
                - GRIP_OPEN_DEG
            ) > 1e-6:

                gripper_deg = set_gripper(
                    ser,
                    q_current,
                    gripper_deg,
                    GRIP_OPEN_DEG
                )


            q_target = step["q"]


            move_joint_interpolated(
                ser,
                q_current,
                q_target,
                gripper_deg,
                duration=3.0,
                steps=60
            )


            q_current = (
                q_target.copy()
            )


# ============================================================
# 34. main
# ============================================================

def main():

    print(
        "\n============================================"
    )

    print(
        "实验三：多层码垛 + 逐层回收"
    )

    print(
        "============================================"
    )


    print(
        f"\nTABLE_Z = "
        f"{TABLE_Z:.1f} mm"
    )


    print(
        f"第一层中心 = "
        f"{TABLE_Z + BLOCK_SIZE / 2:.1f} mm"
    )


    print(
        "PICK =",
        PICK_POINT
    )


    print(
        f"STACK XY = "
        f"({STACK_X:.1f}, "
        f"{STACK_Y:.1f})"
    )


    print(
        f"APPROACH_HEIGHT = "
        f"{APPROACH_HEIGHT:.1f} mm"
    )


    print(
        f"近水平范围 = "
        f"{LEVEL_TARGET_DEG - LEVEL_TOL_DEG:.1f}° "
        f"~ "
        f"{LEVEL_TARGET_DEG + LEVEL_TOL_DEG:.1f}°"
    )


    print(
        f"J6 OPEN = "
        f"{GRIP_OPEN_DEG:.1f}°"
    )


    print(
        f"J6 CLOSE = "
        f"{GRIP_CLOSE_DEG:.1f}°"
    )


    print(
        f"TARGET_LAYERS = "
        f"{TARGET_LAYERS}"
    )


    if not (
        1
        <= TARGET_LAYERS
        <= MAX_DESIGNED_LAYERS
    ):

        raise ValueError(
            f"TARGET_LAYERS 必须在 "
            f"1~{MAX_DESIGNED_LAYERS}"
        )


    # --------------------------------------------------------
    # 建模
    # --------------------------------------------------------

    robot = build_robot_model()


    # --------------------------------------------------------
    # 离线规划
    # --------------------------------------------------------

    print(
        "\n[阶段 1] "
        "离线规划..."
    )


    plan = build_motion_plan(
        robot,
        TARGET_LAYERS
    )


    print_plan(
        plan
    )


    # --------------------------------------------------------
    # 打印各层
    # --------------------------------------------------------

    print(
        "\n============================================"
    )

    print(
        "各层坐标"
    )

    print(
        "============================================"
    )


    for layer in range(
        1,
        TARGET_LAYERS + 1
    ):


        print(
            f"\nLayer {layer}"
        )


        print(
            "STACK     =",
            stack_point(layer)
        )


        print(
            "APPROACH  =",
            stack_approach_point(layer)
        )


        print(
            "SAFE      =",
            stack_safe_point(layer)
        )


    # --------------------------------------------------------
    # 离线模式
    # --------------------------------------------------------

    if not ENABLE_HARDWARE:

        print(
            "\n仅完成离线规划。"
        )

        return


    # --------------------------------------------------------
    # 实机确认
    # --------------------------------------------------------

    print(
        "\n============================================"
    )

    print(
        "实机运行前确认："
    )

    print(
        f"1. TABLE_Z = {TABLE_Z:.1f} mm"
    )

    print(
        f"2. 第一层中心 = {TABLE_Z + BLOCK_SIZE / 2.0:.1f} mm"
    )

    print(
        "3. J6=0° 张开"
    )

    print(
        "4. J6=-30° 闭合"
    )

    print(
        "5. 大范围搬运允许夹爪倾斜"
    )

    print(
        "6. 最后 15 mm 强制进入近水平状态"
    )

    print(
        "7. 第一次仍建议 TARGET_LAYERS=1"
    )

    print(
        "============================================"
    )


    answer = input(
        "\n输入 YES 开始控制真机："
    )


    if (
        answer.strip().upper()
        != "YES"
    ):

        print(
            "取消运行。"
        )

        return


    ser = init_serial()


    if ser is None:

        return


    try:

        execute_plan(
            ser,
            plan
        )


        print(
            "\n============================================"
        )

        print(
            f"实验完成："
            f"{TARGET_LAYERS} 层码垛 "
            f"+ "
            f"{TARGET_LAYERS} 层回收"
        )

        print(
            "============================================"
        )


    except KeyboardInterrupt:

        print(
            "\n[用户中断]"
        )

        print(
            "保持当前姿态。"
        )


    finally:

        if (
            ser is not None
            and ser.is_open
        ):

            ser.close()

            print(
                "[硬件] 串口已关闭。"
            )


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":

    main()
