import time
import serial
import numpy as np
import spatialmath as sm
import roboticstoolbox as rtb


# ============================================================
# 实验三：多层码垛与逐层回收
# ============================================================


# ============================================================
# 0. 串口配置
# ============================================================

SERIAL_PORT = "COM5"
BAUDRATE = 115200


# ============================================================
# 1. 场景几何参数
# ============================================================

# 木块尺寸
BLOCK_SIZE = 30.0


# ------------------------------------------------------------
# 桌面 / 木块放置平面在当前 D-H 基坐标系中的 z 坐标。
# 当前采用实机测量与调试后确认的标定值 52.0 mm；若机械臂或桌面
# 的安装位置改变，必须重新测量，不能仅由连杆 D-H 尺寸推算。
# ------------------------------------------------------------

TABLE_Z = 52.0


# ============================================================
# 2. PICK / RETURN 工位
# ============================================================

PICK_X = 200.0
PICK_Y = 100.0


# 第一块木块中心高度
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


# 抓取以后先向上提升
PICK_CLEARANCE = 80.0


PICK_SAFE_POINT = np.array(
    [
        PICK_X,
        PICK_Y,
        PICK_Z + PICK_CLEARANCE
    ],
    dtype=np.float64
)


# ============================================================
# 3. STACK 工位
# ============================================================

# 当前先沿用这个点。
#
# 如果高层连续出现 IK 无解，
# 后续再重新搜索 STACK_X。
STACK_X = 180.0
STACK_Y = -40.0


# 放置/抓取之后向上抬高
STACK_CLEARANCE = 60.0


# ============================================================
# 4. 层数
# ============================================================

# 第一次先跑 1 层
TARGET_LAYERS = 6


# 这里只是软件保护值，
# 并不代表实际一定能达到 12 层。
MAX_DESIGNED_LAYERS = 12


# ============================================================
# 5. 第六轴夹爪
# ============================================================

# J6 与其他舵机一样使用角度控制

# 张开
GRIP_OPEN_DEG = 0.0

# 闭合
GRIP_CLOSE_DEG = -50.0


# ============================================================
# 6. 实机控制
# ============================================================

ENABLE_HARDWARE = True

# 每完成一块后暂停，
# 方便人工补料 / 取走回收木块
WAIT_FOR_HUMAN = True


# ============================================================
# 7. 前五轴舵机方向
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
# 8. 前五轴机械限位
# ============================================================

JOINT_LIMITS_DEG = np.array(
    [
        [-90.0, 90.0],      # J1
        [-90.0, 90.0],      # J2
        [-90.0, 90.0],      # J3
        [-90.0, 25.0],      # J4
        [-180.0, 180.0],    # J5，目前保持 0°
    ],
    dtype=np.float64
)


# ============================================================
# 9. D-H 参数
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
# 10. 标准 D-H 矩阵
# ============================================================

def calc_dh_matrix(theta, alpha, d, a):

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
# 11. FK
# ============================================================

def fk_matrix(q_deg):

    T = np.eye(
        4,
        dtype=np.float64
    )

    for i, p in enumerate(DH_PARAMS):

        theta = (
            np.deg2rad(q_deg[i])
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
# 12. Robotics Toolbox 模型
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
# 13. XYZ -> SE3
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
# 14. 角度归一化
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
# 15. 关节限位检查
# ============================================================

def joint_limits_ok(q_deg):

    q_deg = np.asarray(
        q_deg,
        dtype=np.float64
    )

    for i in range(5):

        low = JOINT_LIMITS_DEG[i, 0]
        high = JOINT_LIMITS_DEG[i, 1]

        if not (
            low <= q_deg[i] <= high
        ):
            return False

    return True


# ============================================================
# 16. IK
# ============================================================

def solve_ik(
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

    # 只约束 XYZ
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


    # 多初值搜索
    seed_deg_list = [

        q_previous_deg,

        np.array(
            [
                q1_guess,
                -20.0,
                -80.0,
                -75.0,
                0.0
            ]
        ),

        np.array(
            [
                q1_guess,
                10.0,
                -80.0,
                -80.0,
                0.0
            ]
        ),

        np.array(
            [
                q1_guess,
                40.0,
                -80.0,
                -80.0,
                0.0
            ]
        ),

        np.array(
            [
                q1_guess,
                -40.0,
                -40.0,
                -40.0,
                0.0
            ]
        ),

        np.array(
            [
                q1_guess,
                0.0,
                -50.0,
                -50.0,
                0.0
            ]
        ),

        np.array(
            [
                q1_guess,
                -60.0,
                30.0,
                -50.0,
                0.0
            ]
        ),

        np.array(
            [
                q1_guess,
                60.0,
                -40.0,
                -70.0,
                0.0
            ]
        )
    ]


    candidates = []


    for seed_deg in seed_deg_list:

        try:

            result = robot.ikine_LM(
                target,
                q0=np.deg2rad(
                    seed_deg
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


        # 当前 J5 固定为 0°
        q_deg[4] = 0.0


        if not joint_limits_ok(
            q_deg
        ):
            continue


        fk_xyz = fk_position(
            q_deg
        )


        error = np.linalg.norm(
            fk_xyz - xyz
        )


        if error > 1.0:
            continue


        # 优先选择与上一姿态最接近的解
        score = np.linalg.norm(
            q_deg[:4]
            - q_previous_deg[:4]
        )


        candidates.append(
            (
                score,
                q_deg,
                error
            )
        )


    if not candidates:

        return None


    candidates.sort(
        key=lambda x: x[0]
    )


    _, q_best, _ = (
        candidates[0]
    )


    return q_best


# ============================================================
# 17. 第 k 层坐标
# ============================================================

def stack_point(layer):

    # 木块中心高度：
    #
    # TABLE_Z + (k - 1/2) * BLOCK_SIZE

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


def stack_safe_point(layer):

    p = stack_point(
        layer
    ).copy()

    p[2] += (
        STACK_CLEARANCE
    )

    return p


# ============================================================
# 18. 关节空间路径预检
# ============================================================

def check_joint_path(
    q_start_deg,
    q_end_deg,
    samples=40
):

    q_start_deg = np.asarray(
        q_start_deg,
        dtype=np.float64
    )

    q_end_deg = np.asarray(
        q_end_deg,
        dtype=np.float64
    )


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
                "插值路径出现关节超限"
            )


        xyz = fk_position(q)


        # D-H 末端不能明显进入桌面以下
        if xyz[2] < (
            TABLE_Z - 2.0
        ):

            return (
                False,
                (
                    "末端进入桌面以下："
                    f"z={xyz[2]:.2f} mm"
                )
            )


    return (
        True,
        "OK"
    )


# ============================================================
# 19. 生成完整运动计划
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


    def add_move(
        name,
        xyz
    ):

        nonlocal q_current


        q_target = solve_ik(
            robot,
            xyz,
            q_current
        )


        if q_target is None:

            raise RuntimeError(
                "\nIK 求解失败"
                f"\n动作：{name}"
                f"\n目标 XYZ：{xyz}"
            )


        ok, msg = check_joint_path(
            q_current,
            q_target
        )


        if not ok:

            raise RuntimeError(
                "\n轨迹预检失败"
                f"\n动作：{name}"
                f"\n原因：{msg}"
            )


        fk_xyz = fk_position(
            q_target
        )


        error = np.linalg.norm(
            fk_xyz - xyz
        )


        plan.append(
            {
                "type": "move",
                "name": name,
                "xyz": xyz.copy(),
                "q": q_target.copy(),
                "error": error
            }
        )


        q_current = (
            q_target.copy()
        )


    def add_grip(
        name,
        angle_deg
    ):

        plan.append(
            {
                "type": "grip",
                "name": name,
                "angle_deg": float(
                    angle_deg
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
    # A. 逐层码垛
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


        # 先到取料点上方
        add_move(
            f"L{layer} -> PICK_SAFE",
            PICK_SAFE_POINT
        )


        # 下降
        add_move(
            f"L{layer} -> PICK",
            PICK_POINT
        )


        # 夹紧
        add_grip(
            f"L{layer} 夹紧",
            GRIP_CLOSE_DEG
        )


        # 提升
        add_move(
            f"L{layer} PICK -> PICK_SAFE",
            PICK_SAFE_POINT
        )


        # 到码垛点上方
        add_move(
            f"L{layer} -> STACK_SAFE",
            stack_safe_point(
                layer
            )
        )


        # 下降到第 layer 层
        add_move(
            f"L{layer} -> STACK",
            stack_point(
                layer
            )
        )


        # 松开
        add_grip(
            f"L{layer} 松开",
            GRIP_OPEN_DEG
        )


        # 退出
        add_move(
            f"L{layer} STACK -> STACK_SAFE",
            stack_safe_point(
                layer
            )
        )


    # ========================================================
    # B. 等待开始回收
    # ========================================================

    add_wait(
        "码垛完成",
        (
            f"{layers} 层码垛完成。"
            "确认木块稳定后按 Enter 开始回收。"
        )
    )


    # ========================================================
    # C. 从最高层向下回收
    # ========================================================

    for layer in range(
        layers,
        0,
        -1
    ):


        add_move(
            f"回收 L{layer} -> STACK_SAFE",
            stack_safe_point(
                layer
            )
        )


        add_move(
            f"回收 L{layer} -> STACK",
            stack_point(
                layer
            )
        )


        add_grip(
            f"回收 L{layer} 夹紧",
            GRIP_CLOSE_DEG
        )


        add_move(
            f"回收 L{layer} STACK -> STACK_SAFE",
            stack_safe_point(
                layer
            )
        )


        add_move(
            f"回收 L{layer} -> PICK_SAFE",
            PICK_SAFE_POINT
        )


        add_move(
            f"回收 L{layer} -> PICK",
            PICK_POINT
        )


        add_grip(
            f"回收 L{layer} 松开",
            GRIP_OPEN_DEG
        )


        add_move(
            f"回收 L{layer} PICK -> PICK_SAFE",
            PICK_SAFE_POINT
        )


        add_wait(
            f"移走第 {layer} 块",
            (
                "请人工取走已经放回 PICK 工位的木块，"
                "然后按 Enter。"
            )
        )


    # ========================================================
    # D. 回零
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
# 20. 打印离线规划
# ============================================================

def print_plan(plan):

    print(
        "\n============================================"
    )

    print(
        "完整运动计划"
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
                ),
                "deg"
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


    print(
        "\n============================================"
    )

    print(
        "离线规划完成"
    )

    print(
        "============================================"
    )


# ============================================================
# 21. 六轴串口控制帧
#
# J1~J6 均为角度舵机
#
# 输入：
#     rad
#
# STM32 协议：
#     int(rad * 1000)
# ============================================================

def pack_frame(q6_rad):

    q6_rad = np.asarray(
        q6_rad,
        dtype=np.float64
    )


    if len(q6_rad) != 6:

        raise ValueError(
            "必须传入 J1~J6 六个关节"
        )


    tx = bytearray(16)


    # 帧头
    tx[0] = 0xAA


    # 六轴
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
            1 + i * 2
        ] = (
            val >> 8
        ) & 0xFF


        tx[
            2 + i * 2
        ] = (
            val
        ) & 0xFF


    # 控制模式
    tx[13] = 0x01


    # XOR 校验
    check = 0

    for b in tx[:14]:

        check ^= b


    tx[14] = check


    # 帧尾
    tx[15] = 0xBB


    return tx


# ============================================================
# 22. 构造六轴目标
# ============================================================

def build_six_axis_command(
    q5_deg,
    gripper_deg
):

    q5_deg = np.asarray(
        q5_deg,
        dtype=np.float64
    )


    if len(q5_deg) != 5:

        raise ValueError(
            "q5_deg 必须为五轴数组"
        )


    # 前五轴
    q5_rad = np.deg2rad(
        q5_deg
    )


    q5_rad *= (
        SERVO_DIRECTION_MAP
    )


    # 第六轴夹爪
    gripper_rad = np.deg2rad(
        gripper_deg
    )


    # 六轴合并
    q6_rad = np.zeros(
        6,
        dtype=np.float64
    )


    q6_rad[:5] = q5_rad

    q6_rad[5] = (
        gripper_rad
    )


    return q6_rad


# ============================================================
# 23. 串口初始化
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
            f"\n[硬件] 已连接 "
            f"{SERIAL_PORT}"
        )


        time.sleep(1.0)


        # J1~J5 = 0°
        # J6 = 0° OPEN
        q5_zero = np.zeros(
            5,
            dtype=np.float64
        )


        q6_rad = (
            build_six_axis_command(
                q5_zero,
                GRIP_OPEN_DEG
            )
        )


        ser.write(
            pack_frame(
                q6_rad
            )
        )


        ser.flush()


        print(
            "[硬件] 初始状态："
            "J1~J5=0°，"
            f"J6={GRIP_OPEN_DEG:.1f}°"
        )


        time.sleep(1.5)


        return ser


    except Exception as e:

        print(
            f"[错误] 串口初始化失败：{e}"
        )


        return None


# ============================================================
# 24. 前五轴平滑插值
#
# J6 保持当前状态
# ============================================================

def move_joint_interpolated(
    ser,
    q_start_deg,
    q_target_deg,
    gripper_deg,
    duration=2.0,
    steps=40
):

    q_start_deg = np.asarray(
        q_start_deg,
        dtype=np.float64
    )


    q_target_deg = np.asarray(
        q_target_deg,
        dtype=np.float64
    )


    dt = (
        duration / steps
    )


    for k in range(
        1,
        steps + 1
    ):


        ratio = (
            k / steps
        )


        q5_deg = (
            q_start_deg
            + ratio
            * (
                q_target_deg
                - q_start_deg
            )
        )


        q6_rad = (
            build_six_axis_command(
                q5_deg,
                gripper_deg
            )
        )


        ser.write(
            pack_frame(
                q6_rad
            )
        )


        ser.flush()


        time.sleep(dt)


# ============================================================
# 25. 夹爪 J6 平滑控制
# ============================================================

def set_gripper(
    ser,
    q_current_deg,
    current_gripper_deg,
    target_gripper_deg,
    duration=1.0,
    steps=20
):

    q_current_deg = np.asarray(
        q_current_deg,
        dtype=np.float64
    )


    print(
        f"[夹爪] J6: "
        f"{current_gripper_deg:.1f}°"
        f" -> "
        f"{target_gripper_deg:.1f}°"
    )


    dt = (
        duration / steps
    )


    for k in range(
        1,
        steps + 1
    ):


        ratio = (
            k / steps
        )


        gripper_deg = (
            current_gripper_deg
            + ratio
            * (
                target_gripper_deg
                - current_gripper_deg
            )
        )


        q6_rad = (
            build_six_axis_command(
                q_current_deg,
                gripper_deg
            )
        )


        ser.write(
            pack_frame(
                q6_rad
            )
        )


        ser.flush()


        time.sleep(dt)


    time.sleep(0.5)


    return float(
        target_gripper_deg
    )


# ============================================================
# 26. 执行完整计划
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


        # ====================================================
        # 人工等待
        # ====================================================

        if step["type"] == "wait":


            if WAIT_FOR_HUMAN:

                input(
                    "\n"
                    + step["message"]
                    + "\n"
                )


            continue


        # ====================================================
        # J1~J5
        # ====================================================

        if step["type"] == "move":


            q_target = step["q"]


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
                    q_target,
                    2
                )
            )


            print(
                f"J6 保持 "
                f"{gripper_deg:.1f}°"
            )


            move_joint_interpolated(
                ser,
                q_current,
                q_target,
                gripper_deg,
                duration=2.0,
                steps=40
            )


            q_current = (
                q_target.copy()
            )


            continue


        # ====================================================
        # J6
        # ====================================================

        if step["type"] == "grip":


            target_gripper_deg = (
                step["angle_deg"]
            )


            gripper_deg = (
                set_gripper(
                    ser,
                    q_current,
                    gripper_deg,
                    target_gripper_deg,
                    duration=1.0,
                    steps=20
                )
            )


            continue


        # ====================================================
        # HOME
        # ====================================================

        if step["type"] == "home":


            # 回零前先打开夹爪
            if abs(
                gripper_deg
                - GRIP_OPEN_DEG
            ) > 1e-6:


                gripper_deg = (
                    set_gripper(
                        ser,
                        q_current,
                        gripper_deg,
                        GRIP_OPEN_DEG
                    )
                )


            q_target = step["q"]


            print(
                "机械臂返回零位..."
            )


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
# 27. 主函数
# ============================================================

def main():

    print(
        "\n============================================"
    )

    print(
        "实验三：多层码垛与逐层回收"
    )

    print(
        "============================================"
    )


    print(
        f"\nTABLE_Z = "
        f"{TABLE_Z:.1f} mm"
    )


    print(
        f"BLOCK_SIZE = "
        f"{BLOCK_SIZE:.1f} mm"
    )


    print(
        f"第一层理论中心高度 = "
        f"{TABLE_Z + BLOCK_SIZE / 2:.1f} mm"
    )


    print(
        "PICK =",
        PICK_POINT
    )


    print(
        "PICK_SAFE =",
        PICK_SAFE_POINT
    )


    print(
        f"STACK XY = "
        f"({STACK_X:.1f}, "
        f"{STACK_Y:.1f})"
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
            f"1~{MAX_DESIGNED_LAYERS} 之间"
        )


    # ========================================================
    # 建模
    # ========================================================

    robot = build_robot_model()


    # ========================================================
    # 离线规划
    # ========================================================

    print(
        "\n[阶段 1] "
        "生成完整码垛 + 回收轨迹..."
    )


    plan = build_motion_plan(
        robot,
        TARGET_LAYERS
    )


    print_plan(
        plan
    )


    # ========================================================
    # 打印层高
    # ========================================================

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
            f"Layer {layer:2d}:"
        )


        print(
            "  STACK =",
            np.round(
                stack_point(layer),
                2
            )
        )


        print(
            "  SAFE  =",
            np.round(
                stack_safe_point(layer),
                2
            )
        )


    # ========================================================
    # 只做仿真
    # ========================================================

    if not ENABLE_HARDWARE:


        print(
            "\nENABLE_HARDWARE=False"
        )


        print(
            "未连接真机。"
        )


        return


    # ========================================================
    # 真机确认
    # ========================================================

    print(
        "\n============================================"
    )


    print(
        "真机运行前确认："
    )


    print(
        f"1. 当前 TABLE_Z = {TABLE_Z:.1f} mm"
    )


    print(
        f"2. 第一层目标高度 = {TABLE_Z + BLOCK_SIZE / 2.0:.1f} mm"
    )


    print(
        "3. PICK 工位已放置木块"
    )


    print(
        "4. J6=0° 为张开"
    )


    print(
        "5. J6=-30° 为闭合"
    )


    print(
        "6. 第一次建议 TARGET_LAYERS=1"
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
            "取消实机运行。"
        )

        return


    # ========================================================
    # 串口
    # ========================================================

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
            f"{TARGET_LAYERS} 层完整回收"
        )


        print(
            "============================================"
        )


    except KeyboardInterrupt:


        print(
            "\n[用户中断]"
        )


        print(
            "机械臂保持当前位置。"
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
# 28. 程序入口
# ============================================================

if __name__ == "__main__":

    main()
