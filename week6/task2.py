import time
import serial
import numpy as np
import spatialmath as sm
import roboticstoolbox as rtb
import matplotlib

try:
    matplotlib.use('TkAgg')
except Exception:
    pass

import matplotlib.pyplot as plt


# ============================================================
# 基本配置
# ============================================================

SERIAL_PORT = "COM5"
BAUDRATE = 115200

# 当前机械臂舵机方向。
# 你的实验一直接发送 q 就能够正常运动，因此这里先使用恒等映射。
SERVO_DIRECTION_MAP = np.array(
    [1, 1, 1, 1, 1, 1],
    dtype=np.float64
)

# 软件限位
# 前 4 轴按当前实验机械臂的实际范围进行保护
JOINT_LIMITS_DEG = [
    (-90.0, 90.0),   # J1
    (-90.0, 90.0),   # J2
    (-90.0, 90.0),   # J3
    (-90.0, 25.0),   # J4
]

# 是否执行 A 点上摆解的实机演示
ENABLE_HARDWARE_A = True


# ============================================================
# 通信帧
# ============================================================

def pack_frame(q_rad):
    """
    将 6 轴弧度数据打包成 STM32 16 字节控制帧。
    """

    tx = bytearray(16)

    tx[0] = 0xAA

    for i in range(6):

        val = int(q_rad[i] * 1000)

        val = max(
            min(val, 32767),
            -32768
        )

        tx[1 + i * 2] = (val >> 8) & 0xFF
        tx[2 + i * 2] = val & 0xFF

    tx[13] = 0x01

    check = 0

    for b in tx[:14]:
        check ^= b

    tx[14] = check
    tx[15] = 0xBB

    return tx


# ============================================================
# 串口初始化
# ============================================================

def init_serial(port="COM5", baudrate=115200):

    ser = serial.Serial()

    ser.port = port
    ser.baudrate = baudrate
    ser.timeout = 1

    ser.dtr = False
    ser.rts = False

    try:

        ser.open()

        print(
            f"\n[硬件] 成功建立串口通信: {port}"
        )

        time.sleep(1.0)

        print(
            "[硬件] 正在发送六轴零位目标..."
        )

        ser.write(
            pack_frame(
                np.zeros(
                    6,
                    dtype=np.float64
                )
            )
        )

        ser.flush()

        time.sleep(1.5)

        print(
            "[硬件] 零位目标已发送。"
        )

        return ser

    except Exception as e:

        print(
            f"[提示] 串口初始化失败: {e}"
        )

        print(
            "[提示] 后续仅执行数值计算和仿真。"
        )

        return None


# ============================================================
# 实机关节插值
# ============================================================

def move_joint_interpolated(
    ser,
    q_curr_deg,
    q_target_deg,
    duration=2.0,
    steps=40
):

    q_start_rad = np.deg2rad(
        q_curr_deg
    )

    q_target_rad = np.deg2rad(
        q_target_deg
    )

    dt = duration / steps

    for k in range(1, steps + 1):

        ratio = k / steps

        q_k = (
            q_start_rad
            + ratio
            * (
                q_target_rad
                - q_start_rad
            )
        )

        if ser and ser.is_open:

            q_servo = (
                q_k
                * SERVO_DIRECTION_MAP
            )

            ser.write(
                pack_frame(q_servo)
            )

            ser.flush()

        time.sleep(dt)

    return q_target_deg.copy()


# ============================================================
# RoboticsToolbox 机器人模型
# ============================================================

def build_robot_model(
    dh_params,
    name="Physical_Servo_Arm"
):

    links = [

        rtb.RevoluteDH(
            d=p['d'],
            a=p['a'],
            alpha=p['alpha'],
            offset=p.get(
                'offset',
                0.0
            )
        )

        for p in dh_params
    ]

    return rtb.DHRobot(
        links,
        name=name
    )


# ============================================================
# 动画
# ============================================================

def animate_robot_motion(
    robot,
    q_start_deg,
    q_target_deg,
    title="IK Solution",
    steps=40,
    dt=0.04
):

    print(
        f"\n>> [虚拟仿真] {title}"
    )

    q_start_rad = np.deg2rad(
        q_start_deg[:5]
    )

    q_target_rad = np.deg2rad(
        q_target_deg[:5]
    )

    traj = rtb.jtraj(
        q_start_rad,
        q_target_rad,
        steps
    )

    try:

        robot.plot(
            traj.q,
            block=False,

            # B 点下摆可能进入 z < 0
            limits=[
                -250, 450,
                -300, 300,
                -150, 550
            ],

            jointaxes=True,
            eeframe=True,
            shadow=True,
            dt=dt
        )

        plt.pause(1.5)

    except Exception as e:

        print(
            "[动画提示] 动画播放失败：",
            e
        )


# ============================================================
# 标准 D-H 单级矩阵
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
# 使用 FK 验证 IK 解
# ============================================================

def verify_fk_position(
    dh_params,
    q_deg
):

    T = np.eye(
        4,
        dtype=np.float64
    )

    for i, p in enumerate(
        dh_params
    ):

        theta = (
            np.deg2rad(
                q_deg[i]
            )
            + p.get(
                'offset',
                0.0
            )
        )

        T_link = calc_dh_matrix(
            theta,
            p['alpha'],
            p['d'],
            p['a']
        )

        T = T @ T_link

    return T[:3, 3]


# ============================================================
# 求机械臂 D-H 各坐标原点中的最低高度
# ============================================================

def minimum_dh_frame_height(
    dh_params,
    q_deg
):

    T = np.eye(
        4,
        dtype=np.float64
    )

    heights = [0.0]

    for i, p in enumerate(
        dh_params
    ):

        theta = (
            np.deg2rad(
                q_deg[i]
            )
            + p.get(
                'offset',
                0.0
            )
        )

        T_link = calc_dh_matrix(
            theta,
            p['alpha'],
            p['d'],
            p['a']
        )

        T = T @ T_link

        heights.append(
            T[2, 3]
        )

    return float(
        np.min(heights)
    )


# ============================================================
# 机械关节限位检查
# ============================================================

def check_joint_limits(
    q_deg
):

    valid = True

    print(
        "\n关节限位检查："
    )

    for i, (
        q,
        limits
    ) in enumerate(
        zip(
            q_deg[:4],
            JOINT_LIMITS_DEG
        )
    ):

        low, high = limits

        ok = (
            low <= q <= high
        )

        status = (
            "OK"
            if ok
            else "超限"
        )

        print(
            f"J{i+1}: "
            f"{q:8.2f}°   "
            f"允许 [{low:.1f}, {high:.1f}]°   "
            f"{status}"
        )

        if not ok:
            valid = False

    return valid


# ============================================================
# 输出单个 IK 解
# ============================================================

def analyse_solution(
    name,
    result,
    target_xyz,
    dh_params
):

    print(
        f"\n----- {name} -----"
    )

    print(
        f"success = {result.success}"
    )

    if not result.success:

        print(
            "该初值未获得有效 IK 解。"
        )

        return None

    # 原始 IK 解
    q_rad = np.array(
        result.q,
        dtype=np.float64
    )

    q_deg_raw = np.rad2deg(
        q_rad
    )

    # 按指导书保留两位小数后做 FK 验算
    q_deg = np.round(
        q_deg_raw,
        2
    )

    print(
        "IK 关节角 (deg)："
    )

    print(
        q_deg
    )

    # FK 反向验证
    pos_fk = verify_fk_position(
        dh_params,
        q_deg
    )

    error = np.linalg.norm(
        pos_fk
        - target_xyz
    )

    print(
        "舍入角度 FK 位置："
    )

    print(
        "X = %.4f mm, "
        "Y = %.4f mm, "
        "Z = %.4f mm"
        % tuple(pos_fk)
    )

    print(
        f"位置误差 = "
        f"{error:.6f} mm"
    )

    # 最低 D-H 坐标原点
    min_z = (
        minimum_dh_frame_height(
            dh_params,
            q_deg
        )
    )

    print(
        f"最低 D-H 原点高度 "
        f"z_min = {min_z:.4f} mm"
    )

    return {
        "q_rad": q_rad,
        "q_deg": q_deg,
        "position": pos_fk,
        "error": error,
        "min_z": min_z
    }


# ============================================================
# 实验二主函数
# ============================================================

def run_ik_experiment():

    print(
        "\n"
        "============================================"
    )
    print(
        "实验二：机械臂逆运动学 IK"
    )
    print(
        "============================================"
    )


    # ========================================================
    # 1. D-H 参数
    # ========================================================

    dh_params_servo = [

        {
            'd': 108.5,
            'a': 0.0,
            'alpha': np.pi / 2,
            'offset': 0.0
        },

        {
            'd': 0.0,
            'a': 105.0,
            'alpha': 0.0,
            'offset': np.pi / 2
        },

        {
            'd': 0.0,
            'a': 97.5,
            'alpha': 0.0,
            'offset': 0.0
        },

        {
            'd': 0.0,
            'a': 28.22,
            'alpha': np.pi / 2,
            'offset': np.pi / 2
        },

        {
            'd': 172.70,
            'a': 0.0,
            'alpha': 0.0,
            'offset': 0.0
        }
    ]


    # ========================================================
    # 2. 建立 RoboticsToolbox 模型
    # ========================================================

    robot = build_robot_model(
        dh_params_servo
    )


    # ========================================================
    # 3. 指导书给定目标 A / B / C
    # ========================================================

    targets_xyz = {

        "A": np.array(
            [120.0, 80.0, 440.0],
            dtype=np.float64
        ),

        "B": np.array(
            [160.0, 0.0, 80.0],
            dtype=np.float64
        ),

        "C": np.array(
            [350.0, 200.0, 420.0],
            dtype=np.float64
        )
    }


    # ========================================================
    # 4. NumPy 构造纯平移齐次矩阵，再转 SE3
    # ========================================================

    targets_se3 = {}

    for name, xyz in targets_xyz.items():

        X, Y, Z = xyz

        T_mat = np.array(
            [
                [1.0, 0.0, 0.0, X],
                [0.0, 1.0, 0.0, Y],
                [0.0, 0.0, 1.0, Z],
                [0.0, 0.0, 0.0, 1.0]
            ],
            dtype=np.float64
        )

        targets_se3[name] = (
            sm.SE3(T_mat)
        )


    # ========================================================
    # 5. 两组 IK 搜索初值
    # ========================================================

    q0_up = np.deg2rad(
        [
            0.0,
            45.0,
            -80.0,
            -80.0,
            0.0
        ]
    )

    q0_down = np.deg2rad(
        [
            0.0,
            -110.0,
            10.0,
            110.0,
            0.0
        ]
    )

    # 只要求 XYZ
    ik_mask = [
        1, 1, 1,
        0, 0, 0
    ]


    # 用于保存结果
    solutions = {}


    # ========================================================
    # 6. 对 A、B、C 分别求两组 IK
    # ========================================================

    for target_name in [
        "A",
        "B",
        "C"
    ]:

        print(
            "\n\n"
            "============================================"
        )

        print(
            f"Target {target_name} | "
            f"XYZ = {targets_xyz[target_name]} mm"
        )

        print(
            "============================================"
        )

        target = (
            targets_se3[target_name]
        )


        # -------------------------
        # 上摆初值
        # -------------------------

        result_up = (
            robot.ikine_LM(
                target,
                q0=q0_up,
                mask=ik_mask
            )
        )


        # -------------------------
        # 下摆初值
        # -------------------------

        result_down = (
            robot.ikine_LM(
                target,
                q0=q0_down,
                mask=ik_mask
            )
        )


        # -------------------------
        # 分析两组解
        # -------------------------

        up_data = analyse_solution(
            f"Target {target_name} - 上摆初值",
            result_up,
            targets_xyz[target_name],
            dh_params_servo
        )

        down_data = analyse_solution(
            f"Target {target_name} - 下摆初值",
            result_down,
            targets_xyz[target_name],
            dh_params_servo
        )


        solutions[target_name] = {
            "up": up_data,
            "down": down_data
        }


        # -------------------------
        # 两组解差异
        # -------------------------

        if (
            up_data is not None
            and down_data is not None
        ):

            dist = np.linalg.norm(
                up_data["q_rad"]
                - down_data["q_rad"]
            )

            print(
                "\n两组原始 IK 关节向量"
                f"二范数差 = {dist:.6f} rad"
            )

            if dist > 0.15:

                print(
                    "=> 两组解存在明显构型差异。"
                )

            else:

                print(
                    "=> 两组解较接近。"
                )


        # ====================================================
        # C 点额外做几何不可达论证
        # ====================================================

        if target_name == "C":

            target_distance = (
                np.linalg.norm(
                    targets_xyz["C"]
                )
            )

            conservative_bound = sum(
                abs(p['d'])
                + abs(p['a'])
                for p in dh_params_servo
            )

            print(
                "\n[C 点几何分析]"
            )

            print(
                "目标点距基座原点 = "
                f"{target_distance:.2f} mm"
            )

            print(
                "D-H 连杆长度保守上界 = "
                f"{conservative_bound:.2f} mm"
            )

            if (
                target_distance
                > conservative_bound
            ):

                print(
                    "=> C 点超过模型的保守几何距离上界，"
                    "属于几何不可达目标。"
                )


    # ========================================================
    # 7. A 点多解动画
    # ========================================================

    print(
        "\n\n"
        "============================================"
    )
    print(
        "A 点：观察两种不同 IK 构型"
    )
    print(
        "============================================"
    )

    zero5 = np.zeros(5)

    if solutions["A"]["up"] is not None:

        animate_robot_motion(
            robot,
            zero5,
            solutions["A"]["up"]["q_deg"],
            title="A 点 - 上摆解"
        )

        plt.close('all')


    if solutions["A"]["down"] is not None:

        animate_robot_motion(
            robot,
            zero5,
            solutions["A"]["down"]["q_deg"],
            title="A 点 - 下摆解"
        )

        plt.close('all')


    # ========================================================
    # 8. B 点动画与安全分析
    # ========================================================

    print(
        "\n\n"
        "============================================"
    )
    print(
        "B 点：数学解与物理可行性分析"
    )
    print(
        "============================================"
    )


    if solutions["B"]["up"] is not None:

        print(
            "\n[B 上摆] 软件限位检查："
        )

        check_joint_limits(
            solutions["B"]["up"]["q_deg"]
        )

        print(
            "\n[B 上摆] 仅播放动画，不发送实机。"
        )

        animate_robot_motion(
            robot,
            zero5,
            solutions["B"]["up"]["q_deg"],
            title="B 点 - 上摆解"
        )

        plt.close('all')


    if solutions["B"]["down"] is not None:

        print(
            "\n[B 下摆] 软件限位检查："
        )

        check_joint_limits(
            solutions["B"]["down"]["q_deg"]
        )

        print(
            "\n!!! B 点下摆仅允许虚拟仿真 !!!"
        )

        print(
            "最低 D-H 原点："
            f"{solutions['B']['down']['min_z']:.2f} mm"
        )

        if (
            solutions["B"]["down"]["min_z"]
            < 0
        ):

            print(
                "=> 模型骨架已进入 z < 0 区域，"
                "存在底座/桌面干涉风险。"
            )

        animate_robot_motion(
            robot,
            zero5,
            solutions["B"]["down"]["q_deg"],
            title="B 点 - 下摆危险解（仅仿真）"
        )

        plt.close('all')


    # ========================================================
    # 9. A 点上摆解实机安全检查
    # ========================================================

    print(
        "\n\n"
        "============================================"
    )
    print(
        "实机演示准备"
    )
    print(
        "============================================"
    )


    A_up = solutions["A"]["up"]

    hardware_safe = False

    if A_up is not None:

        print(
            "\n准备检查 A 点上摆解："
        )

        print(
            A_up["q_deg"]
        )

        hardware_safe = (
            check_joint_limits(
                A_up["q_deg"]
            )
        )

        if A_up["min_z"] < 0:

            hardware_safe = False

            print(
                "A 点存在 z < 0 的 D-H 原点，"
                "禁止实机执行。"
            )


    # ========================================================
    # 10. 实机执行 A 点上摆
    # ========================================================

    if (
        ENABLE_HARDWARE_A
        and hardware_safe
    ):

        print(
            "\n[A 点上摆] "
            "通过当前软件限位检查。"
        )

        ser = init_serial(
            port=SERIAL_PORT,
            baudrate=BAUDRATE
        )

        if ser is not None:

            current_pose = np.zeros(
                6,
                dtype=np.float64
            )

            q_target_6 = np.array(
                [
                    A_up["q_deg"][0],
                    A_up["q_deg"][1],
                    A_up["q_deg"][2],
                    A_up["q_deg"][3],
                    A_up["q_deg"][4],
                    0.0
                ],
                dtype=np.float64
            )

            try:

                print(
                    "\n[实机] "
                    "运动到 A 点上摆 IK 解..."
                )

                print(
                    "目标角度 =",
                    q_target_6
                )

                current_pose = (
                    move_joint_interpolated(
                        ser,
                        current_pose,
                        q_target_6,
                        duration=3.0,
                        steps=60
                    )
                )

                time.sleep(2.0)


                print(
                    "\n[实机] 返回零位..."
                )

                current_pose = (
                    move_joint_interpolated(
                        ser,
                        current_pose,
                        np.zeros(6),
                        duration=3.0,
                        steps=60
                    )
                )

                time.sleep(1.0)


            finally:

                if ser.is_open:

                    ser.close()

                    print(
                        "[硬件] 串口已关闭。"
                    )

    else:

        print(
            "\n未执行实机 A 点运动。"
        )

        if not hardware_safe:

            print(
                "原因：IK 解未通过当前软件安全检查。"
            )


    # ========================================================
    # 11. 实验总结
    # ========================================================

    print(
        "\n\n"
        "============================================"
    )
    print(
        "实验二运行结束"
    )
    print(
        "============================================"
    )

    print(
        """
应观察到：

1. A 点：
   两次 IK 均成功；
   得到两组不同关节角；
   FK 验算均接近目标 (120, 80, 440)。

2. B 点：
   两次 IK 可以获得数学解；
   但存在关节超限和/或 z<0 干涉；
   因此只做虚拟动画，不发送实机。

3. C 点：
   两组 IK 通常均失败；
   目标距离超过 D-H 模型保守几何上界；
   因此属于几何不可达目标。

4. 实机：
   仅执行通过检查的 A 点上摆解，
   随后返回零位。
"""
    )


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':

    run_ik_experiment()