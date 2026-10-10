import time
import serial
import numpy as np
import roboticstoolbox as rtb
import matplotlib

try:
    matplotlib.use('TkAgg')
except Exception:
    pass

import matplotlib.pyplot as plt


# ============================================================
# 预提供函数：通信帧打包
# ============================================================
def pack_frame(q_rad):
    """
    将 6 轴关节弧度数组打包成 STM32 控制协议的 16 字节通信帧。
    """
    tx = bytearray(16)

    tx[0] = 0xAA

    for i in range(6):
        val = int(q_rad[i] * 1000)

        # 这里只防止 int16 溢出，不代表机械限位
        val = max(min(val, 32767), -32768)

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
# 预提供函数：串口初始化
# ============================================================
def init_serial(port="COM5", baudrate=115200):
    """
    初始化 STM32 串口，并发送一次六轴全零目标。
    """
    ser = serial.Serial()

    ser.port = port
    ser.baudrate = baudrate
    ser.timeout = 1
    ser.dtr = False
    ser.rts = False

    try:
        ser.open()

        print(f"[硬件提示] 成功建立控制板串口通信: {port}")

        time.sleep(1.0)

        print("[硬件提示] 正在下发六轴零位目标...")

        ser.write(
            pack_frame(
                np.zeros(6, dtype=np.float64)
            )
        )

        ser.flush()

        time.sleep(1.5)

        print("[硬件提示] 零位目标已下发，请人工确认实际姿态。")

        return ser

    except Exception as e:
        print(
            f"[仿真提示] 串口 {port} 初始化失败 ({e})，"
            "继续进行计算和动画，不发送实机指令。"
        )

        return None


# ============================================================
# 预提供函数：关节线性插值
# ============================================================
def move_joint_interpolated(
    ser,
    q_curr_deg,
    q_target_deg,
    duration=2.0,
    steps=40
):
    """
    从当前指令角线性插值到目标指令角。
    """

    q_start_rad = np.deg2rad(q_curr_deg)
    q_target_rad = np.deg2rad(q_target_deg)

    dt = duration / steps

    for k in range(1, steps + 1):

        ratio = k / steps

        q_k = (
            q_start_rad
            + ratio * (q_target_rad - q_start_rad)
        )

        if ser and ser.is_open:
            ser.write(pack_frame(q_k))
            ser.flush()

        time.sleep(dt)

    return q_target_deg.copy()


# ============================================================
# 预提供函数：构建 RoboticsToolbox 机器人模型
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
            offset=p.get('offset', 0.0)
        )
        for p in dh_params
    ]

    return rtb.DHRobot(
        links,
        name=name
    )


# ============================================================
# 预提供函数：机械臂运动动画
# ============================================================
def animate_robot_motion(
    robot,
    q_start_deg,
    q_target_deg,
    steps=30,
    dt=0.04
):
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

    robot.plot(
        traj.q,
        block=False,
        limits=[
            -150, 450,
            -250, 250,
            0, 500
        ],
        jointaxes=True,
        shadow=True,
        dt=dt
    )

    plt.pause(1.5)


# ============================================================
# 学生完成部分 1：
# 标准 D-H 单级齐次变换矩阵
# ============================================================
def calc_dh_matrix(theta, alpha, d, a):
    """
    返回标准 D-H 4×4 齐次变换矩阵。

    theta, alpha:
        单位 rad

    d, a:
        单位 mm
    """

    ct = np.cos(theta)
    st = np.sin(theta)

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T = np.array(
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

    return T


# ============================================================
# 学生完成部分 2：
# 实验一 正运动学
# ============================================================
def run_fk_experiment():

    print("\n========================================")
    print("实验一：机械臂运动学正解 FK")
    print("========================================")


    # --------------------------------------------------------
    # 1. 标准 D-H 参数
    # --------------------------------------------------------

    dh_params_servo = [

        # Joint 1
        {
            'd': 108.5,
            'a': 0.0,
            'alpha': np.pi / 2,
            'offset': 0.0
        },

        # Joint 2
        {
            'd': 0.0,
            'a': 105.0,
            'alpha': 0.0,
            'offset': np.pi / 2
        },

        # Joint 3
        {
            'd': 0.0,
            'a': 97.5,
            'alpha': 0.0,
            'offset': 0.0
        },

        # Joint 4
        {
            'd': 0.0,
            'a': 28.22,
            'alpha': np.pi / 2,
            'offset': np.pi / 2
        },

        # Joint 5
        {
            'd': 172.70,
            'a': 0.0,
            'alpha': 0.0,
            'offset': 0.0
        }
    ]


    # --------------------------------------------------------
    # 2. 指导书测试关节角
    # --------------------------------------------------------

    q_test_deg = np.array(
        [
            0.0,
            0.0,
            20.0,
            -45.0,
            15.0
        ],
        dtype=np.float64
    )

    print("\n测试关节角：")
    print(
        "q =",
        q_test_deg,
        "deg"
    )


    # --------------------------------------------------------
    # 3. 手写 D-H 矩阵连乘
    # --------------------------------------------------------

    T_manual = np.eye(
        4,
        dtype=np.float64
    )

    print("\n========================================")
    print("逐级 D-H 矩阵")
    print("========================================")

    for i, p in enumerate(
        dh_params_servo
    ):

        # 当前关节角
        q_rad = np.deg2rad(
            q_test_deg[i]
        )

        # D-H 实际 theta = q + offset
        theta_actual = (
            q_rad
            + p['offset']
        )

        # 单级变换矩阵
        T_link = calc_dh_matrix(
            theta_actual,
            p['alpha'],
            p['d'],
            p['a']
        )

        print(
            f"\nT_{i}{i + 1} ="
        )

        print(
            np.round(
                T_link,
                4
            )
        )

        # 连续右乘
        T_manual = (
            T_manual
            @ T_link
        )


    # --------------------------------------------------------
    # 4. 输出手写 FK
    # --------------------------------------------------------

    print("\n========================================")
    print("手写矩阵连乘结果 T_manual")
    print("========================================")

    print(
        np.round(
            T_manual,
            4
        )
    )


    pos_manual = (
        T_manual[:3, 3]
    )

    print("\n手写 FK 末端坐标：")

    print(
        f"X = {pos_manual[0]:.4f} mm"
    )

    print(
        f"Y = {pos_manual[1]:.4f} mm"
    )

    print(
        f"Z = {pos_manual[2]:.4f} mm"
    )


    # --------------------------------------------------------
    # 5. RoboticsToolbox 建模
    # --------------------------------------------------------

    robot = build_robot_model(
        dh_params_servo
    )

    q_test_rad = np.deg2rad(
        q_test_deg
    )

    T_toolbox = robot.fkine(
        q_test_rad
    )


    print("\n========================================")
    print("RoboticsToolbox FK")
    print("========================================")

    print(
        np.round(
            T_toolbox.A,
            4
        )
    )


    print(
        "\nRoboticsToolbox 末端位置："
    )

    print(
        f"X = {T_toolbox.t[0]:.4f} mm"
    )

    print(
        f"Y = {T_toolbox.t[1]:.4f} mm"
    )

    print(
        f"Z = {T_toolbox.t[2]:.4f} mm"
    )


    # --------------------------------------------------------
    # 6. 数值误差验证
    # --------------------------------------------------------

    diff = np.max(
        np.abs(
            T_manual
            - T_toolbox.A
        )
    )

    print("\n========================================")
    print("手写 FK 与 RoboticsToolbox 对比")
    print("========================================")

    print(
        f"最大绝对误差 = "
        f"{diff:.12e}"
    )

    if diff < 1e-6:

        print(
            "[验证通过] "
            "手写 D-H 连乘与 RoboticsToolbox 一致。"
        )

    else:

        print(
            "[验证失败] "
            "请检查 D-H 参数或矩阵实现。"
        )


    # --------------------------------------------------------
    # 7. 构造六轴控制目标
    # --------------------------------------------------------

    # 第六轴为夹爪，不参与 D-H 模型
    q_zero_deg = np.zeros(
        6,
        dtype=np.float64
    )

    q_target_deg = np.array(
        [
            q_test_deg[0],
            q_test_deg[1],
            q_test_deg[2],
            q_test_deg[3],
            q_test_deg[4],
            0.0
        ],
        dtype=np.float64
    )


    # --------------------------------------------------------
    # 8. 三维动画
    # --------------------------------------------------------

    print("\n========================================")
    print("播放机械臂三维运动动画")
    print("========================================")

    try:

        animate_robot_motion(
            robot,
            q_zero_deg,
            q_target_deg,
            steps=30,
            dt=0.04
        )

    except Exception as e:

        print(
            "[动画提示] 动画显示失败：",
            e
        )

        print(
            "[动画提示] 不影响 FK 数值实验和实机控制。"
        )


    # --------------------------------------------------------
    # 9. 串口连接
    # --------------------------------------------------------

    print("\n========================================")
    print("初始化实机")
    print("========================================")

    ser = init_serial(
        port="COM5",
        baudrate=115200
    )


    # --------------------------------------------------------
    # 10. 实机运动
    # --------------------------------------------------------

    if ser is not None:

        current_pose = (
            q_zero_deg.copy()
        )

        try:

            print(
                "\n[实机] "
                "开始运动到实验目标姿态..."
            )

            print(
                "[实机] 目标角度："
            )

            print(
                q_target_deg
            )

            current_pose = (
                move_joint_interpolated(
                    ser,
                    current_pose,
                    q_target_deg,
                    duration=2.5,
                    steps=40
                )
            )

            print(
                "[实机] "
                "已发送目标姿态。"
            )

            time.sleep(1.5)


            # ----------------------------------------------
            # 返回零位
            # ----------------------------------------------

            print(
                "\n[实机] "
                "开始返回零位..."
            )

            current_pose = (
                move_joint_interpolated(
                    ser,
                    current_pose,
                    q_zero_deg,
                    duration=2.0,
                    steps=40
                )
            )

            print(
                "[实机] "
                "零位目标已发送。"
            )

            time.sleep(1.0)


        finally:

            if (
                ser is not None
                and ser.is_open
            ):

                ser.close()

                print(
                    "[硬件提示] "
                    "串口已安全关闭。"
                )

    else:

        print(
            "\n[仿真模式] "
            "未连接实机，只完成 FK 计算和动画。"
        )


    print("\n========================================")
    print("实验一结束")
    print("========================================")


# ============================================================
# 主程序入口
# ============================================================
if __name__ == '__main__':
    run_fk_experiment()