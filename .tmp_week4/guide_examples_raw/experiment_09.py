import time
import serial
import numpy as np

#协议打包函数
def pack_frame(q_rad):
    """将 6 轴弧度数组打包为 16 字节控制帧"""
    tx = bytearray(16)
    tx[0] = 0xAA  # 帧头
    for i in range(6):
        val = int(q_rad[i] * 1000)
        val = max(min(val, 32767), -32768)  #幅值限幅
        tx[1 + i * 2] = (val >> 8) & 0xFF  #高8位(大端模式)
        tx[2 + i * 2] = val & 0xFF  # 低8位

    tx[13] = 0x01  #控制模式

    #计算前14字节XOR校验和
    check = 0
    for b in tx[:14]:
        check ^= b
    tx[14] = check
    tx[15] = 0xBB  #帧尾
    return tx


#线性插值平滑驱动函数
def move_joint_interpolated(ser, q_curr_deg, q_target_deg, duration=1.5, steps=30):
    """
    平滑驱动机械臂从当前姿态插值过渡到目标姿态
    :param ser: 已成功打开的 serial.Serial 串口对象
    :param q_curr_deg: 当前姿态角度数组 (度)
    :param q_target_deg: 目标姿态角度数组 (度)
    :param duration: 动作过渡总时间 (秒)
    :param steps: 离散插值细分步数
    """

    q_start_rad = np.deg2rad(q_curr_deg)
    q_target_rad = np.deg2rad(q_target_deg)
    dt = duration / steps

    for k in range(1, steps + 1):
        ratio = k / steps
        q_k = q_start_rad + ratio * (q_target_rad - q_start_rad)
        frame = pack_frame(q_k)
        ser.write(frame)
        ser.flush()
        time.sleep(dt)

    return q_target_deg.copy()


#实机控制主程序

def main():
    SERIAL_PORT = "COM5"  # 请实验人员根据实际设备端口号修改

    ser = serial.Serial()
    ser.port = SERIAL_PORT
    ser.baudrate = 115200
    ser.timeout = 1
    # 核心安全配置：禁用 DTR/RTS 规避硬件复位陷阱
    ser.dtr = False
    ser.rts = False

    try:
        ser.open()
        print(f"成功连接控制板串口: {SERIAL_PORT}")
        print("等待 STM32 底层系统就绪 (1.5 秒)...")
        time.sleep(1.5)

        #初始全零姿态 [J1, J2, J3, J4, J5, J6]
        current_pose = np.zeros(6, dtype=np.float64)

        #动作1:控制第一个关节(底座)旋转+45.0°
        pose_step1 = current_pose.copy()
        pose_step1[0] = 45.0
        print("\n>> 动作1:关节 1 (底座) 偏转 +45.0°")
        current_pose = move_joint_interpolated(ser, current_pose, pose_step1, duration=1.5)
        time.sleep(1.0)

        #动作2:控制第三个关节(肘部小臂)旋转+30.0°
        pose_step2 = current_pose.copy()
        pose_step2[2] = 30.0
        print("\n>> 动作2:关节 3 (肘部小臂) 抬升 +30.0°")
        current_pose = move_joint_interpolated(ser, current_pose, pose_step2, duration=1.5)
        time.sleep(1.0)

        #动作3:控制最后一个关节(夹爪开合)开合一次
        pose_gripper_open = current_pose.copy()
        pose_gripper_open[5] = 50.0
        print("\n动作3.1:夹爪张开 (+50.0°)")
        current_pose = move_joint_interpolated(ser, current_pose, pose_gripper_open, duration=1.0)
        time.sleep(1.0)

        #夹爪闭合复位至0.0°
        pose_gripper_close = current_pose.copy()
        pose_gripper_close[5] = 0.0
        print("动作3.2:夹爪闭合复位 (0.0°)")
        current_pose = move_joint_interpolated(ser, current_pose, pose_gripper_close, duration=1.0)
        time.sleep(1.0)

        #动作4:控制 1, 2, 3, 4, 5 号关节同时旋转到达复合目标姿态
        pose_multi_joints = np.array([30.0, -20.0, 35.0, -15.0, 20.0, 0.0], dtype=np.float64)
        print("\n>> 动作4:关节 1~5 协同多轴联动 ->", pose_multi_joints[:5])
        current_pose = move_joint_interpolated(ser, current_pose, pose_multi_joints, duration=2.0)
        time.sleep(2.0)

        #动作5:全机平滑复位至全零初始位
        pose_home = np.zeros(6, dtype=np.float64)
        print("\n>> 动作5:全机平滑复位至全零位")
        current_pose = move_joint_interpolated(ser, current_pose, pose_home, duration=2.0)

    except Exception as e:
        print(f"硬件驱动异常: {e}")
    finally:
        if ser.is_open:
            ser.close()
            print("串口物理链路已安全释放。")

if __name__ == '__main__':
    main()
