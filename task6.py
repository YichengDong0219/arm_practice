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
    """平滑驱动机械臂从当前姿态插值过渡到目标姿态"""
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



#实验硬件驱动主程序
def main():
    SERIAL_PORT = "COM5"  #请实验人员根据实际端口号修改

    ser = serial.Serial()
    ser.port = SERIAL_PORT
    ser.baudrate = 115200
    ser.timeout = 1
    ser.dtr = False  #禁用DTR/RTS防止硬件复位卡死
    ser.rts = False

    try:
        ser.open()
        print(f"成功连接控制板串口: {SERIAL_PORT}")
        print("等待 STM32 底层系统就绪 (1.5 秒)...")
        time.sleep(1.5)

        # 初始全零姿态
        current_pose = np.zeros(6, dtype=np.float64)

        #目标点位1:关节4偏转+45°
        pose_1 = np.zeros(6, dtype=np.float64)
        pose_1[3] = 45.0  # 倒数第三个舵机(索引 3)
        print("\n>> 动作阶段1:关节 4 偏转 +45.0°")
        current_pose = move_joint_interpolated(ser, current_pose, pose_1, duration=1.5)
        time.sleep(1.0)

        #目标点位2:关节4偏转-30°
        pose_2 = np.zeros(6, dtype=np.float64)
        pose_2[3] = -30.0
        print("\n>> 动作阶段 2: 关节 4 偏转 -30.0°")
        current_pose = move_joint_interpolated(ser, current_pose, pose_2, duration=1.5)
        time.sleep(1.0)

        #目标点位3:原点复位
        pose_home = np.zeros(6, dtype=np.float64)
        print("\n>> 动作阶段 3: 复位至 0.0°")
        current_pose = move_joint_interpolated(ser, current_pose, pose_home, duration=1.5)
    except Exception as e:
        print(f"硬件驱动异常: {e}")
    finally:
        if ser.is_open:
            ser.close()
            print("串口链路已释放。")

if __name__ == '__main__':
    main()