import time
import serial
import numpy as np



#协议组包与多轴同步插值函数

def pack_frame(q_rad):
    """打包 16 字节控制帧"""
    tx = bytearray(16)
    tx[0] = 0xAA
    for i in range(6):
        val = int(q_rad[i] * 1000)
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


def move_multiaxis_interpolated(ser, q_curr_deg, q_target_deg, duration=1.5, steps=30):
    """多轴协同平滑插值函数 (通用支持单轴与多轴状态过渡)"""
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



#硬件控制主程序
def main():
    SERIAL_PORT = "COM5"  # 请根据实际端口号修改
    ser = serial.Serial()
    ser.port = SERIAL_PORT
    ser.baudrate = 115200
    ser.timeout = 1
    ser.dtr = False
    ser.rts = False

    try:
        ser.open()
        print(f"成功连接控制板串口: {SERIAL_PORT}")
        print("等待 STM32 初始化就绪 (1.5 秒)...")
        time.sleep(1.5)

        #初始全零姿态
        current_pose = np.zeros(6, dtype=np.float64)

        
        #轮流移动 2、3、4 号关节 (单轴依次使能)

        #步骤1:仅移动 2 号舵机
        pose_step1 = current_pose.copy()
        pose_step1[1] = 30.0
        print("仅驱动 2 号舵机运动至 30.0° (大臂抬起，末端大范围公转)")
        current_pose = move_multiaxis_interpolated(ser, current_pose, pose_step1, duration=1.5)
        time.sleep(1.0)

        # 步骤2:仅移动 3 号舵机
        pose_step2 = current_pose.copy()
        pose_step2[2] = -45.0
        print("仅驱动 3 号舵机运动至 -45.0° (肘部折弯，末端中范围公转)")
        current_pose = move_multiaxis_interpolated(ser, current_pose, pose_step2, duration=1.5)
        time.sleep(1.0)

        # 步骤3:仅移动 4 号舵机
        pose_step3 = current_pose.copy()
        pose_step3[3] = 15.0
        print("仅驱动 4 号舵机运动至 15.0° (手腕调姿，末端纯自转调整)")
        current_pose = move_multiaxis_interpolated(ser, current_pose, pose_step3, duration=1.5)
        time.sleep(1.5)


        #同时联动 2、3、4 号关节 (多轴同步协同)
        #2、3、4 号舵机同时协同动作至前伸构型
        pose_step4 = current_pose.copy()
        pose_step4[1] = 45.0  # 2 号轴
        pose_step4[2] = -30.0  # 3 号轴
        pose_step4[3] = -15.0  # 4 号轴
        print("2、3、4 号舵机同时协同动作至前伸构型 [45°, -30°, -15°]")
        current_pose = move_multiaxis_interpolated(ser, current_pose, pose_step4, duration=2.0, steps=40)
        time.sleep(1.5)

        #2、3、4 号舵机同时协同复位
        pose_home = np.zeros(6, dtype=np.float64)
        print("2、3、4 号舵机同时协同平滑复位至基准位 [0°, 0°, 0°]")
        current_pose = move_multiaxis_interpolated(ser, current_pose, pose_home, duration=2.0, steps=40)


    except Exception as e:
        print(f"实物运行异常: {e}")
    finally:
        if ser.is_open:
            ser.close()
            print("物理串口通信链路已释放。")


if __name__ == '__main__':
    main()