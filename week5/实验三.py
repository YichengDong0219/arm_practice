import matplotlib
if __name__ == "__main__":
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass

import numpy as np
import roboticstoolbox as rtb


def build_and_render_servo_arm(dh_params, arm_name="ServoArm_Real", limits=[-150, 450, -300, 300, 0, 500]):
    links = [
        rtb.RevoluteDH(
            d=p['d'],
            a=p['a'],
            alpha=p['alpha'],
            offset=p.get('offset', 0.0)
        )
        for p in dh_params
    ]
    robot = rtb.DHRobot(links, name=arm_name)

    print(f"\n=== {arm_name} 实测标准 D-H 参数矩阵表 (单位: 毫米 / 弧度) ===")
    print(robot)

    q_deg = [p['theta'] for p in dh_params]
    q_rad = np.deg2rad(q_deg)
    print(f"当前输入控制关节角 q: {q_deg}")
    robot.plot(
        q_rad,
        block=True,
        limits=limits,
        jointaxes=True,
        shadow=True
    )
    return robot



#依据实测物理尺寸构建标准D-H参数列表
dh_params_servo = [
    {'theta':   0, 'alpha':  90, 'd': 108.5, 'a': 0,  'offset': 0 },
    {'theta':   0, 'alpha':  0, 'd': 0, 'a': 105, 'offset': 90 },
    {'theta':   0 , 'alpha': 0, 'd': 0,'a': 97.5,  'offset': 0 },
    {'theta':   0, 'alpha':  90, 'd': 0, 'a': 28.22,  'offset': 90 },
    {'theta':   0, 'alpha':  0 , 'd': 172.83, 'a': 0,  'offset': 0 }
]


if __name__ == "__main__":
    # 仅直接运行实验三时显示模型。实验四导入 D-H 参数时不应弹出并阻塞窗口。
    build_and_render_servo_arm(dh_params_servo, arm_name="Physical_Servo_Manipulator")
