import numpy as np


#定义基础单轴旋转矩阵构造函数
def rot_x(deg):
    """构建绕 X 轴旋转矩阵 (输入参数 deg 为角度制)"""
    r = np.deg2rad(deg)
    return np.array([
        [1.0, 0.0,        0.0],
        [0.0, np.cos(r), -np.sin(r)],
        [0.0, np.sin(r),  np.cos(r)]
    ], dtype=np.float64)

def rot_y(deg):
    """构建绕 Y 轴旋转矩阵 (输入参数 deg 为角度制)"""
    r = np.deg2rad(deg)
    return np.array([
        [ np.cos(r), 0.0, np.sin(r)],
        [ 0.0,       1.0, 0.0      ],
        [-np.sin(r), 0.0, np.cos(r)]
    ], dtype=np.float64)

def rot_z(deg):
    """构建绕 Z 轴旋转矩阵 (输入参数 deg 为角度制)"""
    r = np.deg2rad(deg)
    return np.array([
        [np.cos(r), -np.sin(r), 0.0],
        [np.sin(r),  np.cos(r), 0.0],
        [0.0,        0.0,       1.0]
    ], dtype=np.float64)


#构造基础旋转矩阵
Rx = rot_x(45.0)  # 绕机体自身 X 轴旋转 45 度 (Roll 滚转)
Ry = rot_y(30.0)  # 绕机体自身 Y' 轴旋转 30 度 (Pitch 俯仰)
Rz = rot_z(60.0)  # 绕机体自身 Z'' 轴旋转 60 度 (Yaw 偏航)


#复合旋转矩阵计算：内旋右乘 vs 外旋左乘
#内旋（绕自身运动局部坐标轴）：依次右乘 Rx @ Ry @ Rz
R_int = Rx @ Ry @ Rz

#外旋（绕固定世界基坐标轴）：依次左乘 Rz @ Ry @ Rx
R_ext = Rz @ Ry @ Rx


#空间探测基准点变换计算
p_test = np.array([1.0, 1.0, 1.0], dtype=np.float64)
p_prime_int = R_int @ p_test
p_prime_ext = R_ext @ p_test


#控制台输出与代数非对易性校验

print("复合内旋总矩阵 R_int = Rx @ Ry @ Rz (3x3):\n", np.round(R_int, 4))
print("\n复合外旋总矩阵 R_ext = Rz @ Ry @ Rx (3x3):\n", np.round(R_ext, 4))


#[选做] 调用分步动画模块可视化内旋过程
try:
    from animator_3d import play_3d_step_animation
    #列表形式传入旋转矩阵动画按照Rx->Ry->Rz顺序旋转，但一个是绕局部坐标系一个绕世界坐标系
    play_3d_step_animation(
        rotation_matrices_inc=[Rx, Ry, Rz],   #传入3个基础旋转矩阵
        #rotation_matrices_inc=R_int          #(可选打开注释)传入1个复合旋转矩阵
        frame_mode='body',                    #指定为机身坐标系内旋模式 (矩阵右乘)
        order='rotate_first'
    )
    play_3d_step_animation(
        rotation_matrices_inc=[Rx, Ry, Rz],   #传入3个基础旋转矩阵
        #rotation_matrices_inc=R_ext          #(可选打开注释)传入1个复合旋转矩阵
        frame_mode='world',                    #指定为世界坐标系外旋模式 (矩阵左乘)
        order='rotate_first'
    )
except ImportError:
    pass
