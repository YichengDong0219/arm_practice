import numpy as np


#定义单轴旋转矩阵构造函数

def rot_x(deg):
    """构建绕 X 轴旋转矩阵 (输入参数 deg 为角度制)"""
    r = np.deg2rad(deg)  # 角度转弧度
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
Rx = rot_x(45.0)  # 绕世界 X 轴旋转 45 度
Ry = rot_y(30.0)  # 绕世界 Y 轴旋转 30 度
Rz = rot_z(60.0)  # 绕世界 Z 轴旋转 60 度


#外旋连乘计算：绕固定世界基坐标系依次旋转 X -> Y -> Z (采用矩阵左乘)
R_ext = Rz @ Ry @ Rx


#空间探测基准点变换计算

# 引入非零基准点以反映姿态变化 (若为 [0,0,0]^T 则旋转前后恒为 [0,0,0]^T)
p_test = np.array([1.0, 1.0, 1.0], dtype=np.float64)
p_prime = R_ext @ p_test


#控制台输出计算结果

print("单轴旋转矩阵 Rx (45°):\n", np.round(Rx, 4))
print("\n单轴旋转矩阵 Ry (30°):\n", np.round(Ry, 4))
print("\n单轴旋转矩阵 Rz (60°):\n", np.round(Rz, 4))
print("\n复合外旋总矩阵 R_ext = Rz @ Ry @ Rx (3x3):\n", np.round(R_ext, 4))
print("经复合外旋变换后的空间点 p_prime 坐标: ", np.round(p_prime, 4))

#[选做] 调用分步动画模块可视化外旋过程

try:
    from animator_3d import play_3d_step_animation
    play_3d_step_animation(
        rotation_matrices_inc=[Rx, Ry, Rz],   # 传入 3 个基础旋转矩阵
        frame_mode='world',                   # 指定为世界基坐标系外旋模式 (矩阵左乘)
        order='rotate_first'
    )
except ImportError:
    pass
