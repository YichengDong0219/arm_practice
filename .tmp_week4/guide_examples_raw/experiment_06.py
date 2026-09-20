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

#计算复合旋转矩阵与平移向量
#经历X(45°)->Y(30°)->Z(60°)三次连续外旋(矩阵左乘)
Rx = rot_x(45.0)
Ry = rot_y(30.0)
Rz = rot_z(60.0)
R_composite = Rz @ Ry @ Rx

#空间平移向量
t_vec = np.array([3.0, 2.0, 2.5], dtype=np.float64)


#构造4x4齐次变换矩阵T
T = np.eye(4, dtype=np.float64)
T[:3, :3] = R_composite  #填充左上角3x3旋转子矩阵
T[:3, 3] = t_vec         #填充右上角3x1平移子向量


#齐次坐标变换计算(依托物体几何中心)
# 物体自身的几何中心原点坐标
p_B = np.array([0.0, 0.0, 0.0], dtype=np.float64)

#扩展为4维齐次坐标向量 [0, 0, 0, 1.0]
p_B_homo = np.append(p_B, 1.0)

#执行单一齐次矩阵向量积
p_0_homo = T @ p_B_homo

#提取前三维绝对空间物理坐标
p_0 = p_0_homo[:3]

#控制台输出计算结果
print(" 4x4 齐次变换矩阵 T:\n", np.round(T, 4))
print(f"\n几何中心在自身坐标系中的坐标 p_B:       {p_B}")
print(f"扩展后的4D齐次坐标向量 p_B_homo:         {p_B_homo}")
print(f"矩阵相乘得到的4D结果向量 p_0_homo:       {np.round(p_0_homo, 4)}")
print(f"最终在世界基坐标系下的三维物理坐标 p_0:     {np.round(p_0, 4)}")


#[选做]演示动画
try:
    from animator_3d import play_3d_step_animation

    play_3d_step_animation(
        T_inc=T,           #传入4x4齐次变换增量矩阵
        frames_per_step=60,     #插值细分帧数
        fps=30
    )
except ImportError:
    pass
