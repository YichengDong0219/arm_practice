import numpy as np



#定义基础单轴旋转矩阵与齐次矩阵构造函数
def rot_x(deg):
    """构建绕 X 轴旋转矩阵 (输入参数 deg 为角度制)"""
    r = np.deg2rad(deg)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(r), -np.sin(r)],
        [0.0, np.sin(r), np.cos(r)]
    ], dtype=np.float64)


def rot_y(deg):
    """构建绕 Y 轴旋转矩阵 (输入参数 deg 为角度制)"""
    r = np.deg2rad(deg)
    return np.array([
        [np.cos(r), 0.0, np.sin(r)],
        [0.0, 1.0, 0.0],
        [-np.sin(r), 0.0, np.cos(r)]
    ], dtype=np.float64)


def rot_z(deg):
    """构建绕 Z 轴旋转矩阵 (输入参数 deg 为角度制)"""
    r = np.deg2rad(deg)
    return np.array([
        [np.cos(r), -np.sin(r), 0.0],
        [np.sin(r), np.cos(r), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def build_T(R, t):
    """将 3x3 旋转矩阵与 3x1 平移向量组装为 4x4 齐次变换矩阵"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T



#构造各级局部相对齐次变换矩阵
#坐标系{0}->坐标系{B}
R_0B = rot_z(90.0)
t_0B = np.array([0.0, 0.0, 2.0], dtype=np.float64)
T_0_B = build_T(R_0B, t_0B)

#坐标系{B}->坐标系{C}
R_BC = rot_x(45.0)
t_BC = np.array([2.0, 0.0, 0.0], dtype=np.float64)
T_B_C = build_T(R_BC, t_BC)

#坐标系{C}->目标物体P
R_CP = rot_y(30.0)
t_CP = np.array([0.0, 1.5, 0.0], dtype=np.float64)
T_C_P = build_T(R_CP, t_CP)


#矩阵链式连乘求解多级复合位姿

#计算坐标系{C}在基坐标系{A}下的总位姿
T_0_C = T_0_B @ T_B_C

#链式连乘：计算物体P在基坐标系{A}下的总齐次变换矩阵
T_0_P = T_0_B @ T_B_C @ T_C_P

#提取物体几何中心在世界基坐标系{A}下的三维绝对物理坐标
p_final_world = T_0_P[:3, 3]


#控制台输出计算结果
print("坐标系 {B} 相对于 {0} 的齐次矩阵 T_A_B:\n", np.round(T_0_B, 4))
print("\n坐标系 {C} 相对于 {B} 的齐次矩阵 T_B_C:\n", np.round(T_B_C, 4))
print("\n坐标系 {C} 在世界坐标系 {0} 下的复合齐次矩阵 T_A_C = T_A_B @ T_B_C:\n", np.round(T_0_C, 4))
print("\n物体 P 在时间坐标系 {0} 下的总齐次矩阵 T_A_P:\n", np.round(T_0_P, 4))
print(f"\n物体几何中心 P 在世界基坐标系 {{0}} 下的最终三维坐标 [X, Y, Z]:\n{np.round(p_final_world, 4)}")


#[选做]调用动画模块可视化链式多坐标系运动
try:
    from animator_3d import play_3d_step_animation


    frame_B = {'T': T_0_B, 'label': '{B}'}
    frame_C = {'T': T_B_C, 'label': '{C}'}

    play_3d_step_animation(
        reference_frames=[frame_B, frame_C],  #链式派生坐标系{B}与{C}
        T_start=np.eye(4),  #初始位于坐标系{C}的原点
        T_inc=T_C_P,  #相对末端系{C}执行变换T_C_P
        xlim=(-2, 4), ylim=(-1, 5), zlim=(-1, 5),
        frames_per_step=30,
        fps=30
    )
except ImportError:
    pass
