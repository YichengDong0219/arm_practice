import numpy as np

#基础矩阵构造工具函数
def rot_x(deg):
    """构建绕 X 轴旋转矩阵 (输入参数 deg 为角度制)"""
    r = np.deg2rad(deg)
    return np.array([
        [1.0, 0.0,        0.0],
        [0.0, np.cos(r), -np.sin(r)],
        [0.0, np.sin(r),  np.cos(r)]
    ], dtype=np.float64)

def rot_z(deg):
    """构建绕 Z 轴旋转矩阵 (输入参数 deg 为角度制)"""
    r = np.deg2rad(deg)
    return np.array([
        [np.cos(r), -np.sin(r), 0.0],
        [np.sin(r),  np.cos(r), 0.0],
        [0.0,        0.0,       1.0]
    ], dtype=np.float64)

def build_T(R, t):
    """将 3x3 旋转矩阵与 3x1 平移向量组装为 4x4 齐次变换矩阵"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

# 初始各级局部相对齐次变换矩阵构建 (0 -> A -> B -> C -> D -> E)
# 0 -> A
T_0_A_init = build_T(np.eye(3), [0.0, 0.0, 1.0])

# A -> B
T_A_B_init = build_T(rot_x(90.0), [0.0, 0.0, 0.5])

# B -> C
T_B_C_init = build_T(np.eye(3), [1.5, 0.0, 0.0])

# C -> D
T_C_D_init = build_T(np.eye(3), [1.2, 0.0, 0.0])

# D -> E
T_D_E_init = build_T(np.eye(3), [0.8, 0.0, 0.0])


#局部坐标系自转更新与代数链式复合传递
#坐标系{A}绕自身垂直转轴Z_A自转45.0°(局部右乘)
R_rot_A = rot_z(45.0)
T_0_A_new = T_0_A_init @ build_T(R_rot_A, [0.0, 0.0, 0.0])

#坐标系{D}绕自身俯仰转轴Z_D自转-30.0°(局部右乘)
R_rot_D = rot_z(-30.0)
T_C_D_new = T_C_D_init @ build_T(R_rot_D, [0.0, 0.0, 0.0])

#链式复合连乘求解末端E在世界基准系{0}下的总位姿
T_0_E = T_0_A_new @ T_A_B_init @ T_B_C_init @ T_C_D_new @ T_D_E_init

#提取旋转矩阵与平移位置
R_0_E = T_0_E[:3, :3]
t_0_E = T_0_E[:3, 3]


#控制台格式化输出
print("末端工具坐标系 {E} 相对于世界坐标系 {0} 的总齐次变换矩阵 T_0_E:\n", np.round(T_0_E, 4))
print("\n末端工具坐标系 {E} 相对于世界坐标系 {0} 的最终姿态旋转矩阵 R_0_E:\n", np.round(R_0_E, 4))
print(f"\n末端工具原点在世界坐标系 {{0}} 下的最终物理空间位置 [X, Y, Z]:\n{np.round(t_0_E, 4)}")


#[选做]调用动画模块动态展示多级坐标系拓扑与连续局部自转

try:
    from animator_3d import play_3d_step_animation

    frames = [
        {'T': T_0_A_init, 'label': '{A}'},
        {'T': T_A_B_init, 'label': '{B}'},
        {'T': T_B_C_init, 'label': '{C}'},
        {'T': T_C_D_init, 'label': '{D}'},
        {'T': T_D_E_init, 'label': '{E}'}
    ]

    rotate_frames_seq = ['A', 'D']
    rotate_R_seq = [
        [rot_z(45.0)],
        [rot_z(-30.0)]
    ]

    play_3d_step_animation(
        reference_frames=frames,
        translation_vec_start=[0.0, 0.0, 0.0],
        rotate_frame=rotate_frames_seq,
        rotate_R=rotate_R_seq,
        frame_rot_timing='before',
        xlim=(-1, 4), ylim=(-1, 4), zlim=(0, 4),
        frames_per_step=30,
        fps=30
    )
except ImportError:
    pass