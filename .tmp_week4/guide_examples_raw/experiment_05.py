import numpy as np



#定义单轴旋转矩阵构造函数
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



#构建参考坐标系{A}相对于世界坐标系{0}的位姿
#坐标系{A}原点在世界坐标系中的位置
t_0_A = np.array([2.0, 1.5, 1.0], dtype=np.float64)

#坐标系{A}姿态：绕世界轴外旋X(0°)->Y(30°)->Z(45°)
R_0_A = rot_z(45.0) @ rot_y(30.0) @ rot_x(0.0)

#定义物体在坐标系{A}中的初始状态与运动增量
#初始点位与初始姿态(相对坐标系{A})
p_start_A = np.array([1.0, 0.5, 0.2], dtype=np.float64)
R_start_A = rot_z(15.0)

#运动增量：相对坐标系{A}轴外旋X(30°)->Y(20°)->Z(45°)
R_inc_x = rot_x(30.0)
R_inc_y = rot_y(20.0)
R_inc_z = rot_z(45.0)
R_inc_A = R_inc_z @ R_inc_y @ R_inc_x  # 外旋左乘

#平移位移增量(相对坐标系 {A})
t_inc_A = np.array([1.2, 0.8, 0.5], dtype=np.float64)


#坐标解算：局部坐标最终点位->全局世界坐标映射

#计算在坐标系{A}内部经历“先外旋后平移”后的最终局部坐标
p_final_A = R_inc_A @ p_start_A + t_inc_A

#跨坐标系线性映射：求解在世界基坐标系{0}下的绝对三维坐标
p_final_world = R_0_A @ p_final_A + t_0_A

#控制台输出计算结果
print("坐标系 {A} 相对世界基坐标系的旋转矩阵 R_0_A (3x3):\n", np.round(R_0_A, 4))
print("坐标系 {A} 相对世界基坐标系的原点平移 t_0_A:", t_0_A)
print("\n物体在坐标系 {A} 内部的复合旋转矩阵 R_inc_A:\n", np.round(R_inc_A, 4))
print(f"物体在坐标系 {{A}} 内部的初始点位 p_start_A: {p_start_A}")
print(f"物体在坐标系 {{A}} 内部的平移与旋转后的点位 p_final_A:{np.round(p_final_A, 4)}")
print(f"\n物体最终在全局世界坐标系 {{0}} 下的绝对坐标:{np.round(p_final_world, 4)}")


#[选做]调用动画模块可视化多坐标系及相对运动过程
try:
    from animator_3d import play_3d_step_animation

    #定义参考坐标系A
    frame_A = {
        't': t_0_A,
        'R': R_0_A,
        'label': '{A}'#参考坐标系名称
    }


    play_3d_step_animation(
        reference_frames=[frame_A],  #建立参考坐标系{A}
        translation_vec_start=p_start_A,  #基于坐标系{A}的初始位置
        rotation_matrices_start=R_start_A,  #基于坐标系{A}的初始姿态
        translation_vec_inc=t_inc_A,  #基于坐标系{A}的平移增量
        rotation_matrices_inc=[R_inc_x, R_inc_y, R_inc_z],  #基于坐标系{A}的旋转增量
        order='rotate_first',  #先旋转，后平移
        frame_mode='world',  #相对坐标系{A}的基准轴外旋
        xlim=(-1, 6), ylim=(-1, 6), zlim=(-1, 6),
        frames_per_step=30,
        fps=30
    )
except ImportError:
    pass
