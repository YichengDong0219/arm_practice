import numpy as np



#定义基础单轴旋转矩阵构造函数

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



#参数定义
#目标平移向量
t_vec = np.array([2.5, 1.5, 2.0], dtype=np.float64)

#构造 X -> Y -> Z 三次连续旋转的基础矩阵
Rx = rot_x(45.0)  # 绕 X 轴旋转 45 度
Ry = rot_y(30.0)  # 绕 Y 轴旋转 30 度
Rz = rot_z(60.0)  # 绕 Z 轴旋转 60 度

#计算复合外旋总矩阵 (世界基轴左乘: Rz @ Ry @ Rx)
R_ext = Rz @ Ry @ Rx


#两种时序下的物体几何中心位置代数计算

#先在原点经历三次连续自转，后平移定位

p_final_1 = t_vec.copy()

#先平移至 t_vec，后绕世界基坐标轴经历三次连续外旋

p_final_2 = R_ext @ t_vec



#控制台输出计算与对比结果
print(f"目标平移向量 t:                          {t_vec}")
print("\n复合外旋总矩阵R_ext(Rz @ Ry @ Rx):\n", np.round(R_ext, 4))
print(f"\n (先旋转后平移)最终中心坐标: {np.round(p_final_1, 4)}")
print(f"(先平移后外旋)最终中心坐标: {np.round(p_final_2, 4)}")



#调用动画模块可视化对比
try:
    from animator_3d import play_3d_step_animation


    play_3d_step_animation(
        translation_vec_inc=t_vec,
        rotation_matrices_inc=[Rx, Ry, Rz],
        order='rotate_first',
        frame_mode='body'
    )


    play_3d_step_animation(
        translation_vec_inc=t_vec,
        rotation_matrices_inc=[Rx, Ry, Rz],
        order='translate_first',
        frame_mode='world'
    )
except ImportError:
    pass
