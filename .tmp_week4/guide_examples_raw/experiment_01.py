import numpy as np

#定义初始位置点 A 与平移向量 t (三维空间向量)
p_A = np.array([1.0, 0.5, 0.0], dtype=np.float64)
t_vec = np.array([2.0, 1.5, 2.5], dtype=np.float64)

#执行向量代数加法运算求解目标点 B
p_B = p_A + t_vec

#打印输出计算结果
print(f"初始点 A 位置坐标: {p_A}")
print(f"空间平移位移向量 t: {t_vec}")
print(f"平移变换后目标点 B 坐标: {p_B}")

#[选做] 调用动画模块可视化纯平移动态过程
try:
    #导入小飞机模块，传入平移矩阵，旋转矩阵置空
    from animator_3d import play_3d_step_animation
    play_3d_step_animation(translation_vec_inc=t_vec, translation_vec_start=p_A)
except ImportError:
    pass
