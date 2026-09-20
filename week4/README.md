# 第四周实验 1.8 至 1.10

本目录只覆盖指导书的实验 1.8、1.9 和 1.10。工程根目录原有的 `task8.py` 与
`task8_reverse.py` 是平面圆周运动项目，不等同于第四周指导书中的“实验八”，因此不移动、
不覆盖。

## 目录用途

- `common/`：三个实验共用的矩阵、通信帧和插值驱动代码。
- `experiments/`：整理后的实验 1.8、1.9、1.10 可执行程序。
- `reference/guide_examples/`：从指导书提取的原始示例代码，仅作对照。
- `assets/`：指导书中与 1.9/1.10 有关的接口接线图。
- `docs/`：任务分工、硬件检查和待完成工作说明。
- `tests/`：无需连接机械臂即可运行的软件测试。
- `outputs/`：保存控制台输出、照片、视频和实测记录，不提交临时文件。

工作区中同时存在 `week4/task8/main.py`、`task9/main.py`、`task10/main.py`。经逐文件比较，
它们与指导书提取版本仅有文件末尾换行差异，可视为原始单文件示例；新的模块化程序不会覆盖它们。

## 环境准备

当前 PowerShell 中的 `python` 和 `python3` 均指向 `D:\MSYS2\ucrt64\bin\python.exe`，
该解释器目前没有安装 NumPy。若原来的 `ra_class` 环境仍可用，应先激活它；否则可使用
Windows 的 Python 3.10 创建独立环境：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r week4\requirements.txt
```

## 推荐执行顺序

```powershell
# 1.8：纯数学计算，无需硬件
python -m week4.experiments.experiment_08_fk_chain

# 1.9：默认仅预演，不会打开串口
python -m week4.experiments.experiment_09_multijoint_gripper

# 1.10：默认仅预演，不会打开串口
python -m week4.experiments.experiment_10_pick_and_place

# 实验10扩展：固定供料、30 mm物块三层堆叠（默认仅预演）
python week4\task10\stacking.py --block-height-mm 30 --count 3

# 软件测试
python -m unittest discover -s week4\tests -v
```

堆叠程序直接使用 `task10/main.py` 未注释动作中的抓取、抬升和第一层放置角度。
程序用 120/120/140 mm 模型反算第一层坐标，后续层保持 x、y 不变并将 z 每层增加
30 mm，再求连续 IK 解。旧的标定文件不会被读取。

确认第一层动作在实机上安全后运行：

```powershell
python week4\task10\stacking.py --execute --port COM5 --block-height-mm 30 --count 3
```

堆叠程序每轮等待回车确认物块已放到固定抓取位。每块依次执行抓取、夹紧、抬升、
放置、松开和撤离，随后立即平滑回到全零位，再等待下一块物料。原始实验10的
`main.py` 保持不变。

放置前先停在目标点正上方40 mm，再以10段笛卡尔高度点近似竖直下降。下降、放置和
竖直撤离过程中始终满足 `J2 + J3 + J4 = -90°`，由J4补偿J2/J3以维持夹爪水平。
为避免第一层水平姿态使J2超过-90°限位，模型堆叠点相对原放置点向基座微调5 mm。

抓取阶段采用相同的水平约束：先到取料点上方40 mm，再以10段竖直下降，夹紧后沿
原路径抬升。原取料坐标在水平姿态下会使J3约为-103.6°，因此固定供料点需要沿原来的
J1=30°方向向外移动30 mm，使抓取与抬升全过程保持 `J2 + J3 + J4 = -90°`。

确认机械臂零位、供电、串口号和活动空间后，才使用 `--execute`：

```powershell
python -m week4.experiments.experiment_09_multijoint_gripper --execute --port COM5
python -m week4.experiments.experiment_10_pick_and_place --execute --port COM5
```

1.8 的动画是选做项。指导书引用了 `animator_3d.py` 和 `airplane_geometry.py`，但当前工程
没有这两个实际模块；取得教师提供的文件后，将其放入 `week4/vendor/` 或工程根目录并配置
Python 搜索路径，再使用 `--animate`。
