# Arm Teleoperation v1.0

[中文](#中文) | [English](#english)

## 中文

这是机械臂键盘遥操作的第一版稳定实现。仓库只保留 teleop 运行逻辑和最基本的项目文件，不包含课程实验代码、实验报告、临时目录或开发更新脚本。

### 安装

```powershell
pip install -r requirements.txt
```

### 启动

水平夹爪模式：

```powershell
python teleop/keyboard_position_ik.py --live --port COM5 --orientation-mode horizontal
```

自由姿态模式：

```powershell
python teleop/keyboard_position_ik.py --live --port COM5 --orientation-mode free
```

省略 `--live` 时为 dry run。

### 控制

| 按键 | 功能 |
|---|---|
| `P` | 开始控制 |
| `A / ←` | J1 右转 / 俯视顺时针 |
| `D / →` | J1 左转 / 俯视逆时针 |
| `W / ↑` | 径向前进 |
| `S / ↓` | 径向后退 |
| `Q` | 上升 |
| `E` | 下降 |
| `O` | 打开夹爪 |
| `C` | 闭合夹爪 |
| `N` | 夹爪回参考位 |
| `H` | `HORIZONTAL` / `FREE` 切换 |
| `ESC` | 正常结束并平滑回零 |

默认线速度为 `60 mm/s`，默认 J1 转速为 `20 deg/s`。

水平模式严格保持：

```text
J2 + J3 + J4 = -90°
```

自由模式只控制末端的径向距离 `r` 和高度 `z`，不固定 J2/J3/J4 的角度和。

### 夹爪

```text
O -> J6 +50°  打开
C -> J6 -40°  闭合
N -> J6   0°  参考位
```

### 轨迹

轨迹默认写入：

```text
teleop/recordings/
```

该目录中的生成文件不会提交到 Git。

回放：

```powershell
python teleop/replay_trajectory.py teleop/recordings/trajectory_xxx.csv --live --port COM5
```

当前系统没有真实关节反馈，因此轨迹记录的是下发的目标关节角，而不是编码器实测值。

### 模型

```text
L1 = 120 mm
L2 = 120 mm
L3 = 140 mm
J2 model angle = J2 servo angle + 90°
```

### 最终目录

```text
arm_practice/
├─ .gitignore
├─ README.md
├─ requirements.txt
└─ teleop/
   ├─ __init__.py
   ├─ hardware.py
   ├─ keyboard_position_ik.py
   ├─ position_ik.py
   ├─ replay_trajectory.py
   ├─ trajectory_io.py
   └─ recordings/
      └─ .gitignore
```

## English

This is the first stable keyboard-teleoperation release. The repository intentionally contains only the teleop runtime and minimal project metadata; course experiments, reports, temporary directories, and development update scripts are excluded.

### Install

```powershell
pip install -r requirements.txt
```

### Run

Horizontal mode:

```powershell
python teleop/keyboard_position_ik.py --live --port COM5 --orientation-mode horizontal
```

Free-orientation mode:

```powershell
python teleop/keyboard_position_ik.py --live --port COM5 --orientation-mode free
```

Omit `--live` for dry run.

### Controls

| Key | Action |
|---|---|
| `P` | Start |
| `A / ←` | Rotate J1 right / CW |
| `D / →` | Rotate J1 left / CCW |
| `W / ↑` | Radially forward |
| `S / ↓` | Radially backward |
| `Q` | Up |
| `E` | Down |
| `O` | Open gripper |
| `C` | Close gripper |
| `N` | Gripper reference |
| `H` | Toggle `HORIZONTAL` / `FREE` |
| `ESC` | Normal stop and smooth homing |

Default TCP speed is `60 mm/s`; default J1 speed is `20 deg/s`.

Horizontal mode enforces:

```text
J2 + J3 + J4 = -90°
```

Free mode controls only radial distance `r` and height `z`.

### Trajectories

Generated trajectories are stored under `teleop/recordings/` and ignored by Git.

Hardware replay:

```powershell
python teleop/replay_trajectory.py teleop/recordings/trajectory_xxx.csv --live --port COM5
```

Recorded joint values are commanded targets, not encoder measurements.

### Final Layout

```text
arm_practice/
├─ .gitignore
├─ README.md
├─ requirements.txt
└─ teleop/
   ├─ __init__.py
   ├─ hardware.py
   ├─ keyboard_position_ik.py
   ├─ position_ik.py
   ├─ replay_trajectory.py
   ├─ trajectory_io.py
   └─ recordings/
      └─ .gitignore
```
