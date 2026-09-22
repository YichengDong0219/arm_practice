# Arm Practice — Keyboard Teleoperation v1.0

[中文](#中文) | [English](#english)

---

## 中文

### 项目简介

这是 `arm_practice` 的第一版稳定键盘遥操作实现。上位机在 Windows 上读取键盘输入，实时计算机械臂目标关节角，并通过串口向 STM32 发送 6 轴控制指令。

当前控制逻辑采用更符合机械臂结构的**圆柱坐标控制**：

- `A / D`：只控制 J1 底座旋转；
- `W / S`：控制末端沿当前朝向前进 / 后退；
- `Q / E`：控制末端竖直上升 / 下降；
- `O / C / N`：控制 J6 夹爪打开 / 闭合 / 回参考位；
- `H`：运行中切换 `HORIZONTAL` / `FREE` 末端姿态模式；
- `ESC`：正常结束控制，并平滑回到六轴全零位。

默认控制频率为 **50 Hz**，默认末端线速度为 **60 mm/s**。

### 主要功能

- Windows 键盘实时遥操作；
- 圆柱坐标 `(r, θ, z)` 控制；
- J1 与前后 / 上下运动解耦；
- 两种末端姿态模式：
  - `HORIZONTAL`：严格保持 `J2 + J3 + J4 = -90°`；
  - `FREE`：只约束末端位置，J2/J3/J4 自由协同；
- 启动时可通过命令行选择姿态模式；
- 运行中可按 `H` 平滑切换姿态模式；
- J6 夹爪开闭控制；
- 自动记录控制轨迹；
- 支持轨迹校验与 replay；
- 正常按 `ESC` 后自动平滑回零；
- `Ctrl+C` / 异常情况下不强制自动运动。

### 环境要求

推荐：

- Windows 10 / 11
- Python 3.10+
- NumPy
- pySerial

安装依赖：

```powershell
pip install -r requirements.txt
```

### 真机启动

默认串口为 `COM5`，但仍建议显式指定。

#### 水平夹爪模式

```powershell
python teleop/keyboard_position_ik.py --live --port COM5 --orientation-mode horizontal
```

在该模式下：

```text
J2 + J3 + J4 = -90°
```

程序会严格保持这一约束。

#### 自由姿态模式

```powershell
python teleop/keyboard_position_ik.py --live --port COM5 --orientation-mode free
```

该模式只控制末端的 `r,z` 位置，不限制 `J2+J3+J4`。

#### Dry run

省略 `--live` 时不会打开串口：

```powershell
python teleop/keyboard_position_ik.py --orientation-mode horizontal
```

### 按键

| 按键 | 功能 |
|---|---|
| `P` | 进入准备姿态并开始控制 |
| `A / ←` | 左转 / 俯视逆时针，仅控制 J1 |
| `D / →` | 右转 / 俯视顺时针，仅控制 J1 |
| `W / ↑` | 径向前进，末端远离底座 |
| `S / ↓` | 径向后退，末端靠近底座 |
| `Q` | 竖直上升 |
| `E` | 竖直下降 |
| `O` | 打开夹爪，J6 → `+50°` |
| `C` | 闭合夹爪，J6 → `-40°` |
| `N` | 夹爪回参考位，J6 → `0°` |
| `H` | `HORIZONTAL` / `FREE` 模式切换 |
| `ESC` | 正常结束，平滑回全零位 |

### 速度参数

默认末端线速度：

```text
60 mm/s
```

可以修改：

```powershell
python teleop/keyboard_position_ik.py --live --port COM5 --speed 40
```

A/D 控制 J1，使用独立角速度参数，默认：

```text
20 deg/s
```

修改示例：

```powershell
python teleop/keyboard_position_ik.py --live --port COM5 --j1-speed 30
```

如果实机 J1 正方向与程序定义相反：

```powershell
python teleop/keyboard_position_ik.py --live --port COM5 --j1-sign -1
```

### 姿态模式

#### HORIZONTAL

严格满足：

```text
J2 + J3 + J4 = -90°
```

适用于希望夹爪始终保持水平的操作。

#### FREE

仅追踪末端的径向距离 `r` 与高度 `z`。J2/J3/J4 通过 DLS IK 自由协同，因此末端俯仰角不会被固定。

从 `HORIZONTAL` 切换到 `FREE` 不会产生额外机械运动。

从 `FREE` 切换回 `HORIZONTAL` 时，程序会先规划完整的姿态校正轨迹，在尽量保持当前 `r,z` 不变的情况下平滑恢复到 `-90°`。如果当前位置不存在水平构型，则拒绝切换并继续保持 `FREE`。

### 轨迹记录

键盘控制默认记录每个控制周期的：

```text
frame
timestamp
target X / Y / Z
commanded J1 ... J6
```

记录目录：

```text
teleop/recordings/
```

每次运行会生成一对：

```text
trajectory_YYYYMMDD_HHMMSS.csv
trajectory_YYYYMMDD_HHMMSS.json
```

注意：当前 STM32 接口只有上位机下行控制，没有舵机编码器反馈。因此记录的是**下发目标角度**，不是实测关节角。

临时关闭记录：

```powershell
python teleop/keyboard_position_ik.py --live --port COM5 --no-record
```

### Replay

先只检查轨迹文件：

```powershell
python teleop/replay_trajectory.py teleop/recordings/trajectory_xxx.csv
```

真机 replay：

```powershell
python teleop/replay_trajectory.py teleop/recordings/trajectory_xxx.csv --live --port COM5
```

播放速度倍率：

```powershell
--speed-scale 0.5
```

表示半速；`2` 表示两倍速。

Replay 直接重放记录的 J1–J6 目标角，不会重新执行 IK。

### 当前运动学模型

当前模型采用：

```text
L1 = 120 mm
L2 = 120 mm
L3 = 140 mm
```

并使用：

```text
J2 model angle = J2 servo angle + 90°
```

当前软件语义：

- J1：底座旋转；
- J2/J3/J4：平面臂部运动与末端俯仰；
- J5：当前位置遥操作中保持不变；
- J6：夹爪开合。

### 验证

姿态模式与 IK：

```powershell
python teleop/validate_orientation_modes.py
```

轨迹记录 / 读取：

```powershell
python teleop/validate_trajectory_io.py
```

### 安全说明

这是上位机命令控制系统，不是带关节反馈的闭环控制器。

真机使用时请注意：

- 启动前确认机械臂真实姿态接近软件假设；
- 清空机械臂运动范围；
- 第一次使用新的速度或运动范围时从低速开始；
- 准备随时断电；
- `ESC` 属于正常结束，会自动回零；
- `Ctrl+C` 或异常退出不会强制回零；
- 软件显示的 XYZ 和关节角是模型 / 指令值，不代表真实传感器测量值。

### 目录结构

```text
arm_practice/
├─ README.md
├─ requirements.txt
├─ task8.py
├─ task8_reverse.py
└─ teleop/
   ├─ __init__.py
   ├─ keyboard_position_ik.py
   ├─ position_ik.py
   ├─ trajectory_io.py
   ├─ replay_trajectory.py
   ├─ validate_orientation_modes.py
   ├─ validate_trajectory_io.py
   └─ recordings/
      └─ .gitignore
```

---

## English

### Overview

This is the first stable keyboard-teleoperation release of `arm_practice`. A Windows PC reads keyboard input, computes target joint commands in real time, and sends six-axis command frames to an STM32 controller over serial.

The operator-facing motion model uses **cylindrical coordinates**:

- `A / D`: rotate the base through J1 only;
- `W / S`: move the TCP radially forward / backward;
- `Q / E`: move the TCP vertically up / down;
- `O / C / N`: open / close / reset the J6 gripper;
- `H`: toggle end-effector orientation between `HORIZONTAL` and `FREE`;
- `ESC`: stop normal teleoperation and smoothly return all joints to zero.

The default control rate is **50 Hz** and the default TCP linear speed is **60 mm/s**.

### Features

- Real-time Windows keyboard teleoperation;
- Cylindrical `(r, θ, z)` control;
- J1 decoupled from radial and vertical motion;
- Two orientation modes:
  - `HORIZONTAL`: strictly enforces `J2 + J3 + J4 = -90°`;
  - `FREE`: controls position only and lets J2/J3/J4 coordinate freely;
- Orientation mode selectable from the command line;
- Runtime mode switching with `H`;
- J6 gripper control;
- Automatic command-trajectory recording;
- Trajectory validation and replay;
- Smooth all-zero homing after a normal `ESC` stop;
- No forced motion after `Ctrl+C` or an unexpected failure.

### Requirements

Recommended:

- Windows 10 / 11
- Python 3.10+
- NumPy
- pySerial

Install dependencies:

```powershell
pip install -r requirements.txt
```

### Running on Hardware

The current default serial port is `COM5`, although explicitly passing the port is recommended.

#### Horizontal mode

```powershell
python teleop/keyboard_position_ik.py --live --port COM5 --orientation-mode horizontal
```

This mode strictly enforces:

```text
J2 + J3 + J4 = -90°
```

#### Free-orientation mode

```powershell
python teleop/keyboard_position_ik.py --live --port COM5 --orientation-mode free
```

This mode controls TCP `r,z` position without constraining `J2+J3+J4`.

#### Dry run

Omit `--live` to avoid opening the serial port:

```powershell
python teleop/keyboard_position_ik.py --orientation-mode horizontal
```

### Controls

| Key | Action |
|---|---|
| `P` | Move to the preparation pose and start |
| `A / ←` | Turn left / counter-clockwise from the top view; J1 only |
| `D / →` | Turn right / clockwise from the top view; J1 only |
| `W / ↑` | Radially forward, away from the base |
| `S / ↓` | Radially backward, toward the base |
| `Q` | Move vertically up |
| `E` | Move vertically down |
| `O` | Open gripper, J6 → `+50°` |
| `C` | Close gripper, J6 → `-40°` |
| `N` | Gripper reference position, J6 → `0°` |
| `H` | Toggle `HORIZONTAL` / `FREE` |
| `ESC` | Normal stop and smooth all-zero homing |

### Speed

Default TCP linear speed:

```text
60 mm/s
```

Override it with:

```powershell
python teleop/keyboard_position_ik.py --live --port COM5 --speed 40
```

J1 rotation has a separate default speed:

```text
20 deg/s
```

Override it with:

```powershell
python teleop/keyboard_position_ik.py --live --port COM5 --j1-speed 30
```

If the physical J1 direction is reversed relative to the software convention:

```powershell
python teleop/keyboard_position_ik.py --live --port COM5 --j1-sign -1
```

### Orientation Modes

#### HORIZONTAL

Strictly enforces:

```text
J2 + J3 + J4 = -90°
```

Use this when the gripper should remain horizontal.

#### FREE

Only radial distance `r` and height `z` are tracked. J2/J3/J4 are coordinated by DLS IK without a fixed end-effector pitch.

Switching from `HORIZONTAL` to `FREE` does not cause an extra robot motion.

When switching from `FREE` back to `HORIZONTAL`, the program first plans the full transition and then smoothly restores the `-90°` joint-sum constraint while keeping the current `r,z` as fixed as the model allows. If no horizontal configuration exists at that position, the switch is rejected and the robot stays in `FREE`.

### Trajectory Recording

Teleoperation records each control cycle by default:

```text
frame
timestamp
target X / Y / Z
commanded J1 ... J6
```

Recordings are stored in:

```text
teleop/recordings/
```

Each session creates:

```text
trajectory_YYYYMMDD_HHMMSS.csv
trajectory_YYYYMMDD_HHMMSS.json
```

The current STM32 interface is command-only and does not provide joint encoder feedback. Recorded joint values are therefore **commanded targets**, not measured joint states.

Disable recording with:

```powershell
python teleop/keyboard_position_ik.py --live --port COM5 --no-record
```

### Replay

Validate a trajectory without opening serial:

```powershell
python teleop/replay_trajectory.py teleop/recordings/trajectory_xxx.csv
```

Replay on hardware:

```powershell
python teleop/replay_trajectory.py teleop/recordings/trajectory_xxx.csv --live --port COM5
```

Time scaling:

```powershell
--speed-scale 0.5
```

means half speed; `2` means double speed.

Replay sends the recorded J1–J6 commands directly and does not rerun IK.

### Kinematic Model

The current model uses:

```text
L1 = 120 mm
L2 = 120 mm
L3 = 140 mm
```

with:

```text
J2 model angle = J2 servo angle + 90°
```

Software joint semantics:

- J1: base yaw;
- J2/J3/J4: planar arm motion and tool pitch;
- J5: held unchanged by the current position teleoperation;
- J6: gripper actuator.

### Validation

Orientation-mode and IK regression:

```powershell
python teleop/validate_orientation_modes.py
```

Trajectory I/O regression:

```powershell
python teleop/validate_trajectory_io.py
```

### Safety Notes

This is a command-side controller, not a closed-loop joint-feedback controller.

Before hardware operation:

- verify the real robot starts near the assumed software pose;
- clear the robot workspace;
- start slowly when testing a new speed or workspace region;
- keep power-off access available;
- `ESC` performs a normal smooth return to zero;
- `Ctrl+C` or unexpected failures do not force an automatic return;
- displayed XYZ and joint angles are model / command values, not sensor measurements.

### Project Layout

```text
arm_practice/
├─ README.md
├─ requirements.txt
├─ task8.py
├─ task8_reverse.py
└─ teleop/
   ├─ __init__.py
   ├─ keyboard_position_ik.py
   ├─ position_ik.py
   ├─ trajectory_io.py
   ├─ replay_trajectory.py
   ├─ validate_orientation_modes.py
   ├─ validate_trajectory_io.py
   └─ recordings/
      └─ .gitignore
```
