"""第八周：机械臂定点搬运（必做 + 选做）。

任务（作业纸上 s1、s2、s3 为夹取点，t1、t2、t3 为落料点，木块 30 mm 立方体）：
  必做  s1→t1、s2→t3、s3→t2（按此顺序）
  选做  在必做完成后的状态上继续：t1→s2、t2→s1、t3→s3
验收：木块底面完全位于对应圆形区域内、不与边界相交、稳定落平，无错放、掉落或碰倒其他木块。

运行方式（先离线，后实机）::

    python week8/transport.py --mode scan                       # 各点俯仰窗口、每趟倾角、推荐作业纸摆放
    python week8/transport.py --mode plan --task 1              # 规划+离线检查，导出 CSV/JSON/PNG
    python week8/transport.py --mode simulate --task 2          # 三维动画（必做后接选做）
    python week8/transport.py --mode calib --port COM5          # 实机：键盘（同 teleop 按键）把夹爪移到 6 个圆心并记录 x、y
    python week8/transport.py --mode calib --port COM5 --only t3   # 只重标某几个点
    python week8/transport.py --mode calib --port COM5 --text-jog  # 非 Windows：输入指令微调
    python week8/transport.py --mode hardware --task 1 --port COM5 [--step]
    python week8/transport.py --mode hardware --task 2 --port COM5        # 必做 + 选做连续执行

设计要点
--------
1. 状态机：每一趟搬运 = 越障到取料点上方 → 竖直下降 → 静置 → 慢合 → 慢起 → 高处横移 →
   竖直下降到落料点 → 静置 → 慢开 → 慢起撤离；趟与趟之间不回零，全部完成后回零。
2. 逆解动态调用：沿每段笛卡尔直线逐点调用固定俯仰解析逆解（夹爪俯仰 = q2+q3+q4+90°），
   以上一点的解为种子保持肘部分支连续；越障横移若需切换构型，在高处用关节空间运动切换。
3. 流式下发：所有轨迹按 50 Hz（20 ms）逐帧下发 16 字节控制帧，开合夹爪也按帧插值。
4. 倾斜放置：桌面高度上每个半径只对应一段窄的可用俯仰。作业纸跨度约 200 mm，夹取点与落料点
   的俯仰无法完全一致，放下时木块会带一个倾角（推荐摆放下 10.5°~20.5°）。所选俯仰对 ±5 mm
   的径向摆放误差仍可达（鲁棒窗口），纸没摆准也不至于无解。程序按手中木块的
   实际姿态，把最低棱停在纸面上方 RELEASE_GAP 处再松爪：倾斜绕切向水平轴，着地的是整条棱，
   木块翻平时底面中心水平移动 < 1 mm。离线检查给出每趟倾角和“底面距圆边界”的余量。
5. 安全：木块只在越障高度横移；中止（Ctrl+C）只停止下发、绝不松爪。
6. 圆心标定（--mode calib）：夹爪以实际夹取姿态停在各圆心估计位置，用 teleop 同款实时键盘
   （W/S 径向、A/D 绕底座，按住移动）微调到两指中点对准圆心后回车记录，存入 week8/sheet_calib.json；
   夹取姿态到不了时自动切到自由遥控（末端 x/y/z/俯仰，或 teleop JOINT 式直接转关节），M 键切换。
   标定不设人为限制（只保留舵机物理限位）。plan / simulate / hardware 自动读取这套 x、y，高度固定按
   桌面 TABLE_Z，并以标定为准：不再留摆放容差；某点常规俯仰窗口为空时按标定时的俯仰兜底并取消关节余量；
   离线检查只提示、不阻止实机运行。标定后纸不能再动。
   注意：teleop 显示的 r、z 用的是旧的简化连杆长度，与这里的实测 D-H 不同，不能直接抄过来。

坐标系与 week5~week7 的 D-H 模型一致（基座系，mm），桌面 z = TABLE_Z。
作业纸坐标：u 轴由 s 列指向 t 列，v 轴由第 3 行指向第 1 行；默认 u = +X_B（远离机械臂），
v = +Y_B（站在机械臂身后看是左侧），即 s 列靠近机械臂、s1/t1 在左侧。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:  # 课程环境中可用；缺失时规划/检查仍可运行（使用下方 numpy 正解）。
    import roboticstoolbox as rtb
except Exception:  # pragma: no cover
    rtb = None

try:
    import serial
except Exception:  # pragma: no cover
    serial = None


# ============================================================
# 1. 机械臂模型与实机参数（week5~week7、teleop 已验证）
# ============================================================

DH_PARAMS = [
    {"d": 108.5, "a": 0.0, "alpha": np.pi / 2, "offset": 0.0},
    {"d": 0.0, "a": 105.0, "alpha": 0.0, "offset": np.pi / 2},
    {"d": 0.0, "a": 97.5, "alpha": 0.0, "offset": 0.0},
    {"d": 0.0, "a": 28.22, "alpha": np.pi / 2, "offset": np.pi / 2},
    {"d": 172.70, "a": 0.0, "alpha": 0.0, "offset": 0.0},
]
JOINT_LIMITS_DEG = np.array(
    [[-90.0, 90.0], [-90.0, 90.0], [-90.0, 90.0], [-90.0, 25.0], [-180.0, 180.0]]
)
JOINT_MARGIN_DEG = 2.0           # 任意时刻离限位的最小余量
WINDOW_MARGIN_DEG = 4.0          # 计算取放点俯仰窗口时要求的余量
JOINT_SPEED_LIMIT_DEG_S = 60.0

SERIAL_PORT = "COM5"
BAUDRATE = 115200
SEND_PERIOD_S = 0.02             # 50 Hz 流式下发

GRIP_OPEN_DEG = 0.0              # J6：0° 张开
GRIP_CLOSE_DEG = -40.0           # J6：-40° 闭合（teleop 实测；夹不紧时可调到 -50°）
GRIP_TIME_S = 1.0
SETTLE_S = 0.4


# ============================================================
# 2. 作业纸与木块
# ============================================================
# 作业纸几何来自实物 A4 纸扫描（1855×2683 px）：先拟合纸的四条边，按 210×297 mm 做单应变换，
# 再对六个圆做最小二乘圆拟合（每圆 4000+ 个边缘点，残差 0.2~0.3 mm），最后拟合一个正交网格：
#   s 列 → t 列 97.3 mm；第 2 行 → 第 1 行 95.5 mm，第 2 行 → 第 3 行 93.8 mm（两行间距确实不等）；
#   圆直径 61.0~61.6 mm；图案相对纸边转了约 0.4°（打印歪斜），六个圆心对网格的残差 ≤ 0.3 mm。
# 下面是以 s2、t2 连线中点为原点、u 轴 s→t、v 轴指向第 1 行的局部坐标（mm）。
# 手机测距仪交叉核对（整厘米读数）：纸左边→s1 5、s1→t1 10、t1→纸右边 6、纸上边→t1 4、
#   行距 9 / 9、t3→纸下边 7 cm，与扫描值在 1 cm 读数精度内一致。
# 扫描比例由纸边推得，误差约 ±1%：用钢尺量一下 s1→s3 圆心距（约 189.2 mm）用 --s1s3 传入即可校准。
SHEET_UV = {"s1": (-48.65, 95.49), "s2": (-48.65, 0.0), "s3": (-48.65, -93.75),
            "t1": (48.65, 95.49), "t2": (48.65, 0.0), "t3": (48.65, -93.75)}
SHEET_S1S3 = SHEET_UV["s1"][1] - SHEET_UV["s3"][1]   # 189.24 mm
SHEET_SCALE_U = 1.0              # --s1s3 / --pitch 校准后的缩放
SHEET_SCALE_V = 1.0

TABLE_Z = 52.0                   # 桌面（作业纸）在基座系中的高度（第六周标定）
BLOCK_SIZE = 30.0                # 木块边长 h

CIRCLE_DIAMETER = 61.0           # 区域圆直径（扫描最小值，偏保守）
SHEET_CENTER_XY = (260.0, 0.0)   # 作业纸中心（s2、t2 圆心连线中点）在基座系中的位置（--mode scan：俯仰窗口最宽的摆放）
SHEET_YAW_DEG = 0.0              # u 轴（s→t）相对 +X_B 的转角

POINT_XY_OVERRIDE = {}           # 实测圆心：{"s1": (x, y), ...}；给出后覆盖上面的作业纸几何
CALIB_FILE = Path(__file__).resolve().parent / "sheet_calib.json"   # --mode calib 的记录，存在即自动加载
CALIB_SOURCE = ""
TAUGHT = {}                      # 标定时记录的实际姿态 {"s1": {"pitch": .., "z": .., "q": [..]}}，逆解兜底用
PLACE_XY_COMP = {}               # 逐点放置补偿 {"t1": (dx, dy), ...}（实机标定）
PICK_XY_COMP = {}                # 逐点夹取补偿
GRASP_Z_OFFSET = 0.0             # 夹取点相对木块中心的高度（+ 为偏上）
HOLD_SAG_COMP = 0.0              # 持块时末端下垂补偿（+ 为抬高指令）

# 竖直进出、放置与越障
APPROACH = 12.0                  # 在取放点上方以固定俯仰竖直进出的距离
RELEASE_GAP = 1.0                # 松爪时木块最低棱高出纸面的距离
SLOW_ZONE = 5.0
TRANSIT_CLEAR_HELD = 25.0        # 横移时手中木块底面高出最高木块顶面
TRANSIT_CLEAR_EMPTY = 25.0       # 空爪横移时夹指最低点高出最高木块顶面
MAX_PLACE_TILT_DEG = 26.0        # 放置倾角上限：棱先着地、绕棱翻平，<45° 不会翻倒；留给纸转角/比例误差的余量
POSITION_TOLERANCE = 5.0         # 取放点位置容差：所选俯仰对圆心沿半径 ±该值仍然可行（纸的摆放/标定误差）
REGION_MARGIN = 2.0              # 规划时木块底面到圆边界至少留的余量（mm）

# 速度
TRANSIT_SPEED = 40.0             # mm/s
VERTICAL_SPEED = 20.0
SLOW_SPEED = 5.0
MOVEJ_SPEED_DEG_S = 30.0
MOVEJ_MIN_TIME = 1.5

# 夹爪几何（工具系：z 沿夹爪轴线朝外，y 为张合方向，原点为 D-H 末端=夹取点）
FINGER_TIP = 8.0
FINGER_BACK = 15.0
FINGER_HALF_W = 6.0
FINGER_THICK = 4.0
OPEN_HALF_GAP = 22.0
PALM_DEPTH = 25.0
FINGER_TABLE_CLEAR = 2.0         # 夹指最低点至少高出纸面

# 碰撞检查
ARM_RADIUS = 15.0
ARM_CLEARANCE = 4.0
MAX_FK_ERROR = 0.5

OUTPUT_DIR = Path(__file__).resolve().parent / "transport_output"
PITCH_GRID = np.arange(-85.0, 20.0, 0.5)

POINT_NAMES = ("s1", "s2", "s3", "t1", "t2", "t3")
TASK1 = (("s1", "t1"), ("s2", "t3"), ("s3", "t2"))   # 必做
TASK2 = (("t1", "s2"), ("t2", "s1"), ("t3", "s3"))   # 选做（接在必做之后）


# ============================================================
# 3. 正运动学（numpy，与 roboticstoolbox 标准 D-H 一致）
# ============================================================

def dh_matrix(theta, d, a, alpha):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


def fk_frames(q_deg):
    T = np.eye(4)
    out = [T]
    for qi, p in zip(np.deg2rad(np.asarray(q_deg, float)[:5]), DH_PARAMS):
        T = T @ dh_matrix(qi + p["offset"], p["d"], p["a"], p["alpha"])
        out.append(T)
    return out


def tool_pose(q_deg):
    return fk_frames(q_deg)[-1]


def grip_point(q_deg):
    return tool_pose(q_deg)[:3, 3]


def tool_pitch(q_deg):
    """夹爪轴线俯仰：0° 水平朝外，-90° 竖直朝下（= q2+q3+q4+90°）。"""
    return float(q_deg[1] + q_deg[2] + q_deg[3] + 90.0)


def build_rtb_model():
    if rtb is None:
        return None
    return rtb.DHRobot([rtb.RevoluteDH(d=p["d"], a=p["a"], alpha=p["alpha"], offset=p["offset"])
                        for p in DH_PARAMS], name="Week8_Transport_Arm")


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ============================================================
# 4. 固定俯仰解析逆解
# ============================================================

class IKError(RuntimeError):
    pass


_W2T = {}


def wrist_to_tip(pitch):
    """q1=0 时，J4 轴心到夹取点的向量（径向、侧向、竖直），只与俯仰有关。"""
    key = float(pitch)
    if key not in _W2T:
        fr = fk_frames([0.0, 0.0, 0.0, pitch - 90.0, 0.0])
        _W2T[key] = fr[-1][:3, 3] - fr[3][:3, 3]
    return _W2T[key]


def joint_margin(q):
    q = np.asarray(q, float)[:4]
    return float(np.min(np.minimum(q - JOINT_LIMITS_DEG[:4, 0], JOINT_LIMITS_DEG[:4, 1] - q)))


def ik_candidates(p, pitch):
    p = np.asarray(p, float)
    w_r, w_y, w_z = wrist_to_tip(pitch)
    rho2 = p[0] ** 2 + p[1] ** 2
    if rho2 < w_y ** 2:
        return []
    radial = np.sqrt(rho2 - w_y ** 2)
    q1 = np.arctan2(p[1], p[0]) - np.arctan2(w_y, radial)
    wr, wz = radial - w_r, p[2] - w_z - DH_PARAMS[0]["d"]
    l2, l3 = DH_PARAMS[1]["a"], DH_PARAMS[2]["a"]
    c3 = (wr * wr + wz * wz - l2 * l2 - l3 * l3) / (2 * l2 * l3)
    if abs(c3) > 1.0 + 1e-9:
        return []
    c3 = float(np.clip(c3, -1.0, 1.0))
    out = []
    for s in (-1.0, 1.0):
        q3 = s * np.arccos(c3)
        q2 = np.arctan2(wz, wr) - np.arctan2(l3 * np.sin(q3), l2 + l3 * np.cos(q3)) - np.pi / 2
        q4 = np.deg2rad(pitch - 90.0) - q2 - q3
        q = np.rad2deg([q1, q2, q3, q4, 0.0])
        q[:4] = (q[:4] + 180.0) % 360.0 - 180.0
        out.append(q)
    return out


def best_margin(p, pitch):
    return max((joint_margin(q) for q in ik_candidates(p, pitch)), default=-np.inf)


def _m(extra=0.0):
    """规划用关节余量。标定模式下若有点只能贴着限位到达，JOINT_MARGIN_DEG 会被置 0，余量全部取消。"""
    return JOINT_MARGIN_DEG + extra if JOINT_MARGIN_DEG > 0 else 0.0


def solve_ik(p, pitch, q_prev=None, margin=None):
    margin = _m() if margin is None else margin
    valid = [q for q in ik_candidates(p, pitch) if joint_margin(q) >= margin]
    if not valid:
        raise IKError(f"点 {np.round(p, 1)} mm 在俯仰 {pitch:.1f}° 下无限位内逆解")
    if q_prev is None:
        return max(valid, key=joint_margin)
    return min(valid, key=lambda q: float(np.sum((q[:4] - q_prev[:4]) ** 2)))


FLEX_PITCH_GRID = np.arange(-90.0, 30.01, 0.25)


def solve_ik_flex(p, pitch, q_prev=None):
    """优先按给定俯仰求逆解；无解时就近换俯仰（不要求前后夹爪角度一致），先保留余量、再贴限位。"""
    try:
        return solve_ik(p, pitch, q_prev)
    except IKError:
        pass
    for margin in (_m(), 0.0):
        for x in sorted(FLEX_PITCH_GRID, key=lambda v: abs(v - pitch)):
            valid = [q for q in ik_candidates(p, x) if joint_margin(q) >= margin]
            if valid:
                if q_prev is None:
                    return max(valid, key=joint_margin)
                return min(valid, key=lambda q: float(np.sum((q[:4] - q_prev[:4]) ** 2)))
    raise IKError(f"点 {np.round(p, 1)} mm 在任何俯仰下都没有限位内逆解（超出机械臂可达范围）")


def pitch_window(p_low, p_high, margin=None):
    """在 p_low→p_high 的竖直段上全程可行的俯仰区间（可能为空）。"""
    margin = WINDOW_MARGIN_DEG if margin is None else margin
    pts = [np.asarray(p_low) + t * (np.asarray(p_high) - np.asarray(p_low)) for t in (0, 0.5, 1)]
    ok = [pt for pt in PITCH_GRID if all(best_margin(p, pt) >= margin for p in pts)]
    return (float(min(ok)), float(max(ok))) if ok else None


def best_pitch(p, margin=None):
    margin = _m(2.0) if margin is None else margin
    m, pt = max((best_margin(p, x), x) for x in PITCH_GRID)
    if m < margin:
        raise IKError(f"点 {np.round(p, 1)} mm 在任何俯仰下都缺少关节余量")
    return float(pt)


# ============================================================
# 5. 作业纸几何、取放点俯仰与每趟倾角
# ============================================================

def sheet_points(center_xy=None, yaw_deg=None):
    """六个圆心在基座系中的 (x, y)。"""
    if POINT_XY_OVERRIDE and center_xy is None and yaw_deg is None:
        return {k: np.asarray(v, float) for k, v in POINT_XY_OVERRIDE.items()}
    cx, cy = SHEET_CENTER_XY if center_xy is None else center_xy
    yaw = np.deg2rad(SHEET_YAW_DEG if yaw_deg is None else yaw_deg)
    u_axis = np.array([np.cos(yaw), np.sin(yaw)])
    v_axis = np.array([-np.sin(yaw), np.cos(yaw)])
    return {k: np.array([cx, cy]) + u * SHEET_SCALE_U * u_axis + v * SHEET_SCALE_V * v_axis
            for k, (u, v) in SHEET_UV.items()}


def col_pitch():
    return (SHEET_UV["t2"][0] - SHEET_UV["s2"][0]) * SHEET_SCALE_U


def row_pitches():
    return (SHEET_UV["s1"][1] * SHEET_SCALE_V, -SHEET_UV["s3"][1] * SHEET_SCALE_V)


def block_center_at(xy):
    return np.array([xy[0], xy[1], TABLE_Z + BLOCK_SIZE / 2])


TILT_LIFT_ALLOW = 8.0            # 倾斜放置时夹取点比水平放置高出的余量（≈ h/2·sin(倾角)，18° 时约 4.6 mm）


_WIN_CACHE = {}


def point_window(xy, tol=None):
    """取放点的俯仰窗口：覆盖从夹取高度到悬停点的整段竖直行程（含倾斜放置时的抬高量），
    并且对圆心沿半径偏移 ±tol 仍然可行（取三处窗口的交集）。窗口只与半径有关。"""
    tol = POSITION_TOLERANCE if tol is None else tol
    r = float(np.hypot(xy[0], xy[1]))
    lo, hi = -np.inf, np.inf
    for dr in ((-tol, 0.0, tol) if tol > 0 else (0.0,)):
        key = (round(r + dr, 2), GRASP_Z_OFFSET, RELEASE_GAP, APPROACH, HOLD_SAG_COMP, WINDOW_MARGIN_DEG)
        if key not in _WIN_CACHE:
            g = np.array([r + dr, 0.0, TABLE_Z + BLOCK_SIZE / 2 + GRASP_Z_OFFSET])
            top = g + np.array([0.0, 0.0, RELEASE_GAP + TILT_LIFT_ALLOW + APPROACH + HOLD_SAG_COMP])
            _WIN_CACHE[key] = pitch_window(g - np.array([0.0, 0.0, 1.0]), top)
        w = _WIN_CACHE[key]
        if w is None:
            return None
        lo, hi = max(lo, w[0]), min(hi, w[1])
    return (lo, hi) if hi - lo >= 1.0 else None


def choose_pitches(W_pick, W_place):
    """夹取俯仰 p1 ∈ W_pick、放置俯仰 p2 ∈ W_place，使 |p2 - p1| 最小。"""
    lo, hi = max(W_pick[0], W_place[0]), min(W_pick[1], W_place[1])
    if hi >= lo:                         # 有公共俯仰：取公共区间中点，放下时木块完全水平
        p = (lo + hi) / 2
        return p, p
    if W_pick[1] < W_place[0]:           # 取料点窗口整体更“竖直”
        return W_pick[1], W_place[0]
    return W_pick[0], W_place[1]


def fallback_window(name, xy):
    """常规俯仰窗口为空时的兜底：只要求夹取高度这一点可达（不再要求整段竖直行程同一俯仰、不留余量），
    优先用标定时实际到达的俯仰。返回退化窗口 (p, p)；仍不可达返回 None。"""
    g = np.array([xy[0], xy[1], TABLE_Z + BLOCK_SIZE / 2 + GRASP_Z_OFFSET])
    cands = []
    if name in TAUGHT:
        cands.append(float(TAUGHT[name]["pitch"]))
    cands += sorted(PITCH_GRID, key=lambda x: -best_margin(g, x))[:1]
    for pt in cands:
        if best_margin(g, pt) >= 0.0:
            return (pt, pt)
    return None


FALLBACK_POINTS = set()


def plan_transfers(task, points=None, fallback=None):
    global JOINT_MARGIN_DEG
    points = sheet_points() if points is None else points
    fallback = True if fallback is None else fallback
    windows = {k: point_window(v) for k, v in points.items()}
    for k, w in list(windows.items()):
        if w is None and fallback:
            windows[k] = fallback_window(k, points[k])
            if windows[k] is not None:
                FALLBACK_POINTS.add(k)
                JOINT_MARGIN_DEG = 0.0           # 有点只能贴限位到达：取消全部关节余量
    bad = [k for k, w in windows.items() if w is None]
    if bad:
        raise IKError(f"点 {bad} 在夹取高度（z = {TABLE_Z + BLOCK_SIZE / 2 + GRASP_Z_OFFSET:.1f} mm）任何俯仰都无逆解："
                      "用 --mode calib 重新遥控标定，或挪动作业纸")
    moves = list(TASK1) + (list(TASK2) if task == 2 else [])
    out = []
    for a, b in moves:
        p1, p2 = choose_pitches(windows[a], windows[b])
        out.append({"from": a, "to": b, "pick_pitch": p1, "place_pitch": p2, "tilt": p2 - p1,
                    "W_from": windows[a], "W_to": windows[b]})
    return out, windows


# ============================================================
# 6. 轨迹段
# ============================================================

@dataclass
class Segment:
    label: str
    kind: str                 # movej / line / grip / dwell / wait
    q: np.ndarray             # N x 6 指令角（deg）
    duration: float
    ref: np.ndarray | None = None
    pitch: np.ndarray | None = None
    held: str | None = None
    target: str | None = None
    scene: dict = field(default_factory=dict)
    phase: str = ""
    state: str = ""           # 状态机状态名
    message: str = ""
    held_rel: np.ndarray | None = None   # 本段手中木块相对夹爪的位姿（同一木块可能被夹多次）


def quintic(n):
    t = np.linspace(0.0, 1.0, n)
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def n_samples(duration):
    return max(2, int(round(duration / SEND_PERIOD_S)) + 1)


def add_grip_col(q5, grip):
    q5 = np.atleast_2d(q5)
    return np.hstack([q5, np.full((len(q5), 1), grip)])


def line_points(p0, p1, pitch0, pitch1, q_start, duration):
    s = quintic(n_samples(duration))
    pts = p0 + s[:, None] * (p1 - p0)
    pitches = (1 - s) * pitch0 + s * pitch1
    qs, q = [], q_start
    for p, pt in zip(pts, pitches):
        q = solve_ik_flex(p, pt, q)
        qs.append(q)
    qs = np.array(qs)
    return qs, pts, np.array([tool_pitch(x) for x in qs])


def const_pitch_top(p_low, pitch, z_max, step=2.0, margin=None):
    margin = _m(1.0) if margin is None else margin
    p = np.asarray(p_low, float)
    z = p[2]
    while z + step <= z_max + 1e-9 and best_margin((p[0], p[1], z + step), pitch) >= margin:
        z += step
    if z_max - z < step and best_margin((p[0], p[1], z_max), pitch) >= margin:
        z = z_max
    return float(z)


def approach_top(g, pitch):
    """竖直进出的悬停高度：常规为 g + APPROACH；兜底点在该俯仰下够不到时取能到的最高处。"""
    return max(float(g[2]), const_pitch_top(g, pitch, g[2] + APPROACH, step=1.0, margin=_m()))


def line_feasible(p0, p1, pitch0, pitch1, n=7, margin=None):
    margin = _m(1.0) if margin is None else margin
    return all(best_margin(p0 + t * (p1 - p0), (1 - t) * pitch0 + t * pitch1) >= margin
               for t in np.linspace(0.0, 1.0, n))


# ============================================================
# 7. 规划器（状态机）
# ============================================================

class Planner:
    def __init__(self, task, points=None, plan=True):
        self.task = task
        self.points = sheet_points() if points is None else points
        self.transfers, self.windows = plan_transfers(task, self.points) if plan else ([], {})
        self.segments: list[Segment] = []
        self.q = np.zeros(5)
        self.grip = GRIP_OPEN_DEG
        self.held = None
        self.held_rel = {}
        # 木块按起始位置命名：B1 从 s1 出发，B2 从 s2，B3 从 s3；木块一边正对基座
        self.blocks = {}
        for i, s in enumerate(("s1", "s2", "s3"), start=1):
            xy = self.points[s] + np.asarray(PICK_XY_COMP.get(s, (0.0, 0.0)))
            self.blocks[f"B{i}"] = (block_center_at(xy), float(np.arctan2(xy[1], xy[0])))
        self.occupant = {"s1": "B1", "s2": "B2", "s3": "B3", "t1": None, "t2": None, "t3": None}
        self.phase = ""
        self.state = ""
        self.under = None
        self.placements = []     # 每次放置：木块、目标点、倾角、底面到圆边界余量

    # ---------- 状态 ----------
    def scene(self):
        return {k: (c.copy(), yaw) for k, (c, yaw) in self.blocks.items() if k != self.held}

    def top_of_obstacles(self):
        tops = [c[2] + BLOCK_SIZE / 2 for k, (c, _) in self.blocks.items() if k != self.held]
        return max(tops + [TABLE_Z])

    def _push(self, seg):
        seg.held, seg.scene, seg.phase, seg.state = self.held, self.scene(), self.phase, self.state
        seg.held_rel = self.held_rel[self.held].copy() if self.held else None
        if self.segments:
            seg.q[0] = self.segments[-1].q[-1]
        self.segments.append(seg)
        self.q = seg.q[-1, :5].copy()
        self.grip = float(seg.q[-1, 5])

    # ---------- 基本动作 ----------
    def movej(self, label, q_to, duration=None):
        dq = float(np.max(np.abs(q_to[:5] - self.q)))
        duration = duration or max(MOVEJ_MIN_TIME, 1.875 * dq / MOVEJ_SPEED_DEG_S)
        s = quintic(n_samples(duration))[:, None]
        q = self.q + s * (q_to[:5] - self.q)
        self._push(Segment(label, "movej", add_grip_col(q, self.grip), duration))

    def line(self, label, p1, pitch1, speed, target=None):
        p0 = grip_point(self.q)
        pitch0 = tool_pitch(self.q)
        dist = float(np.linalg.norm(p1 - p0))
        duration = max(0.5, dist / speed, 1.875 * abs(pitch1 - pitch0) / MOVEJ_SPEED_DEG_S)
        base_duration = duration
        for _ in range(6):
            if duration > 8.0 * base_duration:
                raise IKError(f"{label}: 逆解分支跳变（放慢也无法连续）")
            qs, pts, pitches = line_points(p0, np.asarray(p1, float), pitch0, pitch1, self.q, duration)
            vel = np.max(np.abs(np.diff(qs[:, :4], axis=0))) / SEND_PERIOD_S if len(qs) > 1 else 0
            if vel <= 0.9 * JOINT_SPEED_LIMIT_DEG_S:
                break
            duration *= vel / (0.85 * JOINT_SPEED_LIMIT_DEG_S)
        else:
            raise IKError(f"{label}: 关节角速度无法满足（可能经过奇异或分支切换）")
        self._push(Segment(label, "line", add_grip_col(qs, self.grip), duration,
                           ref=pts, pitch=pitches, target=target))

    def vertical(self, label, z_to, speed, target=None):
        p = grip_point(self.q).copy()
        p[2] = z_to
        self.line(label, p, tool_pitch(self.q), speed, target=target)

    def gripper(self, label, angle, target=None):
        n = n_samples(GRIP_TIME_S)
        q = add_grip_col(np.repeat(self.q[None, :], n, axis=0), 0.0)
        q[:, 5] = np.linspace(self.grip, angle, n)
        self._push(Segment(label, "grip", q, GRIP_TIME_S, target=target))

    def dwell(self, label, seconds, target=None):
        n = n_samples(seconds)
        q = add_grip_col(np.repeat(self.q[None, :], n, axis=0), self.grip)
        self._push(Segment(label, "dwell", q, seconds, target=target))

    # ---------- 越障 ----------
    def transit_z(self):
        top = self.top_of_obstacles()
        if self.held:
            return top + TRANSIT_CLEAR_HELD + BLOCK_SIZE / 2 + abs(GRASP_Z_OFFSET) + 6.0
        return top + TRANSIT_CLEAR_EMPTY + FINGER_TIP + FINGER_HALF_W

    def travel_to_hover(self, label, hover, hover_pitch, target):
        """竖直上升 → 高处横移 → 竖直下降到目标上方。贴近物体的一段夹爪姿态不变、纯竖直平动。"""
        here = grip_point(self.q)
        zt = max(self.transit_z(), hover[2], here[2])
        pitch_here = tool_pitch(self.q)
        z_up_c = const_pitch_top(here, pitch_here, zt)
        z_dn_c = const_pitch_top(hover, hover_pitch, zt)
        up_c = np.array([here[0], here[1], z_up_c])
        up = np.array([here[0], here[1], zt])
        above = np.array([hover[0], hover[1], zt])
        dn_c = np.array([hover[0], hover[1], z_dn_c])
        cand_a = [x for x in sorted(PITCH_GRID, key=lambda x: abs(x - pitch_here))
                  if best_margin(up, x) >= _m(1)][:40]
        cand_b = [x for x in sorted(PITCH_GRID, key=lambda x: abs(x - hover_pitch))
                  if best_margin(above, x) >= _m(1)][:40]

        def candidates():
            for a in cand_a:
                if not line_feasible(up_c, up, pitch_here, a):
                    continue
                for b in cand_b:
                    if line_feasible(up, above, a, b, n=9) and line_feasible(above, dn_c, b, hover_pitch):
                        yield a, b

        saved = (len(self.segments), self.q.copy(), self.grip, self.under)
        last_err, tried, ok = None, 0, False
        for a, b in candidates():
            tried += 1
            try:
                if z_up_c - here[2] > 0.5:
                    self.line(f"{label}/竖直上升", up_c, pitch_here, VERTICAL_SPEED, target=self.under)
                if zt - z_up_c > 0.5 or abs(a - pitch_here) > 0.1:
                    self.line(f"{label}/上升转角", up, a, VERTICAL_SPEED, target=self.under)
                self.line(f"{label}/横移", above, b, TRANSIT_SPEED)
                if zt - z_dn_c > 0.5 or abs(b - hover_pitch) > 0.1:
                    self.line(f"{label}/下降转角", dn_c, hover_pitch, VERTICAL_SPEED, target=target)
                if z_dn_c - hover[2] > 0.5:
                    self.line(f"{label}/竖直下降", hover, hover_pitch, VERTICAL_SPEED, target=target)
                ok = True
                break
            except IKError as e:
                last_err = e
                del self.segments[saved[0]:]
                self.q, self.grip, self.under = saved[1].copy(), saved[2], saved[3]
                if tried >= 30:
                    break
        if not ok and not self._transit_movej(label, here, pitch_here, up_c, z_up_c, up, zt,
                                              above, hover, hover_pitch, target) \
                and not self._transit_free(label, up, above, hover, hover_pitch, target):
            raise IKError(f"{label}: 找不到可行的越障路线（越障高度 {zt:.0f} mm）"
                          + (f"；{last_err}" if last_err else ""))
        self.under = target

    def _transit_free(self, label, up, above, hover, hover_pitch, target):
        """最后的越障方案：不要求俯仰一致——升到越障高度（俯仰随逆解就近变化）→ 关节空间移到目标上方 →
        降到悬停点（俯仰逐渐过渡到取放俯仰）。"""
        saved = (len(self.segments), self.q.copy(), self.grip, self.under)
        try:
            self.line(f"{label}/上升", up, tool_pitch(self.q), VERTICAL_SPEED, target=self.under)
            self.movej(f"{label}/高处关节移动", solve_ik_flex(above, hover_pitch, self.q))
            self.line(f"{label}/下降", hover, hover_pitch, VERTICAL_SPEED, target=target)
            return True
        except IKError:
            del self.segments[saved[0]:]
            self.q, self.grip, self.under = saved[1].copy(), saved[2], saved[3]
            return False

    def _transit_movej(self, label, here, pitch_here, up_c, z_up_c, up, zt, above, hover, hover_pitch, target):
        """备选越障：竖直升到越障高度 → 高处关节空间运动（可切换肘部构型）→ 竖直下降。"""
        saved = (len(self.segments), self.q.copy(), self.grip, self.under)
        cand_a = [x for x in sorted(PITCH_GRID, key=lambda x: abs(x - pitch_here))
                  if line_feasible(up_c, up, pitch_here, x)][:20]
        cand_b = [x for x in sorted(PITCH_GRID, key=lambda x: abs(x - hover_pitch))
                  if best_margin(above, x) >= _m(1) and line_feasible(above, hover, x, hover_pitch)][:20]
        for a in cand_a:
            for b in cand_b:
                for q_top in sorted((q for q in ik_candidates(above, b) if joint_margin(q) >= _m(1)),
                                    key=lambda q: -joint_margin(q)):
                    try:
                        if z_up_c - here[2] > 0.5:
                            self.line(f"{label}/竖直上升", up_c, pitch_here, VERTICAL_SPEED, target=self.under)
                        if zt - z_up_c > 0.5 or abs(a - pitch_here) > 0.1:
                            self.line(f"{label}/上升转角", up, a, VERTICAL_SPEED, target=self.under)
                        self.movej(f"{label}/高处关节换构型", q_top)
                        if np.linalg.norm(grip_point(self.q) - hover) > 0.5 or abs(b - hover_pitch) > 0.1:
                            self.line(f"{label}/下降", hover, hover_pitch, VERTICAL_SPEED, target=target)
                        return True
                    except IKError:
                        del self.segments[saved[0]:]
                        self.q, self.grip, self.under = saved[1].copy(), saved[2], saved[3]
        return False

    # ---------- 状态机：取、放 ----------
    def pick(self, name, pitch, label):
        """MOVE_TO_PICK → DESCEND → GRASP → LIFT。木块平放，任何可行俯仰都能贴合两侧面。"""
        center, yaw = self.blocks[name]
        g = center + np.array([0.0, 0.0, GRASP_Z_OFFSET])
        hover = np.array([g[0], g[1], approach_top(g, pitch)])
        self.state = "MOVE_TO_PICK"
        self.travel_to_hover(f"{label}/去取料点", hover, pitch, target=name)
        self.state = "DESCEND_PICK"
        self.descend(label, g, name)
        self.dwell(f"{label}/静置", SETTLE_S, target=name)
        self.state = "GRASP"
        self.gripper(f"{label}/夹紧", GRIP_CLOSE_DEG, target=name)
        self.dwell(f"{label}/保持", SETTLE_S, target=name)
        B = np.eye(4)
        B[:3, :3] = rot_z(yaw)
        B[:3, 3] = center
        self.held_rel[name] = np.linalg.inv(tool_pose(self.q)) @ B
        self.held = name
        self.state = "LIFT"
        self.ascend(label, g, hover[2], HOLD_SAG_COMP, name)
        self.under = name

    def release_grip_point(self, name, target_xy, pitch):
        """求放置夹取点：手中木块（可能倾斜）中心 xy 对准目标、最低棱高出纸面 RELEASE_GAP。"""
        g = np.array([target_xy[0], target_xy[1], TABLE_Z + BLOCK_SIZE / 2 + RELEASE_GAP + 5.0])
        q = self.q
        for _ in range(4):
            q = solve_ik_flex(g, pitch, q)
            Tb = tool_pose(q) @ self.held_rel[name]
            local = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * BLOCK_SIZE / 2
            corners = (Tb[:3, :3] @ local.T).T + Tb[:3, 3]
            g[:2] += np.asarray(target_xy) - Tb[:2, 3]
            g[2] += TABLE_Z + RELEASE_GAP - corners[:, 2].min()
        return g + np.array([0.0, 0.0, HOLD_SAG_COMP])

    def place(self, name, point, pitch, label):
        """MOVE_TO_PLACE → DESCEND → RELEASE → RETREAT。"""
        target_xy = self.points[point] + np.asarray(PLACE_XY_COMP.get(point, (0.0, 0.0)))
        g = self.release_grip_point(name, target_xy, pitch)
        hover = np.array([g[0], g[1], approach_top(g, pitch)])
        self.state = "MOVE_TO_PLACE"
        self.travel_to_hover(f"{label}/去落料点", hover, pitch, target=name)
        self.state = "DESCEND_PLACE"
        self.descend(label, g, name)
        self.dwell(f"{label}/静置", SETTLE_S, target=name)
        Tb = tool_pose(self.q) @ self.held_rel[name]
        yaw = float(np.arctan2(Tb[1, 0], Tb[0, 0]))
        tilt = float(np.degrees(np.arccos(np.clip(Tb[2, 2], -1.0, 1.0))))
        self.state = "RELEASE"
        self.gripper(f"{label}/松开", GRIP_OPEN_DEG, target=name)
        self.dwell(f"{label}/保持", SETTLE_S, target=name)
        # 松爪后木块绕着地棱翻平：底面中心落在当前中心的水平投影处（偏移 < 1 mm）
        settled = np.array([Tb[0, 3], Tb[1, 3], TABLE_Z + BLOCK_SIZE / 2])
        self.held = None
        self.blocks[name] = (settled, yaw)
        self.occupant[point] = name
        c = self.points[point]
        off = float(np.linalg.norm(settled[:2] - c))
        margin = CIRCLE_DIAMETER / 2 - (off + BLOCK_SIZE / np.sqrt(2))
        self.placements.append({"block": name, "point": point, "tilt": tilt, "center_offset": off,
                                "region_margin": margin, "phase": self.phase})
        self.state = "RETREAT"
        self.ascend(label, g, hover[2], 0.0, name)
        self.under = name

    def descend(self, label, g, name):
        """悬停点 → 夹取/松爪点：先常速到上方 SLOW_ZONE，再慢降（悬停点较低时直接慢降）。"""
        if grip_point(self.q)[2] - g[2] > SLOW_ZONE + 0.5:
            self.vertical(f"{label}/下降", g[2] + SLOW_ZONE, VERTICAL_SPEED, target=name)
        self.vertical(f"{label}/慢降", g[2], SLOW_SPEED, target=name)

    def ascend(self, label, g, z_top, sag, name):
        z_slow = min(g[2] + SLOW_ZONE, z_top) + sag
        self.vertical(f"{label}/慢起", z_slow, SLOW_SPEED, target=name)
        if z_top + sag - z_slow > 0.5:
            self.vertical(f"{label}/{'抬起' if self.held else '撤离'}", z_top + sag, VERTICAL_SPEED, target=name)

    def go_home(self, label):
        self.state = "HOME"
        self.rise(label)
        self.movej(f"{label}/回零", np.zeros(5), duration=3.0)

    def rise(self, label):
        """原地竖直升到越障高度（先保持俯仰，到头后再边升边转）。"""
        here = grip_point(self.q)
        zt = max(self.transit_z(), here[2])
        if zt - here[2] > 0.5:
            pt = tool_pitch(self.q)
            z_c = const_pitch_top(here, pt, zt)
            up_c = np.array([here[0], here[1], z_c])
            if z_c - here[2] > 0.5:
                self.line(f"{label}/竖直上升", up_c, pt, VERTICAL_SPEED, target=self.under)
            if zt - z_c > 0.5:
                for a in sorted(PITCH_GRID, key=lambda x: abs(x - pt)):
                    if line_feasible(up_c, np.array([here[0], here[1], zt]), pt, a):
                        self.line(f"{label}/上升转角", np.array([here[0], here[1], zt]), a,
                                  VERTICAL_SPEED, target=self.under)
                        break

    def start_from_zero(self, first_xy):
        self.state = "READY"
        p = np.array([first_xy[0], first_xy[1], self.transit_z()])
        try:
            pt = best_pitch(p)
        except IKError:
            pt = -45.0
        self.movej("零位→准备", solve_ik_flex(p, pt))

    # ---------- 任务 ----------
    def run(self):
        for i, tr in enumerate(self.transfers):
            a, b = tr["from"], tr["to"]
            self.phase = "必做" if i < len(TASK1) else "选做"
            name = self.occupant[a]
            if name is None:
                raise IKError(f"{a} 处没有木块，无法执行 {a}→{b}")
            if self.occupant[b] is not None:
                raise IKError(f"{b} 处已有木块 {self.occupant[b]}，无法执行 {a}→{b}")
            if not self.segments:
                self.start_from_zero(self.points[a])
            label = f"{self.phase}{i % len(TASK1) + 1}:{a}→{b}"
            self.pick(name, tr["pick_pitch"], f"{label}/取")
            self.occupant[a] = None
            self.place(name, b, tr["place_pitch"], f"{label}/放")
        self.go_home("完成")


def build_plan(task):
    pl = Planner(task)
    pl.run()
    return pl


# ============================================================
# 8. 离线检查
# ============================================================

def block_corners(center, yaw, h=BLOCK_SIZE):
    R = rot_z(yaw)
    d = h / 2
    return np.array([R @ np.array([sx * d, sy * d, sz * d]) + center
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def held_corners(rel, q):
    T = tool_pose(q) @ rel
    d = BLOCK_SIZE / 2
    local = np.array([[sx * d, sy * d, sz * d, 1.0] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
                     + [[0, 0, 0, 1.0]])
    return (T @ local.T).T[:, :3], T


def point_in_block(p, center, yaw, expand):
    """连杆采样点（半径 expand 的球）与木块是否相交：按点到长方体的真实距离判断，
    避免把木块各向外扩成方盒后在棱角处过度保守。"""
    d = rot_z(yaw).T @ (np.asarray(p) - center)
    out = np.maximum(np.abs(d) - BLOCK_SIZE / 2, 0.0)
    return bool(np.linalg.norm(out) <= expand)


def obb_penetration(c1, R1, c2, R2, half=BLOCK_SIZE / 2):
    axes = [R1[:, i] for i in range(3)] + [R2[:, i] for i in range(3)]
    axes += [np.cross(R1[:, i], R2[:, j]) for i in range(3) for j in range(3)]
    d = np.asarray(c2, float) - np.asarray(c1, float)
    best = -np.inf
    for L in axes:
        n = np.linalg.norm(L)
        if n < 1e-9:
            continue
        L = L / n
        r = half * (np.abs(R1.T @ L).sum() + np.abs(R2.T @ L).sum())
        best = max(best, abs(d @ L) - r)
    return float(best)


def arm_samples(q, step=10.0):
    fr = fk_frames(q)
    nodes = [f[:3, 3] for f in fr[1:]]
    lens = [float(np.linalg.norm(b - a)) for a, b in zip(nodes[:-1], nodes[1:])]
    remain = np.cumsum(lens[::-1])[::-1]
    pts, arc = [], []
    for i, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
        n = max(2, int(np.ceil(lens[i] / step)) + 1)
        for t in np.linspace(0, 1, n):
            pts.append(a + t * (b - a))
            arc.append(remain[i] - t * lens[i])
    return np.array(pts), np.array(arc)


def finger_points(q, grip_deg):
    T = tool_pose(q)
    closed = BLOCK_SIZE / 2 + FINGER_THICK / 2
    opened = OPEN_HALF_GAP + FINGER_THICK / 2
    frac = np.clip((grip_deg - GRIP_CLOSE_DEG) / (GRIP_OPEN_DEG - GRIP_CLOSE_DEG + 1e-9), 0.0, 1.0)
    y = closed + frac * (opened - closed)
    pts = []
    for side in (-1, 1):
        for x in (-FINGER_HALF_W, 0.0, FINGER_HALF_W):
            for z in (-FINGER_BACK, 0.0, FINGER_TIP):
                pts.append(T @ np.array([x, side * y, z, 1.0]))
    return np.array(pts)[:, :3]


def check_plan(pl, stride=2, raise_on_fail=True):
    issues = {}

    def issue(msg):
        issues.setdefault(re.sub(r"[-\d.\[\], ]", "", msg), msg)

    st = {"min_joint_margin": np.inf, "max_joint_speed": 0.0, "max_fk_error": 0.0,
          "min_finger_table": np.inf, "min_held_clear": np.inf, "min_held_gap": np.inf}
    prev = None
    for seg in pl.segments:
        q = seg.q
        if prev is not None and np.max(np.abs(q[0] - prev)) > 1e-6:
            issue(f"{seg.label}: 与上一段不连续")
        prev = q[-1]
        m = min(joint_margin(x) for x in q)
        st["min_joint_margin"] = min(st["min_joint_margin"], m)
        if m < JOINT_MARGIN_DEG - 1e-6:
            issue(f"{seg.label}: 关节离限位仅 {m:.1f}°")
        if len(q) > 1:
            v = np.max(np.abs(np.diff(q[:, :5], axis=0))) / SEND_PERIOD_S
            st["max_joint_speed"] = max(st["max_joint_speed"], float(v))
            if v > JOINT_SPEED_LIMIT_DEG_S:
                issue(f"{seg.label}: 关节角速度 {v:.0f}°/s 超限")
        if seg.ref is not None:
            err = max(np.linalg.norm(grip_point(x) - p) for x, p in zip(q, seg.ref))
            st["max_fk_error"] = max(st["max_fk_error"], float(err))
            if err > MAX_FK_ERROR:
                issue(f"{seg.label}: 正解误差 {err:.3f} mm")
        for x in q[::stride]:
            pts, arc = arm_samples(x)
            for p, a in zip(pts, arc):
                if a >= PALM_DEPTH and p[2] - ARM_RADIUS - TABLE_Z < ARM_CLEARANCE:
                    issue(f"{seg.label}: 连杆点 z={p[2]:.0f} 过低（桌面）")
                    break
                for name, (c, yaw) in seg.scene.items():
                    near_tip = a < PALM_DEPTH + ARM_RADIUS
                    if near_tip and name == seg.target:
                        continue
                    exp = (FINGER_THICK if near_tip else ARM_RADIUS) + ARM_CLEARANCE
                    if point_in_block(p, c, yaw, exp):
                        issue(f"{seg.label}: 连杆点 {np.round(p, 0)} 碰到木块 {name}")
                        break
            fp = finger_points(x, x[5])
            st["min_finger_table"] = min(st["min_finger_table"], float(fp[:, 2].min() - TABLE_Z))
            if fp[:, 2].min() < TABLE_Z + FINGER_TABLE_CLEAR:
                issue(f"{seg.label}: 夹指最低点 z={fp[:, 2].min():.1f}，离纸面不足 {FINGER_TABLE_CLEAR:.0f} mm")
            for p in fp:
                for name, (c, yaw) in seg.scene.items():
                    if name == seg.target:
                        continue
                    if point_in_block(p, c, yaw, 0.5):
                        issue(f"{seg.label}: 夹指碰到木块 {name}")
                        break
            if seg.held:
                corners, Tb = held_corners(seg.held_rel, x)
                cz = corners[:, 2].min() - TABLE_Z
                st["min_held_clear"] = min(st["min_held_clear"], float(cz))
                if cz < -0.3:
                    issue(f"{seg.label}: 手中木块 {seg.held} 碰到纸面")
                for name, (c, yaw) in seg.scene.items():
                    sep = obb_penetration(Tb[:3, 3], Tb[:3, :3], c, rot_z(yaw))
                    st["min_held_gap"] = min(st["min_held_gap"], sep)
                    if sep < -0.3:
                        issue(f"{seg.label}: 手中木块 {seg.held} 碰到木块 {name}")
                        break
    for pc in pl.placements:
        if abs(pc["tilt"]) > MAX_PLACE_TILT_DEG:
            issue(f"{pc['block']}→{pc['point']}: 放置倾角 {pc['tilt']:.1f}° 超过 {MAX_PLACE_TILT_DEG:.0f}°")
        if pc["region_margin"] < REGION_MARGIN:
            issue(f"{pc['block']}→{pc['point']}: 底面距圆边界仅 {pc['region_margin']:.1f} mm")
    st["placements"] = pl.placements
    st["issues"] = list(issues.values())
    pl.checks = st
    if st["issues"] and raise_on_fail:
        raise RuntimeError("离线检查未通过：\n  " + "\n  ".join(st["issues"][:15]))
    return st


# ============================================================
# 9. 摘要、导出与可视化
# ============================================================

def describe_layout(points):
    lines = []
    for k in POINT_NAMES:
        p = points[k]
        lines.append(f"  {k}: ({p[0]:6.1f}, {p[1]:6.1f}) mm  r={np.hypot(*p):.0f}  方位 {np.degrees(np.arctan2(p[1], p[0])):5.1f}°")
    return "\n".join(lines)


def summarize(pl):
    L = [f"\n========== 第八周 定点搬运（任务 {'必做' if pl.task == 1 else '必做 + 选做'}） =========="]
    if POINT_XY_OVERRIDE:
        L.append(f"圆心来源：手动标定（{CALIB_SOURCE or '程序内 POINT_XY_OVERRIDE'}）；圆直径 {CIRCLE_DIAMETER:.1f} mm；"
                 f"桌面 z={TABLE_Z}")
    else:
        L.append(f"作业纸：中心 {SHEET_CENTER_XY}，u 轴转角 {SHEET_YAW_DEG:.0f}°，列距 {col_pitch():.1f}、"
                 f"行距 {row_pitches()[0]:.1f}/{row_pitches()[1]:.1f}、圆直径 {CIRCLE_DIAMETER:.1f} mm；桌面 z={TABLE_Z}")
    L.append("圆心位置（基座系）：\n" + describe_layout(pl.points))
    L.append("每趟搬运（夹取俯仰 → 放置俯仰，差值即放下时木块倾角）：")
    for tr, pc in zip(pl.transfers, pl.placements):
        L.append(f"  [{pc['phase']}] {tr['from']}→{tr['to']}（{pc['block']}）：{tr['pick_pitch']:6.1f}° → "
                 f"{tr['place_pitch']:6.1f}°  倾角 {pc['tilt']:4.1f}°  中心偏差 {pc['center_offset']:.1f} mm  "
                 f"底面距圆边界 {pc['region_margin']:.1f} mm")
    c = pl.checks
    total = sum(s.duration for s in pl.segments)
    L.append(f"总时长 {total:.0f} s；关节最小余量 {c['min_joint_margin']:.1f}°；最大角速度 {c['max_joint_speed']:.0f}°/s；"
             f"最大正解误差 {c['max_fk_error']:.4f} mm")
    L.append(f"夹指离纸面最低 {c['min_finger_table']:.1f} mm；横移时手中木块与其他木块最小间隙 "
             f"{c['min_held_gap']:.1f} mm（0 为起吊/落下贴合）")
    if FALLBACK_POINTS:
        L.append(f"兜底点 {sorted(FALLBACK_POINTS)}：常规俯仰窗口为空，改用夹取高度可达的俯仰（有标定时用标定俯仰），"
                 "进出时俯仰随逆解就近变化；已取消关节余量")
    L.append("离线检查：" + ("通过" if not c["issues"] else f"{len(c['issues'])} 项提示（不阻止实机运行）"))
    return "\n".join(L)


def state_sequence(pl):
    seq, last = [], None
    for s in pl.segments:
        if s.state != last:
            seq.append(s.state)
            last = s.state
    return seq


def export(pl, out_dir, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{tag}_trajectory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "phase", "state", "segment", "held", "q1", "q2", "q3", "q4", "q5", "q6",
                    "pitch", "x", "y", "z"])
        t = 0.0
        for seg in pl.segments:
            for x in seg.q:
                p = grip_point(x)
                w.writerow([f"{t:.3f}", seg.phase, seg.state, seg.label, seg.held or ""]
                           + [f"{v:.3f}" for v in x] + [f"{tool_pitch(x):.2f}"] + [f"{v:.2f}" for v in p])
                t += SEND_PERIOD_S
    summary = {
        "task": pl.task, "points": {k: v.round(2).tolist() for k, v in pl.points.items()},
        "sheet": {"center": SHEET_CENTER_XY, "yaw_deg": SHEET_YAW_DEG, "col_pitch": col_pitch(),
                  "row_pitch_12_23": row_pitches(), "points_uv": {k: [u * SHEET_SCALE_U, v * SHEET_SCALE_V] for k, (u, v) in SHEET_UV.items()}, "circle_diameter": CIRCLE_DIAMETER},
        "transfers": [{k: v for k, v in tr.items()} for tr in pl.transfers],
        "duration_s": sum(s.duration for s in pl.segments),
        "checks": pl.checks,
    }
    with open(out_dir / f"{tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)


COLORS = ["#d1495b", "#2e86ab", "#3b8b5a"]


def _use_cjk_font():
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                                              "WenQuanYi Zen Hei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def save_overview(pl, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Polygon
    _use_cjk_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    for k, p in pl.points.items():
        ax1.add_patch(Circle(p, CIRCLE_DIAMETER / 2, fill=False, ls="--", color="0.45"))
        ax1.annotate(k, p + np.array([0, CIRCLE_DIAMETER / 2 + 4]), ha="center", fontsize=9)
    for tr in pl.transfers:
        a, b = pl.points[tr["from"]], pl.points[tr["to"]]
        col = "#d1495b" if (tr["from"], tr["to"]) in TASK1 else "#3b8b5a"
        ax1.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=col, lw=1.6,
                                      connectionstyle="arc3,rad=0.08"))
        lab = a + 0.3 * (b - a)            # 标在路线前 30% 处，避免交叉路线的标注重叠
        ax1.annotate(f"{tr['tilt']:+.0f}°", lab, xytext=(0, 5), textcoords="offset points",
                     fontsize=8, color=col, ha="center")
    for name, (c, yaw) in pl.blocks.items():
        cs = block_corners(c, yaw)[[0, 2, 6, 4], :2]
        ax1.add_patch(Polygon(cs, closed=True, fc=COLORS[int(name[1]) - 1], alpha=0.5, ec="k"))
        ax1.annotate(name, c[:2], ha="center", va="center", fontsize=8)
    ax1.plot(0, 0, "ks")
    ax1.annotate("机械臂基座", (0, 0), xytext=(5, -12), textcoords="offset points", fontsize=8)
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.3)
    ax1.set_xlabel("X_base (mm)")
    ax1.set_ylabel("Y_base (mm)")
    ax1.set_title("俯视：取放区域、搬运路线（红=必做，绿=选做）与放置倾角；色块为最终位置", fontsize=10)
    for seg in pl.segments:
        if seg.kind == "grip":
            x = seg.q[-1]
            fr = fk_frames(x)
            u = np.array([np.cos(np.deg2rad(x[0])), np.sin(np.deg2rad(x[0])), 0.0])
            pts = np.array([[f[:3, 3] @ u, f[2, 3]] for f in fr])
            closing = seg.q[-1, 5] == GRIP_CLOSE_DEG
            ax2.plot(pts[:, 0], pts[:, 1], "-o", ms=3, lw=1.2, color="#2e86ab" if closing else "#c77d1a",
                     alpha=0.7)
    ax2.axhline(TABLE_Z, color="sienna")
    ax2.plot([], [], color="#2e86ab", label="夹紧时姿态")
    ax2.plot([], [], color="#c77d1a", label="松开时姿态")
    ax2.legend(fontsize=8)
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.3)
    ax2.set_xlabel("径向距离 (mm)")
    ax2.set_ylabel("z (mm)")
    ax2.set_title("侧视：每次夹紧/松开时的手臂构型", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def animate(pl, speed=4.0, save_frames=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    _use_cjk_font()
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    faces_idx = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    circle_t = np.linspace(0, 2 * np.pi, 48)
    stride = max(1, int(round(speed)))
    frame = 0
    for seg in pl.segments:
        for i in range(0, len(seg.q), stride):
            x = seg.q[i]
            ax.cla()
            ax.set_xlim(-50, 400)
            ax.set_ylim(-225, 225)
            ax.set_zlim(0, 450)
            ax.set_box_aspect((450, 450, 450))
            for k, p in pl.points.items():
                ax.plot(p[0] + CIRCLE_DIAMETER / 2 * np.cos(circle_t), p[1] + CIRCLE_DIAMETER / 2 * np.sin(circle_t),
                        TABLE_Z, color="0.5", lw=0.8)
            fr = fk_frames(x)
            pts = np.array([f[:3, 3] for f in fr])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", color="0.2", lw=3, ms=4)
            fp = finger_points(x, x[5])
            ax.scatter(fp[:, 0], fp[:, 1], fp[:, 2], s=4, color="k")
            blocks = list(seg.scene.items())
            if seg.held:
                corners, Tb = held_corners(seg.held_rel, x)
                blocks.append((seg.held, (Tb[:3, 3], 0.0, corners[:8])))
            for name, val in blocks:
                cs = val[2] if len(val) == 3 else block_corners(val[0], val[1])
                ax.add_collection3d(Poly3DCollection([cs[f] for f in faces_idx], alpha=0.75,
                                                     facecolor=COLORS[int(name[1]) - 1], edgecolor="k"))
            ax.set_title(f"[{seg.state}] {seg.label}   俯仰 {tool_pitch(x):.0f}°", fontsize=9)
            if save_frames:
                fig.savefig(Path(save_frames) / f"frame_{frame:05d}.png", dpi=70)
            else:
                plt.pause(SEND_PERIOD_S * stride / speed)
            frame += 1
    if not save_frames:
        plt.show()


# ============================================================
# 10. 实机通信与执行（50 Hz 流式下发）
# ============================================================

def pack_frame(q_rad):
    q_rad = np.asarray(q_rad, dtype=np.float64)
    if q_rad.shape != (6,) or not np.all(np.isfinite(q_rad)):
        raise ValueError("必须传入 6 个有限的关节弧度值")
    tx = bytearray(16)
    tx[0] = 0xAA
    for i, v in enumerate(q_rad):
        v = max(min(int(v * 1000), 32767), -32768)
        tx[1 + 2 * i] = (v >> 8) & 0xFF
        tx[2 + 2 * i] = v & 0xFF
    tx[13] = 0x01
    c = 0
    for b in tx[:14]:
        c ^= b
    tx[14] = c
    tx[15] = 0xBB
    return tx


def init_serial(port):
    if serial is None:
        raise RuntimeError("未安装 pyserial，无法连接实机")
    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = port, BAUDRATE, 1
    ser.dtr = False
    ser.rts = False
    ser.open()
    print(f"已连接 {port}，发送零位（夹爪 {GRIP_OPEN_DEG:.0f}° 张开）。")
    time.sleep(1.0)
    q = np.zeros(6)
    q[5] = np.deg2rad(GRIP_OPEN_DEG)
    ser.write(pack_frame(q))
    ser.flush()
    time.sleep(1.5)
    return ser


def stream(ser, rows):
    t0 = time.perf_counter()
    for i, row in enumerate(rows):
        ser.write(pack_frame(np.deg2rad(row)))
        ser.flush()
        rem = t0 + (i + 1) * SEND_PERIOD_S - time.perf_counter()
        if rem > 0:
            time.sleep(rem)


def execute(ser, pl, step=False, pause_between=False):
    t0 = time.perf_counter()
    last_phase, last_state = None, None
    for seg in pl.segments:
        if pause_between and last_phase is not None and seg.phase != last_phase:
            input(f"\n{last_phase} 完成。检查木块后回车继续 {seg.phase}（运动中请勿触碰木块）：")
        last_phase = seg.phase
        if step and seg.kind == "grip":
            ans = input(f"[{seg.label}] 即将{'夹紧' if seg.q[-1, 5] == GRIP_CLOSE_DEG else '松开'}，"
                        "回车继续，q 中止：")
            if ans.strip().lower() == "q":
                raise KeyboardInterrupt
        if seg.state != last_state:
            print(f"  ── 状态 {seg.state}")
            last_state = seg.state
        print(f"     {seg.label}（{seg.duration:.1f} s）")
        stream(ser, seg.q)
    return time.perf_counter() - t0


# ============================================================
# 10b. 手动标定圆心（--mode calib）
# ============================================================
# 思路：夹爪以“实际夹取时的姿态（同一俯仰、同一高度）”停在每个圆心的估计位置，人用键盘微调，
# 直到两指中点正对圆心，记录此刻的指令坐标。这样记录下来的就是“让夹爪对准圆心该发的指令”，
# 机械臂本身的标定误差（连杆、零位、下垂）在该点附近一并被吸收。

CALIB_ORDER = ("s2", "t2", "s1", "t1", "s3", "t3")
CALIB_HELP = """  夹取姿态（默认，固定俯仰与夹取高度）：
    w / s：沿 +X / -X（远离 / 靠近机械臂）   a / d：沿 +Y / -Y（站在机械臂身后看：向左 / 向右）
    可连写（www）；x+1.5、y-2 直接按毫米移动；step 0.5 修改步长；up / down 升起 / 降回；r 回到估计位置
  自由遥控（夹取姿态到不了时用，直接改末端或关节）：
    z+3 / z-3：升降（mm）   p+2 / p-2：改俯仰（deg）   j1+2 … j4-1：直接转关节（deg）
    aim：回到固定的夹取姿态（在当前 x、y）
  c / o：夹紧 / 张开     ok：记录当前 x、y     skip：跳过本点     q：保存并退出     h：帮助"""


def fit_similarity(src, dst):
    """二维相似变换 dst ≈ k·R·src + t（Umeyama）。返回 (k, 转角 deg, t, 变换函数)。"""
    A, B = np.asarray(src, float), np.asarray(dst, float)
    if len(A) == 1:
        t = B[0] - A[0]
        return 1.0, 0.0, t, (lambda p: np.asarray(p, float) + t)
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(B0.T @ A0 / len(A))
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt)) or 1.0])
    R = U @ D @ Vt
    k = float(np.trace(np.diag(S) @ D) * len(A) / (A0 ** 2).sum())
    t = mb - k * R @ ma
    return k, float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), t, (lambda p: k * R @ np.asarray(p, float) + t)


def predict_points(nominal, measured):
    """用已标定的点把名义几何整体平移/旋转/缩放过去，估计其余点。"""
    if not measured:
        return {k: np.asarray(v, float) for k, v in nominal.items()}
    keys = list(measured)
    f = fit_similarity([nominal[k] for k in keys], [measured[k] for k in keys])[3]
    return {k: (np.asarray(measured[k], float) if k in measured else f(nominal[k])) for k in nominal}


def point_pitches(points):
    """每个圆心在任务中实际使用的俯仰（取或放），标定就在这个姿态下做；算不出来的点不给（改为自由遥控）。"""
    try:
        trs, _ = plan_transfers(2, points, fallback=True)
    except IKError:                      # 有点完全不可达：其余点各自取窗口中值，不可达的点不给俯仰
        out = {}
        for k, xy in points.items():
            w = point_window(xy) or fallback_window(k, xy)
            if w is not None:
                out[k] = (w[0] + w[1]) / 2
        return out
    out = {}
    for tr in trs:
        out.setdefault(tr["from"], tr["pick_pitch"])
        out.setdefault(tr["to"], tr["place_pitch"])
    return out


def load_calib(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: np.asarray(v, float) for k, v in data["points"].items()}, data


def save_calib(path, nominal, measured, taught):
    pts = predict_points(nominal, measured)
    data = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "points": {k: [round(float(v[0]), 2), round(float(v[1]), 2)] for k, v in pts.items()},
            "measured": sorted(measured),
            "taught": {k: v for k, v in taught.items() if k in measured},
            "nominal": {k: [round(float(v[0]), 2), round(float(v[1]), 2)] for k, v in nominal.items()}}
    lines = []
    if len(measured) >= 3:
        keys = sorted(measured)
        k, th, t, f = fit_similarity([nominal[x] for x in keys], [measured[x] for x in keys])
        res = {x: float(np.linalg.norm(f(nominal[x]) - measured[x])) for x in keys}
        data["fit"] = {"scale": round(k, 4), "rot_deg": round(th, 2), "shift_mm": [round(float(v), 2) for v in t],
                       "residual_mm": {x: round(v, 2) for x, v in res.items()}}
        lines.append(f"相对名义摆放：平移 ({t[0]:+.1f}, {t[1]:+.1f}) mm，转角 {th:+.2f}°，比例 {k:.4f}")
        lines.append("各点对整体刚性拟合的残差（仅供参考，运行时以标定值为准）："
                     + "，".join(f"{x} {v:.1f}" for x, v in res.items()) + " mm")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    print(f"\n已保存 {path}")
    for k in POINT_NAMES:
        tag = "标定" if k in measured else "推算"
        d = pts[k] - nominal[k]
        extra = ""
        if k in taught and k in measured:
            extra = f"  俯仰 {taught[k]['pitch']:.1f}°  z {taught[k]['z']:.1f}  [{taught[k]['mode']}]"
        print(f"  {k}: ({pts[k][0]:6.1f}, {pts[k][1]:6.1f}) mm  [{tag}]  相对名义 ({d[0]:+.1f}, {d[1]:+.1f}){extra}")
    for ln in lines:
        print(ln)


# ---------- 实时键盘（复用 teleop/keyboard_position_ik.py 的按键读取，仅 Windows） ----------
KEY_HELP = """  M：切换模式  夹取姿态 → 自由末端 → 关节 → 夹取姿态（夹取姿态到不了时自动进入自由末端）
  夹取姿态（固定俯仰、固定夹取高度，只改 x、y）：
    W/S（↑/↓）径向前进/后退   A/D（←/→）绕底座右转/左转（同 teleop）   Q/E 升起/降回   R 回到估计位置
  自由末端：W/S、A/D 同上，Q/E 上升/下降，R/F 俯仰 +/-（夹爪更平 / 更竖直）
  关节（同 teleop JOINT）：A/D=J1 -/+  W/S=J2 +/-  Q/E=J3 +/-  R/F=J4 +/-
  按住移动：{fast:g} mm/s、{jfast:g}°/s；同时按住 Shift 为 {slow:g} mm/s、{jslow:g}°/s
  O/C 张开/夹紧   Enter 记录当前 x、y   Backspace 跳过本点   Esc 保存并回零退出"""
VK = {"ESC": 0x1B, "ENTER": 0x0D, "BACK": 0x08, "SHIFT": 0x10, "W": ord("W"), "A": ord("A"), "S": ord("S"),
      "D": ord("D"), "Q": ord("Q"), "E": ord("E"), "R": ord("R"), "F": ord("F"), "O": ord("O"), "C": ord("C"),
      "M": ord("M"), "LEFT": 0x25, "UP": 0x26, "RIGHT": 0x27, "DOWN": 0x28}
JOG_FAST, JOG_SLOW = 10.0, 2.0           # mm/s
JOG_JFAST, JOG_JSLOW = 10.0, 2.0         # deg/s（俯仰、关节）
MODE_NAMES = {"grasp": "夹取姿态", "free": "自由末端", "joint": "关节"}


def _load_teleop_keys():
    """导入仓库里 teleop 的按键函数（GetAsyncKeyState）；非 Windows 或缺依赖时返回 None。"""
    import sys
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from teleop import keyboard_position_ik as kb
        return kb
    except Exception:
        return None


def _flush_console_input():
    """实时按键期间按下的字母会进入控制台输入缓冲，下一次 input() 前清掉。"""
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getwch()
    except Exception:
        pass


def clip_joints(q):
    q = np.asarray(q, float).copy()
    q[:4] = np.clip(q[:4], JOINT_LIMITS_DEG[:4, 0], JOINT_LIMITS_DEG[:4, 1])
    return q


def run_calib(port, only=None, path=CALIB_FILE, use_keyboard=True):
    """逐点标定圆心 x、y。不设任何人为限制（只保留舵机物理限位），以实际标定为准。"""
    global POSITION_TOLERANCE
    POSITION_TOLERANCE = 0.0
    nominal = sheet_points(SHEET_CENTER_XY, SHEET_YAW_DEG)          # 名义几何（不读已有标定）
    measured, taught = {}, {}
    if only and Path(path).exists():
        old, data = load_calib(path)
        measured = {k: v for k, v in old.items() if k not in only and k in data.get("measured", old)}
        taught = {k: v for k, v in data.get("taught", {}).items() if k in measured}
        TAUGHT.update(taught)
    order = [k for k in CALIB_ORDER if not only or k in only]
    zg = TABLE_Z + BLOCK_SIZE / 2 + GRASP_Z_OFFSET

    kb = _load_teleop_keys() if use_keyboard else None
    if use_keyboard and kb is None:
        print("（未能加载 teleop 实时键盘：非 Windows 或缺少依赖，改用输入指令方式）")

    pl = Planner(1, predict_points(nominal, measured), plan=False)
    pl.occupant = {}
    # 标定时纸上可能放着一块试夹用的木块：越障高度按“任意圆内有一块木块”计算
    pl.blocks = {"标定块": (block_center_at(nominal["s2"]), 0.0)}

    print("手动标定圆心：夹爪先以实际夹取时的姿态（固定俯仰、夹取高度 z = "
          f"{zg:.1f} mm，指尖离纸约 5 mm）停在每个圆心的估计位置，")
    print("把两指中点移到圆心正上方后记录。到不了时自动切到自由遥控，你把夹爪开到圆心上方即可。")
    print("记录的是 x、y（及当时的俯仰，供逆解兜底）；运行时高度固定按桌面 TABLE_Z 计算。纸固定后不要再动。")
    print(KEY_HELP.format(fast=JOG_FAST, slow=JOG_SLOW, jfast=JOG_JFAST, jslow=JOG_JSLOW) if kb else CALIB_HELP)
    print("顺序：" + " → ".join(order))
    if input("确认纸上没有木块、撤手后输入 YES 开始：").strip().upper() != "YES":
        return
    ser = init_serial(port)
    sent = [0]

    def flush():
        for seg in pl.segments[sent[0]:]:
            stream(ser, seg.q)
        sent[0] = len(pl.segments)

    def safe(fn, *a, quiet=False):
        n0, q0, g0, u0 = len(pl.segments), pl.q.copy(), pl.grip, pl.under
        try:
            fn(*a)
            flush()
            return True
        except IKError as e:
            del pl.segments[n0:]
            pl.q, pl.grip, pl.under = q0, g0, u0
            if not quiet:
                print(f"\n    不可达：{e}")
            return False

    def go_aim(xy, pitch, label):
        def _do():
            pl.travel_to_hover(label, np.array([xy[0], xy[1], approach_top(np.array([xy[0], xy[1], zg]), pitch)]),
                               pitch, None)
            pl.vertical(f"{label}/下降", zg, VERTICAL_SPEED)
        return safe(_do, quiet=True)

    def reach(k, xy, pitch):
        """去本点的夹取姿态；失败则回零后从上方再试；仍失败返回 False（转自由遥控）。"""
        if pitch is None:
            return False
        if go_aim(xy, pitch, f"标定/{k}"):
            return True
        safe(pl.rise, "标定/升起", quiet=True)
        if safe(pl.movej, "标定/回零", np.zeros(5), 3.0, quiet=True) and \
                safe(pl.start_from_zero, xy, quiet=True) and go_aim(xy, pitch, f"标定/{k}"):
            return True
        return False

    def sync_jog(q5):
        """把实时遥控到达的姿态记入规划器（已经下发过，不再重复下发）。"""
        q = add_grip_col(np.vstack([pl.q, q5]), pl.grip)
        pl._push(Segment("标定/遥控", "jog", q, SEND_PERIOD_S))
        sent[0] = len(pl.segments)

    def taught_entry(q, mode):
        g = grip_point(q)
        return {"pitch": round(float(tool_pitch(q)), 2), "z": round(float(g[2]), 2),
                "q": [round(float(v), 3) for v in q[:5]], "mode": MODE_NAMES[mode]}

    # ---------- 每个点：实时键盘 ----------
    def jog_keys(k, guess, pitch, mode):
        down = kb.key_down
        while any(down(v) for v in VK.values()):
            time.sleep(0.01)
        up = False
        q = pl.q.copy()
        aim_xy = grip_point(q)[:2].copy()
        last_print = last_err = 0.0
        tick = time.perf_counter()

        def note(msg):
            nonlocal last_err
            now = time.perf_counter()
            if now - last_err > 1.0:
                print("\n    " + msg)
                last_err = now

        def blocking(fn, *a):
            nonlocal q
            if not np.allclose(q, pl.q):
                sync_jog(q)
            ok = fn(*a)
            q = pl.q.copy()
            return ok

        def wait_release(*names):
            for n in names:
                while down(VK[n]):
                    time.sleep(0.01)

        def send(q_new):
            nonlocal q
            q = q_new
            ser.write(pack_frame(np.deg2rad(np.append(q, pl.grip))))
            ser.flush()

        try:
            while True:
                if down(VK["ESC"]):
                    wait_release("ESC")
                    return "quit", None, None
                if down(VK["ENTER"]):
                    wait_release("ENTER")
                    if up:
                        return "ok", aim_xy.copy(), {"pitch": round(float(pitch), 2), "z": round(zg, 2),
                                                     "q": None, "mode": MODE_NAMES["grasp"]}
                    return "ok", grip_point(q)[:2].copy(), taught_entry(q, mode)
                if down(VK["BACK"]):
                    wait_release("BACK")
                    return "skip", None, None
                if down(VK["M"]):
                    wait_release("M")
                    nxt = {"grasp": "free", "free": "joint", "joint": "grasp"}[mode]
                    if mode == "grasp" and up:
                        note("先按 E 降下再切换模式")
                    elif nxt == "grasp":
                        xy = grip_point(q)[:2]
                        if pitch is not None and blocking(go_aim, xy, pitch, f"标定/{k}"):
                            mode = "grasp"
                        else:
                            mode = "free"
                            note("当前 x、y 在固定夹取姿态下不可达，切到自由末端")
                    else:
                        mode = nxt
                    print(f"\n    模式 → {MODE_NAMES[mode]}")
                    tick = time.perf_counter()
                    continue
                if down(VK["O"]) or down(VK["C"]):
                    close = down(VK["C"])
                    wait_release("O", "C")
                    blocking(safe, pl.gripper, "标定/试夹" if close else "标定/张开",
                             GRIP_CLOSE_DEG if close else GRIP_OPEN_DEG)
                    tick = time.perf_counter()
                    continue
                if mode == "grasp":
                    if down(VK["Q"]):
                        wait_release("Q")
                        if not up:
                            aim_xy = grip_point(q)[:2].copy()
                            if pl.grip != GRIP_OPEN_DEG:
                                blocking(safe, pl.gripper, "标定/张开", GRIP_OPEN_DEG)
                            up = blocking(safe, pl.rise, "标定/升起")
                            if up:
                                print("\n    已升起：可在圆内放 / 取木块；W/S/A/D 只改目标位置，按 E 降下")
                        tick = time.perf_counter()
                        continue
                    if down(VK["E"]):
                        wait_release("E")
                        if up:
                            if blocking(go_aim, aim_xy, pitch, f"标定/{k}"):
                                up = False
                            else:
                                note("该 x、y 在夹取姿态下不可达；按 M 切自由遥控（需先降下：按 R 回估计位置再 E）")
                        tick = time.perf_counter()
                        continue
                    if down(VK["R"]):
                        wait_release("R")
                        if up:
                            aim_xy = guess.copy()
                        else:
                            blocking(go_aim, guess, pitch, f"标定/{k}")
                        tick = time.perf_counter()
                        continue

                shift = down(VK["SHIFT"])
                v_lin = JOG_SLOW if shift else JOG_FAST
                v_ang = JOG_JSLOW if shift else JOG_JFAST
                dt = SEND_PERIOD_S
                if mode == "joint":
                    dj = np.zeros(5)
                    dj[0] = float(down(VK["D"])) - float(down(VK["A"]))
                    dj[1] = float(down(VK["W"])) - float(down(VK["S"]))
                    dj[2] = float(down(VK["Q"])) - float(down(VK["E"]))
                    dj[3] = float(down(VK["R"])) - float(down(VK["F"]))
                    if np.any(dj):
                        send(clip_joints(q + dj * v_ang * dt))
                else:
                    vr = float(down(VK["W"]) or down(VK["UP"])) - float(down(VK["S"]) or down(VK["DOWN"]))
                    vt = float(down(VK["D"]) or down(VK["RIGHT"])) - float(down(VK["A"]) or down(VK["LEFT"]))
                    vz = (float(down(VK["Q"])) - float(down(VK["E"]))) if mode == "free" else 0.0
                    vp = (float(down(VK["R"])) - float(down(VK["F"]))) if mode == "free" else 0.0
                    if vr or vt or vz or vp:
                        g = None if up else grip_point(q)
                        cur = aim_xy if up else grip_point(q)[:2]
                        rr = float(np.hypot(*cur)) or 1.0
                        er = cur / rr
                        et = np.array([-er[1], er[0]])
                        nxt = cur + dt * v_lin * (vr * er + vt * et)
                        if up:
                            aim_xy = nxt
                        else:
                            z_new = (g[2] + dt * v_lin * vz) if mode == "free" else zg
                            pt_new = (tool_pitch(q) + dt * v_ang * vp) if mode == "free" else pitch
                            try:
                                send(solve_ik_flex(np.array([nxt[0], nxt[1], z_new]), pt_new, q))
                            except IKError:
                                note("这里任何俯仰都够不到（超出可达范围）：按 M 切到关节模式直接转关节")
                now = time.perf_counter()
                if now - last_print > 0.2:
                    if up:
                        cur, zz, pp = aim_xy, None, None
                    else:
                        gg = grip_point(q)
                        cur, zz, pp = gg[:2], gg[2], tool_pitch(q)
                    d = cur - guess
                    tail = "  [已升起，目标]" if up else f"  z={zz:6.1f}  俯仰={pp:6.1f}°"
                    print(f"\r  [{k}|{MODE_NAMES[mode]}] x={cur[0]:7.1f}  y={cur[1]:7.1f}  相对估计 ({d[0]:+5.1f}, "
                          f"{d[1]:+5.1f}){tail}  {'慢' if shift else '常'}速   ", end="", flush=True)
                    last_print = now
                tick += SEND_PERIOD_S
                rem = tick - time.perf_counter()
                if rem > 0:
                    time.sleep(rem)
                else:
                    tick = time.perf_counter()
        finally:
            if not np.allclose(q, pl.q):
                sync_jog(q)
            print()
            _flush_console_input()

    # ---------- 每个点：输入指令 ----------
    def jog_text(k, guess, pitch, mode):
        step, up = 1.0, False
        aim_xy = grip_point(pl.q)[:2].copy()

        def move_to(q_to, label):
            dq = float(np.max(np.abs(q_to[:4] - pl.q[:4])))
            return safe(pl.movej, label, np.asarray(q_to, float), max(0.4, 1.875 * dq / MOVEJ_SPEED_DEG_S))

        while True:
            g = grip_point(pl.q)
            here = aim_xy if up else g[:2]
            d = here - guess
            cmd = input(f"  [{k}|{MODE_NAMES[mode]}] ({here[0]:.1f}, {here[1]:.1f})  相对估计 ({d[0]:+.1f}, {d[1]:+.1f})"
                        f"  z {g[2]:.1f}  俯仰 {tool_pitch(pl.q):.1f}°  步长 {step:g}{'  [已升起]' if up else ''} > "
                        ).strip().lower().replace(" ", "")
            if not cmd:
                continue
            if cmd in ("h", "help", "?"):
                print(CALIB_HELP)
                continue
            if cmd == "q":
                return "quit", None, None
            if cmd == "skip":
                return "skip", None, None
            if cmd == "ok":
                if up:
                    return "ok", here.copy(), {"pitch": round(float(pitch), 2), "z": round(zg, 2), "q": None,
                                               "mode": MODE_NAMES["grasp"]}
                return "ok", here.copy(), taught_entry(pl.q, mode)
            if cmd.startswith("step"):
                try:
                    step = max(0.1, float(cmd[4:]))
                except ValueError:
                    print("    用法：step 0.5")
                continue
            if cmd in ("c", "o"):
                safe(pl.gripper, "标定/试夹" if cmd == "c" else "标定/张开",
                     GRIP_CLOSE_DEG if cmd == "c" else GRIP_OPEN_DEG)
                continue
            if cmd == "up":
                if mode == "grasp" and not up:
                    if pl.grip != GRIP_OPEN_DEG:
                        safe(pl.gripper, "标定/张开", GRIP_OPEN_DEG)
                    aim_xy = g[:2].copy()
                    up = safe(pl.rise, "标定/升起")
                continue
            if cmd in ("down", "aim"):
                xy = aim_xy if up else g[:2]
                if pitch is not None and go_aim(xy, pitch, f"标定/{k}"):
                    up, mode = False, "grasp"
                else:
                    print("    该 x、y 在固定夹取姿态下不可达，继续用 z / p / j 指令自由遥控")
                    mode = "free"
                continue
            if cmd == "r":
                if up:
                    aim_xy = guess.copy()
                elif pitch is not None:
                    go_aim(guess, pitch, f"标定/{k}")
                continue
            m = re.fullmatch(r"j([1-4])([+-]\d+(?:\.\d*)?)", cmd)
            if m:
                qn = pl.q.copy()
                qn[int(m.group(1)) - 1] += float(m.group(2))
                if move_to(clip_joints(qn), "标定/关节遥控"):
                    mode = "joint"
                continue
            m = re.fullmatch(r"([xyzp])([+-]\d+(?:\.\d*)?)", cmd)
            delta, dz, dp = np.zeros(2), 0.0, 0.0
            if m:
                v = float(m.group(2))
                if m.group(1) == "x":
                    delta[0] = v
                elif m.group(1) == "y":
                    delta[1] = v
                elif m.group(1) == "z":
                    dz = v
                else:
                    dp = v
            elif re.fullmatch(r"[wasd]+", cmd):
                delta = step * np.array([cmd.count("w") - cmd.count("s"), cmd.count("a") - cmd.count("d")], float)
            else:
                print("    无法识别，输入 h 查看指令")
                continue
            if up:
                aim_xy = aim_xy + delta          # 升起状态只改目标，down 时到位
                continue
            if dz or dp:
                mode = "free"
            z_new = g[2] + dz if mode != "grasp" else zg
            p_new = tool_pitch(pl.q) + dp if mode != "grasp" else pitch
            try:
                qn = solve_ik_flex(np.array([g[0] + delta[0], g[1] + delta[1], z_new]), p_new, pl.q)
            except IKError as e:
                print(f"    末端逆解无解（{e}）：可用 j1+2 等指令直接转关节")
                continue
            move_to(qn, "标定/遥控")

    try:
        start_xy = predict_points(nominal, measured)[order[0]]
        if not safe(pl.start_from_zero, start_xy, quiet=True):
            print("准备姿态不可达，从零位开始自由遥控")
        for k in order:
            guess = predict_points(nominal, measured)[k]
            for kk, v in taught.items():
                TAUGHT[kk] = v
            pitch = point_pitches(predict_points(nominal, measured)).get(k)
            if pl.grip != GRIP_OPEN_DEG:
                safe(pl.gripper, "标定/张开", GRIP_OPEN_DEG)
            ok = reach(k, guess, pitch)
            mode = "grasp" if ok else "free"
            ptxt = f"俯仰 {pitch:.1f}°" if pitch is not None else "无固定俯仰"
            print(f"\n── {k}：估计 ({guess[0]:.1f}, {guess[1]:.1f}) mm，{ptxt}；"
                  + ("已到夹取姿态" if ok else "夹取姿态到不了，请自由遥控到圆心上方（M 可切关节模式）"))
            res, xy, info = (jog_keys if kb else jog_text)(k, guess, pitch, mode)
            if res == "ok":
                measured[k] = np.asarray(xy, float)
                taught[k] = info
                print(f"    已记录 {k} = ({xy[0]:.1f}, {xy[1]:.1f}) mm  [{info['mode']}]")
            elif res == "quit":
                break
        if pl.grip != GRIP_OPEN_DEG:
            safe(pl.gripper, "标定/张开", GRIP_OPEN_DEG)
        if not safe(pl.go_home, "标定", quiet=True):
            safe(pl.movej, "标定/回零", np.zeros(5), 3.0)
    except KeyboardInterrupt:
        print("\n已中止下发。")
    finally:
        ser.close()
    if measured:
        save_calib(path, nominal, measured, taught)
        print(f"之后的 plan / simulate / hardware 会自动使用这套 x、y（z 固定按桌面 {TABLE_Z} mm），--no-calib 可忽略。")
    else:
        print("没有记录任何点，未保存。")


# ============================================================
# 11. 扫描：推荐作业纸摆放
# ============================================================

def run_scan():
    pts = sheet_points()
    print("当前作业纸摆放下各圆心的俯仰窗口：")
    for k in POINT_NAMES:
        W = point_window(pts[k])
        print(f"  {k}: r={np.hypot(*pts[k]):.0f} mm  窗口 {W}")
    try:
        trs, _ = plan_transfers(2, pts, fallback=False)
        print("每趟倾角：" + "，".join(f"{t['from']}→{t['to']} {t['tilt']:+.1f}°" for t in trs))
    except IKError as e:
        print("当前摆放不可行：", e)
    if POINT_XY_OVERRIDE:
        return
    print(f"\n搜索作业纸摆放（中心沿 +X_B，转角 -10~10°；位置容差 ±{POSITION_TOLERANCE:.0f} mm），按最大倾角排序：")
    rows = []
    for R in np.arange(190.0, 300.1, 2.5):
        for yaw in np.arange(-10.0, 10.1, 2.5):
            p = sheet_points((R, 0.0), yaw)
            try:
                trs, W = plan_transfers(2, p, fallback=False)
            except IKError:
                continue
            worst = max(abs(t["tilt"]) for t in trs)
            narrow = min(w[1] - w[0] for w in W.values())
            rows.append((round(worst, 1), -narrow, R, yaw))
    rows.sort()
    for worst, nn, R, yaw in rows[:8]:
        print(f"  中心 ({R:.1f}, 0)  u 轴转角 {yaw:+.1f}°  最大倾角 {worst:.1f}°  最窄俯仰窗口 {-nn:.1f}°")
    print("\n摆放方法：作业纸 s 列靠近机械臂、t 列远离，s1/t1 在左侧（站在机械臂身后看）。"
          "按上面的位置摆好纸并用胶带固定，再用 --mode calib 逐点标定圆心。")


# ============================================================
# 12. 主流程
# ============================================================

def main():
    global SHEET_CENTER_XY, SHEET_YAW_DEG, SHEET_SCALE_U, SHEET_SCALE_V, CIRCLE_DIAMETER, GRIP_CLOSE_DEG, RELEASE_GAP
    global POSITION_TOLERANCE, POINT_XY_OVERRIDE, CALIB_SOURCE, JOINT_MARGIN_DEG
    ap = argparse.ArgumentParser(description="第八周：机械臂定点搬运（必做 + 选做）")
    ap.add_argument("--mode", choices=("plan", "simulate", "hardware", "calib", "scan"), default="plan")
    ap.add_argument("--task", type=int, choices=(1, 2), default=1, help="1：必做；2：必做后接着做选做")
    ap.add_argument("--port", default=SERIAL_PORT)
    ap.add_argument("--sheet", type=float, nargs=2, metavar=("X", "Y"), help="作业纸中心（基座系 mm）")
    ap.add_argument("--sheet-yaw", type=float, help="作业纸 u 轴（s→t）转角（deg）")
    ap.add_argument("--s1s3", type=float, help="实测 s1→s3 圆心距（mm，扫描值 189.2），等比校准整张作业纸")
    ap.add_argument("--pitch", type=float, nargs=2, metavar=("COL", "ROW"), help="实测 s→t 列距、s1→s3 距离（mm），两方向分别校准")
    ap.add_argument("--tol", type=float, help="取放点位置容差（mm），默认 5")
    ap.add_argument("--circle", type=float, help="实测区域圆直径（mm）")
    ap.add_argument("--grip-close", type=float, help="夹爪闭合角（deg）")
    ap.add_argument("--release-gap", type=float, help="松爪时木块最低棱离纸面（mm）")
    ap.add_argument("--calib-file", default=str(CALIB_FILE), help="圆心标定文件（--mode calib 生成）")
    ap.add_argument("--no-calib", action="store_true", help="忽略标定文件，用作业纸名义几何")
    ap.add_argument("--only", help="calib：只标定这些点，如 s1,t3（其余沿用已有标定）")
    ap.add_argument("--text-jog", action="store_true", help="calib：不用实时键盘，改用输入指令微调")
    ap.add_argument("--step", action="store_true", help="每次夹紧/松开前回车确认")
    ap.add_argument("--pause-between", action="store_true", help="必做与选做之间暂停确认")
    ap.add_argument("--speed", type=float, default=4.0, help="仿真回放倍速")
    ap.add_argument("--save-frames", help="仿真帧导出目录（不弹窗）")
    args = ap.parse_args()
    if args.sheet:
        SHEET_CENTER_XY = tuple(args.sheet)
    if args.sheet_yaw is not None:
        SHEET_YAW_DEG = args.sheet_yaw
    if args.s1s3:
        SHEET_SCALE_U = SHEET_SCALE_V = args.s1s3 / SHEET_S1S3
        CIRCLE_DIAMETER *= SHEET_SCALE_V
    if args.pitch:
        SHEET_SCALE_U = args.pitch[0] / (SHEET_UV["t2"][0] - SHEET_UV["s2"][0])
        SHEET_SCALE_V = args.pitch[1] / SHEET_S1S3
    if args.tol is not None:
        POSITION_TOLERANCE = args.tol
    if args.circle:
        CIRCLE_DIAMETER = args.circle
    if args.grip_close is not None:
        GRIP_CLOSE_DEG = args.grip_close
    if args.release_gap is not None:
        RELEASE_GAP = args.release_gap

    if args.mode == "scan":
        run_scan()
        return

    if args.mode == "calib":
        only = [x.strip() for x in args.only.split(",")] if args.only else None
        if only and any(x not in POINT_NAMES for x in only):
            raise SystemExit(f"--only 只能是 {POINT_NAMES} 中的点")
        run_calib(args.port, only=only, path=Path(args.calib_file), use_keyboard=not args.text_jog)
        return

    calib_path = Path(args.calib_file)
    if not args.no_calib and calib_path.exists():
        pts, data = load_calib(calib_path)
        POINT_XY_OVERRIDE = pts
        TAUGHT.update(data.get("taught", {}))
        POSITION_TOLERANCE = 0.0          # 以标定为准：不再给摆放误差留余量
        CALIB_SOURCE = f"{calib_path.name}，{data.get('time', '')}，实测 {','.join(data.get('measured', []))}"
        print(f"使用圆心标定：{CALIB_SOURCE}（--no-calib 可忽略）")

    pl = build_plan(args.task)
    check_plan(pl, raise_on_fail=False)
    print(summarize(pl))
    print("状态机序列（前 12 个）：" + " → ".join(state_sequence(pl)[:12]) + " …")
    if pl.checks["issues"]:
        print("离线检查提示（仅供参考）：\n  " + "\n  ".join(pl.checks["issues"][:20]))
    robot = build_rtb_model()
    if robot is not None:
        worst = max(float(np.linalg.norm(robot.fkine(np.deg2rad(s.q[i, :5])).t - grip_point(s.q[i])))
                    for s in pl.segments for i in (0, len(s.q) - 1))
        print(f"roboticstoolbox 正解复核：最大差异 {worst:.2e} mm")
    tag = f"task{args.task}"
    export(pl, OUTPUT_DIR, tag)
    save_overview(pl, OUTPUT_DIR / f"{tag}_overview.png")
    print(f"已导出 {tag}_trajectory.csv / _summary.json / _overview.png（目录 {OUTPUT_DIR}）")

    if args.mode == "simulate":
        import matplotlib
        if args.save_frames:
            matplotlib.use("Agg")
            Path(args.save_frames).mkdir(parents=True, exist_ok=True)
        else:
            try:
                matplotlib.use("TkAgg")
            except Exception:
                pass
        animate(pl, speed=args.speed, save_frames=args.save_frames)
        return

    if args.mode == "hardware":
        print("\n实机运行：三块木块放在 s1、s2、s3 圆心，一边正对机械臂基座；t1~t3 为空。撤手，运动中不得伸手。")
        print("Ctrl+C 立即停止下发；夹爪保持当前状态，不会松开手中木块。")
        if pl.checks["issues"]:
            print(f"注意：离线检查有 {len(pl.checks['issues'])} 项提示（见上），按标定结果照常执行。")
        if input("输入 YES 开始：").strip().upper() != "YES":
            print("已取消。")
            return
        ser = init_serial(args.port)
        try:
            used = execute(ser, pl, step=args.step, pause_between=args.pause_between)
            print(f"完成，用时 {used:.0f} s。")
        except KeyboardInterrupt:
            print("\n已中止：停止下发，舵机保持最后一帧（夹爪状态不变）。确认安全后手动处理。")
        finally:
            ser.close()
            print("串口已关闭。")


if __name__ == "__main__":
    main()
