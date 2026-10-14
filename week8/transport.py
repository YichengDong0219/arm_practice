"""第八周：机械臂定点搬运（必做 + 选做）。

任务（作业纸上 s1、s2、s3 为夹取点，t1、t2、t3 为落料点，木块 30 mm 立方体）：
  必做  s1→t1、s2→t3、s3→t2（按此顺序）
  选做  在必做完成后的状态上继续：t1→s2、t2→s1、t3→s3
验收：木块底面完全位于对应圆形区域内、不与边界相交、稳定落平，无错放、掉落或碰倒其他木块。

运行方式（先离线，后实机）::

    python week8/transport.py --mode scan                       # 各点俯仰窗口、每趟倾角、推荐作业纸摆放
    python week8/transport.py --mode plan --task 1              # 规划+离线检查，导出 CSV/JSON/PNG
    python week8/transport.py --mode simulate --task 2          # 三维动画（必做后接选做）
    python week8/transport.py --mode calib --port COM5          # 实机：张爪依次悬停在 6 个圆心上方，核对位置
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
   的俯仰无法完全一致，放下时木块会带一个小倾角（推荐摆放下最大约 15°）。程序按手中木块的
   实际姿态，把最低棱停在纸面上方 RELEASE_GAP 处再松爪：倾斜绕切向水平轴，着地的是整条棱，
   木块翻平时底面中心水平移动 < 1 mm。离线检查给出每趟倾角和“底面距圆边界”的余量。
5. 安全：木块只在越障高度横移；中止（Ctrl+C）只停止下发、绝不松爪。

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
# 2. 作业纸与木块（作业纸尺寸按课件图估计，请按实物测量后修改）
# ============================================================

TABLE_Z = 52.0                   # 桌面（作业纸）在基座系中的高度（第六周标定）
BLOCK_SIZE = 30.0                # 木块边长 h

COL_PITCH = 100.0                # s 列与 t 列圆心距（mm）【估计值，请实测】
ROW_PITCH = 100.0                # 相邻两行圆心距（mm）【估计值，请实测】
CIRCLE_DIAMETER = 64.0           # 区域圆直径（mm）【估计值，请实测】
SHEET_CENTER_XY = (245.0, 0.0)   # 作业纸中心（第 2 行两圆心连线中点）在基座系中的位置（--mode scan 推荐）
SHEET_YAW_DEG = 0.0              # u 轴（s→t）相对 +X_B 的转角

POINT_XY_OVERRIDE = {}           # 实测圆心：{"s1": (x, y), ...}；给出后覆盖上面的作业纸几何
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
MAX_PLACE_TILT_DEG = 18.0        # 允许的放置倾角（夹取与放置俯仰差）上限
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


def solve_ik(p, pitch, q_prev=None, margin=JOINT_MARGIN_DEG):
    valid = [q for q in ik_candidates(p, pitch) if joint_margin(q) >= margin]
    if not valid:
        raise IKError(f"点 {np.round(p, 1)} mm 在俯仰 {pitch:.1f}° 下无限位内逆解")
    if q_prev is None:
        return max(valid, key=joint_margin)
    return min(valid, key=lambda q: float(np.sum((q[:4] - q_prev[:4]) ** 2)))


def pitch_window(p_low, p_high, margin=WINDOW_MARGIN_DEG):
    """在 p_low→p_high 的竖直段上全程可行的俯仰区间（可能为空）。"""
    pts = [np.asarray(p_low) + t * (np.asarray(p_high) - np.asarray(p_low)) for t in (0, 0.5, 1)]
    ok = [pt for pt in PITCH_GRID if all(best_margin(p, pt) >= margin for p in pts)]
    return (float(min(ok)), float(max(ok))) if ok else None


def best_pitch(p, margin=JOINT_MARGIN_DEG + 2.0):
    m, pt = max((best_margin(p, x), x) for x in PITCH_GRID)
    if m < margin:
        raise IKError(f"点 {np.round(p, 1)} mm 在任何俯仰下都缺少关节余量")
    return float(pt)


# ============================================================
# 5. 作业纸几何、取放点俯仰与每趟倾角
# ============================================================

def sheet_points(center_xy=None, yaw_deg=None):
    """六个圆心在基座系中的 (x, y)。"""
    if POINT_XY_OVERRIDE:
        return {k: np.asarray(v, float) for k, v in POINT_XY_OVERRIDE.items()}
    cx, cy = SHEET_CENTER_XY if center_xy is None else center_xy
    yaw = np.deg2rad(SHEET_YAW_DEG if yaw_deg is None else yaw_deg)
    u_axis = np.array([np.cos(yaw), np.sin(yaw)])
    v_axis = np.array([-np.sin(yaw), np.cos(yaw)])
    uv = {"s1": (-COL_PITCH / 2, ROW_PITCH), "s2": (-COL_PITCH / 2, 0.0), "s3": (-COL_PITCH / 2, -ROW_PITCH),
          "t1": (COL_PITCH / 2, ROW_PITCH), "t2": (COL_PITCH / 2, 0.0), "t3": (COL_PITCH / 2, -ROW_PITCH)}
    return {k: np.array([cx, cy]) + u * u_axis + v * v_axis for k, (u, v) in uv.items()}


def block_center_at(xy):
    return np.array([xy[0], xy[1], TABLE_Z + BLOCK_SIZE / 2])


TILT_LIFT_ALLOW = 8.0            # 倾斜放置时夹取点比水平放置高出的余量（≈ h/2·sin(倾角)，18° 时约 4.6 mm）


def point_window(xy):
    """取放点的俯仰窗口：覆盖从夹取高度到悬停点的整段竖直行程（含倾斜放置时的抬高量）。"""
    g = block_center_at(xy) + np.array([0.0, 0.0, GRASP_Z_OFFSET])
    top = g + np.array([0.0, 0.0, RELEASE_GAP + TILT_LIFT_ALLOW + APPROACH + HOLD_SAG_COMP])
    return pitch_window(g - np.array([0.0, 0.0, 1.0]), top)


def choose_pitches(W_pick, W_place):
    """夹取俯仰 p1 ∈ W_pick、放置俯仰 p2 ∈ W_place，使 |p2 - p1| 最小。"""
    lo, hi = max(W_pick[0], W_place[0]), min(W_pick[1], W_place[1])
    if hi >= lo:                         # 有公共俯仰：取公共区间中点，放下时木块完全水平
        p = (lo + hi) / 2
        return p, p
    if W_pick[1] < W_place[0]:           # 取料点窗口整体更“竖直”
        return W_pick[1], W_place[0]
    return W_pick[0], W_place[1]


def plan_transfers(task, points=None):
    points = sheet_points() if points is None else points
    windows = {k: point_window(v) for k, v in points.items()}
    bad = [k for k, w in windows.items() if w is None]
    if bad:
        raise IKError(f"点 {bad} 在桌面高度不可达，请调整作业纸位置（见 --mode scan）")
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
        q = solve_ik(p, pt, q)
        qs.append(q)
    return np.array(qs), pts, pitches


def const_pitch_top(p_low, pitch, z_max, step=2.0, margin=JOINT_MARGIN_DEG + 1.0):
    p = np.asarray(p_low, float)
    z = p[2]
    while z + step <= z_max + 1e-9 and best_margin((p[0], p[1], z + step), pitch) >= margin:
        z += step
    if z_max - z < step and best_margin((p[0], p[1], z_max), pitch) >= margin:
        z = z_max
    return float(z)


def line_feasible(p0, p1, pitch0, pitch1, n=7, margin=JOINT_MARGIN_DEG + 1.0):
    return all(best_margin(p0 + t * (p1 - p0), (1 - t) * pitch0 + t * pitch1) >= margin
               for t in np.linspace(0.0, 1.0, n))


# ============================================================
# 7. 规划器（状态机）
# ============================================================

class Planner:
    def __init__(self, task, points=None):
        self.task = task
        self.points = sheet_points() if points is None else points
        self.transfers, self.windows = plan_transfers(task, self.points)
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
                  if best_margin(up, x) >= JOINT_MARGIN_DEG + 1][:40]
        cand_b = [x for x in sorted(PITCH_GRID, key=lambda x: abs(x - hover_pitch))
                  if best_margin(above, x) >= JOINT_MARGIN_DEG + 1][:40]

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
                                              above, hover, hover_pitch, target):
            raise IKError(f"{label}: 找不到可行的越障路线（越障高度 {zt:.0f} mm）"
                          + (f"；{last_err}" if last_err else ""))
        self.under = target

    def _transit_movej(self, label, here, pitch_here, up_c, z_up_c, up, zt, above, hover, hover_pitch, target):
        """备选越障：竖直升到越障高度 → 高处关节空间运动（可切换肘部构型）→ 竖直下降。"""
        saved = (len(self.segments), self.q.copy(), self.grip, self.under)
        cand_a = [x for x in sorted(PITCH_GRID, key=lambda x: abs(x - pitch_here))
                  if line_feasible(up_c, up, pitch_here, x)][:20]
        cand_b = [x for x in sorted(PITCH_GRID, key=lambda x: abs(x - hover_pitch))
                  if best_margin(above, x) >= JOINT_MARGIN_DEG + 1 and line_feasible(above, hover, x, hover_pitch)][:20]
        for a in cand_a:
            for b in cand_b:
                for q_top in sorted((q for q in ik_candidates(above, b) if joint_margin(q) >= JOINT_MARGIN_DEG + 1),
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
        hover = g + np.array([0.0, 0.0, APPROACH])
        self.state = "MOVE_TO_PICK"
        self.travel_to_hover(f"{label}/去取料点", hover, pitch, target=name)
        self.state = "DESCEND_PICK"
        self.vertical(f"{label}/下降", g[2] + SLOW_ZONE, VERTICAL_SPEED, target=name)
        self.vertical(f"{label}/慢降", g[2], SLOW_SPEED, target=name)
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
        self.vertical(f"{label}/慢起", g[2] + SLOW_ZONE + HOLD_SAG_COMP, SLOW_SPEED, target=name)
        self.vertical(f"{label}/抬起", g[2] + APPROACH + HOLD_SAG_COMP, VERTICAL_SPEED, target=name)
        self.under = name

    def release_grip_point(self, name, target_xy, pitch):
        """求放置夹取点：手中木块（可能倾斜）中心 xy 对准目标、最低棱高出纸面 RELEASE_GAP。"""
        g = np.array([target_xy[0], target_xy[1], TABLE_Z + BLOCK_SIZE / 2 + RELEASE_GAP + 5.0])
        q = self.q
        for _ in range(4):
            q = solve_ik(g, pitch, q)
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
        hover = g + np.array([0.0, 0.0, APPROACH])
        self.state = "MOVE_TO_PLACE"
        self.travel_to_hover(f"{label}/去落料点", hover, pitch, target=name)
        self.state = "DESCEND_PLACE"
        self.vertical(f"{label}/下降", g[2] + SLOW_ZONE, VERTICAL_SPEED, target=name)
        self.vertical(f"{label}/慢降", g[2], SLOW_SPEED, target=name)
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
        self.vertical(f"{label}/慢起", g[2] + SLOW_ZONE, SLOW_SPEED, target=name)
        self.vertical(f"{label}/撤离", g[2] + APPROACH, VERTICAL_SPEED, target=name)
        self.under = name

    def go_home(self, label):
        here = grip_point(self.q)
        zt = max(self.transit_z(), here[2])
        self.state = "HOME"
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
        self.movej(f"{label}/回零", np.zeros(5), duration=3.0)

    def start_from_zero(self, first_xy):
        self.state = "READY"
        p = np.array([first_xy[0], first_xy[1], self.transit_z()])
        self.movej("零位→准备", solve_ik(p, best_pitch(p)))

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
    d = rot_z(yaw).T @ (np.asarray(p) - center)
    return bool(np.all(np.abs(d) <= BLOCK_SIZE / 2 + expand))


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
    L.append(f"作业纸：中心 {SHEET_CENTER_XY}，u 轴转角 {SHEET_YAW_DEG:.0f}°，列距 {COL_PITCH:.0f}、行距 {ROW_PITCH:.0f}、"
             f"圆直径 {CIRCLE_DIAMETER:.0f} mm；桌面 z={TABLE_Z}")
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
    L.append("离线检查：" + ("通过" if not c["issues"] else f"{len(c['issues'])} 项未通过"))
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
        "sheet": {"center": SHEET_CENTER_XY, "yaw_deg": SHEET_YAW_DEG, "col_pitch": COL_PITCH,
                  "row_pitch": ROW_PITCH, "circle_diameter": CIRCLE_DIAMETER},
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
# 11. 扫描：推荐作业纸摆放
# ============================================================

def run_scan():
    pts = sheet_points()
    print("当前作业纸摆放下各圆心的俯仰窗口：")
    for k in POINT_NAMES:
        W = point_window(pts[k])
        print(f"  {k}: r={np.hypot(*pts[k]):.0f} mm  窗口 {W}")
    try:
        trs, _ = plan_transfers(2, pts)
        print("每趟倾角：" + "，".join(f"{t['from']}→{t['to']} {t['tilt']:+.1f}°" for t in trs))
    except IKError as e:
        print("当前摆放不可行：", e)
    if POINT_XY_OVERRIDE:
        return
    print("\n搜索作业纸摆放（中心沿 +X_B，u 轴转角 0~175°），按最大倾角排序：")
    rows = []
    for R in np.arange(180.0, 320.0, 5.0):
        for yaw in np.arange(0.0, 180.0, 5.0):
            p = sheet_points((R, 0.0), yaw)
            try:
                trs, W = plan_transfers(2, p)
            except IKError:
                continue
            worst = max(abs(t["tilt"]) for t in trs)
            narrow = min(w[1] - w[0] for w in W.values())
            rows.append((round(worst, 1), -narrow, R, yaw))
    rows.sort()
    for worst, nn, R, yaw in rows[:8]:
        print(f"  中心 ({R:.0f}, 0)  u 轴转角 {yaw:.0f}°  最大倾角 {worst:.1f}°  最窄俯仰窗口 {-nn:.1f}°")


# ============================================================
# 12. 主流程
# ============================================================

def main():
    global SHEET_CENTER_XY, SHEET_YAW_DEG, COL_PITCH, ROW_PITCH, CIRCLE_DIAMETER, GRIP_CLOSE_DEG, RELEASE_GAP
    ap = argparse.ArgumentParser(description="第八周：机械臂定点搬运（必做 + 选做）")
    ap.add_argument("--mode", choices=("plan", "simulate", "hardware", "calib", "scan"), default="plan")
    ap.add_argument("--task", type=int, choices=(1, 2), default=1, help="1：必做；2：必做后接着做选做")
    ap.add_argument("--port", default=SERIAL_PORT)
    ap.add_argument("--sheet", type=float, nargs=2, metavar=("X", "Y"), help="作业纸中心（基座系 mm）")
    ap.add_argument("--sheet-yaw", type=float, help="作业纸 u 轴（s→t）转角（deg）")
    ap.add_argument("--pitch", type=float, nargs=2, metavar=("COL", "ROW"), help="实测列距、行距（mm）")
    ap.add_argument("--circle", type=float, help="实测区域圆直径（mm）")
    ap.add_argument("--grip-close", type=float, help="夹爪闭合角（deg）")
    ap.add_argument("--release-gap", type=float, help="松爪时木块最低棱离纸面（mm）")
    ap.add_argument("--step", action="store_true", help="每次夹紧/松开前回车确认")
    ap.add_argument("--pause-between", action="store_true", help="必做与选做之间暂停确认")
    ap.add_argument("--speed", type=float, default=4.0, help="仿真回放倍速")
    ap.add_argument("--save-frames", help="仿真帧导出目录（不弹窗）")
    args = ap.parse_args()
    if args.sheet:
        SHEET_CENTER_XY = tuple(args.sheet)
    if args.sheet_yaw is not None:
        SHEET_YAW_DEG = args.sheet_yaw
    if args.pitch:
        COL_PITCH, ROW_PITCH = args.pitch
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
        pl = Planner(1)
        pl.blocks = {}                     # 标定时纸上不放木块
        pts = pl.points
        pl.start_from_zero(pts["s1"])
        stops = []
        for k in POINT_NAMES:
            hover = np.array([pts[k][0], pts[k][1], TABLE_Z + BLOCK_SIZE + 20.0])
            pl.travel_to_hover(f"标定/{k}", hover, best_pitch(hover), target=None)
            stops.append((k, len(pl.segments)))
        pl.go_home("标定")
        print("标定：取走纸上所有木块。夹爪将张开依次停在 6 个圆心上方约 20 mm（块顶高度以上）。")
        print(describe_layout(pts))
        if input("撤手后输入 YES 开始：").strip().upper() != "YES":
            return
        ser = init_serial(args.port)
        try:
            start = 0
            for k, end in stops:
                for seg in pl.segments[start:end]:
                    stream(ser, seg.q)
                start = end
                input(f"  已停在 {k} 上方：核对夹爪中心与圆心的偏差（写入 PICK/PLACE_XY_COMP 或修正作业纸位置），回车继续")
            for seg in pl.segments[start:]:
                stream(ser, seg.q)
        finally:
            ser.close()
        return

    pl = build_plan(args.task)
    check_plan(pl, raise_on_fail=(args.mode == "hardware"))
    print(summarize(pl))
    print("状态机序列（前 12 个）：" + " → ".join(state_sequence(pl)[:12]) + " …")
    if pl.checks["issues"]:
        print("未通过项：\n  " + "\n  ".join(pl.checks["issues"][:20]))
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
