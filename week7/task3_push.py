"""第七周实验 1.4（附加实验）：桌面推块。

把外围四块木块沿桌面拖到中心，组成 2×2 方阵。夹爪全程闭合，用夹指压住
块顶中心，靠摩擦带动木块滑动；每块到位后只抬第 2 轴脱离，再高位换块。

运行方式（先离线，后实机）::

    python week7/task3_push.py --mode scan                   # 扫描工位中心与外围间距
    python week7/task3_push.py --mode plan --task 1          # 只规划+检查，导出 CSV/JSON/PNG
    python week7/task3_push.py --mode simulate --task 1      # 不连串口，播放俯视+侧视动画
    python week7/task3_push.py --mode calib --port COM5      # 实机标定：悬停在块顶上方供测量
    python week7/task3_push.py --mode hardware --task 1 --port COM5 [--step]

--task 1 必做“就近聚合”，2 选做一“同排换位”，3 选做二“跨排换位”。

坐标约定
--------
* 基座坐标系 B：与 week5~week7 的 D-H 模型一致，单位 mm；桌面为 z = TABLE_Z。
* 图坐标 (u, v)：指导书俯视图，u 向右、v 向上，原点为方阵中心（黑点）。
  默认图的“上”= 远离机械臂的 +X_B，“右”= -Y_B（站在机械臂身后俯视）；
  WORK_YAW_DEG 可把整个工作区绕中心旋转，起点和目标一起旋转。
* 接触点：闭合夹指压在块顶上的点，等于 D-H 末端加上末端坐标系中的
  固定偏移 CONTACT_OFFSET_TOOL，p_c(q) = t(q) + R(q)·Δp。
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
except Exception:  # pragma: no cover - 取决于运行环境
    rtb = None

try:
    import serial
except Exception:  # pragma: no cover
    serial = None


# ============================================================
# 1. 机械臂模型与实机参数（来自 week5~week7、week6、teleop）
# ============================================================

DH_PARAMS = [
    {"d": 108.5, "a": 0.0, "alpha": np.pi / 2, "offset": 0.0},
    {"d": 0.0, "a": 105.0, "alpha": 0.0, "offset": np.pi / 2},
    {"d": 0.0, "a": 97.5, "alpha": 0.0, "offset": 0.0},
    {"d": 0.0, "a": 28.22, "alpha": np.pi / 2, "offset": np.pi / 2},
    {"d": 172.70, "a": 0.0, "alpha": 0.0, "offset": 0.0},
]

# 实机确认的指令范围（deg）。J5 不影响末端位置，全程锁零。
JOINT_LIMITS_DEG = np.array(
    [[-90.0, 90.0], [-90.0, 90.0], [-90.0, 90.0], [-90.0, 25.0], [-180.0, 180.0]]
)
JOINT_MARGIN_DEG = 2.0          # 规划时离限位至少保留的余量
JOINT_SPEED_LIMIT_DEG_S = 60.0  # 任意关节的最大指令角速度

# 夹爪：teleop 中 O=+50° 张开、C=-40° 闭合、N=0° 参考位。全程保持闭合。
GRIPPER_CLOSED_DEG = -40.0
GRIPPER_CLOSE_TIME_S = 1.0

SERIAL_PORT = "COM5"   # 本组 teleop/week6 使用 COM5；指导书默认 COM3
BAUDRATE = 115200
SEND_PERIOD_S = 0.02   # 50 Hz 流式下发，与 teleop 一致


# ============================================================
# 2. 桌面、木块与工位（需按本组实物标定）
# ============================================================

TABLE_Z = 52.0          # 桌面在基座坐标系中的高度（week6 实机标定）
BLOCK_SIZE = 30.0       # h：按实物测量边长后修改
GAP = 6.0               # g：内围相邻块名义间隙（理论 0；取 6 mm 留出放置误差，避免相撞）
LANE_MARGIN = 8.0       # m：绕行通道与已到位木块的额外间距

WORK_CENTER_XY = (230.0, 0.0)  # 方阵中心 (x_c, y_c)，见 --mode scan
OUTER_OFFSET = 62.0            # D：外围块中心相对方阵中心的 u、v 偏距（须 ≥ h + s + m + g，见 min_outer_offset）
WORK_YAW_DEG = 0.0             # 工作坐标系绕中心旋转角

# 接触与补偿
PRESS_DEPTH = 2.0                          # δ：接触点指令低于块顶的下压量
CONTACT_OFFSET_TOOL = np.array([0.0, 0.0, 0.0])  # Δp：闭合夹指接触点在末端系中的偏移
CONTACT_Z_COMP = 0.0                       # 全局接触高度补偿（+ 为抬高）
BLOCK_Z_COMP = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}       # 逐块 Z 补偿
BLOCK_XY_COMP = {                          # 逐块终点 XY 补偿（基座系，mm）
    "a": (0.0, 0.0), "b": (0.0, 0.0), "c": (0.0, 0.0), "d": (0.0, 0.0),
}

# 速度与高度
HOVER_HEIGHT = 40.0       # 悬停点高出块顶的距离
PRE_TOUCH_HEIGHT = 6.0    # 慢速压下段的起始高度（高出块顶）
DESCEND_SPEED = 30.0      # mm/s，悬停 → 预接触
PRESS_SPEED = 5.0         # mm/s，预接触 → 压入
DRAG_SPEED = 15.0         # mm/s，拖动段平均速度（五次插值峰值约 1.9 倍）
MOVEJ_SPEED_DEG_S = 30.0  # MoveJ 按最大关节转角估算时长
MOVEJ_MIN_TIME = 1.2
READY_HEIGHT = 80.0       # 高位换块/起止姿态：方阵中心上方高出块顶的距离

# 抬离：只改 J2，正向使大臂后仰、接触点上抬（略向外 1~2 mm）
LIFT_DQ2_DEG = 5.0
LIFT_TIME = 0.8
LIFT_MIN_CLEAR = 5.0      # 抬离后接触点至少高出块顶

# 倾角：tool_pitch = 夹爪轴线相对水平面的俯仰（负值朝下），等于 q2+q3+q4+90°
TILT_MODE = "auto"          # auto：能固定就固定，否则按半径排程；或 "fixed"/"radial"
TILT_FIXED_MIN_MARGIN = 6.0  # 固定倾角整段的最小关节余量要求（deg）
TILT_SEARCH_DEG = np.arange(-85.0, -15.0, 1.0)
CONST_DESCENT_MARGIN = 5.0     # 下降段“倾角不变纯竖直”部分要求的关节余量（宁短勿贴限位）

# 净空检查
ARM_RADIUS = 15.0         # 连杆/夹爪主体包络半径
TIP_RADIUS = 5.0          # 闭合夹指尖端（距接触点 TIP_EXCLUDE 以内）的包络半径
BLOCK_CLEARANCE = 5.0     # 连杆包络高出木块顶面的最小距离
TIP_CLEARANCE = 2.0       # 夹指尖端包络高出非当前木块顶面的最小距离
TABLE_CLEARANCE = 3.0     # 连杆包络高出桌面的最小距离
TIP_EXCLUDE = 25.0        # 距接触点该链长以内视为夹指尖端；接触段内对当前木块豁免
SWEEP_MIN_GAP = 3.0       # 被拖木块与其他木块的最小间隙
MAX_FK_ERROR = 0.5        # 笛卡尔段接触点正解误差上限（mm）

OUTPUT_DIR = Path(__file__).resolve().parent / "push_output"
BLOCK_NAMES = ("a", "b", "c", "d")


# ============================================================
# 3. 正运动学（numpy，与 roboticstoolbox 标准 D-H 一致）
# ============================================================

def dh_matrix(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [[ct, -st * ca, st * sa, a * ct],
         [st, ct * ca, -ct * sa, a * st],
         [0.0, sa, ca, d],
         [0.0, 0.0, 0.0, 1.0]]
    )


def fk_frames(q_deg):
    """返回基座 + 五个关节坐标系的 6 个 4x4 齐次矩阵。q_deg 至少含前 5 轴。"""
    T = np.eye(4)
    frames = [T]
    for qi, p in zip(np.deg2rad(np.asarray(q_deg, dtype=float)[:5]), DH_PARAMS):
        T = T @ dh_matrix(qi + p["offset"], p["d"], p["a"], p["alpha"])
        frames.append(T)
    return frames


def contact_point(q_deg, offset=None):
    """接触点 p_c(q) = t(q) + R(q)·Δp。"""
    offset = CONTACT_OFFSET_TOOL if offset is None else offset
    T = fk_frames(q_deg)[-1]
    return T[:3, 3] + T[:3, :3] @ offset


def tool_pitch_deg(q_deg):
    """夹爪轴线俯仰角：0° 水平向外，-90° 竖直向下（= q2+q3+q4+90°）。"""
    return float(q_deg[1] + q_deg[2] + q_deg[3] + 90.0)


def build_rtb_model():
    if rtb is None:
        return None
    links = [
        rtb.RevoluteDH(d=p["d"], a=p["a"], alpha=p["alpha"], offset=p["offset"])
        for p in DH_PARAMS
    ]
    return rtb.DHRobot(links, name="Week7_Push_Arm")


# ============================================================
# 4. 固定倾角解析逆解（接触点 + 夹爪俯仰角 → 5 轴）
# ============================================================

class IKError(RuntimeError):
    pass


_WRIST_TO_CONTACT_CACHE: dict = {}


def wrist_to_contact(pitch_deg, offset):
    """q1=0 时，从 J4 轴心（坐标系 3 原点）到接触点的向量 (径向, 侧向, 竖直)。

    只与夹爪俯仰角和 Δp 有关，用正解数值求得，避免手推符号出错。
    """
    key = (round(float(pitch_deg), 6), tuple(np.round(offset, 6)))
    if key not in _WRIST_TO_CONTACT_CACHE:
        q = np.array([0.0, 0.0, 0.0, pitch_deg - 90.0, 0.0])
        frames = fk_frames(q)
        T5 = frames[-1]
        c = T5[:3, 3] + T5[:3, :3] @ offset
        _WRIST_TO_CONTACT_CACHE[key] = c - frames[3][:3, 3]
    return _WRIST_TO_CONTACT_CACHE[key]


def joint_margin(q_deg):
    """前四轴离限位的最小余量（deg），负值表示越限。"""
    q = np.asarray(q_deg, dtype=float)[:4]
    return float(np.min(np.minimum(q - JOINT_LIMITS_DEG[:4, 0], JOINT_LIMITS_DEG[:4, 1] - q)))


def ik_candidates(p, pitch_deg, offset=None):
    """返回所有几何上可行的 5 轴解（deg，含越限解），按肘部两个分支。"""
    offset = CONTACT_OFFSET_TOOL if offset is None else offset
    p = np.asarray(p, dtype=float)
    w_r, w_y, w_z = wrist_to_contact(pitch_deg, offset)
    rho2 = p[0] ** 2 + p[1] ** 2
    if rho2 < w_y ** 2:
        return []
    radial = np.sqrt(rho2 - w_y ** 2)
    q1 = np.arctan2(p[1], p[0]) - np.arctan2(w_y, radial)
    wr = radial - w_r                       # 腕点到肩轴的径向距离
    wz = p[2] - w_z - DH_PARAMS[0]["d"]     # 腕点相对肩轴的高度
    l2, l3 = DH_PARAMS[1]["a"], DH_PARAMS[2]["a"]
    c3 = (wr * wr + wz * wz - l2 * l2 - l3 * l3) / (2.0 * l2 * l3)
    if c3 < -1.0 - 1e-9 or c3 > 1.0 + 1e-9:
        return []
    c3 = float(np.clip(c3, -1.0, 1.0))
    out = []
    for sign in (-1.0, 1.0):
        q3 = sign * np.arccos(c3)
        th2 = np.arctan2(wz, wr) - np.arctan2(l3 * np.sin(q3), l2 + l3 * np.cos(q3))
        q2 = th2 - np.pi / 2
        q4 = np.deg2rad(pitch_deg - 90.0) - q2 - q3
        q = np.rad2deg([q1, q2, q3, q4, 0.0])
        q[:4] = (q[:4] + 180.0) % 360.0 - 180.0
        out.append(q)
    return out


def solve_ik(p, pitch_deg, q_prev=None, offset=None, margin=JOINT_MARGIN_DEG):
    """限位内的固定俯仰逆解；有多个解时取离上一时刻最近者，保证分支连续。"""
    valid = [q for q in ik_candidates(p, pitch_deg, offset) if joint_margin(q) >= margin]
    if not valid:
        raise IKError(
            f"接触点 {np.round(p, 1)} mm 在夹爪俯仰 {pitch_deg:.1f}° 下无限位内逆解"
        )
    if q_prev is None:
        return max(valid, key=joint_margin)
    return min(valid, key=lambda q: float(np.sum((q[:4] - q_prev[:4]) ** 2)))


def best_margin(p, pitch_deg):
    cands = ik_candidates(p, pitch_deg)
    return max((joint_margin(q) for q in cands), default=-np.inf)


# ============================================================
# 5. 倾角选择：固定倾角 / 按半径排程
# ============================================================

@dataclass
class TiltPlan:
    mode: str              # "fixed" or "radial"
    value: float = 0.0     # fixed 时的倾角
    coef: tuple = (0.0, 0.0, 0.0)  # radial 时 pitch = k_r·r + k_z·z + b

    def pitch(self, p):
        if self.mode == "fixed":
            return self.value
        r = float(np.hypot(p[0], p[1]))
        return float(self.coef[0] * r + self.coef[1] * p[2] + self.coef[2])

    def describe(self):
        if self.mode == "fixed":
            return f"固定 {self.value:.1f}°"
        k_r, k_z, b = self.coef
        return f"按位置排程 {k_r:.3f}·r {k_z:+.3f}·z {b:+.1f}°"


def fit_tilt_schedule(z_contact):
    """在接触高度及其上方若干高度，逐点找关节余量最大的倾角，再做 (r, z) 线性拟合。"""
    rows, pitches = [], []
    for z in z_contact + np.array([0.0, 20.0, 45.0, 90.0]):
        for r in np.arange(160.0, 340.0, 10.0):
            m, pt = max((best_margin((r, 0.0, z), pt), pt) for pt in TILT_SEARCH_DEG)
            if m > 0:
                rows.append([r, z, 1.0])
                pitches.append(pt)
    if len(rows) < 3:
        raise IKError("可达点过少，无法建立倾角排程")
    coef, *_ = np.linalg.lstsq(np.array(rows), np.array(pitches), rcond=None)
    return TiltPlan("radial", coef=tuple(float(c) for c in coef))


def const_pitch_top(p_low, pitch, z_max, step=2.0, margin=JOINT_MARGIN_DEG + 1.0):
    """从 p_low 起以固定倾角竖直向上，最高能到多少（不超过 z_max）。"""
    p = np.asarray(p_low, float)
    z = p[2]
    while z + step <= z_max + 1e-9 and best_margin((p[0], p[1], z + step), pitch) >= margin:
        z += step
    if z_max - z < step and best_margin((p[0], p[1], z_max), pitch) >= margin:
        z = z_max
    return float(z)


def blend_start_pitch(top, bottom, pitch_bottom, margin=JOINT_MARGIN_DEG + 1.0):
    """悬停点倾角：尽量接近 pitch_bottom，且 top→bottom 倾角线性过渡全程可行。"""
    top, bottom = np.asarray(top, float), np.asarray(bottom, float)
    for pt in sorted(np.arange(-85.0, -15.0, 0.5), key=lambda x: abs(x - pitch_bottom)):
        if all(best_margin(top + t * (bottom - top), (1 - t) * pt + t * pitch_bottom) >= margin
               for t in np.linspace(0.0, 1.0, 9)):
            return float(pt)
    raise IKError(f"悬停点 {np.round(top, 1)} 下降过程找不到连续可行的倾角过渡")


def point_pitch(p, schedule, min_margin=TILT_FIXED_MIN_MARGIN):
    """非接触点（悬停/准备位）的倾角：先用排程，余量不足时单独搜索最优。"""
    pt = schedule.pitch(p)
    if best_margin(p, pt) >= min_margin:
        return pt
    m, pt = max((best_margin(p, x), x) for x in TILT_SEARCH_DEG)
    if m < JOINT_MARGIN_DEG:
        raise IKError(f"点 {np.round(p, 1)} mm 在任何倾角下都没有足够的关节余量")
    return float(pt)


def choose_tilt(points, schedule):
    """为一块木块的全部关键接触点选倾角：优先整段固定倾角。"""
    points = np.atleast_2d(points)
    if TILT_MODE in ("auto", "fixed"):
        best = (-np.inf, None)
        for pt in TILT_SEARCH_DEG:
            m = min(best_margin(p, pt) for p in points)
            if m > best[0]:
                best = (m, pt)
        if best[0] >= TILT_FIXED_MIN_MARGIN or (TILT_MODE == "fixed" and best[0] > JOINT_MARGIN_DEG):
            return TiltPlan("fixed", value=float(best[1])), best[0]
        if TILT_MODE == "fixed":
            raise IKError("该木块路径不存在整段共同可达的固定倾角，请改用 auto/radial 或调整工位")
    m = min(best_margin(p, schedule.pitch(p)) for p in points)
    return schedule, m


# ============================================================
# 6. 工位几何与三种任务的目标对应、顺序、绕行点（图坐标 u 右 v 上）
# ============================================================

def fig_to_base(uv, z):
    """图坐标 (u, v) → 基座坐标：v 沿工作区 +X，u 沿工作区 -Y。"""
    yaw = np.deg2rad(WORK_YAW_DEG)
    u, v = uv
    local = np.array([v, -u])
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    xy = np.asarray(WORK_CENTER_XY) + R @ local
    return np.array([xy[0], xy[1], z])


def base_to_fig(xy):
    yaw = np.deg2rad(WORK_YAW_DEG)
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    local = R.T @ (np.asarray(xy[:2]) - np.asarray(WORK_CENTER_XY))
    return np.array([-local[1], local[0]])


def min_outer_offset():
    """选做二中 c 的右侧通道、b 的转向点与外围 d 保持至少 g 的间隙所需的最小 D。"""
    s = (BLOCK_SIZE + GAP) / 2.0
    return BLOCK_SIZE + s + LANE_MARGIN + GAP


def task_layout(task):
    """返回 (外围起点, 内围目标名, 执行顺序, 每块图坐标路径)。"""
    h, m, D = BLOCK_SIZE, LANE_MARGIN, OUTER_OFFSET
    s = (BLOCK_SIZE + GAP) / 2.0
    slots = {"TL": (-s, s), "TR": (s, s), "BL": (-s, -s), "BR": (s, -s)}
    outer = {"a": (-D, D), "b": (D, D), "c": (-D, -D), "d": (D, -D)}

    if task == 1:      # 就近聚合：上排 a b，下排 c d
        target = {"a": "TL", "b": "TR", "c": "BL", "d": "BR"}
        order = ["a", "b", "c", "d"]
        routes = {k: [outer[k], slots[target[k]]] for k in order}
    elif task == 2:    # 同排换位：上排 b a，下排 c d
        target = {"a": "TR", "b": "TL", "c": "BL", "d": "BR"}
        order = ["a", "b", "c", "d"]          # a 必须先到右上，b 才能从上方通道绕行
        lane_v = s + h + m                    # b 下沿高出 a 上沿 m
        side_u = -s - m                       # b 下行时与 a 侧面相距 g+m
        routes = {
            "a": [outer["a"], slots["TR"]],
            "b": [outer["b"], (side_u, lane_v), (side_u, s), slots["TL"]],
            "c": [outer["c"], slots["BL"]],
            "d": [outer["d"], slots["BR"]],
        }
    elif task == 3:    # 跨排换位：上排 a c，下排 b d
        target = {"a": "TL", "b": "BL", "c": "TR", "d": "BR"}
        order = ["a", "b", "c", "d"]          # 右下区域是转向通道，d 必须最后
        routes = {
            "a": [outer["a"], slots["TL"]],
            # b：斜穿空着的右上/右下区域到转向点，再沿下方通道左移，最后上插 m
            "b": [outer["b"], (s + m, -s - m), (-s, -s - m), slots["BL"]],
            # c：先沿已到位 b 下方的通道右移（与选做一 b 的上方通道对称，
            #    比沿外围下排 v=-D 走离基座更远，关节余量更大），再沿右侧通道
            #    上行穿过空着的右下区域，最后左插 m
            "c": [outer["c"], (s + m, -(s + h + m)), (s + m, s), slots["TR"]],
            "d": [outer["d"], slots["BR"]],
        }
    else:
        raise ValueError("task 只能是 1、2、3")
    return outer, {k: slots[v] for k, v in target.items()}, target, order, routes


# ============================================================
# 7. 轨迹段生成
# ============================================================

@dataclass
class Segment:
    label: str
    kind: str                      # gripper/movej/descend/press/drag/lift
    block: str | None
    q: np.ndarray                  # N x 6 指令角（deg，含 J6）
    duration: float
    contact_ref: np.ndarray | None = None   # N x 3 规划接触点
    in_contact: bool = False
    pitch: np.ndarray | None = None


@dataclass
class Plan:
    task: int
    segments: list = field(default_factory=list)
    blocks: dict = field(default_factory=dict)   # 每块的布局/倾角/关键点信息
    order: list = field(default_factory=list)    # 本次实际执行的木块顺序
    checks: dict = field(default_factory=dict)
    initial_xy: dict = field(default_factory=dict)  # 开始时四块的位置（基座系 XY）


def quintic(n):
    tau = np.linspace(0.0, 1.0, n)
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


def n_samples(duration):
    return max(2, int(round(duration / SEND_PERIOD_S)) + 1)


def with_gripper(q5):
    q5 = np.atleast_2d(q5)
    return np.hstack([q5, np.full((len(q5), 1), GRIPPER_CLOSED_DEG)])


def movej(label, q_from, q_to, block=None, duration=None):
    dq = float(np.max(np.abs(np.asarray(q_to[:5]) - np.asarray(q_from[:5]))))
    if duration is None:
        duration = max(MOVEJ_MIN_TIME, 1.875 * dq / MOVEJ_SPEED_DEG_S)
    s = quintic(n_samples(duration))[:, None]
    q = np.asarray(q_from[:5]) + s * (np.asarray(q_to[:5]) - np.asarray(q_from[:5]))
    return Segment(label, "movej", block, with_gripper(q), duration)


def movel(label, kind, p0, p1, tilt, q_prev, speed, block, in_contact, blend=None):
    """直线段五次插值 + 逐点固定俯仰逆解。blend=(俯仰0, 俯仰1) 时俯仰随进度线性过渡。"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    q_start = q_prev
    duration = max(0.6, float(np.linalg.norm(p1 - p0)) / speed)
    for _ in range(8):  # 若关节角速度超限，则放慢该段重新插值
        s = quintic(n_samples(duration))[:, None]
        pts = p0 + s * (p1 - p0)
        qs, pitches, q_prev = [], [], q_start
        for si, p in zip(s[:, 0], pts):
            pt = tilt.pitch(p) if blend is None else (1 - si) * blend[0] + si * blend[1]
            q_prev = solve_ik(p, pt, q_prev)
            qs.append(q_prev)
            pitches.append(pt)
        qs = np.array(qs)
        vel = np.max(np.abs(np.diff(qs[:, :5], axis=0))) / SEND_PERIOD_S
        if vel <= 0.9 * JOINT_SPEED_LIMIT_DEG_S:
            break
        duration *= vel / (0.85 * JOINT_SPEED_LIMIT_DEG_S)
    return Segment(label, kind, block, with_gripper(qs), duration,
                   contact_ref=pts, in_contact=in_contact, pitch=np.array(pitches))


def lift_segment(label, q_from, block):
    q_to = np.asarray(q_from[:5], float).copy()
    q_to[1] += LIFT_DQ2_DEG
    s = quintic(n_samples(LIFT_TIME))[:, None]
    q = np.asarray(q_from[:5]) + s * (q_to - np.asarray(q_from[:5]))
    return Segment(label, "lift", block, with_gripper(q), LIFT_TIME, in_contact=True)


def block_geometry(task):
    """计算每块的基座系关键点：起点、绕行点、终点（含补偿）。"""
    outer, slots, target, order, routes = task_layout(task)
    top = TABLE_Z + BLOCK_SIZE
    info = {}
    for k in order:
        z_c = top - PRESS_DEPTH + CONTACT_Z_COMP + BLOCK_Z_COMP[k]
        pts = [fig_to_base(uv, z_c) for uv in routes[k]]
        pts[-1][:2] += np.asarray(BLOCK_XY_COMP[k])
        info[k] = {
            "outer_uv": outer[k], "target_slot": target[k], "target_uv": slots[k],
            "route_uv": routes[k], "contact_z": z_c, "route_base": np.array(pts),
            "hover": np.array([pts[0][0], pts[0][1], top + HOVER_HEIGHT]),
            "pre_touch": np.array([pts[0][0], pts[0][1], top + PRE_TOUCH_HEIGHT]),
        }
    return info, order


def build_plan(task, only=None):
    """only：只执行其中的木块（单块验证用）。顺序在它们之前的木块视为已在
    内围目标，之后的仍在外围起点；请在运动前按此手工摆放。"""
    info, full_order = block_geometry(task)
    only = set(only or full_order)
    unknown = only - set(full_order)
    if unknown:
        raise ValueError(f"未知木块：{sorted(unknown)}")
    order = [k for k in full_order if k in only]
    initial_xy = {}
    for i, k in enumerate(full_order):
        placed_before = k not in only and i < max(full_order.index(x) for x in only)
        route = info[k]["route_base"]
        initial_xy[k] = (route[-1] if placed_before else route[0])[:2].copy()
    top = TABLE_Z + BLOCK_SIZE
    schedule = fit_tilt_schedule(top - PRESS_DEPTH)

    # 每块选择接触倾角：覆盖预接触点与整条拖动路径（悬停点单独定倾角）
    for k in order:
        route = info[k]["route_base"]
        dense = [info[k]["pre_touch"]]
        for a, b in zip(route[:-1], route[1:]):
            dense += [a + t * (b - a) for t in np.linspace(0, 1, 8)]
        tilt, m = choose_tilt(np.array(dense), schedule)
        info[k]["tilt"], info[k]["tilt_margin"] = tilt, m
        # 下降段分两段：贴近木块的一段保持接触倾角、纯竖直平动（夹爪不转动），
        # 尽量拉长；倾角变化只在其上方进行（夹爪末端仍沿竖直线）。
        pc = tilt.pitch(info[k]["pre_touch"])
        pre = info[k]["pre_touch"]
        z_c = const_pitch_top(pre, pc, info[k]["hover"][2], margin=CONST_DESCENT_MARGIN)
        info[k]["descend_mid"] = np.array([pre[0], pre[1], z_c])
        info[k]["hover_pitch"] = (pc if info[k]["hover"][2] - z_c < 0.5
                                  else blend_start_pitch(info[k]["hover"], info[k]["descend_mid"], pc,
                                                         margin=CONST_DESCENT_MARGIN))

    ready_p = fig_to_base((0.0, 0.0), top + READY_HEIGHT)
    q_zero = np.zeros(5)
    q_ready = solve_ik(ready_p, point_pitch(ready_p, schedule))

    plan = Plan(task=task, order=order, blocks=info, initial_xy=initial_xy)
    plan.blocks["_ready"] = {"point": ready_p, "q": q_ready, "schedule": schedule}
    segs = plan.segments

    # 起始：零位闭合夹爪 → 高位准备姿态
    n = n_samples(GRIPPER_CLOSE_TIME_S)
    q6 = np.zeros((n, 6))
    q6[:, 5] = np.linspace(0.0, GRIPPER_CLOSED_DEG, n)
    segs.append(Segment("close_gripper", "gripper", None, q6, GRIPPER_CLOSE_TIME_S))
    segs.append(movej("zero->ready", q_zero, q_ready))
    q_cur = q_ready

    for k in order:
        b = info[k]
        tilt = b["tilt"]
        q_hover = solve_ik(b["hover"], b["hover_pitch"], q_cur)
        segs.append(movej(f"{k}/to_hover", q_cur, q_hover, block=k))
        pc = tilt.pitch(b["pre_touch"])
        seg = None
        if b["hover"][2] - b["descend_mid"][2] > 0.5:
            seg = movel(f"{k}/descend_turn", "descend", b["hover"], b["descend_mid"], tilt, q_hover,
                        DESCEND_SPEED, k, in_contact=False, blend=(b["hover_pitch"], pc))
            segs.append(seg)
        seg = movel(f"{k}/descend", "descend", b["descend_mid"], b["pre_touch"], tilt,
                    seg.q[-1, :5] if seg is not None else q_hover,
                    DESCEND_SPEED, k, in_contact=False, blend=(pc, pc))
        segs.append(seg)
        seg = movel(f"{k}/press", "press", b["pre_touch"], b["route_base"][0], tilt,
                    seg.q[-1, :5], PRESS_SPEED, k, in_contact=True)
        segs.append(seg)
        for i, (p0, p1) in enumerate(zip(b["route_base"][:-1], b["route_base"][1:])):
            seg = movel(f"{k}/drag{i + 1}", "drag", p0, p1, tilt, seg.q[-1, :5],
                        DRAG_SPEED, k, in_contact=True)
            segs.append(seg)
        b["q_contact_start"] = segs[-1 - (len(b["route_base"]) - 1)].q[-1, :5].copy()
        b["q_contact_end"] = seg.q[-1, :5].copy()
        lift = lift_segment(f"{k}/lift_J2", seg.q[-1], k)
        segs.append(lift)
        q_cur = lift.q[-1, :5]

    segs.append(movej("exit->ready", q_cur, q_ready))
    segs.append(movej("ready->zero", q_ready, q_zero))
    return plan


# ============================================================
# 8. 离线检查
# ============================================================

def arm_samples(q_deg, step=10.0):
    """沿连杆链采样包络中心点；返回 (点, 到接触点的链长)。不含底座立柱。"""
    frames = fk_frames(q_deg)
    nodes = [f[:3, 3] for f in frames[1:]] + [contact_point(q_deg)]
    pts, arc = [], []
    seg_len = [float(np.linalg.norm(b - a)) for a, b in zip(nodes[:-1], nodes[1:])]
    remaining = np.cumsum(seg_len[::-1])[::-1]       # 各段起点到接触点的链长
    for i, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
        n = max(2, int(np.ceil(seg_len[i] / step)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            pts.append(a + t * (b - a))
            arc.append(remaining[i] - t * seg_len[i])
    return np.array(pts), np.array(arc)


def square_gap(c1, c2):
    """两个与工作系对齐的正方形（边长 h）之间的间隙；负值为重叠深度。"""
    d = np.abs(base_to_fig(c1) - base_to_fig(c2))
    return float(np.max(d) - BLOCK_SIZE)


def inside_footprint(p, center, expand):
    d = np.abs(base_to_fig(p) - base_to_fig(center))
    return bool(np.all(d <= BLOCK_SIZE / 2.0 + expand))


def check_plan(plan, stride=1, raise_on_fail=True):
    top = TABLE_Z + BLOCK_SIZE
    positions = {k: v.copy() for k, v in plan.initial_xy.items()}
    issue_map = {}

    class _Issues(list):
        """同一段、同一类问题只保留最严重的一条。"""

        def append(self, msg):
            # 去掉数字后作为键：同段同类问题（如同一木块的净空不足）只记第一条
            issue_map.setdefault(re.sub(r"[-\d.\[\], ]", "", msg), msg)

    issues = _Issues()
    stats = {"min_joint_margin": np.inf, "max_joint_speed": 0.0, "max_fk_error": 0.0,
             "min_block_gap": np.inf, "min_arm_block_clear": np.inf,
             "min_table_clear": np.inf, "lift": {}}
    q_all = np.vstack([s.q for s in plan.segments])
    stats["joint_range_deg"] = {
        f"J{i + 1}": [float(q_all[:, i].min()), float(q_all[:, i].max())] for i in range(6)
    }

    prev_last = None
    for seg in plan.segments:
        q = seg.q
        # 段间连续
        if prev_last is not None and np.max(np.abs(q[0] - prev_last)) > 1e-6:
            issues.append(f"{seg.label}: 与上一段首尾不连续")
        prev_last = q[-1]
        # 限位与角速度
        m = min(joint_margin(x) for x in q)
        stats["min_joint_margin"] = min(stats["min_joint_margin"], m)
        if m < JOINT_MARGIN_DEG - 1e-6:
            issues.append(f"{seg.label}: 关节离限位仅 {m:.1f}°")
        vel = np.max(np.abs(np.diff(q, axis=0))) / SEND_PERIOD_S
        stats["max_joint_speed"] = max(stats["max_joint_speed"], float(vel))
        if vel > JOINT_SPEED_LIMIT_DEG_S:
            issues.append(f"{seg.label}: 关节角速度 {vel:.0f}°/s 超过 {JOINT_SPEED_LIMIT_DEG_S:.0f}")
        # 正解误差
        if seg.contact_ref is not None:
            err = max(np.linalg.norm(contact_point(x) - p) for x, p in zip(q, seg.contact_ref))
            stats["max_fk_error"] = max(stats["max_fk_error"], float(err))
            if err > MAX_FK_ERROR:
                issues.append(f"{seg.label}: 接触点正解误差 {err:.3f} mm")
        # 木块扫掠
        if seg.kind == "drag":
            for p in seg.contact_ref[::stride]:
                for other, c in positions.items():
                    if other == seg.block:
                        continue
                    gap = square_gap(p, c)
                    stats["min_block_gap"] = min(stats["min_block_gap"], gap)
                    if gap < SWEEP_MIN_GAP:
                        issues.append(f"{seg.label}: 木块 {seg.block} 与 {other} 间隙 {gap:.1f} mm")
                        break
        # 连杆/夹爪净空
        for x in q[::stride]:
            pts, arc = arm_samples(x)
            for p, a in zip(pts, arc):
                tip = a < TIP_EXCLUDE
                radius = TIP_RADIUS if tip else ARM_RADIUS
                need = TIP_CLEARANCE if tip else BLOCK_CLEARANCE
                table_clear = p[2] - radius - TABLE_Z
                stats["min_table_clear"] = min(stats["min_table_clear"], table_clear)
                if table_clear < TABLE_CLEARANCE:
                    issues.append(f"{seg.label}: 连杆点 z={p[2]:.1f} mm 过低（桌面）")
                    break
                for other, c in positions.items():
                    # 接触相关段内，夹指尖端压在当前木块上是正常的
                    if tip and seg.block == other and seg.kind in ("descend", "press", "drag", "lift"):
                        continue
                    if inside_footprint(p, c, radius):
                        clear = p[2] - radius - top
                        stats["min_arm_block_clear"] = min(stats["min_arm_block_clear"], clear)
                        if clear < need:
                            issues.append(
                                f"{seg.label}: 连杆点 {np.round(p, 1)} 距木块 {other} 顶面仅 {clear:.1f} mm")
                            break
        # 抬离检查
        if seg.kind == "lift":
            c0, c1 = contact_point(q[0]), contact_point(q[-1])
            rise = c1[2] - top
            stats["lift"][seg.block] = {"rise_above_top": float(rise),
                                        "horizontal_shift": float(np.linalg.norm(c1[:2] - c0[:2]))}
            if rise < LIFT_MIN_CLEAR:
                issues.append(f"{seg.label}: 抬离后接触点仅高出块顶 {rise:.1f} mm")
        # 拖动完成后更新木块位置
        if seg.kind == "drag":
            positions[seg.block] = seg.contact_ref[-1][:2].copy()

    stats["final_positions"] = {k: positions[k].tolist() for k in plan.order}
    issues = list(issue_map.values())
    stats["issues"] = issues
    plan.checks = stats
    if issues and raise_on_fail:
        head = "\n  ".join(issues[:12])
        raise RuntimeError(f"离线检查未通过（共 {len(issues)} 项）：\n  {head}")
    return stats


def cross_check_rtb(plan):
    """用 roboticstoolbox 的 fkine 复核 numpy 正解（课程环境中可用时）。"""
    robot = build_rtb_model()
    if robot is None:
        return None
    worst = 0.0
    for seg in plan.segments:
        for x in seg.q[:: max(1, len(seg.q) // 4)]:
            t_rtb = robot.fkine(np.deg2rad(x[:5])).t
            worst = max(worst, float(np.linalg.norm(t_rtb - fk_frames(x)[-1][:3, 3])))
    return worst


# ============================================================
# 9. 导出与报告摘要
# ============================================================

def summarize(plan):
    lines = [f"\n========== 桌面推块 任务 {plan.task} =========="]
    lines.append(f"方阵中心 {WORK_CENTER_XY} mm，外围偏距 D={OUTER_OFFSET} mm，"
                 f"间隙 g={GAP} mm，块边长 h={BLOCK_SIZE} mm，桌面 z={TABLE_Z} mm")
    lines.append(f"执行顺序：{' → '.join(plan.order)}")
    for k in plan.order:
        b = plan.blocks[k]
        r = b["route_base"]
        lines.append(
            f"  {k}: 外围 {np.round(r[0][:2], 1)} → {b['target_slot']} {np.round(r[-1][:2], 1)}"
            f"，绕行点 {len(r) - 2} 个，接触 z={b['contact_z']:.1f}，倾角 {b['tilt'].describe()}"
            f"（最小余量 {b['tilt_margin']:.1f}°）")
        lines.append(f"     压下关节角 {np.round(b['q_contact_start'], 1)}°，"
                     f"到位关节角 {np.round(b['q_contact_end'], 1)}°")
    c = plan.checks
    total = sum(s.duration for s in plan.segments)
    lines.append(f"总时长 {total:.1f} s；关节最小余量 {c['min_joint_margin']:.1f}°；"
                 f"最大角速度 {c['max_joint_speed']:.1f}°/s；最大接触点正解误差 {c['max_fk_error']:.4f} mm")
    lines.append(f"拖动中木块最小间隙 {c['min_block_gap']:.1f} mm；"
                 f"连杆距木块顶面最小净空 {c['min_arm_block_clear']:.1f} mm")
    for k, v in c["lift"].items():
        lines.append(f"  抬离 {k}: 接触点高出块顶 {v['rise_above_top']:.1f} mm，水平移动 {v['horizontal_shift']:.2f} mm")
    rng = c["joint_range_deg"]
    lines.append("关节范围：" + "，".join(f"{j} [{a:.1f}, {b:.1f}]" for j, (a, b) in rng.items()))
    lines.append("离线检查：" + ("通过" if not c["issues"] else f"{len(c['issues'])} 项未通过"))
    return "\n".join(lines)


def plan_tag(plan):
    tag = f"task{plan.task}"
    if len(plan.order) < len(plan.initial_xy):
        tag += "_only_" + "".join(plan.order)
    return tag


def export_plan(plan, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / plan_tag(plan)
    with open(f"{stem}_trajectory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "segment", "block", "q1", "q2", "q3", "q4", "q5", "q6",
                    "pitch_deg", "ref_x", "ref_y", "ref_z", "fk_x", "fk_y", "fk_z"])
        t = 0.0
        for seg in plan.segments:
            for i, x in enumerate(seg.q):
                ref = seg.contact_ref[i] if seg.contact_ref is not None else [np.nan] * 3
                fk = contact_point(x)
                w.writerow([f"{t:.3f}", seg.label, seg.block or ""]
                           + [f"{v:.3f}" for v in x]
                           + [f"{tool_pitch_deg(x):.2f}"]
                           + [f"{v:.3f}" for v in ref] + [f"{v:.3f}" for v in fk])
                t += SEND_PERIOD_S
    summary = {
        "task": plan.task, "order": plan.order,
        "params": {"TABLE_Z": TABLE_Z, "BLOCK_SIZE": BLOCK_SIZE, "GAP": GAP,
                   "WORK_CENTER_XY": WORK_CENTER_XY, "OUTER_OFFSET": OUTER_OFFSET,
                   "WORK_YAW_DEG": WORK_YAW_DEG, "PRESS_DEPTH": PRESS_DEPTH,
                   "CONTACT_OFFSET_TOOL": CONTACT_OFFSET_TOOL.tolist(),
                   "LIFT_DQ2_DEG": LIFT_DQ2_DEG, "GRIPPER_CLOSED_DEG": GRIPPER_CLOSED_DEG},
        "blocks": {
            k: {"target_slot": plan.blocks[k]["target_slot"],
                "route_base_xy": plan.blocks[k]["route_base"][:, :2].round(2).tolist(),
                "contact_z": plan.blocks[k]["contact_z"],
                "tilt": plan.blocks[k]["tilt"].describe(),
                "q_contact_start_deg": plan.blocks[k]["q_contact_start"].round(2).tolist(),
                "q_contact_end_deg": plan.blocks[k]["q_contact_end"].round(2).tolist()}
            for k in plan.order},
        "duration_s": sum(s.duration for s in plan.segments),
        "checks": {k: v for k, v in plan.checks.items()},
    }
    with open(f"{stem}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    return Path(f"{stem}_trajectory.csv"), Path(f"{stem}_summary.json")


# ============================================================
# 10. 可视化（俯视 + 侧视；可选 roboticstoolbox 三维）
# ============================================================

COLORS = {"a": "#d1495b", "b": "#2e86ab", "c": "#3b8b5a", "d": "#c77d1a"}


def _square(ax, center_xy, **kw):
    import matplotlib.patches as patches
    yaw = np.deg2rad(WORK_YAW_DEG)
    h = BLOCK_SIZE
    corner = np.asarray(center_xy) - np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]) @ np.array([h / 2, h / 2])
    rect = patches.Rectangle(corner, h, h, angle=np.rad2deg(yaw), **kw)
    ax.add_patch(rect)
    return rect


def draw_layout(plan, ax):
    info = plan.blocks
    for k in plan.order:
        b = info[k]
        r = b["route_base"]
        _square(ax, r[0][:2], facecolor="#a9cff0", edgecolor="0.4", lw=0.8)
        _square(ax, r[-1][:2], facecolor="#f6cc96", edgecolor="0.4", lw=0.8, alpha=0.7)
        ax.text(*r[0][:2], k, ha="center", va="center", fontsize=10)
        ax.plot(r[:, 0], r[:, 1], "-", color=COLORS[k], lw=1.6)
        ax.annotate(f"{plan.order.index(k) + 1}", r[0][:2] + np.array([0, 14]),
                    color=COLORS[k], fontsize=8, ha="center")
    for k, xy in plan.initial_xy.items():   # --only 时不动的木块画成灰色障碍
        if k not in plan.order:
            _square(ax, xy, facecolor="0.75", edgecolor="0.3", lw=0.8)
            ax.text(*xy, k, ha="center", va="center", fontsize=10)
    ax.plot(*WORK_CENTER_XY, "k.", ms=6)
    ax.plot(0, 0, "ks", ms=8)
    ax.set_aspect("equal")
    ax.set_xlabel("X_base (mm)")
    ax.set_ylabel("Y_base (mm)")
    ax.grid(alpha=0.3)


def side_view_points(q):
    frames = fk_frames(q)
    pts = [f[:3, 3] for f in frames] + [contact_point(q)]
    q1 = np.deg2rad(q[0])
    radial = np.array([np.cos(q1), np.sin(q1), 0.0])
    return np.array([[p @ radial, p[2]] for p in pts])


def save_overview_figure(plan, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    draw_layout(plan, ax1)
    ax1.set_title(f"Task {plan.task}: top view (blue start, orange target, lines = contact routes)")
    for k in plan.order:
        for key, style in (("q_contact_start", "-"), ("q_contact_end", "--")):
            sv = side_view_points(plan.blocks[k][key])
            ax2.plot(sv[:, 0], sv[:, 1], style, marker="o", ms=3, color=COLORS[k],
                     label=f"{k} {'start' if style == '-' else 'end'}")
    sv = side_view_points(plan.blocks["_ready"]["q"])
    ax2.plot(sv[:, 0], sv[:, 1], ":", marker="o", ms=3, color="0.4", label="ready")
    ax2.axhline(TABLE_Z, color="sienna", lw=1)
    ax2.axhline(TABLE_Z + BLOCK_SIZE, color="sienna", lw=0.6, ls="--")
    ax2.set_aspect("equal")
    ax2.set_xlabel("radial (mm)")
    ax2.set_ylabel("z (mm)")
    ax2.set_title("Side view in each arm plane (contact start / end)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def animate_plan(plan, speed=3.0):
    """俯视（木块随接触点移动）+ 侧视（机械臂骨架）动画。"""
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    draw_layout(plan, ax1)
    positions = {k: v.copy() for k, v in plan.initial_xy.items()}
    patches = {k: _square(ax1, positions[k], facecolor=COLORS[k], alpha=0.85, zorder=5)
               for k in positions}
    (arm_top,) = ax1.plot([], [], "-o", color="0.2", ms=3, zorder=6)
    (trace,) = ax1.plot([], [], color="darkorange", lw=1.2, zorder=7)
    (arm_side,) = ax2.plot([], [], "-o", color="0.2", ms=4)
    ax2.axhline(TABLE_Z, color="sienna")
    ax2.axhline(TABLE_Z + BLOCK_SIZE, color="sienna", ls="--", lw=0.6)
    ax2.set_xlim(-60, 380)
    ax2.set_ylim(0, 420)
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.3)
    title = ax1.set_title("")
    trace_pts = []
    yaw = WORK_YAW_DEG
    stride = max(1, int(round(speed)))
    for seg in plan.segments:
        for i in range(0, len(seg.q), stride):
            x = seg.q[i]
            frames = fk_frames(x)
            pts = np.array([f[:3, 3] for f in frames] + [contact_point(x)])
            arm_top.set_data(pts[:, 0], pts[:, 1])
            if seg.in_contact:
                trace_pts.append(pts[-1][:2])
                trace.set_data(*np.array(trace_pts).T)
            if seg.kind == "drag":
                positions[seg.block] = seg.contact_ref[i][:2]
                h = BLOCK_SIZE
                R = np.array([[np.cos(np.deg2rad(yaw)), -np.sin(np.deg2rad(yaw))],
                              [np.sin(np.deg2rad(yaw)), np.cos(np.deg2rad(yaw))]])
                patches[seg.block].set_xy(positions[seg.block] - R @ np.array([h / 2, h / 2]))
            sv = side_view_points(x)
            arm_side.set_data(sv[:, 0], sv[:, 1])
            title.set_text(f"{seg.label}   pitch {tool_pitch_deg(x):.0f} deg")
            plt.pause(SEND_PERIOD_S * stride / speed)
    plt.show()


def animate_rtb(plan, dt=0.04):
    """可选：沿用 task1/task2 的 roboticstoolbox 三维回放。"""
    robot = build_rtb_model()
    if robot is None:
        print("未安装 roboticstoolbox，跳过三维回放。")
        return
    import matplotlib.pyplot as plt
    q_all = np.vstack([s.q[::3, :5] for s in plan.segments])
    env = robot.plot(np.deg2rad(q_all[0]), block=False,
                     limits=[-150, 450, -250, 250, 0, 500], jointaxes=True, shadow=True, dt=dt)
    ax = env.ax
    for k in plan.order:
        r = plan.blocks[k]["route_base"]
        ax.plot(r[:, 0], r[:, 1], r[:, 2], color=COLORS[k], lw=2)
    for q in q_all:
        robot.q = np.deg2rad(q)
        env.step(dt)
    plt.pause(1.0)


# ============================================================
# 11. 实机通信与执行
# ============================================================

def pack_frame(q_rad):
    q_rad = np.asarray(q_rad, dtype=np.float64)
    if q_rad.shape != (6,) or not np.all(np.isfinite(q_rad)):
        raise ValueError("必须传入 6 个有限的关节弧度值")
    tx = bytearray(16)
    tx[0] = 0xAA
    for i, q_i in enumerate(q_rad):
        v = max(min(int(q_i * 1000), 32767), -32768)
        tx[1 + 2 * i] = (v >> 8) & 0xFF
        tx[2 + 2 * i] = v & 0xFF
    tx[13] = 0x01
    check = 0
    for byte in tx[:14]:
        check ^= byte
    tx[14] = check
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
    print(f"已连接 {port}，发送六轴零位（J6=0 参考位）。")
    time.sleep(1.0)
    ser.write(pack_frame(np.zeros(6)))
    ser.flush()
    time.sleep(1.5)
    return ser


def stream(ser, q_deg_rows):
    """按 SEND_PERIOD_S 等间隔下发；模型角即指令角（方向映射净效果为 +1）。"""
    t0 = time.perf_counter()
    for i, row in enumerate(q_deg_rows):
        ser.write(pack_frame(np.deg2rad(row)))
        ser.flush()
        remaining = t0 + (i + 1) * SEND_PERIOD_S - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)


def execute_plan(ser, plan, step_mode=False):
    start = time.perf_counter()
    for seg in plan.segments:
        if step_mode and seg.kind == "descend":
            ans = input(f"[{seg.block}] 即将下降接触，回车继续，输入 q 中止：")
            if ans.strip().lower() == "q":
                raise KeyboardInterrupt
        print(f"  执行 {seg.label}（{seg.duration:.1f} s）")
        stream(ser, seg.q)
    return time.perf_counter() - start


def build_calib_segments(block):
    """标定：零位 → 准备位 → 外围块上方 PRE_TOUCH_HEIGHT 处悬停（不接触）。"""
    plan = build_plan(1)
    b = plan.blocks[block]
    segs = plan.segments[:2]
    q_ready = plan.blocks["_ready"]["q"]
    q_pre = solve_ik(b["pre_touch"], b["tilt"].pitch(b["pre_touch"]), q_ready)
    segs.append(movej("ready->pre_touch", q_ready, q_pre))
    back = [movej("pre_touch->ready", q_pre, q_ready), movej("ready->zero", q_ready, np.zeros(5))]
    return segs, back, b


# ============================================================
# 12. 工位扫描
# ============================================================

def run_scan():
    global WORK_CENTER_XY, OUTER_OFFSET
    saved = (WORK_CENTER_XY, OUTER_OFFSET)
    print("x_c   D   | 任务1  任务2  任务3 | 最小关节余量  最小木块间隙  最小连杆净空")
    rows = []
    for D in (55.0, 60.0, 65.0):
        for xc in np.arange(210.0, 261.0, 10.0):
            WORK_CENTER_XY, OUTER_OFFSET = (float(xc), 0.0), D
            res, margins = [], []
            for task in (1, 2, 3):
                try:
                    plan = build_plan(task)
                    st = check_plan(plan, stride=4, raise_on_fail=False)
                    ok = not st["issues"]
                    res.append("OK " if ok else f"x{len(st['issues'])}")
                    margins.append((st["min_joint_margin"], st["min_block_gap"], st["min_arm_block_clear"]))
                except IKError:
                    res.append("IK×")
            if margins:
                m = np.min(np.array(margins), axis=0)
                print(f"{xc:5.0f} {D:3.0f} | {'  '.join(f'{r:5s}' for r in res)} | "
                      f"{m[0]:8.1f}°   {m[1]:9.1f} mm   {m[2]:9.1f} mm")
            else:
                print(f"{xc:5.0f} {D:3.0f} | {'  '.join(f'{r:5s}' for r in res)} |")
            rows.append((xc, D, res))
    WORK_CENTER_XY, OUTER_OFFSET = saved
    return rows


# ============================================================
# 13. 主流程
# ============================================================

def main():
    global WORK_CENTER_XY, OUTER_OFFSET, GAP, PRESS_DEPTH, LANE_MARGIN
    ap = argparse.ArgumentParser(description="第七周附加实验：桌面推块")
    ap.add_argument("--mode", choices=("plan", "simulate", "hardware", "calib", "scan"), default="plan")
    ap.add_argument("--task", type=int, choices=(1, 2, 3), default=1)
    ap.add_argument("--port", default=SERIAL_PORT)
    ap.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"), help="方阵中心（基座系 mm）")
    ap.add_argument("--outer", type=float, help="外围偏距 D（mm）")
    ap.add_argument("--gap", type=float, help="内围名义间隙 g（mm）")
    ap.add_argument("--lane", type=float, help="绕行通道额外间距 m（mm）")
    ap.add_argument("--press", type=float, help="下压量 δ（mm）")
    ap.add_argument("--calib-block", choices=BLOCK_NAMES, default="a")
    ap.add_argument("--step", action="store_true", help="实机每块下降前回车确认（调试用）")
    ap.add_argument("--only", nargs="+", choices=BLOCK_NAMES,
                    help="只执行这些木块（单块验证绕行）；其余木块按提示手工摆放")
    ap.add_argument("--viewer", choices=("2d", "rtb", "none"), default="2d")
    ap.add_argument("--speed", type=float, default=3.0, help="仿真回放倍速")
    args = ap.parse_args()

    if args.center:
        WORK_CENTER_XY = tuple(args.center)
    if args.outer:
        OUTER_OFFSET = args.outer
    if args.lane is not None:
        LANE_MARGIN = args.lane
    if args.gap is not None:
        GAP = args.gap
    if args.press is not None:
        PRESS_DEPTH = args.press
    if OUTER_OFFSET < min_outer_offset() - 1e-6:
        print(f"提示：外围偏距 D={OUTER_OFFSET:.1f} mm 小于 {min_outer_offset():.1f} mm，"
              "选做二中绕行木块会贴近外围 d；建议用 --outer 加大。")

    if args.mode == "scan":
        run_scan()
        return

    if args.mode == "calib":
        segs, back, b = build_calib_segments(args.calib_block)
        p = b["pre_touch"]
        print(f"标定：把木块 {args.calib_block} 中心放在 {np.round(b['route_base'][0][:2], 1)} mm，"
              f"夹爪闭合后接触点将停在块顶上方 {PRE_TOUCH_HEIGHT:.1f} mm（模型）。")
        if input("撤手后输入 YES 开始：").strip().upper() != "YES":
            return
        ser = init_serial(args.port)
        try:
            for s in segs:
                stream(ser, s.q)
            print(f"已悬停。模型接触点 {np.round(p, 1)}，倾角 {b['tilt'].pitch(p):.1f}°。")
            print("请测量：①夹指最低点到块顶的实际间隙 e_meas；②最低点相对块顶中心的水平偏差。")
            print(f"  若 e_meas < {PRE_TOUCH_HEIGHT:.1f}，说明接触点比模型低 {PRE_TOUCH_HEIGHT:.1f}-e_meas，"
                  "把差值加到 CONTACT_Z_COMP；水平偏差写入 BLOCK_XY_COMP 或 CONTACT_OFFSET_TOOL。")
            input("测量完成后回车返回零位：")
            for s in back:
                stream(ser, s.q)
        finally:
            ser.close()
        return

    plan = build_plan(args.task, only=args.only)
    raise_now = args.mode == "hardware"
    check_plan(plan, raise_on_fail=raise_now)
    print(summarize(plan))
    if args.only:
        print("单块验证的起始摆放（基座系 XY，mm）：")
        for k, xy in plan.initial_xy.items():
            state = "执行" if k in plan.order else (
                "已在内围目标" if np.allclose(xy, plan.blocks[k]["route_base"][-1][:2]) else "外围起点")
            print(f"  {k}: {np.round(xy, 1)}  [{state}]")
    rtb_err = cross_check_rtb(plan)
    if rtb_err is not None:
        print(f"roboticstoolbox 正解复核：最大差异 {rtb_err:.2e} mm")
    if plan.checks["issues"]:
        print("未通过项：\n  " + "\n  ".join(plan.checks["issues"][:20]))

    out_dir = OUTPUT_DIR
    csv_path, json_path = export_plan(plan, out_dir)
    fig_path = out_dir / f"{plan_tag(plan)}_overview.png"
    save_overview_figure(plan, fig_path)
    print(f"已导出：{csv_path.name}、{json_path.name}、{fig_path.name}（目录 {out_dir}）")

    if args.mode == "simulate":
        if args.viewer == "2d":
            import matplotlib
            try:
                matplotlib.use("TkAgg")
            except Exception:
                pass
            animate_plan(plan, speed=args.speed)
        elif args.viewer == "rtb":
            animate_rtb(plan)
        return

    if args.mode == "hardware":
        print("\n实机运行：启动前摆好四块并撤手；运动中不得用手干预。Ctrl+C 立即停止下发。")
        if input("输入 YES 开始控制真机：").strip().upper() != "YES":
            print("已取消。")
            return
        ser = init_serial(args.port)
        try:
            elapsed = execute_plan(ser, plan, step_mode=args.step)
            print(f"完成，实际用时 {elapsed:.1f} s。请记录编号排列、到位数量和最大间隙。")
        except KeyboardInterrupt:
            print("\n已中止：停止下发，舵机保持最后一帧目标。请确认安全后手动复位。")
        finally:
            ser.close()
            print("串口已关闭。")


if __name__ == "__main__":
    main()
