"""第六周实验三附加题（重构版）：多层叠塔 + 逐层拆塔。

为什么要重构
------------
week6/task3_new.py 用“近水平夹爪”（J2+J3+J4≈-90°±10°）抓取。按实测 D-H
（夹爪工具长 172.7 mm），在 r≈180~225 mm 处这一姿态只有 z≥250 mm 才可达，
桌面上 z≈67 mm 的木块与 1~6 层塔都求不出限位内逆解。另外它在塔附近用关节
插值做大范围移动、拿着木块从侧面靠近塔，存在扫倒塔的风险。

本程序的物理设计
----------------
1. 倾斜夹持：夹爪轴线朝下倾斜（俯仰角 pitch，0°=水平朝外，-90°=竖直朝下）。
   两指沿侧向张合，夹住木块两个侧面；绕侧向轴俯仰不改变指面与侧面的贴合。
2. 俯仰一致：同一块木块从夹起到放下，夹爪俯仰必须相同，否则木块在手中被
   转歪、放到塔上会倾斜。受 J3/J4 限位约束，每层在塔位只有一个窄俯仰窗口，
   因此程序为每层计算一个取料点，使它与该层塔位的俯仰窗口重叠。两种供料方式：
   - 人工递料（--feed hand，默认，可到 11 层）：每层一个“垫块高度 + 水平位置”的
     递料点。机械臂张爪停在递料点上方后静止等待，人把垫块和木块放到两指正下方，
     撤手回车，机械臂再下降夹取；拆塔时反过来。垫块定住木块高度（手持悬空会让
     夹持高度有几毫米随机误差，放到高塔上会压塔或掉落），水平位置随夹爪摆放。
     高层需要更平的俯仰，桌面上够不着，所以 L4、L6~L8 用 30~120 mm 垫块。
   - 桌面预摆（--feed table，≤5 层）：所有木块开机前摆好，叠塔与拆塔全自动。
3. 只在高处横移：在塔和供料位附近只做竖直进出；横移在“越障高度”进行，
   高于当前塔顶与所有木块。横移途中俯仰可以变化（木块暂时倾斜），落下前
   恢复到夹取时的俯仰。
4. 指尖不越过木块底面：闭合的夹指若伸到被夹木块底面以下，放塔时会刮到下层
   木块、拆塔时会把下层一起夹住。程序按夹指几何检查并保留余量。
5. 放置留缝、缓慢开合：放置时木块底面停在塔顶上方 RELEASE_GAP，静置后慢开
   夹爪，再慢速竖直撤离；夹取时先静置再慢合，起步慢速抬升。
6. 中止时绝不松爪：Ctrl+C 只停止下发，舵机保持最后目标，避免木块掉落砸塔。

运行方式（先离线，后实机）::

    python week6/task3_stack.py --mode scan                      # 各层俯仰窗口、递料点与可叠层数
    python week6/task3_stack.py --mode plan --layers 8           # 规划+检查，打印每层递料点和垫块高度
    python week6/task3_stack.py --mode plan --layers 3 --feed table   # 桌面预摆、全自动
    python week6/task3_stack.py --mode simulate --layers 3       # 三维动画（木块随夹爪移动）
    python week6/task3_stack.py --mode calib --port COM5         # 实机：张爪停在 L1 供料块处供核对
    python week6/task3_stack.py --mode hardware --layers 3 --port COM5 [--phase stack|unstack|both] [--step]

坐标系与 week5~week7 的 D-H 模型一致（基座系，mm）；桌面 z = TABLE_Z。
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
WINDOW_MARGIN_DEG = 4.0          # 计算塔位/供料位俯仰窗口时要求的余量
JOINT_SPEED_LIMIT_DEG_S = 60.0

SERIAL_PORT = "COM5"
BAUDRATE = 115200
SEND_PERIOD_S = 0.02             # 50 Hz 流式下发

# 夹爪 J6（week6：0° 张开、-50° 夹紧；teleop：-40° 闭合）。
GRIP_OPEN_DEG = 0.0
GRIP_CLOSE_DEG = -40.0
GRIP_TIME_S = 1.0                # 开合用时（慢速，避免把木块或塔带偏）
SETTLE_S = 0.4                   # 夹取/放置前后的静置时间（舵机余振衰减）


# ============================================================
# 2. 场景几何（需按本组实物标定）
# ============================================================

TABLE_Z = 52.0                   # 桌面在基座系中的高度（week6 实机标定）
BLOCK_SIZE = 30.0                # h
TOWER_XY = (200.0, -45.0)        # 塔位（r≈205 mm，俯仰窗口比 r≈184 宽 2~5 倍）
SUPPLY_AZIMUTHS_DEG = (30.0, 52.0)   # 供料位所在方位（远离塔，交替使用避免互相重叠）
SUPPLY_SPACING = 12.0            # 相邻供料块之间的最小间隙
MAX_LAYERS = 11                  # 极限：12 层能放上去但撤不出来——放完后夹指仍跨在第 12 块两侧，
                                 # 须把夹取点抬到约 426 mm 才能横移离开，而塔位上方手臂最高约 424 mm
BASE_MAX_LAYERS = 8              # ≤8 层时沿用原有全部参数（递料点、垫块、俯仰网格均不变）
HIGH_MAX_LAYERS = 10             # 9~10 层的参数见 HIGH_*；11 层以上另见 EXTREME_*（≤10 层不受影响）

# 供料方式：
#   "hand"  人工递料（默认）：每层一个递料点（垫块高度 + 水平位置），机械臂张爪停在
#           递料点上方，人把“垫块+木块”放到两指正下方，撤手回车后机械臂下降夹取。
#           垫块定住木块高度，水平位置随夹爪摆放，因此 7~10 层也能匹配俯仰
#           （9、10 层需 210~240 mm 的稳固台子，见 HIGH_FEED_*）。
#   "table" 预摆供料：所有木块开机前摆在桌面供料位，全自动（最多约 5 层）。
FEED_MODE = "hand"
FEED_SUPPORT_HEIGHTS = (0.0, 30.0, 60.0, 90.0, 120.0)   # 可用垫块高度（0=直接放桌面）
FEED_AZIMUTH_DEG = 40.0          # 递料点所在方位（远离塔，人手不必靠近塔）
FEED_RADIUS_RANGE = (180.0, 320.0)
FEED_MIN_OVERLAP = 3.0           # 递料点与塔位俯仰窗口的最小重叠（deg）

# 第 9、10 层专用（仅当层数 > BASE_MAX_LAYERS 时启用；L1~L8 的递料点不受影响）：
# 塔高 270~300 mm 时放置需夹爪接近水平甚至略上仰，桌面附近够不到这样的俯仰，
# 必须把木块垫到 210~240 mm 高；递料点也要离基座近一些（远处手臂举不到越障高度）。
HIGH_FEED_SUPPORT_HEIGHTS = (150.0, 180.0, 210.0, 240.0)
HIGH_FEED_RADIUS_RANGE = (150.0, 330.0)

# 第 11 层以上专用（仅当层数 > HIGH_MAX_LAYERS 时启用；L1~L10 完全不变）：
# - 放置时夹爪需上仰 30° 以上，递料点要离基座更近（那里手臂举得更高），木块垫到 270~360 mm；
# - 上仰时指根随之下沉，夹取点相对木块中心上移 EXTREME_GRASP_Z_OFFSET，保证夹指不低于木块底面
#   （否则放塔会刮到下层、在递料台上会刮到台面）。
EXTREME_FEED_SUPPORT_HEIGHTS = (240.0, 270.0, 300.0, 330.0, 360.0)
EXTREME_FEED_RADIUS_RANGE = (130.0, 300.0)
EXTREME_FEED_PREFER_R = 200.0
EXTREME_GRASP_Z_OFFSET = 4.0
EXTREME_PITCH_FROM_LOW = 3.0     # 高层放置俯仰取可行区间下端 +3°：上仰越小，手腕越高，越不易碰塔
EXTREME_RETREAT_R = 150.0        # 11 层以上回零前先撤到该半径（递料方位）上方，避免扫过塔顶
EXTREME_TRANSIT_CLEAR = 15.0     # 11 层以上越障余量（≤10 层仍为 25 mm），手臂高度已接近极限

# 竖直进出与放置
APPROACH = 12.0                  # 在供料位/塔位上方以夹取俯仰竖直进出的距离
RELEASE_GAP = 1.0                # 放置时木块底面高出支撑面的距离
SLOW_ZONE = 5.0                  # 最后/最初这段距离用慢速
TRANSIT_CLEAR_HELD = 25.0        # 横移时木块底面高出最高障碍物的距离
TRANSIT_CLEAR_EMPTY = 25.0       # 空爪横移时夹指最低点高出最高障碍物的距离

# 速度
TRANSIT_SPEED = 40.0             # mm/s
VERTICAL_SPEED = 20.0
SLOW_SPEED = 5.0
MOVEJ_SPEED_DEG_S = 30.0
MOVEJ_MIN_TIME = 1.5

# 补偿（无反馈开环，按实测填写）
GRASP_Z_OFFSET = 0.0             # 夹取点相对木块中心的高度（+ 为偏上）
HOLD_SAG_COMP = 0.0              # 持块时末端下垂补偿（+ 为抬高指令）
TOWER_Z_COMP = {}                # {层号: mm} 逐层放置高度补偿
TOWER_XY_COMP = {}               # {层号: (dx, dy)} 逐层放置水平补偿
SUPPLY_XY_COMP = {}              # {层号: (dx, dy)} 逐个供料位夹取水平补偿

# 夹爪几何（工具坐标系：z 沿夹爪轴线朝外，y 为张合方向，原点为 D-H 末端=夹取点）
FINGER_TIP = 8.0                 # 指尖超出夹取点的长度（沿 z）
FINGER_BACK = 15.0               # 指面向手掌方向的长度（沿 -z）
FINGER_HALF_W = 6.0              # 指面在 x 方向的半宽
FINGER_THICK = 4.0               # 指厚
OPEN_HALF_GAP = 22.0             # 张开时两指内侧到中心的距离（须 > h/2 + 容差）
PALM_DEPTH = 25.0                # 手掌（夹爪本体）距夹取点的深度（沿 -z）
FINGER_BOTTOM_MARGIN = 2.0       # 夹指最低点至少高出被夹木块底面

# 碰撞检查
ARM_RADIUS = 15.0
ARM_CLEARANCE = 4.0
HELD_CLEARANCE = 0.5             # 持块与其他木块/桌面的最小间隙（放置时为 RELEASE_GAP）
MAX_FK_ERROR = 0.5

OUTPUT_DIR = Path(__file__).resolve().parent / "stack_output"
PITCH_GRID_BASE = np.arange(-85.0, 20.0, 0.5)   # ≤8 层：高塔越障时需要接近水平甚至略上仰
PITCH_GRID_HIGH = np.arange(-85.0, 60.0, 0.5)   # 9~10 层：第 10 层放置约 +14°，越障时还需更大上仰
PITCH_GRID_EXTREME = np.arange(-85.0, 80.0, 0.5)   # 11 层以上：第 12 层放置需上仰约 30°~60°
PITCH_GRID = PITCH_GRID_BASE


def use_pitch_grid(layers):
    """按层数选择俯仰搜索网格；≤8 层保持原网格，保证原有规划结果逐帧不变。"""
    global PITCH_GRID
    if layers > HIGH_MAX_LAYERS:
        PITCH_GRID = PITCH_GRID_EXTREME
    elif layers > BASE_MAX_LAYERS:
        PITCH_GRID = PITCH_GRID_HIGH
    else:
        PITCH_GRID = PITCH_GRID_BASE


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
                        for p in DH_PARAMS], name="Week6_Stack_Arm")


# ============================================================
# 4. 固定俯仰解析逆解
# ============================================================

class IKError(RuntimeError):
    pass


_W2T = {}


def wrist_to_tip(pitch):
    """q1=0 时，J4 轴心到夹取点的向量（径向、侧向、竖直），只与俯仰有关。"""
    key = round(float(pitch), 6)
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
# 5. 场景：塔层、供料位（每层俯仰匹配）
# ============================================================

def layer_center(k):
    """第 k 层（从 1 起）木块中心的名义位置。"""
    x, y = TOWER_XY
    dx, dy = TOWER_XY_COMP.get(k, (0.0, 0.0))
    return np.array([x + dx, y + dy, TABLE_Z + BLOCK_SIZE * (k - 0.5)])


def tower_yaw():
    return float(np.arctan2(TOWER_XY[1], TOWER_XY[0]))


def block_index(block):
    """木块编号：接受整数 k 或名字 "Lk"。"""
    if block is None:
        return None
    return int(block[1:]) if isinstance(block, str) else int(block)


def grasp_offset(block=None):
    """夹取点相对木块中心的高度：≤10 层的木块为 GRASP_Z_OFFSET；第 11 层以上的木块再上移。"""
    k = block_index(block)
    extra = EXTREME_GRASP_Z_OFFSET if (k is not None and k > HIGH_MAX_LAYERS) else 0.0
    return GRASP_Z_OFFSET + extra


def grasp_point_from_center(center, k_layer=None, block=None):
    z = (center[2] + grasp_offset(block if block is not None else k_layer)
         + (TOWER_Z_COMP.get(k_layer, 0.0) if k_layer else 0.0))
    return np.array([center[0], center[1], z])


def plan_supply(layers):
    """为每层选供料位：半径使供料位与塔位俯仰窗口重叠最宽，方位交替避免重叠。"""
    if layers > MAX_LAYERS:
        raise IKError(f"层数 {layers} 超过设计上限 {MAX_LAYERS}")
    z_table_grasp = TABLE_Z + BLOCK_SIZE / 2 + GRASP_Z_OFFSET
    supply = {}
    placed = []
    for k in range(1, layers + 1):
        g = grasp_point_from_center(layer_center(k), k)
        place_low = g + np.array([0, 0, RELEASE_GAP])
        W = pitch_window(place_low, place_low + np.array([0, 0, APPROACH]))
        if W is None:
            raise IKError(f"第 {k} 层塔位在任何俯仰下都不可达（塔太高或塔位不合适）")
        best = None
        for az in SUPPLY_AZIMUTHS_DEG:
            u = np.array([np.cos(np.deg2rad(az)), np.sin(np.deg2rad(az))])
            for r in np.arange(160.0, 362.0, 2.0):
                xy = r * u
                if any(np.linalg.norm(xy - q) < BLOCK_SIZE * np.sqrt(2) + SUPPLY_SPACING for q in placed):
                    continue
                if np.linalg.norm(xy - np.asarray(TOWER_XY)) < 80.0:
                    continue
                p0 = np.array([xy[0], xy[1], z_table_grasp])
                S = pitch_window(p0, p0 + np.array([0, 0, APPROACH + RELEASE_GAP]))
                if S is None:
                    continue
                lo, hi = max(W[0], S[0]), min(W[1], S[1])
                if hi - lo < 1.0:
                    continue
                score = (hi - lo) - 0.01 * r      # 重叠越宽越好；同等时取近处
                if best is None or score > best[0]:
                    best = (score, xy, (lo + hi) / 2, (lo, hi), W, S, az)
        if best is None:
            raise IKError(f"第 {k} 层找不到俯仰匹配的供料位（塔位窗口 {W}）")
        _, xy, pitch, overlap, W, S, az = best
        dx, dy = SUPPLY_XY_COMP.get(k, (0.0, 0.0))
        center = np.array([xy[0] + dx, xy[1] + dy, TABLE_Z + BLOCK_SIZE / 2])
        supply[k] = {"center": center, "yaw": float(np.arctan2(xy[1], xy[0])), "pitch": float(pitch),
                     "overlap": overlap, "tower_window": W, "supply_window": S, "azimuth": az}
        placed.append(xy)
    return supply


def plan_feed(layers):
    """人工递料：为每层选“垫块高度 + 半径”，使递料点俯仰窗口覆盖该层塔位窗口。

    在重叠宽度、垫块高度、手臂伸展程度之间折中（见评分）。
    递料点之间不会同时存在，因此都放在同一方位 FEED_AZIMUTH_DEG 上。
    """
    if layers > MAX_LAYERS:
        raise IKError(f"层数 {layers} 超过设计上限 {MAX_LAYERS}")
    az = np.deg2rad(FEED_AZIMUTH_DEG)
    cache = {}

    def feed_window(H, r, k):
        off = grasp_offset(k)
        if (H, r, off) not in cache:
            z = TABLE_Z + H + BLOCK_SIZE / 2 + off
            p0 = np.array([r, 0.0, z])
            cache[(H, r, off)] = pitch_window(p0, p0 + np.array([0, 0, APPROACH + RELEASE_GAP]))
        return cache[(H, r, off)]

    feed = {}
    for k in range(1, layers + 1):
        g = grasp_point_from_center(layer_center(k), k) + np.array([0, 0, RELEASE_GAP])
        W = pitch_window(g, g + np.array([0, 0, APPROACH]))
        if W is None:
            raise IKError(f"第 {k} 层塔位在任何俯仰下都不可达")
        best = None
        for need in (FEED_MIN_OVERLAP, 1.0):
            extreme = k > HIGH_MAX_LAYERS
            high = k > BASE_MAX_LAYERS
            if extreme:
                heights, (r_lo, r_hi) = EXTREME_FEED_SUPPORT_HEIGHTS, EXTREME_FEED_RADIUS_RANGE
            elif high:
                heights, (r_lo, r_hi) = HIGH_FEED_SUPPORT_HEIGHTS, HIGH_FEED_RADIUS_RANGE
            else:
                heights, (r_lo, r_hi) = FEED_SUPPORT_HEIGHTS, FEED_RADIUS_RANGE
            for H in sorted(heights):
                for r in np.arange(r_lo, r_hi + 0.1, 4.0):
                    S = feed_window(H, r, k)
                    if S is None:
                        continue
                    lo, hi = max(W[0], S[0]), min(W[1], S[1])
                    if hi - lo < need:
                        continue
                    # 重叠越宽越好（离限位越远）；垫块越低越稳；半径靠近 250 mm
                    # 时手臂不过分伸展、下垂小。三者折中，宁可多垫一块也不贴着臂展边缘。
                    score = (hi - lo) - 0.03 * H - 0.05 * abs(r - 250.0)
                    if extreme:   # 11 层以上：优先最小的放置上仰角，其次递料点靠近基座
                        if hi - lo < max(need, EXTREME_PITCH_FROM_LOW):
                            continue
                        score = -lo - 0.02 * abs(r - EXTREME_FEED_PREFER_R) - 0.01 * H
                    if best is None or score > best[0]:
                        best = (score, H, r, (lo, hi), S)
            if best:
                break
        if best is None:
            raise IKError(f"第 {k} 层找不到俯仰匹配的递料点（塔位窗口 {W}）；可增加更高的垫块")
        _, H, r, ov, S = best
        center = np.array([r * np.cos(az), r * np.sin(az), TABLE_Z + H + BLOCK_SIZE / 2])
        pitch = (ov[0] + ov[1]) / 2
        if k > HIGH_MAX_LAYERS:
            pitch = ov[0] + min(EXTREME_PITCH_FROM_LOW, (ov[1] - ov[0]) / 2)
        feed[k] = {"center": center, "yaw": float(az), "pitch": float(pitch),
                   "overlap": ov, "tower_window": W, "supply_window": S, "azimuth": FEED_AZIMUTH_DEG,
                   "support": float(H)}
    return feed


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
    held: str | None = None   # 手中木块名
    target: str | None = None  # 正在夹取/放置的木块名（允许夹指靠近）
    scene: dict = field(default_factory=dict)  # 本段内静止木块：名 → (中心, 偏航)
    phase: str = ""
    message: str = ""         # wait 段提示人工操作的内容


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
    """从 p_low 起以固定俯仰竖直向上，最高能到多少（不超过 z_max）。"""
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
# 7. 规划器：带木块状态的动作序列
# ============================================================

class Planner:
    def __init__(self, layers, feed_mode=None):
        use_pitch_grid(layers)
        self.layers = layers
        self.feed_mode = feed_mode or FEED_MODE
        self.supply = plan_feed(layers) if self.feed_mode == "hand" else plan_supply(layers)
        self.segments: list[Segment] = []
        self.q = np.zeros(5)
        self.grip = GRIP_OPEN_DEG
        self.held = None
        self.held_rel = {}           # 木块 → 相对夹爪的位姿（4x4）
        # 预摆模式下所有木块开机就在供料位；递料模式下木块递到夹爪下才出现
        self.blocks = {} if self.feed_mode == "hand" else {
            f"L{k}": (self.supply[k]["center"].copy(), self.supply[k]["yaw"]) for k in range(1, layers + 1)}
        self.tower: list[str] = []
        self.phase = ""
        self.under = None            # 夹指正下方/之间的木块（刚夹起或刚放下的那块）

    # ---------- 状态 ----------
    def scene(self):
        return {k: (c.copy(), yaw) for k, (c, yaw) in self.blocks.items() if k != self.held}

    def top_of_obstacles(self):
        tops = [c[2] + BLOCK_SIZE / 2 for k, (c, _) in self.blocks.items() if k != self.held]
        return max(tops + [TABLE_Z])

    def _push(self, seg):
        seg.held, seg.scene, seg.phase = self.held, self.scene(), self.phase
        if self.segments:
            seg.q[0] = self.segments[-1].q[-1]   # 段首与上一段末严格衔接
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
            if duration > 8.0 * base_duration:   # 放慢 8 倍仍超速：是分支跳变，放慢无效
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

    def wait(self, label, message):
        """静止等待人工操作（递料/取料）；规划与仿真中不占时间，实机执行时回车继续。"""
        q = add_grip_col(np.repeat(self.q[None, :], 2, axis=0), self.grip)
        # 静止时夹指可能仍跨在刚放下/待夹取的木块两侧（之后只做竖直运动）
        self._push(Segment(label, "wait", q, 0.0, message=message, target=self.under))

    # ---------- 组合动作 ----------
    def transit_z(self):
        top = self.top_of_obstacles()
        extreme = self.layers > HIGH_MAX_LAYERS
        clear_held = EXTREME_TRANSIT_CLEAR if extreme else TRANSIT_CLEAR_HELD
        clear_empty = EXTREME_TRANSIT_CLEAR if extreme else TRANSIT_CLEAR_EMPTY
        if self.held:
            return top + clear_held + BLOCK_SIZE / 2 + abs(GRASP_Z_OFFSET) + 6.0
        return top + clear_empty + FINGER_TIP + FINGER_HALF_W

    def travel_to_hover(self, label, hover, hover_pitch, target):
        """竖直上升 → 高处横移 → 竖直下降到目标上方。

        上升和下降都分两段：贴近木块/塔的一段姿态不变、纯竖直平动（夹爪不转动），
        尽量拉长到该俯仰可达范围的上沿；俯仰变化只发生在其上方远离物体处
        （此时夹取点仍沿竖直线运动）和高处横移中。
        """
        here = grip_point(self.q)
        zt = max(self.transit_z(), hover[2], here[2])
        pitch_here = tool_pitch(self.q)
        z_up_c = const_pitch_top(here, pitch_here, zt)       # 上升：姿态不变段的顶端
        z_dn_c = const_pitch_top(hover, hover_pitch, zt)     # 下降：姿态不变段的顶端
        up_c = np.array([here[0], here[1], z_up_c])
        up = np.array([here[0], here[1], zt])
        above = np.array([hover[0], hover[1], zt])
        dn_c = np.array([hover[0], hover[1], z_dn_c])
        # 选择越障高度上的两个俯仰，使转角段与横移段（含俯仰线性过渡）都可行
        cand_a = sorted(PITCH_GRID, key=lambda x: abs(x - pitch_here))
        cand_b = sorted(PITCH_GRID, key=lambda x: abs(x - hover_pitch))
        cand_a = [x for x in cand_a if best_margin(up, x) >= JOINT_MARGIN_DEG + 1][:40]
        cand_b = [x for x in cand_b if best_margin(above, x) >= JOINT_MARGIN_DEG + 1][:40]
        def candidates():
            for a in cand_a:
                if not line_feasible(up_c, up, pitch_here, a):
                    continue
                for b in cand_b:
                    if line_feasible(up, above, a, b, n=9) and line_feasible(above, dn_c, b, hover_pitch):
                        yield a, b

        # 逐个候选尝试生成整条越障路线；某段出现逆解分支跳变时回滚并换下一组俯仰。
        # 原本能成功的路线第一个候选即成功，结果与旧版完全相同。
        saved = (len(self.segments), self.q.copy(), self.grip, self.under)
        last_err, tried = None, 0
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
                break
            except IKError as e:
                last_err = e
                del self.segments[saved[0]:]
                self.q, self.grip, self.under = saved[1].copy(), saved[2], saved[3]
                if tried >= 30:
                    break
        else:
            if tried == 0 and not self._transit_movej(label, here, pitch_here, up_c, z_up_c, up, zt,
                                                      above, hover, hover_pitch, target):
                raise IKError(f"{label}: 找不到可行的越障路线（越障高度 {zt:.0f} mm）")
        if last_err is not None and len(self.segments) == saved[0]:
            # 笛卡尔横移都需要中途切换肘部构型：改在高处用关节空间运动切换（仅 11 层以上会用到）
            if not self._transit_movej(label, here, pitch_here, up_c, z_up_c, up, zt,
                                       above, hover, hover_pitch, target):
                raise IKError(f"{label}: 越障路线均出现逆解分支跳变（{last_err}）")
        self.under = target          # 已离开原来那块，下方变为新的目标（可能为空）

    def _transit_movej(self, label, here, pitch_here, up_c, z_up_c, up, zt, above, hover, hover_pitch, target):
        """备选越障：竖直升到越障高度 → 高处关节空间运动（允许切换肘部构型）→ 竖直下降。

        只在所有笛卡尔越障路线都失败时调用；关节空间段的末端路径由离线碰撞检查覆盖。
        """
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

    def pick(self, name, center, yaw, pitch, label, travel=True):
        """张爪竖直下降 → 静置 → 慢合 → 静置 → 慢起。travel=False 表示已在夹取点上方。"""
        g = grasp_point_from_center(center, block=name)
        hover = g + np.array([0, 0, APPROACH])
        if travel:
            self.travel_to_hover(f"{label}/去夹取", hover, pitch, target=name)
        self.vertical(f"{label}/下降", g[2] + SLOW_ZONE, VERTICAL_SPEED, target=name)
        self.vertical(f"{label}/慢降", g[2], SLOW_SPEED, target=name)
        self.dwell(f"{label}/静置", SETTLE_S, target=name)
        self.gripper(f"{label}/夹紧", GRIP_CLOSE_DEG, target=name)
        self.dwell(f"{label}/保持", SETTLE_S, target=name)
        # 记录木块相对夹爪的位姿
        B = np.eye(4)
        B[:3, :3] = rot_z(yaw)
        B[:3, 3] = center
        self.held_rel[name] = np.linalg.inv(tool_pose(self.q)) @ B
        self.held = name
        self.blocks[name] = (center.copy(), yaw)
        sag = HOLD_SAG_COMP
        self.vertical(f"{label}/慢起", g[2] + SLOW_ZONE + sag, SLOW_SPEED, target=name)
        self.vertical(f"{label}/抬起", g[2] + APPROACH + sag, VERTICAL_SPEED, target=name)
        self.under = name

    def place_hover(self, center, k_layer=None, block=None):
        g = grasp_point_from_center(center, k_layer, block) + np.array([0, 0, RELEASE_GAP + HOLD_SAG_COMP])
        return g, g + np.array([0, 0, APPROACH])

    def place(self, name, center, pitch, label, k_layer=None, travel=True):
        """竖直下降到“底面高出支撑面 RELEASE_GAP” → 静置 → 慢开 → 静置 → 慢起。"""
        g, hover = self.place_hover(center, k_layer, block=name)
        if travel:
            self.travel_to_hover(f"{label}/去放置", hover, pitch, target=name)
        self.vertical(f"{label}/下降", g[2] + SLOW_ZONE, VERTICAL_SPEED, target=name)
        self.vertical(f"{label}/慢降", g[2], SLOW_SPEED, target=name)
        self.dwell(f"{label}/静置", SETTLE_S, target=name)
        # 松开前记录木块实际位姿（偏航随 J1 转动），松开后它成为静止障碍物
        _, Tb = held_corners(self.held_rel[name], self.q)
        yaw = float(np.arctan2(Tb[1, 0], Tb[0, 0]))
        self.gripper(f"{label}/松开", GRIP_OPEN_DEG, target=name)
        self.dwell(f"{label}/保持", SETTLE_S, target=name)
        self.held = None
        self.blocks[name] = (np.asarray(center, float).copy(), yaw)
        self.vertical(f"{label}/慢起", g[2] + SLOW_ZONE, SLOW_SPEED, target=name)
        self.vertical(f"{label}/撤离", g[2] + APPROACH, VERTICAL_SPEED, target=name)
        self.under = name

    def go_home(self, label):
        """先竖直升到越障高度，再关节空间回零（零位为竖直立正，最高最安全）。"""
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
        if (self.layers > HIGH_MAX_LAYERS
                and self.top_of_obstacles() > TABLE_Z + BLOCK_SIZE * HIGH_MAX_LAYERS + 1e-6):
            # 塔高超过 10 层时：塔顶已接近手臂最高点，直接关节回零会扫过塔顶。先在高处横移到
            # 塔外侧（递料方位、靠近基座），再回零。≤10 层不执行此步。
            az = np.deg2rad(FEED_AZIMUTH_DEG)
            p_out = np.array([EXTREME_RETREAT_R * np.cos(az), EXTREME_RETREAT_R * np.sin(az),
                              max(self.transit_z(), grip_point(self.q)[2])])
            if np.linalg.norm(grip_point(self.q)[:2] - p_out[:2]) > 1.0:
                self.travel_to_hover(f"{label}/离开塔顶", p_out, best_pitch(p_out), target=None)
        self.movej(f"{label}/回零", np.zeros(5), duration=3.0)

    def start_from_zero(self, first_hover, pitch):
        """零位 → 越障高度上方的准备姿态（关节空间，远离地面物体）。"""
        zt = self.transit_z()
        p = np.array([first_hover[0], first_hover[1], zt])
        q_ready = solve_ik(p, best_pitch(p))
        self.movej("零位→准备", q_ready)

    # ---------- 任务 ----------
    def stack_phase(self):
        self.phase = "stack"
        for k in range(1, self.layers + 1):
            name = f"L{k}"
            sp = self.supply[k]
            if self.feed_mode == "hand":
                center, yaw = sp["center"].copy(), sp["yaw"]
                hover = grasp_point_from_center(center, block=name) + np.array([0, 0, APPROACH])
                if not self.segments:
                    self.start_from_zero(center, sp["pitch"])
                self.travel_to_hover(f"叠{k}/去递料点", hover, sp["pitch"], target=None)
                self.wait(f"叠{k}/等待递料", feed_message(k, sp, "stack"))
                self.blocks[name] = (center, yaw)
                self.pick(name, center, yaw, sp["pitch"], f"叠{k}/取", travel=False)
            else:
                center, yaw = self.blocks[name]
                if not self.segments:
                    self.start_from_zero(center, sp["pitch"])
                self.pick(name, center, yaw, sp["pitch"], f"叠{k}/取")
            self.place(name, layer_center(k), sp["pitch"], f"叠{k}/放", k_layer=k)
            self.tower.append(name)
        self.go_home("叠塔完成")

    def unstack_phase(self):
        self.phase = "unstack"
        if not self.tower:          # 只拆塔：假设塔已按本程序叠好
            for k in range(1, self.layers + 1):
                self.blocks[f"L{k}"] = (layer_center(k), tower_yaw())
            self.tower = [f"L{k}" for k in range(1, self.layers + 1)]
        for k in range(self.layers, 0, -1):
            name = f"L{k}"
            sp = self.supply[k]
            center, yaw = self.blocks[name]
            if not self.segments or np.allclose(self.q, 0):
                self.start_from_zero(center, sp["pitch"])
            self.pick(name, center, yaw, sp["pitch"], f"拆{k}/取")
            if self.feed_mode == "hand":
                _, hover = self.place_hover(sp["center"], block=name)
                self.travel_to_hover(f"拆{k}/去递料点", hover, sp["pitch"], target=name)
                self.wait(f"拆{k}/等待垫块", feed_message(k, sp, "unstack"))
                self.place(name, sp["center"], sp["pitch"], f"拆{k}/放", travel=False)
                self.wait(f"拆{k}/等待取走", f"请取走第 {k} 块木块和垫块，撤手后回车。")
                del self.blocks[name]
            else:
                self.place(name, sp["center"], sp["pitch"], f"拆{k}/放")
            self.tower.pop()
        self.go_home("拆塔完成")


def feed_message(k, sp, phase):
    c, H = sp["center"], sp.get("support", 0.0)
    pad = ("直接放在桌面上" if H < 0.5 else f"放在高 {H:.0f} mm 的垫块上" if H <= 120.5
           else f"放在高 {H:.0f} mm 的稳固台子上（底面要宽，不要用木块叠）")
    where = f"夹爪正下方（约 ({c[0]:.0f}, {c[1]:.0f}) mm）"
    if phase == "stack":
        return (f"请把第 {k} 块木块{pad}，置于{where}、两指中间，一边正对机械臂基座；"
                "撤手后回车，机械臂将下降夹取。")
    return f"请在{where}放好{('（无需垫块，桌面即可）' if H < 0.5 else f'高 {H:.0f} mm 的垫块')}；撤手后回车，木块将下降放下。"


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def build_plan(layers, phase="both", feed_mode=None):
    pl = Planner(layers, feed_mode)
    if phase in ("stack", "both"):
        pl.stack_phase()
    if phase in ("unstack", "both"):
        pl.unstack_phase()
    return pl


# ============================================================
# 8. 离线检查
# ============================================================

def block_corners(center, yaw, h=BLOCK_SIZE):
    R = rot_z(yaw)
    d = h / 2
    pts = [R @ np.array([sx * d, sy * d, sz * d]) + center
           for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    return np.array(pts)


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
    """两个立方体（任意姿态）的分离轴检测：返回最小分离距离，负值为嵌入深度。"""
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


def block_overlap(c1, y1, c2, y2, gap, R1=None):
    """两块木块是否相交：分离距离 < gap 即判为相交（gap<0 表示允许贴合）。"""
    R1 = rot_z(y1) if R1 is None else R1
    return obb_penetration(c1, R1, c2, rot_z(y2)) < gap


def arm_samples(q, step=10.0):
    """连杆链采样点及其到夹取点的链长（不含底座立柱）。"""
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
    """两片夹指的采样点（工具系 y 为张合方向）。张开/闭合按 J6 线性插值。"""
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
          "min_arm_clear": np.inf, "min_held_clear": np.inf, "min_finger_margin": np.inf,
          "pitch_mismatch": {}}
    held_pitch = {}
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
        # 俯仰一致性：夹紧时记录，松开时比较
        if seg.kind == "grip" and seg.target:
            if seg.q[-1, 5] == GRIP_CLOSE_DEG:
                held_pitch[seg.target] = tool_pitch(q[-1])
            elif seg.target in held_pitch:
                d = tool_pitch(q[-1]) - held_pitch[seg.target]
                st["pitch_mismatch"][f"{seg.phase}:{seg.target}"] = round(d, 3)
                if abs(d) > 0.2:
                    issue(f"{seg.label}: 放下时俯仰与夹取时相差 {d:.1f}°，木块会倾斜")
        for x in q[::stride]:
            # 连杆（含手掌）对木块与桌面
            pts, arc = arm_samples(x)
            for p, a in zip(pts, arc):
                # 手掌以内由夹指模型检查；手掌以外按连杆包络检查桌面
                if a >= PALM_DEPTH and p[2] - ARM_RADIUS - TABLE_Z < ARM_CLEARANCE:
                    issue(f"{seg.label}: 连杆点 z={p[2]:.0f} 过低（桌面）")
                    break
                for name, (c, yaw) in seg.scene.items():
                    near_tip = a < PALM_DEPTH + ARM_RADIUS
                    if near_tip and name == seg.target:
                        continue
                    exp = (FINGER_THICK if near_tip else ARM_RADIUS) + ARM_CLEARANCE
                    if point_in_block(p, c, yaw, exp):
                        d = rot_z(yaw).T @ (p - c)
                        st["min_arm_clear"] = min(st["min_arm_clear"], float(d[2] - BLOCK_SIZE / 2 - ARM_RADIUS))
                        issue(f"{seg.label}: 连杆点 {np.round(p, 0)} 碰到木块 {name}")
                        break
            # 夹指
            fp = finger_points(x, x[5])
            for p in fp:
                if p[2] < TABLE_Z + 1.0:
                    issue(f"{seg.label}: 夹指碰桌面 z={p[2]:.1f}")
                    break
                for name, (c, yaw) in seg.scene.items():
                    if name == seg.target:
                        continue
                    if point_in_block(p, c, yaw, 0.5):
                        issue(f"{seg.label}: 夹指碰到木块 {name}")
                        break
            # 夹指不得低于被夹木块底面（否则刮到/夹住下层）
            if seg.target and (seg.held == seg.target or x[5] < GRIP_OPEN_DEG - 1e-6 or seg.kind == "line"):
                c_t = None
                if seg.held == seg.target and seg.target in pl.held_rel:
                    _, Tb = held_corners(pl.held_rel[seg.target], x)
                    c_t = Tb[:3, 3]
                elif seg.target in seg.scene:
                    c_t = seg.scene[seg.target][0]
                if c_t is not None and np.linalg.norm(grip_point(x)[:2] - c_t[:2]) < BLOCK_SIZE:
                    fm = float(fp[:, 2].min() - (c_t[2] - BLOCK_SIZE / 2))
                    st["min_finger_margin"] = min(st["min_finger_margin"], fm)
                    if fm < FINGER_BOTTOM_MARGIN:
                        issue(f"{seg.label}: 夹指最低点仅高出木块底面 {fm:.1f} mm，"
                              "放塔会刮下层/拆塔会夹住下层；请增大 GRASP_Z_OFFSET")
            # 手中木块对其他木块与桌面
            if seg.held:
                corners, Tb = held_corners(pl.held_rel[seg.held], x)
                cz = corners[:, 2].min() - TABLE_Z
                st["min_held_clear"] = min(st["min_held_clear"], float(cz))
                if cz < -0.3:      # 起吊/落下瞬间允许贴合支撑面，不允许嵌入
                    issue(f"{seg.label}: 手中木块 {seg.held} 碰到桌面")
                yaw_b = float(np.arctan2(Tb[1, 0], Tb[0, 0]))
                for name, (c, yaw) in seg.scene.items():
                    sep = obb_penetration(Tb[:3, 3], Tb[:3, :3], c, rot_z(yaw))
                    st["min_held_gap"] = min(st.get("min_held_gap", np.inf), sep)
                    if sep < -0.3:       # 贴合允许（起吊/落下瞬间），嵌入不允许
                        issue(f"{seg.label}: 手中木块 {seg.held} 碰到木块 {name}")
                        break
    st["issues"] = list(issues.values())
    pl.checks = st
    if st["issues"] and raise_on_fail:
        raise RuntimeError("离线检查未通过：\n  " + "\n  ".join(st["issues"][:15]))
    return st


# ============================================================
# 9. 摘要、导出与可视化
# ============================================================

def straight_descent_stats(pl):
    """每次夹紧/松开之前，夹爪姿态不变、纯竖直下降的连续长度（mm）。"""
    out, run = {}, 0.0
    for seg in pl.segments:
        if seg.kind == "line":
            p0, p1 = grip_point(seg.q[0]), grip_point(seg.q[-1])
            vertical = np.linalg.norm(p1[:2] - p0[:2]) < 0.05 and p1[2] < p0[2]
            const = abs(tool_pitch(seg.q[0]) - tool_pitch(seg.q[-1])) < 0.05
            run = run + (p0[2] - p1[2]) if (vertical and const) else 0.0
        elif seg.kind == "grip":
            out[seg.label] = run
        elif seg.kind not in ("dwell", "wait"):
            run = 0.0
    return out


def summarize(pl):
    L = [f"\n========== 叠塔 {pl.layers} 层 + 拆塔 =========="]
    L.append(f"塔位 {TOWER_XY} mm（r={np.hypot(*TOWER_XY):.0f}），桌面 z={TABLE_Z}，木块 {BLOCK_SIZE} mm")
    if pl.feed_mode == "hand":
        L.append(f"人工递料点（方位 {FEED_AZIMUTH_DEG:.0f}°；机械臂停在上方后，按提示把垫块+木块放到两指正下方）：")
    else:
        L.append("供料位（开机前按此摆放；木块一边朝向基座中心，即正对机械臂）：")
    for k, s in pl.supply.items():
        c = s["center"]
        pad = f"垫块 {s.get('support', 0.0):3.0f} mm  " if pl.feed_mode == "hand" else ""
        L.append(f"  L{k}: {pad}({c[0]:.1f}, {c[1]:.1f}) mm  r={np.hypot(c[0], c[1]):.0f}  "
                 f"朝向 {np.rad2deg(s['yaw']):.1f}°  夹持俯仰 {s['pitch']:.1f}°  "
                 f"(塔位窗口 {s['tower_window'][0]:.1f}~{s['tower_window'][1]:.1f}°，"
                 f"重叠 {s['overlap'][1] - s['overlap'][0]:.1f}°)")
    c = pl.checks
    total = sum(s.duration for s in pl.segments)
    waits = sum(1 for s in pl.segments if s.kind == "wait")
    L.append(f"运动总时长 {total:.0f} s（另有 {waits} 次人工等待）；关节最小余量 {c['min_joint_margin']:.1f}°；"
             f"最大角速度 {c['max_joint_speed']:.0f}°/s；最大正解误差 {c['max_fk_error']:.4f} mm")
    L.append(f"夹指最低点高出被夹木块底面至少 {c['min_finger_margin']:.1f} mm；"
             f"手中木块离桌面最低 {c['min_held_clear']:.1f} mm；"
             f"与其他木块最小间隙 {c.get('min_held_gap', float('nan')):.1f} mm（0 为起吊/落下时贴合）")
    sd = straight_descent_stats(pl)
    if sd:
        L.append(f"夹紧/松开前姿态不变的纯竖直下降：最短 {min(sd.values()):.0f} mm，"
                 f"平均 {np.mean(list(sd.values())):.0f} mm（夹爪在此段不转动）")
    if c["pitch_mismatch"]:
        worst = max(abs(v) for v in c["pitch_mismatch"].values())
        L.append(f"夹取/放下俯仰最大差 {worst:.3f}°（木块保持水平）")
    L.append("离线检查：" + ("通过" if not c["issues"] else f"{len(c['issues'])} 项未通过"))
    return "\n".join(L)


def export(pl, out_dir, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{tag}_trajectory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "phase", "segment", "held", "q1", "q2", "q3", "q4", "q5", "q6",
                    "pitch", "x", "y", "z"])
        t = 0.0
        for seg in pl.segments:
            for x in seg.q:
                p = grip_point(x)
                w.writerow([f"{t:.3f}", seg.phase, seg.label, seg.held or ""]
                           + [f"{v:.3f}" for v in x] + [f"{tool_pitch(x):.2f}"] + [f"{v:.2f}" for v in p])
                t += SEND_PERIOD_S
    summary = {
        "layers": pl.layers, "tower_xy": TOWER_XY, "table_z": TABLE_Z, "block": BLOCK_SIZE,
        "supply": {f"L{k}": {"center": s["center"].round(2).tolist(),
                             "yaw_deg": round(float(np.rad2deg(s["yaw"])), 2),
                             "pitch_deg": round(s["pitch"], 2),
                             "tower_window": s["tower_window"], "supply_window": s["supply_window"]}
                   for k, s in pl.supply.items()},
        "duration_s": sum(s.duration for s in pl.segments),
        "checks": pl.checks,
    }
    with open(out_dir / f"{tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)


COLORS = ["#d1495b", "#2e86ab", "#3b8b5a", "#c77d1a", "#7a4fa3", "#1b998b"]


def _use_cjk_font():
    """让动画标题里的中文正常显示（Windows 用雅黑/黑体，其他系统用 Noto）。"""
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                                              "WenQuanYi Zen Hei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def draw_square(ax, center, yaw, **kw):
    c = block_corners(np.asarray(center, float), yaw)[[0, 2, 6, 4], :2]
    ax.fill(c[:, 0], c[:, 1], **kw)


def save_overview(pl, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    for k, s in pl.supply.items():
        col = COLORS[(k - 1) % len(COLORS)]
        draw_square(ax1, s["center"], s["yaw"], color=col, alpha=0.45)
        pad = f"\n+{s['support']:.0f}mm" if s.get("support", 0) > 0.5 else ""
        ax1.annotate(f"L{k} {s['pitch']:.0f}°{pad}", s["center"][:2], xytext=(18 * ((k % 2) * 2 - 1), 0),
                     textcoords="offset points", ha="center", va="center", fontsize=7)
        ax1.plot([s["center"][0], TOWER_XY[0]], [s["center"][1], TOWER_XY[1]], ":", color=col, lw=0.8)
    draw_square(ax1, layer_center(1), tower_yaw(), color="0.3", alpha=0.7)
    ax1.annotate("tower", TOWER_XY, xytext=(5, -14), textcoords="offset points", fontsize=8)
    ax1.plot(0, 0, "ks")
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.3)
    ax1.set_title("Top view: feed/supply points (pitch, support height) and tower")
    ax1.set_xlabel("X_base (mm)")
    ax1.set_ylabel("Y_base (mm)")
    # 侧视：每层放置瞬间的手臂骨架 + 塔
    for seg in pl.segments:
        if seg.kind == "grip" and seg.phase == "stack" and seg.q[-1, 5] == GRIP_OPEN_DEG:
            k = int(re.search(r"叠(\d+)", seg.label).group(1))
            x = seg.q[-1]
            fr = fk_frames(x)
            az = np.deg2rad(x[0])
            u = np.array([np.cos(az), np.sin(az), 0])
            pts = np.array([[f[:3, 3] @ u, f[2, 3]] for f in fr])
            ax2.plot(pts[:, 0], pts[:, 1], "-o", ms=3, color=COLORS[(k - 1) % len(COLORS)],
                     label=f"place L{k}  pitch {tool_pitch(x):.0f}°")
    r_t = np.hypot(*TOWER_XY)
    for k in range(1, pl.layers + 1):
        z0 = TABLE_Z + BLOCK_SIZE * (k - 1)
        ax2.add_patch(plt.Rectangle((r_t - BLOCK_SIZE / 2, z0), BLOCK_SIZE, BLOCK_SIZE,
                                    fc="0.85", ec="0.3"))
    ax2.axhline(TABLE_Z, color="sienna")
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)
    ax2.set_xlabel("radial (mm)")
    ax2.set_ylabel("z (mm)")
    ax2.set_title("Side view at each placement (arm plane)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def animate(pl, speed=4.0, save_frames=None):
    """三维动画：手臂骨架、夹指、木块（手中木块随夹爪移动）。"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    _use_cjk_font()
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    faces_idx = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    stride = max(1, int(round(speed)))
    frame = 0
    for seg in pl.segments:
        for i in range(0, len(seg.q), stride):
            x = seg.q[i]
            ax.cla()
            ax.set_xlim(-50, 400)
            ax.set_ylim(-200, 250)
            ax.set_zlim(0, 450)
            ax.set_box_aspect((450, 450, 450))
            fr = fk_frames(x)
            pts = np.array([f[:3, 3] for f in fr])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", color="0.2", lw=3, ms=4)
            fp = finger_points(x, x[5])
            ax.scatter(fp[:, 0], fp[:, 1], fp[:, 2], s=4, color="k")
            blocks = list(seg.scene.items())
            if seg.held:
                corners, Tb = held_corners(pl.held_rel[seg.held], x)
                blocks.append((seg.held, (Tb[:3, 3], float(np.arctan2(Tb[1, 0], Tb[0, 0])), corners[:8])))
            for name, val in blocks:
                cs = val[2] if len(val) == 3 else block_corners(val[0], val[1])
                k = int(name[1:])
                ax.add_collection3d(Poly3DCollection([cs[f] for f in faces_idx], alpha=0.7,
                                                     facecolor=COLORS[(k - 1) % len(COLORS)], edgecolor="k"))
            ax.set_title(f"{seg.label}   pitch {tool_pitch(x):.0f}°", fontsize=9)
            if save_frames:
                fig.savefig(Path(save_frames) / f"frame_{frame:05d}.png", dpi=70)
            else:
                plt.pause(SEND_PERIOD_S * stride / speed)
            frame += 1
    if not save_frames:
        plt.show()


# ============================================================
# 10. 实机通信与执行
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


def execute(ser, pl, step=False):
    t0 = time.perf_counter()
    last_phase = None
    for seg in pl.segments:
        if seg.phase != last_phase and last_phase is not None:
            input(f"\n{last_phase} 阶段完成。检查塔/木块状态，撤手后回车继续 {seg.phase}：")
        last_phase = seg.phase
        if seg.kind == "wait":
            ans = input(f"\n[{seg.label}] {seg.message}\n（回车继续，q 中止）：")
            if ans.strip().lower() == "q":
                raise KeyboardInterrupt
            continue
        if step and seg.kind == "grip":
            ans = input(f"[{seg.label}] 即将{'夹紧' if seg.q[-1, 5] == GRIP_CLOSE_DEG else '松开'}，"
                        "回车继续，q 中止：")
            if ans.strip().lower() == "q":
                raise KeyboardInterrupt
        print(f"  {seg.label}（{seg.duration:.1f} s）")
        stream(ser, seg.q)
    return time.perf_counter() - t0


# ============================================================
# 11. 扫描：各层俯仰窗口与可叠层数
# ============================================================

def run_scan(feed_mode=None):
    feed_mode = feed_mode or FEED_MODE
    print(f"塔位 {TOWER_XY}（r={np.hypot(*TOWER_XY):.0f} mm），供料方式 {feed_mode}")
    use_pitch_grid(MAX_LAYERS)
    for k in range(1, MAX_LAYERS + 2):
        g = grasp_point_from_center(layer_center(k), k) + np.array([0, 0, RELEASE_GAP])
        W = pitch_window(g, g + np.array([0, 0, APPROACH]))
        print(f"  L{k}: 中心 z={layer_center(k)[2]:.0f}  塔位俯仰窗口 {W}")
    for n in range(1, MAX_LAYERS + 1):
        use_pitch_grid(n)
        try:
            sup = plan_feed(n) if feed_mode == "hand" else plan_supply(n)
            print(f"{n} 层：可行；" + ", ".join(
                f"L{k}: r={np.hypot(*s['center'][:2]):.0f}"
                + (f"/垫{s.get('support', 0):.0f}" if feed_mode == "hand" else "")
                + f"/{s['pitch']:.0f}°" for k, s in sup.items()))
        except IKError as e:
            print(f"{n} 层：不可行（{e}）")
            break


# ============================================================
# 12. 主流程
# ============================================================

def main():
    global TOWER_XY, GRASP_Z_OFFSET, FEED_SUPPORT_HEIGHTS, RELEASE_GAP
    ap = argparse.ArgumentParser(description="第六周附加题：多层叠塔 + 逐层拆塔（重构版）")
    ap.add_argument("--mode", choices=("plan", "simulate", "hardware", "calib", "scan"), default="plan")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--phase", choices=("stack", "unstack", "both"), default="both")
    ap.add_argument("--port", default=SERIAL_PORT)
    ap.add_argument("--tower", type=float, nargs=2, metavar=("X", "Y"))
    ap.add_argument("--grasp-offset", type=float, help="夹取点相对木块中心的高度（mm）")
    ap.add_argument("--feed", choices=("hand", "table"), default=FEED_MODE,
                    help="hand：人工递料（垫块定高，可到 11 层）；table：开机前预摆在桌面（≤5 层，全自动）")
    ap.add_argument("--supports", help="可用垫块高度列表（mm），如 0,30,60,90,120")
    ap.add_argument("--release-gap", type=float, help="放置时木块底面高出支撑面（mm）")
    ap.add_argument("--step", action="store_true", help="每次夹紧/松开前回车确认")
    ap.add_argument("--speed", type=float, default=4.0, help="仿真回放倍速")
    ap.add_argument("--save-frames", help="仿真帧导出目录（不弹窗）")
    args = ap.parse_args()
    if args.tower:
        TOWER_XY = tuple(args.tower)
    if args.grasp_offset is not None:
        GRASP_Z_OFFSET = args.grasp_offset
    if args.supports:
        FEED_SUPPORT_HEIGHTS = tuple(float(x) for x in args.supports.split(","))
    if args.release_gap is not None:
        RELEASE_GAP = args.release_gap

    if args.mode == "scan":
        run_scan(args.feed)
        return

    if args.mode == "calib":
        pl = Planner(args.layers, args.feed)
        s = pl.supply[1]
        g = grasp_point_from_center(s["center"])
        pl.start_from_zero(s["center"], s["pitch"])
        pl.travel_to_hover("标定", g + np.array([0, 0, APPROACH]), s["pitch"], target="L1")
        pl.vertical("标定/下降", g[2], VERTICAL_SPEED, target="L1")
        pad = f"（放在高 {s.get('support', 0):.0f} mm 的垫块上）" if s.get("support", 0) > 0.5 else ""
        print(f"标定：把 L1{pad}放在 ({s['center'][0]:.1f}, {s['center'][1]:.1f})，一边正对基座。"
              f"夹爪将张开停在夹取高度（俯仰 {s['pitch']:.1f}°），不会闭合。")
        if input("撤手后输入 YES 开始：").strip().upper() != "YES":
            return
        ser = init_serial(args.port)
        try:
            for seg in pl.segments:
                stream(ser, seg.q)
            print("请核对：①两指是否对称地跨在木块两侧；②指尖最低点高出桌面多少；"
                  "③木块中心与两指中心的水平偏差。偏差写入 SUPPLY_XY_COMP / GRASP_Z_OFFSET。")
            input("核对完成后回车，机械臂竖直抬起并回零：")
            back = Planner(args.layers, args.feed)
            back.q, back.grip = pl.q.copy(), pl.grip
            back.blocks = pl.blocks
            back.under = "L1"
            back.go_home("标定")
            for seg in back.segments:
                stream(ser, seg.q)
        finally:
            ser.close()
        return

    pl = build_plan(args.layers, args.phase, args.feed)
    check_plan(pl, raise_on_fail=(args.mode == "hardware"))
    print(summarize(pl))
    if pl.checks["issues"]:
        print("未通过项：\n  " + "\n  ".join(pl.checks["issues"][:20]))
    robot = build_rtb_model()
    if robot is not None:
        worst = max(float(np.linalg.norm(robot.fkine(np.deg2rad(s.q[i, :5])).t - grip_point(s.q[i])))
                    for s in pl.segments for i in (0, len(s.q) - 1))
        print(f"roboticstoolbox 正解复核：最大差异 {worst:.2e} mm")
    tag = f"stack{args.layers}_{args.phase}_{args.feed}"
    export(pl, OUTPUT_DIR, tag)
    save_overview(pl, OUTPUT_DIR / f"{tag}_overview.png")
    print(f"已导出 {tag}_trajectory.csv / _summary.json / _overview.png（目录 {OUTPUT_DIR}）")

    if args.mode == "simulate":
        if args.save_frames:
            import matplotlib
            matplotlib.use("Agg")
            Path(args.save_frames).mkdir(parents=True, exist_ok=True)
        else:
            import matplotlib
            try:
                matplotlib.use("TkAgg")
            except Exception:
                pass
        animate(pl, speed=args.speed, save_frames=args.save_frames)
        return

    if args.mode == "hardware":
        print("\n实机运行：开机前按上表摆好木块（一边正对基座），撤手；运动中不得伸手。")
        print("Ctrl+C 立即停止下发；夹爪保持当前状态，不会松开手中木块。")
        if input("输入 YES 开始：").strip().upper() != "YES":
            print("已取消。")
            return
        ser = init_serial(args.port)
        try:
            used = execute(ser, pl, step=args.step)
            print(f"完成，用时 {used:.0f} s。")
        except KeyboardInterrupt:
            print("\n已中止：停止下发，舵机保持最后一帧（夹爪状态不变）。确认安全后手动处理。")
        finally:
            ser.close()
            print("串口已关闭。")


if __name__ == "__main__":
    main()
