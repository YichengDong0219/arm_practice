"""实验 1.8 使用的三维旋转与齐次变换工具。"""

from __future__ import annotations

import numpy as np


def rot_x(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rad), -np.sin(rad)],
            [0.0, np.sin(rad), np.cos(rad)],
        ],
        dtype=np.float64,
    )


def rot_z(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    return np.array(
        [
            [np.cos(rad), -np.sin(rad), 0.0],
            [np.sin(rad), np.cos(rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def build_transform(rotation, translation) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("旋转矩阵必须为 3x3")
    if translation.shape != (3,):
        raise ValueError("平移向量必须包含 3 个元素")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def assert_rigid_transform(transform: np.ndarray, atol: float = 1e-9) -> None:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("齐次变换矩阵必须为 4x4")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError("旋转子矩阵不满足正交性")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol):
        raise ValueError("旋转子矩阵行列式不为 1")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
        raise ValueError("齐次矩阵最后一行错误")

