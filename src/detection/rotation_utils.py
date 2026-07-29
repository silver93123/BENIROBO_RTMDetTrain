"""6D continuous rotation representation <-> 3x3 회전행렬 변환.

Zhou et al., "On the Continuity of Rotation Representations in Neural
Networks", CVPR 2019 방식. quaternion/axis-angle 대비 회귀 학습이 더
안정적으로 수렴한다고 보고되어 회전 헤드의 출력 형식으로 채택.
"""
from __future__ import annotations

import numpy as np


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """(6,) 또는 (N, 6) -> (3,3) 또는 (N,3,3) 회전행렬.

    앞 3개, 뒤 3개를 회전행렬 1열/2열 후보로 보고 Gram-Schmidt로
    직교화한 뒤 3번째 열은 외적으로 계산한다.
    """
    single = rot6d.ndim == 1
    if single:
        rot6d = rot6d[None, :]

    a1 = rot6d[:, 0:3]
    a2 = rot6d[:, 3:6]

    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    a2_proj = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = a2_proj / np.linalg.norm(a2_proj, axis=1, keepdims=True)
    b3 = np.cross(b1, b2)

    mats = np.stack([b1, b2, b3], axis=-1)
    return mats[0] if single else mats


def matrix_to_rot6d(rot_matrix: np.ndarray) -> np.ndarray:
    """(3,3) 또는 (N,3,3) 회전행렬 -> (6,) 또는 (N,6). 학습 라벨 생성용."""
    single = rot_matrix.ndim == 2
    if single:
        rot_matrix = rot_matrix[None, :, :]

    rot6d = np.concatenate([rot_matrix[:, :, 0], rot_matrix[:, :, 1]], axis=-1)
    return rot6d[0] if single else rot6d


def compose_pose(rot_matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """회전행렬(3,3) + 이동벡터(3,), m 단위 -> 4x4 homogeneous transform."""
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rot_matrix
    pose[:3, 3] = translation
    return pose