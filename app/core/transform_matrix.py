"""카메라 -> 로봇 변환 행렬 직접 입력 / 비교용 순수 계산 (Qt 없음, 13번 탭 '행렬 직접 입력'이 쓴다).

    robot = R · camera + t          (mm)

이동 t와 오일러 회전(deg)을 넣으면 R, 4x4 행렬 T를 만들고, 저장된 변환(data/gripper_calib.json) 등 기준 변환과
행렬 / 점 위치 차이를 비교한다. 오일러 순서는 로봇 컨트롤러마다 다를 수 있어서 두 가지를 고를 수 있게 했다.

    "Rz·Ry·Rx" : R = Rz(rz) @ Ry(ry) @ Rx(rx)   (13번 탭 '변환 계산' 결과 표시와 같은 순서)
    "Rx·Ry·Rz" : R = Rx(rx) @ Ry(ry) @ Rz(rz)
"""
from __future__ import annotations

import math

import numpy as np

ORDER_ZYX = "Rz·Ry·Rx"
ORDER_XYZ = "Rx·Ry·Rz"
EULER_ORDERS = (ORDER_ZYX, ORDER_XYZ)

# 저장된 점 짝이 없을 때 비교에 쓰는 카메라 좌표 (mm): 광축 위 + 좌우/상하 +-200 mm, 깊이 600 mm
DEFAULT_PROBE_POINTS = ((0.0, 0.0, 600.0), (200.0, 0.0, 600.0), (-200.0, 0.0, 600.0),
                        (0.0, 200.0, 600.0), (0.0, -200.0, 600.0))


def _Rx(a: float) -> np.ndarray:
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _Ry(a: float) -> np.ndarray:
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _Rz(a: float) -> np.ndarray:
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def euler_to_R(rx_deg: float, ry_deg: float, rz_deg: float, order: str = ORDER_ZYX) -> np.ndarray:
    """오일러 각(deg) -> 3x3 회전행렬."""
    if order == ORDER_ZYX:
        return _Rz(rz_deg) @ _Ry(ry_deg) @ _Rx(rx_deg)
    if order == ORDER_XYZ:
        return _Rx(rx_deg) @ _Ry(ry_deg) @ _Rz(rz_deg)
    raise ValueError(f"알 수 없는 오일러 순서: {order}")


def R_to_euler(R, order: str = ORDER_ZYX) -> tuple[float, float, float]:
    """3x3 회전행렬 -> (rx, ry, rz) deg. ry=±90도(짐벌락)에서는 rz=0으로 둔다."""
    R = np.asarray(R, dtype=float)
    if order == ORDER_ZYX:
        sy = math.hypot(R[0, 0], R[1, 0])
        if sy > 1e-9:
            rx = math.atan2(R[2, 1], R[2, 2]); rz = math.atan2(R[1, 0], R[0, 0])
        else:
            rx = math.atan2(-R[1, 2], R[1, 1]); rz = 0.0
        ry = math.atan2(-R[2, 0], sy)
    elif order == ORDER_XYZ:
        cy = math.hypot(R[0, 0], R[0, 1])
        if cy > 1e-9:
            rx = math.atan2(-R[1, 2], R[2, 2]); rz = math.atan2(-R[0, 1], R[0, 0])
        else:
            rx = math.atan2(R[2, 1], R[1, 1]); rz = 0.0
        ry = math.atan2(R[0, 2], cy)
    else:
        raise ValueError(f"알 수 없는 오일러 순서: {order}")
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def to_T(R, t) -> np.ndarray:
    """R(3x3), t(3,) -> 동차 4x4 행렬."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def compare_transforms(RA, tA, RB, tB, points=DEFAULT_PROBE_POINTS) -> dict:
    """변환 A(입력)와 B(기준)의 차이 (A - B).

    rot_deg   : 두 회전 사이의 각도 (rot_deg = acos((tr(A^T B) - 1) / 2))
    dt        : 이동 차이 벡터 (tA - tB), dt_norm 크기
    pt_rms/pt_max : 카메라 점들을 각각의 변환으로 옮긴 로봇 좌표의 차이 크기 (mm) - 실제 픽 위치가 얼마나 달라지는지
    """
    RA, RB = np.asarray(RA, dtype=float), np.asarray(RB, dtype=float)
    tA, tB = np.asarray(tA, dtype=float), np.asarray(tB, dtype=float)
    cosv = float(np.clip((np.trace(RA.T @ RB) - 1.0) / 2.0, -1.0, 1.0))
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    d = np.linalg.norm((pts @ RA.T + tA) - (pts @ RB.T + tB), axis=1)
    dt = tA - tB
    return {"rot_deg": math.degrees(math.acos(cosv)), "dt": dt, "dt_norm": float(np.linalg.norm(dt)),
            "pt_rms": float(np.sqrt(np.mean(d ** 2))), "pt_max": float(d.max()), "n_points": int(len(pts))}


def format_T(T, name: str = "") -> str:
    """4x4 행렬을 고정폭 텍스트로 (R 6자리, t 3자리)."""
    T = np.asarray(T, dtype=float)
    rows = []
    for i in range(4):
        r = " ".join(f"{T[i, j]:+10.6f}" for j in range(3)) + f" {T[i, 3]:+11.3f}"
        rows.append(f"[ {r} ]")
    return (f"{name}\n" if name else "") + "\n".join(rows)


def format_numpy(T) -> str:
    """복사해 파이썬에 붙일 수 있는 np.array 문자열 (전체 정밀도)."""
    T = np.asarray(T, dtype=float)
    rows = ",\n    ".join("[" + ", ".join(repr(float(v)) for v in row) + "]" for row in T)
    return f"np.array([\n    {rows},\n])"
