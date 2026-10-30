"""볼트 긴 축 방향(로봇 좌표계) -> 로봇 자세 Rz, Ry (Rz -> Ry 순서, Rx 미사용).

로봇 자세를 R = Rz(rz) @ Ry(ry) 로 만들면 툴 X축 = (cos rz cos ry, sin rz cos ry, -sin ry) 이다.
Ry를 돌려도 툴 Y축은 움직이지 않으므로(Rx를 안 쓰면) 축의 '기울기'를 표현할 수 있는 건 툴 X축(과 Z축)뿐이다.
그래서 볼트의 긴 축을 툴 X축에 대응시킨다:  rz = 수평면 방향각,  ry = -(수평면 대비 올라간 각).

볼트 머리/꼬리(180도)를 구분하지 않으면 축 방향을 +-d 중 rz가 [-90, +90] 인 쪽으로 정규화한다 (그리퍼가 앞뒤 대칭일 때).
GUI/소켓과 무관한 순수 numpy 코드.
"""
from __future__ import annotations

import math

import numpy as np

VERTICAL_EPS = 1e-3     # 수평 성분이 이보다 작으면(축이 거의 수직) rz를 정할 수 없다 -> rz=0


def rz_ry_from_axis(axis, fold180: bool = True) -> tuple[float, float, bool]:
    """(rz_deg, ry_deg, 수직에_가까움). axis: 로봇 좌표계의 볼트 긴 축 방향(아무 길이).

    rz = atan2(dy, dx),  ry = -asin(dz)  [R = Rz(rz) Ry(ry) 의 X축이 axis와 같아지도록].
    fold180=True면 d와 -d를 같은 것으로 보고 rz를 [-90, 90] 범위로 맞춘다.
    """
    d = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        raise ValueError("축 방향이 0 벡터입니다")
    d = d / n
    horiz = math.hypot(d[0], d[1])
    if horiz < VERTICAL_EPS:                       # 거의 수직: 방향각 불능. 위/아래 구분만 ry로 (+-90)
        return 0.0, math.degrees(-math.asin(float(np.clip(d[2], -1, 1)))), True
    if fold180 and d[0] < 0.0:                     # rz가 [-90, 90]을 벗어나면 반대 방향으로 뒤집는다
        d = -d
    elif fold180 and abs(d[0]) < 1e-12 and d[1] < 0.0:   # rz = -90 도 아닌 +90 으로 통일
        d = -d
    rz = math.degrees(math.atan2(d[1], d[0]))
    ry = math.degrees(-math.asin(float(np.clip(d[2], -1, 1))))
    return rz, ry, False


def tool_x_axis(rz_deg: float, ry_deg: float) -> np.ndarray:
    """검증용: R = Rz(rz) Ry(ry) 의 툴 X축 (로봇 좌표계)."""
    a, b = math.radians(rz_deg), math.radians(ry_deg)
    return np.array([math.cos(a) * math.cos(b), math.sin(a) * math.cos(b), -math.sin(b)])
