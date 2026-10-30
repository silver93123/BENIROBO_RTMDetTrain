"""ROI를 '픽점 기준'으로 적용하는 공용 함수 (5/5-2 파이프라인, 4/4-1/3번 탭이 같이 쓴다).

예전에는 ROI 영역만 잘라서 모델에 넣었다 (영역 밖 물체는 검출 자체가 안 됨). 잘라 넣으면 모델 입력 배율/주변 문맥이
바뀌어 전체 이미지 검출과 결과가 달라진다. 지금은 전체 이미지에서 검출 -> 픽점 산출까지 다 한 뒤,
'픽점이 ROI 안에 있는 것'만 남긴다.

ROI = (x1, y1, x2, y2) 이미지 픽셀, x2/y2 미포함.
"""
from __future__ import annotations

import numpy as np

from app.core.camera_intrinsics import locate_pixel, project_points


def uv_in_roi(uv, roi) -> bool:
    """(u, v)가 ROI 안이면 True. roi가 None(ROI 꺼짐)이면 항상 True. uv가 None이면 판단 불가 -> True(제외하지 않음)."""
    if roi is None or uv is None:
        return True
    x1, y1, x2, y2 = roi
    return bool(x1 <= uv[0] < x2 and y1 <= uv[1] < y2)


def pick_pixel(xyz_mm, pcd=None, valid=None, intr=None, uv=None):
    """픽점(카메라 좌표 mm)이 이미지의 어느 픽셀인지. 이미 아는 픽셀(uv)이 있으면 그대로 쓴다.

    점군이 있으면 점군에서 같은 광선의 픽셀을 찾고(화면에 찍히는 마커와 같은 위치), 없으면 핀홀 근사 투영을 쓴다.
    구하지 못하면 None.
    """
    if uv is not None:
        return (float(uv[0]), float(uv[1]))
    if xyz_mm is None:
        return None
    x = np.asarray(xyz_mm, dtype=np.float64).reshape(3)
    if x[2] <= 1.0:
        return None
    guess = None
    if intr is not None:
        guess = project_points(x.reshape(1, 3), intr)[0]
    if pcd is not None and valid is not None:
        found = locate_pixel(pcd, valid, x, guess_uv=guess)
        if found is not None:
            return found
    return None if guess is None else (float(guess[0]), float(guess[1]))


def outside_reason(uv, roi) -> str:
    """픽점이 ROI 밖이면 사유 문자열, 안이면 ''."""
    if uv_in_roi(uv, roi):
        return ""
    x1, y1, x2, y2 = roi
    return f"ROI 밖 (픽점 u={uv[0]:.0f}, v={uv[1]:.0f}; ROI {x1},{y1}~{x2},{y2})"


def mask_centroid_uv(mask):
    """마스크 무게중심 (u, v). 비어 있으면 None. (픽점이 없는 3번 탭의 ROI 판정용)"""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())
