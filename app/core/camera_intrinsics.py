"""카메라 intrinsic 근사 역산 + 3D->2D 투영 공유 유틸리티.

이 레포는 카메라 캘리브레이션 파일을 따로 두지 않는다 - SDK가 왜곡보정까지
끝낸 organized PCD(H,W,3, mm)만 저장하기 때문(src/camera/femto_bolt.py,
lucid_helios.py 참고). 그래서 3D 점을 2D로 투영하려면 핀홀 근사 intrinsic이
필요한데, 이미 가진 (X,Y,Z)<->(u,v) 대응 수만 쌍에서 최소자승으로 fx,fy,cx,cy를
역산하면 별도 캘리브레이션 없이 충분히 정확한 근사치를 얻을 수 있다.

scripts/generate_rotation_labels.py, app/tabs/manual_labeling_tab.py 둘 다
이 모듈을 쓴다 - 같은 계산을 두 번 구현하지 않기 위해 분리해뒀다.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def estimate_intrinsics_from_organized_pcd(
    points_organized: np.ndarray,
    valid_mask: np.ndarray,
    subsample: int = 20000,
) -> Tuple[float, float, float, float]:
    """organized PCD (H,W,3, mm)와 valid_mask -> 근사 핀홀 (fx, fy, cx, cy).

    u = fx * (X/Z) + cx, v = fy * (Y/Z) + cy 는 각 축이 독립적인 1차
    선형회귀이므로 유효 픽셀 (X,Y,Z,u,v) 표본으로 최소자승 적합하면 된다.

    Args:
        points_organized: (H, W, 3) float, mm.
        valid_mask: (H, W) bool.
        subsample: 적합에 쓸 최대 픽셀 수 (전체 다 쓸 필요 없음, 속도용).

    Returns:
        (fx, fy, cx, cy).
    """
    vs, us = np.where(valid_mask)
    if vs.size == 0:
        raise ValueError("valid_mask에 유효 픽셀이 없음 - intrinsic 추정 불가")

    if vs.size > subsample:
        idx = np.random.default_rng(0).choice(vs.size, size=subsample, replace=False)
        vs, us = vs[idx], us[idx]

    xyz = points_organized[vs, us]  # (S, 3)
    z_ok = xyz[:, 2] > 1e-3  # Z<=0은 카메라 뒤/무효값이므로 제외
    x_over_z = xyz[z_ok, 0] / xyz[z_ok, 2]
    y_over_z = xyz[z_ok, 1] / xyz[z_ok, 2]
    us_f = us[z_ok].astype(np.float64)
    vs_f = vs[z_ok].astype(np.float64)

    fx, cx = np.polyfit(x_over_z, us_f, deg=1)
    fy, cy = np.polyfit(y_over_z, vs_f, deg=1)
    return float(fx), float(fy), float(cx), float(cy)


def locate_pixel(points_organized: np.ndarray, valid_mask: np.ndarray, xyz_mm, guess_uv=None,
                 radius: int = 80) -> Tuple[float, float] | None:
    """3D 점(mm)이 점군의 어느 픽셀에 해당하는지 찾는다 -> (u, v) 또는 None.

    화면에 점을 그릴 때 근사 핀홀(estimate_intrinsics_*)로 투영하면, 점군이 그 핀홀과 어긋난 만큼(수~십수 px) 엉뚱한
    픽셀에 찍힌다. 그 마커를 13번 탭에서 클릭해 보면 다른 점을 읽게 된다. 여기서는 점군 자체에서 같은 광선(방향)을
    가진 픽셀을 찾는다: 카메라에서 본 방향이 가장 일치(코사인 최대)하는 유효 픽셀. 광선을 따라 앞뒤로 움직인 점
    (표면 -> 중심 깊이 보정)도 같은 픽셀로 찾아진다.

    guess_uv: 탐색 중심 (근사 투영 결과). 없으면 전체 이미지를 훑는다. radius: 중심에서 +-px 탐색 반경.
    """
    pts_all = np.asarray(points_organized)
    h, w = pts_all.shape[:2]
    x = np.asarray(xyz_mm, dtype=np.float64).reshape(3)
    nx = float(np.linalg.norm(x))
    if nx < 1e-9:
        return None
    if guess_uv is None or not np.all(np.isfinite(guess_uv)):
        x0, x1, y0, y1 = 0, w, 0, h
    else:
        gu, gv = int(round(float(guess_uv[0]))), int(round(float(guess_uv[1])))
        x0, x1 = max(0, gu - radius), min(w, gu + radius + 1)
        y0, y1 = max(0, gv - radius), min(h, gv + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    win = pts_all[y0:y1, x0:x1].astype(np.float64)
    ok = np.asarray(valid_mask[y0:y1, x0:x1], dtype=bool) & np.isfinite(win).all(axis=2) & (win[..., 2] > 1.0)
    if not ok.any():
        return None
    norm = np.linalg.norm(win, axis=2)
    norm[~ok] = 1.0
    cos = (win @ x) / (norm * nx)
    cos[~ok] = -2.0
    iy, ix = np.unravel_index(int(np.argmax(cos)), cos.shape)
    return float(x0 + ix), float(y0 + iy)


def anchor_overlay_uv(uv: np.ndarray, origin_uv) -> np.ndarray:
    """project_points로 투영한 [원점, 축끝1, 축끝2, 축끝3] 픽셀 uv를, 원점이 origin_uv(점군에서 확인한 실제 픽셀)에
    오도록 평행이동한다. 근사 핀홀의 편향(원점이 몇 px 밀림)은 원점 쪽만 틀리고 축의 방향/길이(상대 위치)는 거의 맞다."""
    uv = np.asarray(uv, dtype=np.float64)
    return uv - uv[0] + np.asarray(origin_uv, dtype=np.float64)


def project_points(points_cam_mm: np.ndarray, intrinsics: Tuple[float, float, float, float]) -> np.ndarray:
    """카메라 좌표계 3D점(mm) -> 픽셀 좌표 (K,2), (x,y)=(col,row) 순서."""
    fx, fy, cx, cy = intrinsics
    z = points_cam_mm[:, 2]
    u = fx * points_cam_mm[:, 0] / z + cx
    v = fy * points_cam_mm[:, 1] / z + cy
    return np.stack([u, v], axis=1)