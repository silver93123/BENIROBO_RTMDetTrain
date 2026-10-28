"""13번 탭 (그리퍼 동작 테스트) 로직: 클릭한 픽셀 -> 점군 XYZ, 그리고 로봇 요청에 그 좌표를 돌려주는 워커.

- point_from_cloud(): 클릭한 픽셀 주변 창에서 유효한 점군 점의 중앙값을 읽는다 (왜곡 보정이 끝난 카메라 좌표, mm).
  픽셀 위치를 핀홀로 환산하지 않고 점군을 직접 읽으므로 렌즈 왜곡의 영향을 받지 않는다.
- ClickedPoint / GripperPointWorker: 5번 탭의 PickServer가 그대로 쓰는 워커 인터페이스(is_busy/submit/stop)를 맞춰서,
  요청이 오면 촬영/검출 없이 "마지막으로 클릭된 좌표"를 PickRunResult로 돌려준다 (프로토콜/응답 형식은 5번과 동일).
"""
from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass, field

import numpy as np

from app.core.pick_pipeline import PickCandidate, PickRunResult
from app.core.point_pair_calib import OutputConfig

DEFAULT_WINDOW = 5     # 클릭 픽셀 주변 창 크기 (홀수, 1이면 그 픽셀 한 점만)


@dataclass
class ClickedPoint:
    """클릭 하나의 결과."""
    u: int                                   # 클릭한 픽셀 (이미지 좌표)
    v: int
    xyz_mm: tuple[float, float, float]       # 카메라 좌표계 mm
    n_points: int                            # 좌표 계산에 쓴 유효 점 수
    window: int                              # 사용한 창 크기
    spread_mm: float                         # 창 안 점들의 Z 표준편차 (평평한 면이면 작다)
    stamp: str = ""


def point_from_cloud(points_organized: np.ndarray, valid_mask: np.ndarray | None,
                     u: float, v: float, window: int = DEFAULT_WINDOW) -> ClickedPoint | None:
    """(u, v) 주변 window x window 창의 유효 점 중앙값(XYZ, mm). 유효한 점이 하나도 없으면 None.

    유효 = 유한값이고 (0,0,0)이 아니며, valid_mask가 있으면 그것도 True인 점.
    """
    pts = np.asarray(points_organized)
    if pts.ndim != 3 or pts.shape[2] != 3:
        raise ValueError(f"points_organized 모양이 (H,W,3)이 아닙니다: {pts.shape}")
    h, w = pts.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return None
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    r = window // 2
    v0, v1 = max(0, vi - r), min(h, vi + r + 1)
    u0, u1 = max(0, ui - r), min(w, ui + r + 1)
    patch = pts[v0:v1, u0:u1].reshape(-1, 3).astype(np.float64)
    ok = np.isfinite(patch).all(axis=1) & ~(patch == 0).all(axis=1)
    if valid_mask is not None:
        ok &= np.asarray(valid_mask)[v0:v1, u0:u1].reshape(-1).astype(bool)
    patch = patch[ok]
    if len(patch) == 0:
        return None
    med = np.median(patch, axis=0)
    spread = float(np.std(patch[:, 2])) if len(patch) > 1 else 0.0
    return ClickedPoint(u=ui, v=vi, xyz_mm=(float(med[0]), float(med[1]), float(med[2])),
                        n_points=int(len(patch)), window=window, spread_mm=spread)


class ClickedPointStore:
    """GUI 스레드(클릭)와 소켓 스레드(응답)가 같이 쓰는 '현재 선택된 좌표' 보관소."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._point: ClickedPoint | None = None

    def set(self, point: ClickedPoint | None) -> None:
        with self._lock:
            self._point = point

    def get(self) -> ClickedPoint | None:
        with self._lock:
            return self._point


@dataclass
class GripperPointWorker:
    """PickServer가 기대하는 워커 모양. 'pick' 요청 = 현재 선택된 좌표를 즉시 돌려준다."""
    store: ClickedPointStore
    output: OutputConfig | None = None      # 응답 좌표계 (카메라 / 로봇 변환 적용). None이면 항상 카메라 좌표
    _served: int = field(default=0, init=False)

    # PickServer는 mode 속성을 읽지 않지만 PickWorker와 모양을 맞춘다
    mode = "gripper"

    def is_busy(self) -> bool:
        return False      # 계산이 없으므로 겹치는 요청도 모두 바로 응답한다

    def submit(self, kind: str, keep_camera_open: bool | None = None) -> "Future":
        fut: Future = Future()
        if kind == "pick":
            fut.set_result(self._result())
        else:             # warmup / release_camera: 할 일 없음 (warmup은 None = 성공)
            fut.set_result(None)
        return fut

    def stop(self, timeout: float = 10.0) -> None:
        return None

    def _result(self) -> PickRunResult:
        pt = self.store.get()
        if pt is None:
            return PickRunResult(ok=False, error_code="NO_POINT",
                                 error_message="클릭된 좌표가 없습니다 (이미지에서 위치를 먼저 클릭하세요)")
        xyz = pt.xyz_mm
        if self.output is not None:
            frame, R, t, rms, limit = self.output.snapshot()
            if frame == "robot":
                if R is None or t is None:
                    return PickRunResult(ok=False, error_code="NO_CALIB", error_message="좌표 변환이 아직 계산되지 않았습니다")
                if rms is not None and rms > limit:
                    return PickRunResult(ok=False, error_code="CALIB_BAD",
                                         error_message=f"변환 잔차 RMS {rms:.2f} mm 가 한계 {limit:.2f} mm 를 넘습니다")
                v = R @ np.asarray(xyz, dtype=float) + t
                xyz = (float(v[0]), float(v[1]), float(v[2]))
        self._served += 1
        cand = PickCandidate(instance_id=0, label="click", confidence=1.0, xyz_mm=xyz,
                             rpy_deg=(0.0, 0.0, 0.0), fitness=1.0, rmse_mm=0.0)
        return PickRunResult(ok=True, candidates=[cand], mode="gripper")


# =============================================================================
# 평면 검증: 평평한 판의 점군이 기울어/휘어 보이는지 수치로 본다
# =============================================================================
@dataclass
class PlaneReport:
    n_points: int
    centroid_mm: tuple[float, float, float]
    normal: tuple[float, float, float]        # 카메라 쪽(Z<0)을 향하는 단위 법선
    tilt_deg: float                           # 법선이 카메라 광축(-Z)과 이루는 각 = 판이 광축에 수직인 상태에서 기운 정도
    tilt_x_deg: float                         # X축 둘레로 기운 정도 (판을 수평으로 맞추려면 이만큼 돌린다)
    tilt_y_deg: float                         # Y축 둘레
    rms_mm: float                             # 평면 대비 잔차 RMS (카메라 깊이 정밀도 + 형상 오차)
    max_abs_mm: float
    rms_after_quad_mm: float                  # 완만한 곡면(2차)까지 맞춘 뒤 잔차 RMS
    bow_mm: float                             # 2차 곡면 성분이 ROI 가장자리에서 만드는 높이차 (그릇/볼록 정도, +면 가운데가 카메라 쪽)
    extent_mm: float                          # ROI에 걸친 판의 크기 (대략적인 가로/세로 중 큰 값)
    residual_map: np.ndarray                  # (h, w) float32, ROI 안 각 픽셀의 평면 잔차 mm (무효 = NaN)
    n_outliers: int


def analyze_plane(points_organized: np.ndarray, valid_mask: np.ndarray | None,
                  roi: tuple[int, int, int, int], max_points: int = 30000) -> PlaneReport | None:
    """ROI(x1, y1, x2, y2 - 이미지 좌표, x2/y2 미포함) 안의 점군에 평면을 맞춰 기울기/휘어짐을 계산한다.

    평면은 SVD로 맞추고, 잔차가 3*MAD(1.4826 스케일)를 넘는 점(가장자리 flying pixel 등)을 2회 빼고 다시 맞춘다.
    유효 점이 30개 미만이면 None.
    """
    pts = np.asarray(points_organized)
    h, w = pts.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in roi)
    x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    patch = pts[y1:y2, x1:x2].astype(np.float64)
    ok = np.isfinite(patch).all(axis=2) & ~(patch == 0).all(axis=2)
    if valid_mask is not None:
        ok &= np.asarray(valid_mask)[y1:y2, x1:x2].astype(bool)
    flat = patch[ok]
    if len(flat) < 30:
        return None
    rng = np.random.default_rng(0)
    fit = flat if len(flat) <= max_points else flat[rng.choice(len(flat), max_points, replace=False)]

    def _fit(p):
        c = p.mean(axis=0)
        _, _, vt = np.linalg.svd(p - c, full_matrices=False)
        n = vt[2]
        return c, n

    keep = np.ones(len(fit), bool)
    c, n = _fit(fit)
    for _ in range(2):
        r = (fit - c) @ n
        mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-9
        keep = np.abs(r - np.median(r[keep])) <= 3.0 * mad
        if keep.sum() < 30:
            return None
        c, n = _fit(fit[keep])
    if n[2] > 0:
        n = -n                                           # 카메라 쪽을 향하게 (카메라 Z는 앞쪽 +)
    resid_all = (flat - c) @ n
    r_fit = (fit[keep] - c) @ n

    # 평면 위 좌표계에서 2차 곡면 맞추기 -> 그릇/볼록 성분
    e1 = np.cross(n, [0.0, 1.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(n, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    q = fit[keep] - c
    a, b = q @ e1, q @ e2
    scale = max(np.abs(a).max(), np.abs(b).max(), 1e-6)
    A = np.stack([a * a, b * b, a * b, a, b, np.ones_like(a)], axis=1)
    coef, *_ = np.linalg.lstsq(A / np.array([scale ** 2, scale ** 2, scale ** 2, scale, scale, 1.0]), r_fit, rcond=None)
    pred = (A / np.array([scale ** 2, scale ** 2, scale ** 2, scale, scale, 1.0])) @ coef
    rms_quad = float(np.sqrt(np.mean((r_fit - pred) ** 2)))
    # 좌표를 [-1, 1]로 정규화했으므로 계수 = 가장자리에서 그 성분이 만드는 mm. 법선(+)은 카메라 쪽이라
    # (c0 + c1) / 2 > 0 이면 가장자리가 카메라 쪽으로 올라온 그릇 모양 -> 사람이 읽기 쉽게 부호를 뒤집어
    # bow > 0 = 가운데가 카메라 쪽으로 볼록, bow < 0 = 가운데가 움푹.
    R = float(scale)
    bow = -float((coef[0] + coef[1]) / 2.0)

    rmap = np.full((y2 - y1, x2 - x1), np.nan, dtype=np.float32)
    full_res = ((patch - c) @ n).astype(np.float32)
    rmap[ok] = full_res[ok]
    nz = -n[2]
    tilt = float(np.degrees(np.arccos(np.clip(nz, -1.0, 1.0))))
    return PlaneReport(
        n_points=int(keep.sum()), centroid_mm=(float(c[0]), float(c[1]), float(c[2])),
        normal=(float(n[0]), float(n[1]), float(n[2])), tilt_deg=tilt,
        tilt_x_deg=float(np.degrees(np.arctan2(n[1], -n[2]))), tilt_y_deg=float(np.degrees(np.arctan2(n[0], -n[2]))),
        rms_mm=float(np.sqrt(np.mean(r_fit ** 2))), max_abs_mm=float(np.abs(resid_all[np.abs(resid_all) < 1e9]).max()),
        rms_after_quad_mm=rms_quad, bow_mm=bow, extent_mm=float(2.0 * R),
        residual_map=rmap, n_outliers=int(len(fit) - keep.sum()))
