"""2D 기반 픽 포인트 / 회전: 형상 보정(CAD 실루엣 템플릿 매칭)이 고른 템플릿에서 pose를 바로 구한다.

ICP 정합 없이, 형상 보정 단계에서 이미 결정된 값만 쓴다.

    위치   템플릿 중심 픽셀 (u, v)  +  그 자리 물체 평면의 depth  ->  카메라 좌표 (X, Y, Z) mm
           (pose_from_refine_pcd: 같은 픽셀 주변의 점군을 직접 읽는 방식 - 4-1 탭에서 두 방식을 비교)
    회전   phi (축의 깊이 방향 기울기, 부호로 머리/꼬리 구분) + theta (이미지 평면 회전)
           -> 모델 좌표계 회전  R_model = Rz(theta) @ Ry(-phi)  (SilhouetteBank가 템플릿을 그린 방식 그대로)

한계 (중요)
    * roll(축 둘레 회전)은 영상으로 알 수 없어 고정이다. 템플릿은 CAD 원래 roll 하나로 그려지므로
      roll 값은 신뢰할 수 없고, 축 방향(pitch/yaw 성분)만 의미가 있다.
    * 화면에서 보이는 물체 표면의 depth는 CAD 중심보다 가깝다. depth_correct=True면 중심에서 카메라 쪽으로
      광선을 따라 물체를 벗어나는 거리(원통 근사: 옆면 r/cos(phi), 끝면 L/(2 sin|phi|) 중 작은 쪽)를 더한다.

결과는 ICP 탭과 같은 기준(축보정이 적용된 CAD 좌표계, 단위 m)의 T 로도 돌려줘서 ICP 결과와 직접 비교할 수 있다.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from app.core import icp_runner


@dataclasses.dataclass
class Pose2D:
    ok: bool
    error: str = ""
    T: np.ndarray | None = None              # 4x4, m. ICP 탭과 같은 CAD 좌표계(축보정 적용) -> 카메라
    pose: dict | None = None                 # icp_runner.transform_to_pose(T) 결과 (euler_deg 등)
    pick_point_mm: list[float] | None = None  # CAD 중심(ICP와 같은 정의)의 카메라 좌표
    axis_cam: np.ndarray | None = None       # 부품 긴 축 방향 (카메라 좌표 단위벡터, 머리/꼬리 방향 포함)
    center_cam_mm: np.ndarray | None = None  # 템플릿 모델 중심(bbox 중심)의 카메라 좌표
    z_surface_mm: float = 0.0                # 템플릿 중심 위치의 물체 표면 depth
    z_center_mm: float = 0.0                 # 보정 후 중심 depth
    phi_deg: float = 0.0
    theta_deg: float = 0.0
    source: str = ""                         # 위치(X,Y,Z)를 어디서 얻었는지: "plane" | "pcd(창 N점)" | "plane(폴백 ...)"


# pcd 직접 읽기 방식: 템플릿 중심 픽셀 주변 win x win 창에서 마스크 안 유효 점의 중앙값을 표면 점으로 쓴다.
PCD_WINDOW = 5
PCD_MIN_POINTS = 3   # 창 안 유효 점이 이보다 적으면 읽지 않고 기존 방식(평면 + 역투영)으로 되돌아간다


def _depth_offset(bank, phi_deg: float, depth_correct: bool) -> float:
    """표면 -> 모델 중심 거리 d (mm). 모델 중심에서 카메라 쪽으로 광선을 따라 나가며 물체를 벗어나는 거리
    (원통 근사: 반지름 r, 길이 L). 옆면으로 나가는 거리 r/cos(phi), 끝면으로 나가는 거리 L/(2 sin|phi|) 중 작은 쪽."""
    if not depth_correct:
        return 0.0
    sp, cp = abs(math.sin(math.radians(phi_deg))), abs(math.cos(math.radians(phi_deg)))
    d_side = 0.5 * bank.width_mm / cp if cp > 1e-9 else float("inf")
    d_end = 0.5 * bank.length_mm / sp if sp > 1e-9 else float("inf")
    return min(d_side, d_end)


def _assemble(rr, bank, t_cam, z_surf: float, z_c: float, cad_axis_deg, cad_center_mm, source: str) -> Pose2D:
    """모델 중심의 카메라 좌표 t_cam(mm)에서 회전/ICP 좌표계 T/픽 포인트를 만든다 (위치를 어떻게 구했든 공통)."""
    phi, theta = float(rr.phi_deg), float(rr.theta_deg)
    R_model = icp_runner._Rz(theta) @ icp_runner._Ry(-phi)
    E = np.asarray(bank.model_to_cad_R)
    R_ax = icp_runner._Rz(cad_axis_deg[2]) @ icp_runner._Ry(cad_axis_deg[1]) @ icp_runner._Rx(cad_axis_deg[0])

    # v_icp(축보정 CAD) -> 카메라:  p = R_icp v_icp + t_icp   (mm)
    R_icp = R_model @ E.T @ R_ax.T
    c0 = np.asarray(bank.cad_center_mm)
    origin = np.asarray(bank.model_origin_cad_mm)
    t_icp = t_cam + R_model @ E.T @ (c0 - origin - R_ax.T @ c0)

    center_ref = c0 if cad_center_mm is None else np.asarray(cad_center_mm, dtype=np.float64)
    pick = R_icp @ center_ref + t_icp

    T = np.eye(4)
    T[:3, :3] = R_icp
    T[:3, 3] = t_icp / 1000.0
    return Pose2D(True, T=T, pose=icp_runner.transform_to_pose(T), pick_point_mm=pick.tolist(),
                  axis_cam=R_model[:, 0].copy(), center_cam_mm=np.asarray(t_cam, dtype=np.float64),
                  z_surface_mm=float(z_surf), z_center_mm=float(z_c), phi_deg=phi, theta_deg=theta, source=source)


def pose_from_refine(rr, bank, intrinsics, cad_axis_deg=(0.0, 0.0, 0.0),
                     cad_center_mm=None, depth_correct: bool = True) -> Pose2D:
    """RefineResult(rr) 하나에서 2D 기반 pose를 만든다.

    위치: 템플릿 중심 픽셀 (u, v) + 물체 평면 depth -> 핀홀 역투영 (intrinsic은 점군에서 역산한 근사값)

    rr             : mask_refine.RefineResult (ok=True, center_uv/plane 필요)
    bank           : 그 보정에 쓴 SilhouetteBank
    intrinsics     : (fx, fy, cx, cy)
    cad_axis_deg   : ICP 탭의 CAD 축보정 (roll, pitch, yaw) deg. 이 값이 적용된 CAD 좌표계 기준으로 돌려준다.
    cad_center_mm  : ICP 좌표계에서의 CAD 중심(mm). 주면 pick_point를 ICP와 같은 정의(CAD 점군 중심)로 계산.
                     None이면 정점 평균(bank.cad_center_mm)을 쓴다.
    """
    if rr is None or not rr.ok:
        return Pose2D(False, f"형상 보정 실패({getattr(rr, 'reason', '') or '결과 없음'})")
    if rr.center_uv is None or rr.plane is None:
        return Pose2D(False, "형상 보정 결과에 템플릿 중심/평면 정보가 없습니다")
    fx, fy, cx, cy = (float(v) for v in intrinsics[:4])

    u, v = rr.center_uv
    u0, v0, pa, pb, pc = rr.plane
    z_surf = pa * (u - u0) + pb * (v - v0) + pc

    z_c = z_surf + _depth_offset(bank, float(rr.phi_deg), depth_correct)
    if not np.isfinite(z_c) or z_c <= 1.0:
        return Pose2D(False, f"depth 이상({z_c:.1f}mm)")
    t_cam = np.array([(u - cx) / fx * z_c, (v - cy) / fy * z_c, z_c])
    return _assemble(rr, bank, t_cam, z_surf, z_c, cad_axis_deg, cad_center_mm, source="plane")


def surface_point_from_pcd(pcd, valid, mask, u: float, v: float, win: int = PCD_WINDOW):
    """(u, v) 주변 win x win 창에서 (mask 안 + 유효 + 유한 + Z>1) 점의 중앙값 (X, Y, Z mm)과 점 개수.

    점이 PCD_MIN_POINTS개 미만이면 (None, 개수). pcd는 카메라 SDK가 만든 organized 점군이라
    X, Y에 렌즈 왜곡 보정이 이미 들어 있다 (핀홀 역투영을 거치지 않는다)."""
    h, w = valid.shape
    ui, vi = int(round(u)), int(round(v))
    r = max(1, int(win)) // 2
    x0, x1, y0, y1 = max(0, ui - r), min(w, ui + r + 1), max(0, vi - r), min(h, vi + r + 1)
    if x0 >= x1 or y0 >= y1:
        return None, 0
    pts = np.asarray(pcd[y0:y1, x0:x1], dtype=np.float64).reshape(-1, 3)
    ok = np.asarray(valid[y0:y1, x0:x1], dtype=bool).reshape(-1)   # 아래는 새 배열을 만드는 연산만 쓴다 (원본 valid 보호)
    if mask is not None:
        ok = ok & np.asarray(mask[y0:y1, x0:x1], dtype=bool).reshape(-1)
    ok = ok & np.isfinite(pts).all(axis=1) & (pts[:, 2] > 1.0)
    n = int(ok.sum())
    if n < PCD_MIN_POINTS:
        return None, n
    return np.median(pts[ok], axis=0), n


def pose_from_refine_pcd(rr, bank, intrinsics, pcd, valid, cad_axis_deg=(0.0, 0.0, 0.0),
                         cad_center_mm=None, depth_correct: bool = True, win: int = PCD_WINDOW) -> Pose2D:
    """pose_from_refine과 같은 입력 + 점군(pcd, valid)으로, 위치를 점군에서 직접 읽어 pose를 만든다.

    위치: 템플릿 중심 픽셀 (u, v) 주변 창의 점군 중앙값 = 표면 점 P_s -> 카메라 광선 방향으로 d(표면->중심) 이동
    회전/픽 포인트 계산은 pose_from_refine과 동일 (_assemble). 창 안에 유효 점이 부족하면 pose_from_refine으로 되돌아간다
    (source에 사유 표시). intrinsics는 그 폴백에만 쓰인다.
    """
    if rr is None or not rr.ok:
        return Pose2D(False, f"형상 보정 실패({getattr(rr, 'reason', '') or '결과 없음'})")
    if rr.center_uv is None:
        return Pose2D(False, "형상 보정 결과에 템플릿 중심 정보가 없습니다")

    u, v = rr.center_uv
    p_s, n = surface_point_from_pcd(pcd, valid, getattr(rr, "mask", None), u, v, win)
    if p_s is None:
        p = pose_from_refine(rr, bank, intrinsics, cad_axis_deg, cad_center_mm, depth_correct)
        if p.ok:
            p.source = f"plane(폴백: 창 안 유효 점 {n}개)"
        return p

    d = _depth_offset(bank, float(rr.phi_deg), depth_correct)
    t_cam = p_s + d * p_s / np.linalg.norm(p_s)
    z_c = float(t_cam[2])
    if not np.all(np.isfinite(t_cam)) or z_c <= 1.0:
        return Pose2D(False, f"depth 이상({z_c:.1f}mm)")
    return _assemble(rr, bank, t_cam, float(p_s[2]), z_c, cad_axis_deg, cad_center_mm, source=f"pcd(창 {n}점)")


def compare_methods(base: Pose2D, alt: Pose2D, uv, intrinsics, image_shape) -> dict | None:
    """같은 검출에 대한 두 위치 계산(base: 기존 / alt: pcd 직접 읽기)의 차이 (alt - base). 둘 중 하나라도 실패면 None.

    dx, dy, dz  : 픽 포인트 차이 (mm, 카메라 좌표). 회전이 같으므로 중심 위치 차이와 같다.
    d3_mm, dxy_mm : 3D 거리 / 광축에 수직인 면 안의 거리
    dz_surf_mm  : 표면 depth 차이 (평면 맞춤 Z vs 점군 읽은 Z)
    r_px, r_pct : 템플릿 중심이 주점(cx, cy)에서 떨어진 거리(px) / 가장 먼 모서리까지 거리 대비 % (왜곡은 가장자리일수록 큼)
    """
    if base is None or alt is None or not base.ok or not alt.ok:
        return None
    dp = np.asarray(alt.pick_point_mm, dtype=np.float64) - np.asarray(base.pick_point_mm, dtype=np.float64)
    cx, cy = float(intrinsics[2]), float(intrinsics[3])
    h, w = image_shape[:2]
    r_px = math.hypot(float(uv[0]) - cx, float(uv[1]) - cy)
    r_max = math.hypot(max(cx, w - cx), max(cy, h - cy))
    return {
        "dx": float(dp[0]), "dy": float(dp[1]), "dz": float(dp[2]),
        "d3_mm": float(np.linalg.norm(dp)), "dxy_mm": float(math.hypot(dp[0], dp[1])),
        "dz_surf_mm": float(alt.z_surface_mm - base.z_surface_mm),
        "r_px": r_px, "r_pct": 100.0 * r_px / r_max if r_max > 0 else 0.0,
    }


def compare_with_icp(p2d: Pose2D, icp_result, bank, cad_axis_deg=(0.0, 0.0, 0.0)) -> dict | None:
    """2D 기반 pose와 ICP 결과의 차이. 둘 중 하나라도 실패면 None.

    dpos_mm    : 픽 포인트 거리
    axis_deg   : 부품 긴 축 방향 차이 (방향 포함, 90도 넘으면 머리/꼬리가 반대)
    rot_deg    : 전체 회전 차이 (roll 포함 - roll은 2D에서 고정값이라 크게 나올 수 있음)
    dz_mm      : Z 차이 (2D - ICP)
    """
    if p2d is None or not p2d.ok or icp_result is None or not icp_result.ok or icp_result.T is None:
        return None
    R_ax = icp_runner._Rz(cad_axis_deg[2]) @ icp_runner._Ry(cad_axis_deg[1]) @ icp_runner._Rx(cad_axis_deg[0])
    e1 = np.asarray(bank.model_to_cad_R)[:, 0]
    axis_icp = np.asarray(icp_result.T)[:3, :3] @ R_ax @ e1
    cosv = float(np.clip(np.dot(axis_icp, p2d.axis_cam), -1.0, 1.0))
    dR = np.asarray(icp_result.T)[:3, :3].T @ p2d.T[:3, :3]
    cos_rot = float(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))
    dp = np.asarray(p2d.pick_point_mm) - np.asarray(icp_result.pick_point_mm)
    return {
        "dpos_mm": float(np.linalg.norm(dp)),
        "dz_mm": float(dp[2]),
        "axis_deg": math.degrees(math.acos(cosv)),
        "rot_deg": math.degrees(math.acos(cos_rot)),
    }
