"""2D 기반 픽 포인트 / 회전: 형상 보정(CAD 실루엣 템플릿 매칭)이 고른 템플릿에서 pose를 바로 구한다.

ICP 정합 없이, 형상 보정 단계에서 이미 결정된 값만 쓴다.

    위치   템플릿 중심 픽셀 (u, v)  +  그 자리 물체 평면의 depth  ->  카메라 좌표 (X, Y, Z) mm
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


def pose_from_refine(rr, bank, intrinsics, cad_axis_deg=(0.0, 0.0, 0.0),
                     cad_center_mm=None, depth_correct: bool = True) -> Pose2D:
    """RefineResult(rr) 하나에서 2D 기반 pose를 만든다.

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
    phi, theta = float(rr.phi_deg), float(rr.theta_deg)

    d = 0.0
    if depth_correct:
        # 모델 중심에서 카메라 쪽으로 광선을 따라 나가며 물체를 벗어나는 거리 (원통 근사: 반지름 r, 길이 L).
        # 옆면으로 나가는 거리 r/cos(phi), 끝면으로 나가는 거리 L/(2 sin|phi|) 중 작은 쪽.
        sp, cp = abs(math.sin(math.radians(phi))), abs(math.cos(math.radians(phi)))
        d_side = 0.5 * bank.width_mm / cp if cp > 1e-9 else float("inf")
        d_end = 0.5 * bank.length_mm / sp if sp > 1e-9 else float("inf")
        d = min(d_side, d_end)
    z_c = z_surf + d
    if not np.isfinite(z_c) or z_c <= 1.0:
        return Pose2D(False, f"depth 이상({z_c:.1f}mm)")
    t_cam = np.array([(u - cx) / fx * z_c, (v - cy) / fy * z_c, z_c])

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
                  axis_cam=R_model[:, 0].copy(), center_cam_mm=t_cam, z_surface_mm=float(z_surf),
                  z_center_mm=float(z_c), phi_deg=phi, theta_deg=theta)


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
