"""ICP 포즈 추정 (탭 4: ICP 정합 테스트).

FINE_RTMDet_EXE/scripts/bp_icp.py 를 이식했다. 원본과의 차이:

  - 원본은 인스턴스 포인트를 별도 .ply 파일로 저장했다가 다시 읽어서 ICP를
    돌리지만, 여기서는 탭3의 Detector가 이미 메모리에 들고 있는
    Detection.mask (H,W bool) 를 세션 폴더의 pointcloud_organized/valid_mask
    npy와 바로 조합해서 쓴다 (파일 왕복 없음).
  - CAD가 브라켓 하나로 고정돼 있던 원본과 달리 여기서는 data/cad/ 폴더의
    임의 CAD를 드롭다운으로 고르므로, 원본의 CAD_PICK_LOCAL(그리퍼 픽 오프셋
    하드코딩 좌표)과 2D 마스크 블렌딩/높이포인트 보정은 포함하지 않는다.
    픽 포인트는 "CAD 정합 결과의 중심점"으로 단순화했다 - 실제 그리퍼 픽
    오프셋이 필요해지면 이 파일의 run_icp_for_instance() 안 pick_point_mm
    계산만 확장하면 된다.

튜닝 파라미터는 ICPParams 데이터클래스로 묶여있다. 모듈 상수는 기본값일
뿐이고, 실제 값은 항상 ICPParams 인스턴스를 통해 함수로 전달된다 -
탭4 UI의 스핀박스/슬라이더가 실행마다 다른 ICPParams를 만들어 넘길 수
있게 하기 위해서다 (CAD_AXIS_CORRECTION_DEG, CAD_SAMPLE_POINTS는 CAD 로드
시 1회만 쓰이는 값이라 UI 노출 대상에서 제외했다).

--------------------------------------------------------------------------
2026-07 패치: 3D 정합 정확도 개선 4종 반영
  [1] CAD 가시면(hidden point removal)만 정합에 사용
      - 카메라는 부분(partial) 뷰만 보는데 기존엔 CAD 전체 겉면을 정합
        대상으로 써서 중심(centroid)이 구조적으로 어긋나 있었음.
      - 서버/탭 기동 중 CAD·축보정·초기자세·기준거리가 바뀔 때만 재계산
        (build_visible_cad, ICPTestTab._ensure_cad_loaded에서 캐시).
  [2] Point-to-Point -> Point-to-Plane ICP
      - Open3D 공식 튜토리얼 기준 point-to-plane이 더 빠르고 타이트하게
        수렴 (Rusinkiewicz 2001). 각 stage에서 normal을 추정해서 사용.
  [4] 마스크 침식 (mask erosion)
      - depth 카메라 특성상 마스크 경계 픽셀은 배경/전경 depth가 섞인
        flying pixel일 확률이 높음. extract_instance_points_mm에서 침식
        적용, 포인트가 너무 적어지면 원본 마스크로 자동 폴백.
  [5] 마지막 ICP stage는 원본(다운샘플 이전) 밀도로 정밀 정합
      - 기존엔 고정 voxel(voxel_size_scene/cad)로만 정합해서 정밀도 상한이
        그 voxel 크기 근처였음. icp_stages에 coarse->fine voxel을 단계적으로
        넣고, 마지막 stage는 voxel=None(outlier 제거 후 원본 밀도)으로 정합.
      - 이에 따라 예전의 단일 voxel_size_cad/voxel_size_scene 필드와
        downsample_cad()는 더 이상 쓰이지 않아 제거했다. voxel 크기는 이제
        icp_stages 리스트의 각 stage 딕셔너리 안 "voxel" 키로 지정한다.
--------------------------------------------------------------------------
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import cv2
import numpy as np
import open3d as o3d

# =============================================================================
# CAD 로드 전용 상수
# =============================================================================
CAD_SAMPLE_POINTS = 20000

MIN_POINTS_PER_INSTANCE = 50


# =============================================================================
# ICP 조정 파라미터 (탭4 UI에서 값을 채워 넘긴다)
# =============================================================================
@dataclass
class ICPParams:
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 1.5

    # [5] coarse -> fine 다단계. voxel=None인 stage는 outlier 제거 후
    # "원본 밀도" 그대로 정합한다 (마지막 stage는 항상 None 권장).
    icp_stages: list[dict] = field(default_factory=lambda: [
        {"voxel": 0.006, "max_dist": 0.020, "max_iter": 100},
        {"voxel": 0.003, "max_dist": 0.010, "max_iter": 100},
        {"voxel": None,  "max_dist": 0.003, "max_iter": 50},
    ])

    # [2] point-to-plane normal 추정 반경. voxel 있는 stage는 voxel*factor,
    # voxel=None(원본 밀도) stage는 normal_radius_final을 그대로 쓴다.
    normal_radius_factor: float = 2.5
    normal_radius_final: float = 0.004   # m

    fitness_threshold: float = 0.7
    xyz_max_m: float = 2.0

    # [4] 마스크 침식 픽셀 반경. depth flying pixel(경계 노이즈) 완충용.
    # 침식 후 포인트가 MIN_POINTS_PER_INSTANCE 밑으로 떨어지면 자동으로
    # 원본(침식 전) 마스크로 폴백한다.
    mask_erode_px: int = 1

    # [1] CAD 가시면 계산 파라미터.
    # ref_distance_m: 카메라~부품 대략적인 작업 거리(m). 이 거리에 CAD를
    #   놓고 카메라(원점)에서 봤을 때 보이는 면만 정합에 사용한다.
    #   실제 세팅 거리로 맞출수록 가시면 판정이 정확해진다.
    # radius_factor: Katz2007 hidden_point_removal의 radius = CAD 대각선 * 이 값.
    #   값이 클수록 더 많은 면을 "보인다"고 관대하게 판정한다.
    cad_hpr_ref_distance_m: float = 0.6
    cad_hpr_radius_factor: float = 100.0

    # CAD 로드 시 축 보정 (deg). ICP의 T_init은 두 점군의 중심만 맞추고
    # 회전은 넣지 않기 때문에(아래 build_icp_init 참고), CAD가 로드 직후
    # 실제 물체가 카메라에 놓인 방향과 어긋나 있으면 ICP가 엉뚱한 국소
    # 최적점으로 수렴한다. 물체/CAD가 바뀌면 반드시 다시 맞춰야 하는 값이라
    # 참고용 기본값을 0으로 두지 않고, 브라켓 CAD로 확인됐던 값을 기본값으로
    # 둔다 - 다른 CAD를 쓰면 이 값부터 조정해볼 것.
    cad_axis_roll_deg: float = -90.0
    cad_axis_pitch_deg: float = 90.0
    cad_axis_yaw_deg: float = 90.0

    # ICP 초기 자세 고정값 (deg) - 관찰 후 조정.
    # [1] CAD 가시면 계산에도 이 값이 그대로 쓰인다 (부품이 대략 이 자세로
    # 카메라를 보고 있다고 가정하고 어느 면이 보이는지 판단하기 때문).
    init_roll_deg: float = 0.0
    init_pitch_deg: float = 0.0
    init_yaw_deg: float = 0.0

    # 회전 구속 조건 (deg). 대칭 범위(±roll_limit_deg 등)로 다룬다.
    roll_limit_deg: float = 45.0
    pitch_limit_deg: float = 45.0
    yaw_limit_deg: float = 45.0

    @property
    def roll_range(self) -> tuple[float, float]:
        return (-self.roll_limit_deg, self.roll_limit_deg)

    @property
    def pitch_range(self) -> tuple[float, float]:
        return (-self.pitch_limit_deg, self.pitch_limit_deg)

    @property
    def yaw_range(self) -> tuple[float, float]:
        return (-self.yaw_limit_deg, self.yaw_limit_deg)

    @property
    def cad_axis_correction_deg(self) -> tuple[float, float, float]:
        return (self.cad_axis_roll_deg, self.cad_axis_pitch_deg, self.cad_axis_yaw_deg)

    @property
    def init_rotation_deg(self) -> tuple[float, float, float]:
        return (self.init_roll_deg, self.init_pitch_deg, self.init_yaw_deg)


def default_params() -> ICPParams:
    return ICPParams()


# =============================================================================
# 회전 행렬 헬퍼
# =============================================================================
def _Rx(d):
    c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
    R = np.eye(3); R[1, 1] = c; R[1, 2] = -s; R[2, 1] = s; R[2, 2] = c
    return R

def _Ry(d):
    c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
    R = np.eye(3); R[0, 0] = c; R[0, 2] = s; R[2, 0] = -s; R[2, 2] = c
    return R

def _Rz(d):
    c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
    R = np.eye(3); R[0, 0] = c; R[0, 1] = -s; R[1, 0] = s; R[1, 1] = c
    return R


# =============================================================================
# CAD 로드
# =============================================================================
def load_cad_as_pcd(cad_path, params: ICPParams | None = None) -> o3d.geometry.PointCloud:
    """STL/PLY/OBJ 메쉬를 로드해서 균일 샘플링한 포인트클라우드로 반환한다.
    바운딩박스가 10을 넘으면(=mm 단위로 저장된 CAD) 자동으로 m 단위로 스케일한다.
    params.cad_axis_correction_deg로 축 보정을 적용한다 (CAD/설치 상태마다 다시 맞춰야 함)."""
    p = params if params is not None else default_params()

    mesh = o3d.io.read_triangle_mesh(str(cad_path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"CAD를 읽을 수 없거나 정점이 없습니다: {cad_path}")

    ext = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    if ext.max() > 10.0:
        mesh.scale(1.0 / 1000.0, center=np.zeros(3))

    rx, ry, rz = p.cad_axis_correction_deg
    R = _Rz(rz) @ _Ry(ry) @ _Rx(rx)
    center = np.asarray(mesh.get_center())
    T_fix = np.eye(4); T_fix[:3, :3] = R; T_fix[:3, 3] = center - R @ center
    mesh.transform(T_fix)

    mesh.compute_vertex_normals()
    return mesh.sample_points_poisson_disk(CAD_SAMPLE_POINTS)


# =============================================================================
# [1] CAD 가시면 필터링 (hidden point removal)
# =============================================================================
def build_visible_cad(cad_pcd_full: o3d.geometry.PointCloud, R_fixed: np.ndarray,
                       params: ICPParams) -> o3d.geometry.PointCloud:
    """카메라(원점)에서 R_fixed 자세로, params.cad_hpr_ref_distance_m 만큼
    떨어진 곳에 CAD를 놓았을 때 "보이는 면"만 남긴 CAD 서브셋을 원본(미변환)
    좌표계로 반환한다.

    Katz2007 hidden_point_removal 사용 (표면 재구성/노멀 추정 없이 시점 기준
    가시성을 근사). 부품 자세 변동이 크지 않다는 전제 하에 CAD/축보정/초기
    자세/기준거리가 바뀔 때만 재계산해서 재사용하면 된다 (호출부인
    ICPTestTab._ensure_cad_loaded 에서 캐싱).
    """
    placed = copy.deepcopy(cad_pcd_full)
    T = np.eye(4)
    T[:3, :3] = R_fixed
    T[2, 3] = params.cad_hpr_ref_distance_m
    placed.transform(T)

    diameter = np.linalg.norm(
        np.asarray(placed.get_max_bound()) - np.asarray(placed.get_min_bound()))
    radius = diameter * params.cad_hpr_radius_factor
    _, pt_map = placed.hidden_point_removal([0.0, 0.0, 0.0], radius)

    return cad_pcd_full.select_by_index(pt_map)


def build_visible_cad_pair(cad_pcd_full: o3d.geometry.PointCloud,
                            params: ICPParams) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """정자세 가시면 + 뒤집힌 자세 가시면을 한 번에 만든다.
    (correct_flipped_pose에서 뒤집힘이 감지되면 후자를 재정합 소스로 쓴다.)"""
    R_fixed = _Rz(params.init_yaw_deg) @ _Ry(params.init_pitch_deg) @ _Rx(params.init_roll_deg)
    R_flip = np.diag([-1.0, -1.0, 1.0]) @ R_fixed
    cad_visible_normal = build_visible_cad(cad_pcd_full, R_fixed, params)
    cad_visible_flipped = build_visible_cad(cad_pcd_full, R_flip, params)
    return cad_visible_normal, cad_visible_flipped


# =============================================================================
# ICP 초기값 + 구속 검사
# =============================================================================
def build_icp_init(scene_source, cad_visible, params: ICPParams) -> np.ndarray:
    R_init = _Rz(params.init_yaw_deg) @ _Ry(params.init_pitch_deg) @ _Rx(params.init_roll_deg)
    sc_center = np.asarray(scene_source.get_center())
    cd_center = np.asarray(cad_visible.get_center())
    T_init = np.eye(4)
    T_init[:3, :3] = R_init
    T_init[:3, 3] = sc_center - R_init @ cd_center
    return T_init


def check_rotation_constraint(T: np.ndarray, params: ICPParams) -> tuple[bool, str]:
    R = T[:3, :3]
    pitch = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)))
    cp = np.cos(np.radians(pitch))
    if abs(cp) > 1e-6:
        roll = np.degrees(np.arctan2(R[2, 1] / cp, R[2, 2] / cp))
        yaw = np.degrees(np.arctan2(R[1, 0] / cp, R[0, 0] / cp))
    else:
        roll, yaw = 0.0, np.degrees(np.arctan2(-R[0, 1], R[1, 1]))

    roll_lo, roll_hi = params.roll_range
    pitch_lo, pitch_hi = params.pitch_range
    yaw_lo, yaw_hi = params.yaw_range

    violations = []
    if not (roll_lo <= roll <= roll_hi):
        violations.append(f"roll={roll:.1f}° (허용 [{roll_lo}, {roll_hi}])")
    if not (pitch_lo <= pitch <= pitch_hi):
        violations.append(f"pitch={pitch:.1f}° (허용 [{pitch_lo}, {pitch_hi}])")
    if not (yaw_lo <= yaw <= yaw_hi):
        violations.append(f"yaw={yaw:.1f}° (허용 [{yaw_lo}, {yaw_hi}])")

    if violations:
        return False, "회전 구속 위반: " + ", ".join(violations)
    return True, f"roll={roll:.1f}° pitch={pitch:.1f}° yaw={yaw:.1f}°"


# =============================================================================
# [2]+[5] Point-to-Plane multi-resolution ICP
# =============================================================================
def _prep_stage_cloud(base_pcd: o3d.geometry.PointCloud, voxel: float | None,
                       normal_radius: float) -> o3d.geometry.PointCloud:
    cloud = base_pcd.voxel_down_sample(voxel) if voxel is not None else copy.deepcopy(base_pcd)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    cloud.orient_normals_towards_camera_location([0.0, 0.0, 0.0])
    return cloud


def run_icp_multistage(cad_source, scene_source, T_init: np.ndarray, params: ICPParams):
    """[2]+[5] coarse-to-fine ICP.
    cad_source/scene_source는 voxel 다운샘플 전(outlier 제거 후) 원본 밀도로 넘긴다 -
    각 stage에서 params.icp_stages[i]["voxel"] 기준으로 알아서 리샘플한다.

    추정 방식: voxel이 있는(coarse/mid) stage는 point-to-point를 쓴다 - 초기
    정렬이 많이 틀어져 있을 수 있는 초반 단계에서 point-to-plane은 선형화
    근사 때문에 발산하기 쉽다. voxel=None인 마지막(fine) stage에서만
    point-to-plane으로 정밀화한다 (이미 대략 맞춰진 상태라 안전함).

    반환값에 stage_logs(각 stage의 n_src/n_tgt/fitness/rmse)를 포함해서
    fitness=0.000처럼 완전히 실패하는 경우 어느 단계에서 무너졌는지 바로
    확인할 수 있게 했다."""
    T = T_init.copy()
    stage_logs: list[dict] = []
    for i, stage in enumerate(params.icp_stages):
        voxel = stage.get("voxel")
        n_radius = (voxel * params.normal_radius_factor) if voxel is not None else params.normal_radius_final
        src = _prep_stage_cloud(cad_source, voxel, n_radius)
        tgt = _prep_stage_cloud(scene_source, voxel, n_radius)

        estimation = (
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
            if voxel is None else
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        res = o3d.pipelines.registration.registration_icp(
            src, tgt, stage["max_dist"], T, estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=stage["max_iter"]),
        )
        T = np.asarray(res.transformation)
        stage_logs.append({
            "stage": i, "voxel": voxel, "max_dist": stage["max_dist"],
            "n_src": len(src.points), "n_tgt": len(tgt.points),
            "fitness": float(res.fitness), "rmse": float(res.inlier_rmse),
            "method": "point-to-plane" if voxel is None else "point-to-point",
        })

    final = o3d.pipelines.registration.evaluate_registration(
        cad_source, scene_source, params.icp_stages[-1]["max_dist"], T)
    return T, float(final.fitness), float(final.inlier_rmse), stage_logs


def correct_flipped_pose(T, cad_normal, cad_flipped, scene_source, params: ICPParams):
    """뒤집힘 감지 시 [1]에서 미리 만들어둔 '뒤집힌 자세 기준 가시면 CAD'로
    재정합한다 (가시면 필터링 도입 이후로는 정자세 가시면 그대로 뒤집으면
    보이는 면 자체가 안 맞으므로, 뒤집힌 자세용 가시면을 별도로 써야 한다)."""
    if T[:3, :3][2, 2] >= 0:
        final = o3d.pipelines.registration.evaluate_registration(
            cad_normal, scene_source, params.icp_stages[-1]["max_dist"], T)
        return T, float(final.fitness), float(final.inlier_rmse), False, []
    R_flip = np.diag([-1.0, -1.0, 1.0])
    T_flip = np.eye(4); T_flip[:3, :3] = R_flip
    c = T[:3, 3]; T_flip[:3, 3] = c - R_flip @ c
    T_f, fit, rmse, stage_logs = run_icp_multistage(cad_flipped, scene_source, T_flip @ T, params)
    return T_f, fit, rmse, True, stage_logs


def transform_to_pose(T: np.ndarray) -> dict:
    xyz_mm = (T[:3, 3] * 1000.0).tolist()
    R = T[:3, :3]
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    cp = np.cos(pitch)
    if abs(cp) > 1e-6:
        roll = np.arctan2(R[2, 1] / cp, R[2, 2] / cp)
        yaw = np.arctan2(R[1, 0] / cp, R[0, 0] / cp)
    else:
        roll, yaw = 0.0, np.arctan2(-R[0, 1], R[1, 1])
    e = np.degrees([roll, pitch, yaw]).tolist()
    return {
        "xyz_mm": [round(v, 3) for v in xyz_mm],
        "euler_deg": {"roll_deg": round(e[0], 4), "pitch_deg": round(e[1], 4), "yaw_deg": round(e[2], 4)},
        "transform_matrix": T.tolist(),
    }


# =============================================================================
# [4] 인스턴스 포인트 추출 (마스크 x pointcloud_organized/valid_mask) + 마스크 침식
# =============================================================================
def _erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    kernel = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def extract_instance_points_mm(mask: np.ndarray, pcd_organized: np.ndarray, valid_mask: np.ndarray,
                                erode_px: int = 0, min_points: int = MIN_POINTS_PER_INSTANCE) -> np.ndarray:
    """(H,W) bool 검출 마스크와 세션의 pointcloud_organized(H,W,3 mm)/valid_mask를
    조합해서 이 인스턴스에 해당하는 3D 포인트(mm, N x 3)를 뽑는다.

    [4] erode_px > 0 이면 depth flying pixel(경계 노이즈) 완충을 위해 마스크를
    먼저 침식한다. 침식 후 포인트 수가 min_points 밑으로 떨어지면 침식 없는
    원본 마스크로 자동 폴백한다 (작은 부품/원거리 촬영 시 안전장치)."""
    m = mask.astype(bool)
    vmask = valid_mask.astype(bool)

    if erode_px > 0:
        eroded = _erode_mask(m, erode_px)
        combined = eroded & vmask
        pts = pcd_organized[combined].astype(np.float64)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) >= min_points:
            return pts
        # 폴백: 침식 없이 원본 마스크로 재시도

    combined = m & vmask
    pts = pcd_organized[combined].astype(np.float64)
    return pts[np.isfinite(pts).all(axis=1)]


# =============================================================================
# 결과 데이터클래스
# =============================================================================
@dataclass
class ICPResult:
    instance_id: int
    ok: bool
    error: str | None = None
    fitness: float | None = None
    rmse_m: float | None = None
    was_flipped: bool = False
    num_points_scene: int = 0
    num_points_after_outlier: int = 0
    pose: dict | None = None
    pick_point_mm: list[float] | None = None
    T: np.ndarray | None = None
    scene_pcd: o3d.geometry.PointCloud | None = None
    stage_logs: list[dict] | None = None


# =============================================================================
# 인스턴스 한 개에 대한 ICP 정합
# =============================================================================
def run_icp_for_instance(instance_id: int, pts_mm: np.ndarray, cad_pcd,
                          cad_visible_normal, cad_visible_flipped,
                          params: ICPParams | None = None) -> ICPResult:
    """params를 넘기지 않으면 default_params()를 쓴다.

    cad_pcd            : 축보정만 적용된 CAD 전체 (pick point 중심 계산 / 3D 시각화용)
    cad_visible_normal  : [1] 정자세 기준 가시면 서브셋 (정합 소스)
    cad_visible_flipped : [1] 뒤집힌 자세 기준 가시면 서브셋 (뒤집힘 감지 시 재정합용)
    """
    p = params if params is not None else default_params()

    n_pts = len(pts_mm)
    if n_pts < MIN_POINTS_PER_INSTANCE:
        return ICPResult(instance_id=instance_id, ok=False,
                          error=f"포인트 부족: {n_pts}개", num_points_scene=n_pts)

    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(pts_mm / 1000.0)

    sc, _ = scene_pcd.remove_statistical_outlier(p.outlier_nb_neighbors, p.outlier_std_ratio)
    n_after = len(np.asarray(sc.points))
    if n_after < 10:
        return ICPResult(instance_id=instance_id, ok=False,
                          error="outlier 제거 후 포인트 부족", num_points_scene=n_pts,
                          num_points_after_outlier=n_after)

    T_init = build_icp_init(sc, cad_visible_normal, p)
    T, fit, rmse, stage_logs = run_icp_multistage(cad_visible_normal, sc, T_init, p)
    T, fit, rmse, flipped, flip_stage_logs = correct_flipped_pose(T, cad_visible_normal, cad_visible_flipped, sc, p)
    if flipped:
        stage_logs = flip_stage_logs

    if fit < p.fitness_threshold:
        return ICPResult(instance_id=instance_id, ok=False,
                          error=f"ICP 정합 실패 (fitness={fit:.3f} < {p.fitness_threshold:.2f})",
                          fitness=fit, rmse_m=rmse, was_flipped=flipped,
                          num_points_scene=n_pts, num_points_after_outlier=n_after, scene_pcd=sc,
                          stage_logs=stage_logs)

    if max(abs(v) for v in T[:3, 3]) > p.xyz_max_m:
        return ICPResult(instance_id=instance_id, ok=False,
                          error=f"xyz 범위 이상 (허용 ±{p.xyz_max_m:.2f}m)",
                          fitness=fit, rmse_m=rmse, was_flipped=flipped,
                          num_points_scene=n_pts, num_points_after_outlier=n_after, scene_pcd=sc,
                          stage_logs=stage_logs)

    rot_ok, rot_msg = check_rotation_constraint(T, p)
    if not rot_ok:
        return ICPResult(instance_id=instance_id, ok=False, error=rot_msg,
                          fitness=fit, rmse_m=rmse, was_flipped=flipped,
                          num_points_scene=n_pts, num_points_after_outlier=n_after, scene_pcd=sc,
                          stage_logs=stage_logs)

    pose = transform_to_pose(T)
    cad_center_m = np.asarray(cad_pcd.get_center())
    pick_point_mm = ((T[:3, :3] @ cad_center_m + T[:3, 3]) * 1000.0).tolist()

    return ICPResult(instance_id=instance_id, ok=True, fitness=fit, rmse_m=rmse, was_flipped=flipped,
                      num_points_scene=n_pts, num_points_after_outlier=n_after,
                      pose=pose, pick_point_mm=pick_point_mm, T=T, scene_pcd=sc,
                      stage_logs=stage_logs)


# =============================================================================
# 시각화용 통합 포인트클라우드 (3D 뷰어 창에 그대로 넘길 수 있음)
# =============================================================================
_BG_COLOR = np.array([0.55, 0.55, 0.55], dtype=np.float64)          # 전체 배경: 회색
_INSTANCE_COLOR = np.array([0.9, 0.15, 0.1], dtype=np.float64)      # 마스킹된 인스턴스 포인트: 빨강
_CAD_COLOR = np.array([0.15, 0.85, 0.25], dtype=np.float64)         # 정합된 CAD: 초록
_PICK_COLOR = np.array([1.0, 0.9, 0.05], dtype=np.float64)          # 픽포인트 마커: 노랑 (인스턴스 빨강과 구분)


def build_background_pcd(pcd_organized: np.ndarray, valid_mask: np.ndarray,
                          exclude_mask: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    """세션의 pointcloud_organized(mm)/valid_mask 전체를 회색 배경 포인트클라우드로 만든다.
    exclude_mask(검출된 인스턴스 마스크 합집합)를 주면 그 영역은 배경에서 빼서,
    같은 위치에 인스턴스 색과 배경 회색이 겹쳐 지저분해지는 걸 막는다."""
    valid = valid_mask.astype(bool)
    if exclude_mask is not None:
        valid = valid & ~exclude_mask.astype(bool)
    pts = pcd_organized[valid].astype(np.float64)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]

    pcd = o3d.geometry.PointCloud()
    if len(pts) == 0:
        return pcd
    pcd.points = o3d.utility.Vector3dVector(pts / 1000.0)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(_BG_COLOR, (len(pts), 1)))
    return pcd


def build_scene_geometry(results: list[ICPResult], cad_pcd,
                          background_pcd: o3d.geometry.PointCloud | None = None) -> o3d.geometry.PointCloud:
    """회색 전체 배경 + 인스턴스별 검출 포인트(빨강) + 정합된 CAD(초록) + 픽포인트(노랑 구)를 하나로 합친다.
    시각화는 [1]의 가시면 서브셋이 아니라 항상 CAD 전체(cad_pcd)를 T로 변환해서 보여준다
    (정합 계산에만 가시면 서브셋을 쓰고, 눈으로 확인할 땐 CAD 전체가 자연스럽다)."""
    combined = o3d.geometry.PointCloud()
    if background_pcd is not None and len(background_pcd.points) > 0:
        combined += background_pcd

    for r in results:
        if not r.ok or r.scene_pcd is None or r.T is None:
            continue
        sv = copy.deepcopy(r.scene_pcd)
        sv.colors = o3d.utility.Vector3dVector(np.tile(_INSTANCE_COLOR, (len(sv.points), 1)))

        cv = copy.deepcopy(cad_pcd)
        cv.transform(r.T)
        cv.colors = o3d.utility.Vector3dVector(np.tile(_CAD_COLOR, (len(cv.points), 1)))

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
        sphere.translate(np.array(r.pick_point_mm) / 1000.0)
        sphere.paint_uniform_color(_PICK_COLOR.tolist())
        sphere_pcd = sphere.sample_points_uniformly(400)

        combined += sv + cv + sphere_pcd
    return combined