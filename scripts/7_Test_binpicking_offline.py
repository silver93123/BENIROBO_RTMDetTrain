"""오프라인 빈피킹 정합 테스트 스크립트

카메라/TCP서버/GUI 없이, 이미 저장된 캡처 파일을 읽어
Detection -> ICP 파이프라인만 재현/디버깅하기 위한 배치 스크립트입니다.
(원본 실전 스크립트: 6_Run_binpicking_TCP_UI.py 와 동일한 캡처 폴더 구조를
 그대로 읽습니다: intensity/, pointcloud_organized/, valid_mask/, metadata/)

이번 패치에서 반영한 개선사항 (3D 정합 정확도 저하 대응):
  [1] CAD 가시면(hidden point removal)만 사용
      - 카메라는 항상 부분(partial) 뷰만 보는데, 기존엔 CAD 전체 겉면을
        정합 대상으로 썼음 -> 중심(centroid) 자체가 어긋나 있어 ICP가
        구조적으로 불리한 상태에서 출발했음.
      - 부품 자세 변동이 크지 않다는 전제 하에, 서버 기동 시 1회만
        고정 자세(ICP_INIT_*_DEG) 기준으로 보이는 면만 추려서 재사용.
  [2] Point-to-Point -> Point-to-Plane ICP
      - Open3D 공식 문서 기준으로 point-to-plane이 더 빠르고 타이트하게
        수렴함 (Rusinkiewicz 2001). 각 stage에서 normal을 추정해서 사용.
  [4] 마스크 침식 (mask erosion)
      - depth 카메라 특성상 마스크 경계 픽셀은 배경/전경 depth가 섞인
        flying pixel일 확률이 높음. 1~2px 침식 후 포인트를 뽑고,
        포인트 수가 너무 적어지면 원본 마스크로 폴백.
  [5] 마지막 ICP stage는 원본(다운샘플 이전) 밀도로 정밀 정합
      - 기존엔 3mm 다운샘플 포인트로만 정합 -> 정밀도 상한이 3mm 근처.
      - coarse-to-fine: voxel을 단계적으로 줄이다가 마지막 단계는
        outlier 제거 후 원본 밀도를 그대로 사용.

미반영: PPF/Global Registration (자세 변동이 크지 않다는 전제 하에 우선순위
       낮춤. 필요해지면 build_icp_init() 자리에 추가하면 됨).

사용법:
  cd BENIROBO_RTMDetTrain
  python scripts/7_Test_binpicking_offline.py \
      --capture-dir data/captures/live \
      --cad         data/cad/bracket_v2.stl \
      --config      configs/rtmdet-ins_bracket.py \
      --checkpoint  work_dirs/rtmdet-ins_bracket_v1/best_xxx.pth \
      --out         data/test_results/offline_run1

  특정 프레임 1개만: --frame frame_0001
  (intensity/ 안 파일명에서 확장자만 뺀 것. 예: frame_0001.png -> frame_0001)

입력 폴더 구조 (--capture-dir 는 세션 폴더 1개를 가리켜야 함. collect_dataset.py 저장 규칙과 동일):
  <capture-dir>/intensity/frame_0001.png
  <capture-dir>/pointcloud_organized/frame_0001.npy   (H,W,3) float32 mm, invalid=NaN
  <capture-dir>/valid_mask/frame_0001.npy             (H,W) bool
  예) --capture-dir data/dataset/20260716_165133
  (data/dataset 자체가 아니라 그 아래 세션 폴더 하나!)

출력 (--out 아래):
  results/<ts>_overlay.png       RTMDet 오버레이
  results/<ts>_colored.ply       정합 결과 통합 PLY (배경+씬+CAD+픽포인트)
  results/<ts>_result.json       인스턴스별 pose/pick 상세
  pick_log.csv                   프레임 전체 누적 로그 (CSV)
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("ERROR: open3d 필요. pip install open3d", flush=True)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection import RTMDetInferencer  # noqa: E402

# =============================================================================
# ── 여기 값만 바꿔서 인자 없이 바로 실행 ────────────────────────────────────
# python scripts/7_Test_binpicking_offline.py 만 쳐도 아래 값으로 동작함.
# 필요하면 실행할 때 --capture-dir 등으로 여전히 덮어쓸 수 있음(선택사항).
# =============================================================================
DEFAULT_CAPTURE_DIR = ROOT / "data" / "dataset" / "20260605_095934"   # 세션 폴더 1개!
DEFAULT_CAD_PATH    = ROOT / "data" / "cad" / "bracket_v2.stl"
DEFAULT_CONFIG_PATH = ROOT / "configs" / "rtmdet-ins_bracket.py"
DEFAULT_CHECKPOINT  = ROOT / "work_dirs" / "rtmdet-ins_bracket_v1" / "best_coco_bbox_mAP_epoch_50.pth"
DEFAULT_OUT_DIR     = ROOT / "data" / "test_results" / "offline_run1"
DEFAULT_FRAME       = "frame_0039"          # 특정 프레임 1개만: "frame_0001"
DEFAULT_SCORE_THRESHOLD = 0.3
DEFAULT_DEVICE      = "cuda:0"

# =============================================================================
# 설정 (기본값. CLI 인자로 override 가능한 것들은 parse_args 참고)
# =============================================================================
MIN_POINTS_PER_INSTANCE = 100
MASK_IOU_THRESHOLD       = 0.65
MASK_ERODE_PX             = 1        # [4] 마스크 침식 픽셀 반경. 0이면 침식 비활성.

OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO    = 1.5

# [5] coarse-to-fine ICP stage. voxel=None인 마지막 stage는
#     outlier 제거된 "원본 밀도" 그대로 사용 (다운샘플 안 함).
ICP_STAGES = [
    {"voxel": 0.006, "max_dist": 0.020, "max_iter": 100},
    {"voxel": 0.003, "max_dist": 0.010, "max_iter": 100},
    {"voxel": None,  "max_dist": 0.003, "max_iter": 50},
]
NORMAL_RADIUS_FACTOR = 2.5   # stage voxel 대비 normal 추정 반경 배수
NORMAL_RADIUS_FINAL  = 0.004 # voxel=None 단계에서 쓸 normal 추정 반경 (m)
NORMAL_MAX_NN         = 30

ICP_FITNESS_THRESHOLD   = 0.7
XYZ_MAX_M               = 2.0
CAD_AXIS_CORRECTION_DEG = (0, 90, 90)
CAD_SAMPLE_POINTS       = 20000

# [1] CAD 가시면 필터링 파라미터
CAD_HPR_RADIUS_FACTOR = 100.0   # Katz2007 hidden_point_removal의 radius = diameter * 이 값
CAD_HPR_REF_DISTANCE_M = 0.6    # 카메라~부품 대략적인 작업 거리(m). 이 거리에 놓고 가시성 판단.
                                 # 광축 근처에서 몇 cm 벗어나도 가시 패턴은 거의 안 바뀜.

# ── 고정 초기 자세 (기존과 동일한 의미) ──────────────────────────────────────
ICP_INIT_ROLL_DEG  = 0.0
ICP_INIT_PITCH_DEG = 0.0
ICP_INIT_YAW_DEG   = 0.0

ICP_ROLL_RANGE  = (-45.0, 45.0)
ICP_PITCH_RANGE = (-45.0, 45.0)
ICP_YAW_RANGE   = (-45.0, 45.0)

# ── 픽포인트 ──────────────────────────────────────────────────────────────
CAD_PICK_LOCAL   = np.array([0.000, -0.100, 0.031, 1.0])
PICK_OFFSET_X_MM = 5.0
PICK_OFFSET_Y_MM = 2.0
PICK_OFFSET_Z_MM = 0.0
PICK_2D_MASK_WEIGHT = 0.7
PICK_Z_NEIGHBOR_OFFSETS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
HEIGHT_POINT_OFFSET_Y_MM = 15.0

_PALETTE_BGR = np.array([
    [50, 50, 255], [50, 200, 50], [255, 100, 50],
    [30, 180, 255], [230, 50, 180], [200, 200, 30],
], dtype=np.uint8)
_PALETTE_RGB_FLOAT = _PALETTE_BGR[:, ::-1].astype(np.float64) / 255.0
_BG_COLOR = np.array([0.55, 0.55, 0.55], dtype=np.float64)


def log(msg: str) -> None:
    print(msg, flush=True)


# =============================================================================
# 회전 유틸
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
# CAD 로드 + [1] 가시면 필터링
# =============================================================================
def load_cad_as_pcd(cad_path: Path) -> o3d.geometry.PointCloud:
    mesh = o3d.io.read_triangle_mesh(str(cad_path))
    ext = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    if ext.max() > 10.0:
        mesh.scale(1.0 / 1000.0, center=np.zeros(3))
    rx, ry, rz = CAD_AXIS_CORRECTION_DEG
    R = _Rz(rz) @ _Ry(ry) @ _Rx(rx)
    center = np.asarray(mesh.get_center())
    T_fix = np.eye(4); T_fix[:3, :3] = R; T_fix[:3, 3] = center - R @ center
    mesh.transform(T_fix)
    return mesh.sample_points_poisson_disk(CAD_SAMPLE_POINTS)


def build_visible_cad(cad_pcd_full: o3d.geometry.PointCloud,
                       R_fixed: np.ndarray,
                       ref_distance_m: float = CAD_HPR_REF_DISTANCE_M,
                       radius_factor: float = CAD_HPR_RADIUS_FACTOR) -> o3d.geometry.PointCloud:
    """[1] 카메라(원점)에서 R_fixed 자세로, ref_distance_m 만큼 떨어진 곳에
    CAD를 놓았을 때 "보이는 면"만 남긴 CAD 서브셋을 원본(미변환) 좌표계로 반환.

    Katz2007 hidden_point_removal 사용 (표면 재구성/노멀 추정 없이 시점 기준
    가시성을 근사). 부품 자세 변동이 크지 않다는 전제 하에 1회만 계산해서
    프레임마다 재사용한다 (자세가 크게 바뀌면 다시 계산해야 함).
    """
    placed = copy.deepcopy(cad_pcd_full)
    T = np.eye(4)
    T[:3, :3] = R_fixed
    T[2, 3] = ref_distance_m
    placed.transform(T)

    diameter = np.linalg.norm(
        np.asarray(placed.get_max_bound()) - np.asarray(placed.get_min_bound()))
    radius = diameter * radius_factor
    _, pt_map = placed.hidden_point_removal([0.0, 0.0, 0.0], radius)

    # 원본(회전/이동 적용 전) CAD에서 동일 인덱스만 선택 -> 이후 T_init을
    # 그대로 곱해서 씬에 정합시킬 수 있도록 좌표계를 유지한다.
    visible = cad_pcd_full.select_by_index(pt_map)
    log(f"   [CAD 가시면] 전체 {len(cad_pcd_full.points)}pt -> "
        f"가시 {len(visible.points)}pt ({100*len(visible.points)/len(cad_pcd_full.points):.1f}%)")
    return visible


# =============================================================================
# ICP 초기값 / 회전 구속 검사
# =============================================================================
def build_icp_init(scene_pcd: o3d.geometry.PointCloud,
                    cad_center_ref: np.ndarray) -> np.ndarray:
    R_init = _Rz(ICP_INIT_YAW_DEG) @ _Ry(ICP_INIT_PITCH_DEG) @ _Rx(ICP_INIT_ROLL_DEG)
    sc_center = np.asarray(scene_pcd.get_center())
    T_init = np.eye(4)
    T_init[:3, :3] = R_init
    T_init[:3, 3] = sc_center - R_init @ cad_center_ref
    return T_init


def check_rotation_constraint(T: np.ndarray):
    R = T[:3, :3]
    pitch = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)))
    cp = np.cos(np.radians(pitch))
    if abs(cp) > 1e-6:
        roll = np.degrees(np.arctan2(R[2, 1] / cp, R[2, 2] / cp))
        yaw = np.degrees(np.arctan2(R[1, 0] / cp, R[0, 0] / cp))
    else:
        roll, yaw = 0.0, np.degrees(np.arctan2(-R[0, 1], R[1, 1]))

    violations = []
    if not (ICP_ROLL_RANGE[0] <= roll <= ICP_ROLL_RANGE[1]):
        violations.append(f"roll={roll:.1f}°")
    if not (ICP_PITCH_RANGE[0] <= pitch <= ICP_PITCH_RANGE[1]):
        violations.append(f"pitch={pitch:.1f}°")
    if not (ICP_YAW_RANGE[0] <= yaw <= ICP_YAW_RANGE[1]):
        violations.append(f"yaw={yaw:.1f}°")
    if violations:
        return False, "회전 구속 위반: " + ", ".join(violations)
    return True, f"roll={roll:.1f}° pitch={pitch:.1f}° yaw={yaw:.1f}°"


# =============================================================================
# [2] + [5] Point-to-Plane multi-resolution ICP
# =============================================================================
def _prep_stage_cloud(base_pcd: o3d.geometry.PointCloud, voxel: float | None,
                       normal_radius: float) -> o3d.geometry.PointCloud:
    cloud = base_pcd.voxel_down_sample(voxel) if voxel is not None else base_pcd
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=NORMAL_MAX_NN))
    cloud.orient_normals_towards_camera_location([0.0, 0.0, 0.0])
    return cloud


def run_icp_multistage(cad_source: o3d.geometry.PointCloud,
                        scene_source: o3d.geometry.PointCloud,
                        T_init: np.ndarray):
    """[2]+[5] coarse-to-fine point-to-plane ICP.

    cad_source   : [1]에서 만든 가시면 CAD (voxel 다운샘플 전, meter 단위)
    scene_source : outlier 제거된 씬 포인트클라우드 (voxel 다운샘플 전)
    """
    T = T_init.copy()
    for stage in ICP_STAGES:
        voxel = stage["voxel"]
        n_radius = (voxel * NORMAL_RADIUS_FACTOR) if voxel is not None else NORMAL_RADIUS_FINAL
        src = _prep_stage_cloud(cad_source, voxel, n_radius)
        tgt = _prep_stage_cloud(scene_source, voxel, n_radius)

        res = o3d.pipelines.registration.registration_icp(
            src, tgt, stage["max_dist"], T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=stage["max_iter"]),
        )
        T = np.asarray(res.transformation)

    final = o3d.pipelines.registration.evaluate_registration(
        cad_source, scene_source, ICP_STAGES[-1]["max_dist"], T)
    return T, float(final.fitness), float(final.inlier_rmse)


def correct_flipped_pose(T, cad_normal, cad_flipped, scene_source):
    """뒤집힘 감지 시 [1]에서 미리 만들어둔 '뒤집힌 자세 기준 가시면 CAD'로
    재정합한다. (기존 코드는 하나의 CAD 소스만 썼지만, 가시면 필터링을
    도입한 이상 뒤집힌 상태의 가시면도 별도로 준비해야 정합이 맞는다.)
    """
    if T[:3, :3][2, 2] >= 0:
        final = o3d.pipelines.registration.evaluate_registration(
            cad_normal, scene_source, ICP_STAGES[-1]["max_dist"], T)
        return T, float(final.fitness), float(final.inlier_rmse), False, cad_normal

    R_flip = np.diag([-1.0, -1.0, 1.0])
    T_flip = np.eye(4); T_flip[:3, :3] = R_flip
    c = T[:3, 3]; T_flip[:3, 3] = c - R_flip @ c
    T_f, fit, rmse = run_icp_multistage(cad_flipped, scene_source, T_flip @ T)
    return T_f, fit, rmse, True, cad_flipped


# =============================================================================
# 픽포인트 / 좌표 변환 유틸 (원본 스크립트와 동일 로직)
# =============================================================================
def pick_to_pixel(pick_mm, pcd_organized, valid_mask, fallback_xy):
    target = np.array(pick_mm, dtype=np.float32)
    vr, vc = np.where(valid_mask)
    if len(vr) == 0:
        return fallback_xy
    pts = pcd_organized[vr, vc]
    dist = ((pts - target) ** 2).sum(axis=1)
    idx = int(np.argmin(dist))
    if dist[idx] > 50.0 ** 2:
        return fallback_xy
    return int(vc[idx]), int(vr[idx])


def pixel_to_point(px, py, pcd_organized, valid_mask):
    H, W = pcd_organized.shape[:2]
    ix = min(max(int(round(px)), 0), W - 1)
    iy = min(max(int(round(py)), 0), H - 1)
    if valid_mask[iy, ix]:
        return pcd_organized[iy, ix].astype(np.float64).copy()
    vr, vc = np.where(valid_mask)
    if len(vr) == 0:
        return None
    d2 = (vr - iy) ** 2 + (vc - ix) ** 2
    idx = int(np.argmin(d2))
    return pcd_organized[vr[idx], vc[idx]].astype(np.float64).copy()


def robust_z_from_neighbors(px, py, pcd_organized, valid_mask):
    H, W = pcd_organized.shape[:2]
    ix = min(max(int(round(px)), 0), W - 1)
    iy = min(max(int(round(py)), 0), H - 1)
    zs = []
    for dx, dy in PICK_Z_NEIGHBOR_OFFSETS:
        nx, ny = ix + dx, iy + dy
        if 0 <= nx < W and 0 <= ny < H and valid_mask[ny, nx]:
            zs.append(float(pcd_organized[ny, nx, 2]))
    if not zs:
        return None, 0
    return float(np.median(zs)), len(zs)


def transform_to_pose(T):
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
        "euler_deg": {"roll_deg": round(e[0], 4), "pitch_deg": round(e[1], 4),
                      "yaw_deg": round(e[2], 4)},
        "transform_matrix": T.tolist(),
    }


def compute_pick_point(T, pcd_organized=None, valid_mask=None, cx_2d=None, cy_2d=None):
    pl_icp = CAD_PICK_LOCAL.copy()
    wt_icp = T @ pl_icp
    pos_icp_mm = wt_icp[:3] * 1000.0

    pt2d_mm = None
    if pcd_organized is not None and valid_mask is not None and cx_2d is not None and cy_2d is not None:
        pt2d_mm = pixel_to_point(cx_2d, cy_2d, pcd_organized, valid_mask)

    w = PICK_2D_MASK_WEIGHT
    pos_blend_mm = pos_icp_mm * (1.0 - w) + pt2d_mm * w if pt2d_mm is not None else pos_icp_mm

    z_robust_mm, z_neighbor_count = None, 0
    if pcd_organized is not None and valid_mask is not None:
        fallback_px = (cx_2d, cy_2d) if cx_2d is not None else (0, 0)
        px_blend, py_blend = pick_to_pixel(pos_blend_mm.tolist(), pcd_organized, valid_mask, fallback_px)
        z_robust_mm, z_neighbor_count = robust_z_from_neighbors(px_blend, py_blend, pcd_organized, valid_mask)
    if z_robust_mm is not None:
        pos_blend_mm = np.array([pos_blend_mm[0], pos_blend_mm[1], z_robust_mm])

    pos_final_mm = pos_blend_mm + np.array([PICK_OFFSET_X_MM, PICK_OFFSET_Y_MM, PICK_OFFSET_Z_MM])

    R = T[:3, :3]
    pitch = float(np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))))
    cp = np.cos(np.radians(pitch))
    if abs(cp) > 1e-6:
        roll = float(np.degrees(np.arctan2(R[2, 1] / cp, R[2, 2] / cp)))
        yaw = float(np.degrees(np.arctan2(R[1, 0] / cp, R[0, 0] / cp)))
    else:
        roll, yaw = 0.0, float(np.degrees(np.arctan2(-R[0, 1], R[1, 1])))
    return {
        "position_mm": [round(v, 3) for v in pos_final_mm.tolist()],
        "approach_deg": {"roll_deg": round(roll, 4), "pitch_deg": round(pitch, 4), "yaw_deg": round(yaw, 4)},
        "_pos_icp_mm": pos_icp_mm.tolist(),
        "_pos_2d_mm": None if pt2d_mm is None else pt2d_mm.tolist(),
        "_pos_blend_mm": pos_blend_mm.tolist(),
        "_z_robust_mm": z_robust_mm,
        "_z_neighbor_count": z_neighbor_count,
    }


def compute_height_point(pick_position_mm, pcd_organized=None, valid_mask=None, fallback_xy=(0, 0)):
    target_x = pick_position_mm[0]
    target_y = pick_position_mm[1] + HEIGHT_POINT_OFFSET_Y_MM
    approx_z = pick_position_mm[2]
    z_robust_mm, z_neighbor_count = None, 0
    if pcd_organized is not None and valid_mask is not None:
        px_h, py_h = pick_to_pixel([target_x, target_y, approx_z], pcd_organized, valid_mask, fallback_xy)
        z_robust_mm, z_neighbor_count = robust_z_from_neighbors(px_h, py_h, pcd_organized, valid_mask)
    final_z = approx_z if z_robust_mm is None else z_robust_mm
    return {"position_mm": [round(target_x, 3), round(target_y, 3), round(final_z, 3)],
            "z_neighbor_count": z_neighbor_count, "z_is_fallback": z_robust_mm is None}


def sort_picks_by_priority(picks, x_range_mm=100.0):
    def _key(pk):
        x, _y, z = pk["position_mm"][:3]
        return (0 if abs(x) <= x_range_mm else 1, z)
    return sorted(picks, key=_key)


# =============================================================================
# Detection + [4] 마스크 침식
# =============================================================================
def mask_nms(results, iou_threshold=MASK_IOU_THRESHOLD):
    keep = []
    suppressed = [False] * len(results)
    for i, ri in enumerate(results):
        if suppressed[i]:
            continue
        keep.append(ri)
        area_i = ri.mask.sum()
        if area_i == 0:
            continue
        for j in range(i + 1, len(results)):
            if suppressed[j]:
                continue
            rj = results[j]
            inter = (ri.mask & rj.mask).sum()
            if inter == 0:
                continue
            union = area_i + rj.mask.sum() - inter
            if union > 0 and inter / union >= iou_threshold:
                suppressed[j] = True
    return keep


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """[4] depth flying pixel(경계 노이즈) 완충용 마스크 침식."""
    if px <= 0:
        return mask
    kernel = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def overlay_results(image_bgr, results, valid_mask=None):
    overlay = image_bgr.copy()
    if valid_mask is not None:
        overlay[~valid_mask] = (overlay[~valid_mask] * 0.4).astype(np.uint8)
    for i, r in enumerate(results):
        color = _PALETTE_BGR[i % len(_PALETTE_BGR)]
        layer = np.zeros_like(overlay)
        layer[r.mask] = color
        overlay[r.mask] = (0.5 * overlay[r.mask] + 0.5 * layer[r.mask]).astype(np.uint8)
        x1, y1, x2, y2 = r.bbox.astype(int)
        c = tuple(int(v) for v in color)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), c, 2)
        label = f"#{i} {r.class_name} {r.score:.2f}"
        cv2.putText(overlay, label, (x1 + 2, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def save_instance_pcd(points, out_path, color):
    if points.size == 0:
        return False
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points / 1000.0)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, dtype=np.float64), (len(points), 1)))
    return bool(o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False))


def run_detection(frame_name, gray, pcd_organized, valid_mask, inferencer, result_dir):
    H, W = gray.shape
    bgr = np.stack([gray, gray, gray], axis=-1)
    results = mask_nms(inferencer.infer(bgr))

    cv2.imwrite(str(result_dir / f"{frame_name}_overlay.png"),
                overlay_results(bgr, results, valid_mask))

    instance_plys = []
    for i, r in enumerate(results):
        mask_eroded = erode_mask(r.mask, MASK_ERODE_PX)
        combined = mask_eroded & valid_mask
        obj_pts = pcd_organized[combined]
        used_erode = True
        if len(obj_pts) < MIN_POINTS_PER_INSTANCE:
            # 침식 때문에 점이 너무 적어지면 원본 마스크로 폴백
            combined = r.mask & valid_mask
            obj_pts = pcd_organized[combined]
            used_erode = False
        if len(obj_pts) < MIN_POINTS_PER_INSTANCE:
            log(f"   obj{i}: 점 부족({len(obj_pts)}) -> 스킵")
            continue

        color_rgb = tuple(_PALETTE_RGB_FLOAT[i % len(_PALETTE_RGB_FLOAT)].tolist())
        ply_path = result_dir / f"{frame_name}_obj{i}.ply"
        ok = save_instance_pcd(obj_pts, ply_path, color=color_rgb)
        cx_2d = float((r.bbox[0] + r.bbox[2]) / 2)
        cy_2d = float((r.bbox[1] + r.bbox[3]) / 2)
        if ok:
            instance_plys.append((ply_path, cx_2d, cy_2d, r.bbox, float(r.score),
                                   len(obj_pts), used_erode))

    instance_mask_union = np.zeros(valid_mask.shape, dtype=bool)
    for r in results:
        instance_mask_union |= (r.mask & valid_mask)
    bg_mask = valid_mask & ~instance_mask_union
    bg_pts = pcd_organized[bg_mask]
    bg_pcd = o3d.geometry.PointCloud()
    if len(bg_pts) > 0:
        bg_pcd.points = o3d.utility.Vector3dVector(bg_pts / 1000.0)
        bg_pcd.colors = o3d.utility.Vector3dVector(np.tile(_BG_COLOR, (len(bg_pts), 1)))

    return len(results), instance_plys, bgr, bg_pcd


# =============================================================================
# CSV 로깅
# =============================================================================
PICK_LOG_FIELDS = [
    "timestamp", "frame_name", "instance_id", "status", "error_msg", "det_score",
    "icp_fitness", "icp_rmse_m", "num_points_scene", "num_points_after_outlier_removal",
    "mask_eroded", "was_flipped",
    "final_x_mm", "final_y_mm", "final_z_mm", "roll_deg", "pitch_deg", "yaw_deg",
]


def append_pick_log_csv(csv_path: Path, row: dict) -> None:
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PICK_LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in PICK_LOG_FIELDS})


# =============================================================================
# ICP 프레임 처리
# =============================================================================
def run_icp_for_frame(instance_plys, cad_visible_normal, cad_visible_flipped, cad_center_ref,
                       result_dir, frame_name, bgr_image, pcd_organized, valid_mask, bg_pcd,
                       csv_path: Path):
    icp_results = []
    combined_pcd = bg_pcd if bg_pcd is not None else o3d.geometry.PointCloud()

    for ply_path, cx_2d, cy_2d, bbox, det_score, n_pts_mask, used_erode in instance_plys:
        stem = ply_path.stem
        inst_idx = int(stem.split("obj")[-1])

        scene_pcd = o3d.io.read_point_cloud(str(ply_path))
        n_pts = len(np.asarray(scene_pcd.points))
        if n_pts < 50:
            log(f"   obj{inst_idx}: 포인트 부족({n_pts}) -> 스킵")
            icp_results.append({"instance_id": inst_idx, "error": f"포인트 부족: {n_pts}개"})
            append_pick_log_csv(csv_path, {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "frame_name": frame_name, "instance_id": inst_idx, "status": "error",
                "error_msg": f"포인트 부족: {n_pts}개", "det_score": round(det_score, 4),
                "num_points_scene": n_pts})
            continue

        sc, _ = scene_pcd.remove_statistical_outlier(OUTLIER_NB_NEIGHBORS, OUTLIER_STD_RATIO)
        n_after = len(np.asarray(sc.points))
        log(f"   obj{inst_idx}: {n_pts}pt (마스크침식={used_erode}) -> outlier제거 {n_after}pt")

        T_init = build_icp_init(sc, cad_center_ref)
        T, fit, rmse = run_icp_multistage(cad_visible_normal, sc, T_init)
        T, fit, rmse, flipped, cad_used = correct_flipped_pose(
            T, cad_visible_normal, cad_visible_flipped, sc)
        if flipped:
            log(f"   △ 뒤집힘 보정 후 fitness={fit:.4f}")

        if fit < ICP_FITNESS_THRESHOLD:
            log(f"   ✗ ICP 실패 (fitness={fit:.4f})")
            icp_results.append({"instance_id": inst_idx, "error": "ICP 정합 실패", "icp_fitness": float(fit)})
            append_pick_log_csv(csv_path, {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "frame_name": frame_name, "instance_id": inst_idx, "status": "error",
                "error_msg": "ICP 정합 실패", "det_score": round(det_score, 4),
                "icp_fitness": round(fit, 4), "icp_rmse_m": round(rmse, 6),
                "num_points_scene": n_pts, "num_points_after_outlier_removal": n_after,
                "mask_eroded": used_erode, "was_flipped": flipped})
            ply_path.unlink(missing_ok=True)
            continue

        if max(abs(v) for v in T[:3, 3]) > XYZ_MAX_M:
            icp_results.append({"instance_id": inst_idx, "error": "xyz 범위 이상", "icp_fitness": float(fit)})
            ply_path.unlink(missing_ok=True)
            continue

        rot_ok, rot_msg = check_rotation_constraint(T)
        if not rot_ok:
            log(f"   ✗ {rot_msg} -> 기각")
            icp_results.append({"instance_id": inst_idx, "error": rot_msg, "icp_fitness": float(fit)})
            append_pick_log_csv(csv_path, {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "frame_name": frame_name, "instance_id": inst_idx, "status": "error",
                "error_msg": rot_msg, "det_score": round(det_score, 4),
                "icp_fitness": round(fit, 4), "icp_rmse_m": round(rmse, 6),
                "num_points_scene": n_pts, "num_points_after_outlier_removal": n_after,
                "mask_eroded": used_erode, "was_flipped": flipped})
            ply_path.unlink(missing_ok=True)
            continue
        log(f"   ✓ 회전 OK: {rot_msg}  fitness={fit:.4f} rmse={rmse*1000:.3f}mm")

        pose = transform_to_pose(T)
        pick = compute_pick_point(T, pcd_organized=pcd_organized, valid_mask=valid_mask, cx_2d=cx_2d, cy_2d=cy_2d)
        ppos = pick["position_mm"]
        deg = pick["approach_deg"]
        height_point = compute_height_point(ppos, pcd_organized=pcd_organized, valid_mask=valid_mask,
                                             fallback_xy=(int(cx_2d), int(cy_2d)))

        inst_color = _PALETTE_RGB_FLOAT[inst_idx % len(_PALETTE_RGB_FLOAT)].tolist()
        sv = copy.deepcopy(sc)
        sv.colors = o3d.utility.Vector3dVector(np.tile(inst_color, (len(np.asarray(sv.points)), 1)))
        cv_ = copy.deepcopy(cad_used); cv_.transform(T)
        cv_.colors = o3d.utility.Vector3dVector(np.tile([0.1, 0.9, 0.3], (len(np.asarray(cv_.points)), 1)))
        pm = np.array(ppos) / 1000.0
        sp = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sp.translate(pm); sp.paint_uniform_color([1.0, 0.1, 0.1])
        combined_pcd += sv + cv_ + sp.sample_points_uniformly(500)

        log(f"   ✓ 픽포인트: ({ppos[0]:.1f}, {ppos[1]:.1f}, {ppos[2]:.1f}) mm "
            f"roll={deg['roll_deg']:.2f} pitch={deg['pitch_deg']:.2f} yaw={deg['yaw_deg']:.2f}")

        result = {
            "instance_id": inst_idx, "icp_fitness": float(fit), "icp_rmse_m": float(rmse),
            "was_flipped": flipped, "mask_eroded": used_erode,
            "num_points_scene": n_pts, "num_points_after_outlier_removal": n_after,
            "pose": pose, "pick_point": pick, "height_point": height_point,
        }
        append_pick_log_csv(csv_path, {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "frame_name": frame_name, "instance_id": inst_idx, "status": "ok",
            "det_score": round(det_score, 4), "icp_fitness": round(fit, 4), "icp_rmse_m": round(rmse, 6),
            "num_points_scene": n_pts, "num_points_after_outlier_removal": n_after,
            "mask_eroded": used_erode, "was_flipped": flipped,
            "final_x_mm": ppos[0], "final_y_mm": ppos[1], "final_z_mm": ppos[2],
            "roll_deg": deg["roll_deg"], "pitch_deg": deg["pitch_deg"], "yaw_deg": deg["yaw_deg"]})
        icp_results.append(result)
        ply_path.unlink(missing_ok=True)

    if len(np.asarray(combined_pcd.points)) > 0:
        o3d.io.write_point_cloud(str(result_dir / f"{frame_name}_colored.ply"), combined_pcd, write_ascii=False)

    success = [r for r in icp_results if "error" not in r]
    with (result_dir / f"{frame_name}_result.json").open("w", encoding="utf-8") as f:
        json.dump({"frame": frame_name, "num_total": len(icp_results), "num_success": len(success),
                   "instances": icp_results}, f, indent=2, ensure_ascii=False)

    return icp_results


# =============================================================================
# 캡처 파일 로딩 (오프라인 입력)
#
# collect_dataset.py 가 실제로 저장하는 구조 (세션 폴더 1개 = --capture-dir):
#   <capture-dir>/intensity/frame_0001.png
#   <capture-dir>/pointcloud_organized/frame_0001.npy   (H,W,3) float32 mm, invalid=NaN
#   <capture-dir>/valid_mask/frame_0001.npy             (H,W) bool
#   <capture-dir>/metadata/frame_0001.json
#   <capture-dir>/config_snapshot.yaml
#
# 즉 --capture-dir 는 "data/dataset" 전체가 아니라 그 아래
# 개별 세션 폴더(예: data/dataset/20260716_165133) 하나를 가리켜야 합니다.
# =============================================================================
def find_capture_timestamps(capture_dir: Path) -> list[str]:
    """세션 폴더 안 intensity/frame_NNNN.png 들에서 frame id 목록을 뽑는다.
    반환값은 'frame_0001' 형태의 문자열 (확장자 제외 파일명 그대로).
    """
    intensity_dir = capture_dir / "intensity"
    if not intensity_dir.exists():
        raise FileNotFoundError(
            f"intensity 폴더 없음: {intensity_dir}\n"
            f"  --capture-dir 는 세션 폴더(예: data/dataset/20260716_165133)를 "
            f"가리켜야 합니다. data/dataset 자체를 주면 안 됩니다.")
    ts_list = sorted(p.stem for p in intensity_dir.glob("frame_*.png"))
    if not ts_list:
        raise FileNotFoundError(f"frame_*.png 파일 없음: {intensity_dir}")
    return ts_list


def load_capture_files(capture_dir: Path, ts: str):
    """ts 는 find_capture_timestamps()가 반환한 'frame_0001' 형태 그대로."""
    intensity_path = capture_dir / "intensity" / f"{ts}.png"
    pcd_path = capture_dir / "pointcloud_organized" / f"{ts}.npy"
    mask_path = capture_dir / "valid_mask" / f"{ts}.npy"

    for p in (intensity_path, pcd_path, mask_path):
        if not p.exists():
            raise FileNotFoundError(f"캡처 파일 누락: {p}")

    gray = cv2.imread(str(intensity_path), cv2.IMREAD_GRAYSCALE)
    pcd_organized = np.load(pcd_path).astype(np.float32)   # NaN 포함, mm 단위
    valid_mask = np.load(mask_path).astype(bool)
    return gray, pcd_organized, valid_mask


# =============================================================================
# 프레임 1개 처리
# =============================================================================
def process_one_frame(ts: str, capture_dir: Path, result_dir: Path,
                       inferencer, cad_visible_normal, cad_visible_flipped, cad_center_ref,
                       csv_path: Path):
    log(f"\n{'─'*70}\n[{ts}] 로딩...")
    gray, pcd_organized, valid_mask = load_capture_files(capture_dir, ts)
    frame_name = f"result_{ts}"

    t0 = time.perf_counter()
    n_det, instance_plys, bgr_image, bg_pcd = run_detection(
        frame_name, gray, pcd_organized, valid_mask, inferencer, result_dir)
    det_ms = (time.perf_counter() - t0) * 1000.0
    log(f" [RTMDet] 검출: {n_det}개  유효 PCD: {len(instance_plys)}개  ({det_ms:.0f} ms)")

    if not instance_plys:
        log(" 브라켓 없음")
        return {"status": "No", "num_detected": n_det, "num_icp_ok": 0, "picks": []}

    t0 = time.perf_counter()
    icp_results = run_icp_for_frame(
        instance_plys, cad_visible_normal, cad_visible_flipped, cad_center_ref,
        result_dir, frame_name, bgr_image, pcd_organized, valid_mask, bg_pcd, csv_path)
    icp_ms = (time.perf_counter() - t0) * 1000.0
    success = [r for r in icp_results if "error" not in r]
    log(f" [ICP] 성공: {len(success)}개  실패: {len(icp_results)-len(success)}개  ({icp_ms:.0f} ms)")

    if not success:
        return {"status": "No", "num_detected": n_det, "num_icp_ok": 0, "picks": []}

    picks = [{"position_mm": r["pick_point"]["position_mm"],
              "approach_deg": r["pick_point"]["approach_deg"],
              "icp_fitness": r["icp_fitness"]} for r in success]
    picks = sort_picks_by_priority(picks)
    for i, pk in enumerate(picks):
        pp, deg, fit = pk["position_mm"], pk["approach_deg"], pk["icp_fitness"]
        log(f" #{i} ({pp[0]:.1f}, {pp[1]:.1f}, {pp[2]:.1f}) mm fit={fit:.3f} "
            f"R={deg['roll_deg']:.2f} P={deg['pitch_deg']:.2f} Y={deg['yaw_deg']:.2f}")

    return {"status": "ok", "num_detected": n_det, "num_icp_ok": len(success), "picks": picks}


# =============================================================================
# main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="오프라인 빈피킹 ICP 정합 테스트 "
                     "(인자 없이 실행하면 상단 DEFAULT_* 값을 그대로 사용)")
    p.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR,
                    help=f"캡처 루트 (기본: {DEFAULT_CAPTURE_DIR})")
    p.add_argument("--cad", type=Path, default=DEFAULT_CAD_PATH,
                    help=f"CAD STL 경로 (기본: {DEFAULT_CAD_PATH})")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                    help=f"RTMDet-Ins config .py (기본: {DEFAULT_CONFIG_PATH})")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                    help=f"RTMDet-Ins .pth 체크포인트 (기본: {DEFAULT_CHECKPOINT})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"결과 출력 루트 (기본: {DEFAULT_OUT_DIR})")
    p.add_argument("--frame", type=str, default=DEFAULT_FRAME,
                    help="특정 프레임 1개만 처리, 예: frame_0001 (미지정시 capture-dir 내 전체 처리)")
    p.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    return p.parse_args()


def main():
    args = parse_args()

    for label, p in (("--capture-dir", args.capture_dir), ("--cad", args.cad),
                      ("--config", args.config), ("--checkpoint", args.checkpoint)):
        if not p.exists():
            log(f"ERROR: {label} 경로가 존재하지 않습니다: {p}")
            log("       스크립트 상단 DEFAULT_* 상수를 실제 경로로 바꾸거나, "
                f"{label} 옵션으로 직접 지정하세요.")
            sys.exit(1)

    result_dir = args.out / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "pick_log.csv"

    log(f"모델 로드: {args.checkpoint.name}")
    inferencer = RTMDetInferencer(config=str(args.config), checkpoint=str(args.checkpoint),
                                   device=args.device, score_threshold=args.score_threshold)

    log(f"CAD 로드: {args.cad.name}")
    cad_pcd_full = load_cad_as_pcd(args.cad)
    log(f"CAD 전체 포인트 수: {len(cad_pcd_full.points)}")

    R_fixed = _Rz(ICP_INIT_YAW_DEG) @ _Ry(ICP_INIT_PITCH_DEG) @ _Rx(ICP_INIT_ROLL_DEG)
    R_flip = np.diag([-1.0, -1.0, 1.0]) @ R_fixed
    log("[1] CAD 가시면(정자세) 계산 중...")
    cad_visible_normal = build_visible_cad(cad_pcd_full, R_fixed)
    log("[1] CAD 가시면(뒤집힘 자세) 계산 중...")
    cad_visible_flipped = build_visible_cad(cad_pcd_full, R_flip)
    cad_center_ref = np.asarray(cad_pcd_full.get_center())

    ts_list = [args.frame] if args.frame else find_capture_timestamps(args.capture_dir)
    log(f"\n처리할 프레임: {len(ts_list)}개")

    summary = []
    for ts in ts_list:
        try:
            payload = process_one_frame(ts, args.capture_dir, result_dir, inferencer,
                                         cad_visible_normal, cad_visible_flipped, cad_center_ref, csv_path)
            summary.append((ts, payload))
        except Exception as e:
            log(f"ERROR [{ts}]: {e}")
            summary.append((ts, {"status": "error", "message": str(e)}))

    n_ok = sum(1 for _, p in summary if p.get("status") == "ok")
    log(f"\n{'='*70}\n총 {len(summary)}개 프레임 처리 완료. 성공(ok): {n_ok}개")
    log(f"결과: {result_dir}")
    log(f"CSV 로그: {csv_path}")


if __name__ == "__main__":
    main()