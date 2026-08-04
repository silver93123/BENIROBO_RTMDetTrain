"""PVNet 학습용 2D 키포인트 GT 자동 생성.

generate_rotation_labels.py와 동일한 원리다: RTMDet-Ins(baseline) 검출 +
ICP 정합을 세션의 모든 프레임에 돌리고, fitness가 높은 인스턴스만 골라 그
ICP 결과 pose(T, 4x4)를 "정답"으로 채택한다. 다른 점은 회전행렬 하나가
아니라, CAD 키포인트 K개를 T로 변환한 뒤 2D 픽셀 좌표로 투영해 저장한다는
것 - 이게 PVNet 벡터장 학습에 필요한 실제 라벨이다 (픽셀별 GT 벡터는
학습 시점에 mask + keypoints_2d로부터 즉석에서 계산하면 되므로 여기서
미리 만들어 저장하지 않는다 - 용량도 크고, mask/keypoints_2d만 있으면
언제든 재생성 가능하기 때문).

핀홀 투영 intrinsic을 어떻게 구하는가:
    이 레포는 카메라 캘리브레이션 파일을 따로 두지 않는다. femto_bolt.py/
    lucid_helios.py 드라이버가 SDK 자체 PointCloudFilter로 왜곡보정까지
    끝낸 organized PCD(H,W,3, mm)만 내보내기 때문이다. 따라서 새로
    캘리브레이션을 하는 대신, 이미 가진 (X,Y,Z)<->(u,v) 대응 수만 쌍에서
    최소자승으로 근사 핀홀 intrinsic(fx,fy,cx,cy)을 역산한다
    (estimate_intrinsics_from_organized_pcd 참고). 세션 첫 유효 프레임
    하나로 한 번만 추정하고 이후 모든 프레임에 재사용한다 (카메라 고정
    설치 기준 - 프레임마다 다시 추정할 필요 없음).

실행 (프로젝트 루트에서):
    python scripts/generate_pvnet_labels.py \\
        --dataset 20260521_114500 \\
        --cad data/cad/bracket.stl \\
        --checkpoint checkpoints/rtmdet_ins_bracket/best.pth \\
        --config configs/rtmdet-ins_bracket_used.py \\
        --num-keypoints 8

생성물:
    <mask-out-dir>/<session>_<frame>_obj<i>.npy   - 채택된 인스턴스의 마스크
    <labels-out>                                   - 학습 스크립트가 읽을 라벨 JSON
    <keypoints-3d-out>                             - (K+1,3) CAD 로컬 키포인트,
                                                      labels-out의 keypoints_2d와
                                                      인덱스가 1:1로 대응
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import icp_runner  # noqa: E402
from app.core.detector import Detector  # noqa: E402
from app.core.icp_runner import ICPParams  # noqa: E402
from src.detection.pvnet import farthest_point_sampling  # noqa: E402

# --- 하드코딩 상수 (CLI로 override 가능) ---
DEFAULT_DATASET_ROOT = ROOT / "data" / "dataset"
DEFAULT_MASK_OUT_DIR = ROOT / "data" / "pvnet_labels_masks"
DEFAULT_LABELS_OUT = ROOT / "data" / "pvnet_labels.json"
DEFAULT_KEYPOINTS_3D_OUT = ROOT / "data" / "pvnet_keypoints_3d.npy"
DEFAULT_FITNESS_MIN = 0.85  # generate_rotation_labels.py와 동일 기준 - 학습 라벨은
                            # ICP 탭 기본 threshold(0.6~0.7)보다 엄격해야 함
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_NUM_KEYPOINTS = 8   # keypoints.py DEFAULT_NUM_SURFACE_KEYPOINTS와 동일
DEFAULT_VISIBILITY_TOL_MM = 15.0
DEFAULT_DEVICE = "cuda:0"


# =============================================================================
# 프레임 로딩 (generate_rotation_labels.py와 동일 방식)
# =============================================================================
def _load_frame(session_dir: Path, frame_name: str) -> Tuple[np.ndarray, np.ndarray]:
    pcd_organized = np.load(session_dir / "pointcloud_organized" / f"{frame_name}.npy")
    valid_mask = np.load(session_dir / "valid_mask" / f"{frame_name}.npy")
    return pcd_organized, valid_mask


def _usable_frames(session_dir: Path) -> List[str]:
    intensity_dir = session_dir / "intensity"
    organized_dir = session_dir / "pointcloud_organized"
    mask_dir = session_dir / "valid_mask"
    stems = sorted(f.stem for f in intensity_dir.glob("*.png"))
    return [
        s for s in stems
        if (organized_dir / f"{s}.npy").is_file() and (mask_dir / f"{s}.npy").is_file()
    ]


# =============================================================================
# 핀홀 intrinsic 근사 역산 + 투영
# =============================================================================
def estimate_intrinsics_from_organized_pcd(
    points_organized: np.ndarray,
    valid_mask: np.ndarray,
    subsample: int = 20000,
) -> Tuple[float, float, float, float]:
    """organized PCD (H,W,3, mm)와 valid_mask -> 근사 핀홀 (fx, fy, cx, cy).

    u = fx * (X/Z) + cx, v = fy * (Y/Z) + cy 는 각 축이 독립적인 1차
    선형회귀이므로 유효 픽셀 (X,Y,Z,u,v) 표본으로 최소자승 적합하면 된다.
    별도 캘리브레이션 없이, 이미 로드한 프레임 데이터만으로 충분히 정확한
    근사치를 얻을 수 있다 (SDK가 왜곡보정을 이미 끝낸 뒤의 XYZ이므로).

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


def project_points(points_cam_mm: np.ndarray, intrinsics: Tuple[float, float, float, float]) -> np.ndarray:
    """카메라 좌표계 3D점(mm) -> 픽셀 좌표 (K,2), (x,y)=(col,row) 순서."""
    fx, fy, cx, cy = intrinsics
    z = points_cam_mm[:, 2]
    u = fx * points_cam_mm[:, 0] / z + cx
    v = fy * points_cam_mm[:, 1] / z + cy
    return np.stack([u, v], axis=1)


def _keypoint_visibility(
    keypoints_2d: np.ndarray,
    keypoints_z_mm: np.ndarray,
    points_organized: np.ndarray,
    valid_mask: np.ndarray,
    tol_mm: float,
) -> np.ndarray:
    """투영된 키포인트가 실제로 이 인스턴스의 보이는 표면에 해당하는지 확인.

    투영 좌표 (u,v) 근처 픽셀의 실측 depth와, 그 키포인트의 예상 depth(T로
    변환한 카메라 좌표계 Z)를 비교한다. 차이가 tol_mm 이내면 "보임", 아니면
    다른 물체에 가려졌거나 크롭 밖으로 나간 것으로 보고 "안 보임"으로
    표시한다. 학습 자체에는 필수가 아니지만(벡터장은 가려진 키포인트도
    다른 픽셀에서 추론하도록 설계됨 - PVNet 논문 취지), occlusion 비율이
    너무 높은 인스턴스를 걸러내는 품질 지표로 쓸 수 있다.
    """
    h, w = valid_mask.shape
    visible = np.zeros(len(keypoints_2d), dtype=bool)
    for i, (u, v) in enumerate(keypoints_2d):
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < w and 0 <= vi < h) or not valid_mask[vi, ui]:
            continue
        z_observed = points_organized[vi, ui, 2]
        if abs(float(z_observed) - keypoints_z_mm[i]) < tol_mm:
            visible[i] = True
    return visible


# =============================================================================
# 메인 라벨 생성 루프
# =============================================================================
def generate_labels(
    dataset_names: List[str],
    cad_path: str,
    checkpoint_path: str,
    config_path: str,
    fitness_min: float,
    mask_out_dir: Path,
    labels_out: Path,
    keypoints_3d_out: Path,
    num_keypoints: int = DEFAULT_NUM_KEYPOINTS,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    visibility_tol_mm: float = DEFAULT_VISIBILITY_TOL_MM,
    device: str = DEFAULT_DEVICE,
) -> None:
    mask_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out.parent.mkdir(parents=True, exist_ok=True)

    params = ICPParams()
    print(f"[pvnet-label] CAD 로드 중: {cad_path}")
    cad_pcd = icp_runner.load_cad_as_pcd(cad_path, params)
    cad_visible_normal, cad_visible_flipped = icp_runner.build_visible_cad_pair(cad_pcd, params)

    cad_surface_points = np.asarray(cad_pcd.points)
    keypoints_3d = farthest_point_sampling(cad_surface_points, num_keypoints=num_keypoints, include_centroid=True)
    np.save(keypoints_3d_out, keypoints_3d)
    print(f"[pvnet-label] 키포인트 {keypoints_3d.shape[0]}개(센트로이드 포함) -> {keypoints_3d_out}")

    detector = Detector(
        checkpoint_path=checkpoint_path, config_path=config_path,
        device=device, score_threshold=score_threshold, backend="rtmdet_ins",
    )

    intrinsics: Tuple[float, float, float, float] | None = None  # 세션 전체에서 1회만 추정, 카메라 고정 설치 기준
    items: List[dict] = []
    n_total_instances = 0
    n_accepted = 0

    for dataset_name in dataset_names:
        session_dir = DEFAULT_DATASET_ROOT / dataset_name
        if not session_dir.is_dir():
            print(f"[pvnet-label] ⚠ 세션 폴더 없음, 건너뜀: {session_dir}")
            continue

        frames = _usable_frames(session_dir)
        print(f"[pvnet-label] 세션 {dataset_name}: 프레임 {len(frames)}개")

        for frame_name in frames:
            pcd_organized, valid_mask = _load_frame(session_dir, frame_name)
            image_path = session_dir / "intensity" / f"{frame_name}.png"

            if intrinsics is None:
                intrinsics = estimate_intrinsics_from_organized_pcd(pcd_organized, valid_mask)
                fx, fy, cx, cy = intrinsics
                print(f"[pvnet-label] intrinsic 추정 완료: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

            detections = detector.predict(str(image_path), conf_threshold=score_threshold)
            detections = [d for d in detections if d.mask is not None]

            for i, det in enumerate(detections):
                n_total_instances += 1
                pts_mm = icp_runner.extract_instance_points_mm(
                    det.mask, pcd_organized, valid_mask, erode_px=params.mask_erode_px
                )
                result = icp_runner.run_icp_for_instance(
                    i, pts_mm, cad_pcd, cad_visible_normal, cad_visible_flipped, params=params,
                )

                if not result.ok or result.fitness is None or result.fitness < fitness_min or result.T is None:
                    continue

                T = result.T
                keypoints_cam = (T[:3, :3] @ keypoints_3d.T).T + T[:3, 3]  # (K+1, 3), mm
                keypoints_2d = project_points(keypoints_cam, intrinsics)
                visible = _keypoint_visibility(
                    keypoints_2d, keypoints_cam[:, 2], pcd_organized, valid_mask, visibility_tol_mm
                )

                mask_filename = f"{dataset_name}_{frame_name}_obj{i}.npy"
                mask_path = mask_out_dir / mask_filename
                np.save(mask_path, det.mask.astype(bool))

                items.append({
                    "image": str(image_path),
                    "mask": str(mask_path),
                    "bbox": [float(v) for v in det.bbox],
                    "keypoints_2d": keypoints_2d.tolist(),   # (K+1, 2), keypoints_3d_out과 인덱스 1:1
                    "keypoints_visible": visible.tolist(),   # (K+1,) bool - 품질 필터링/curriculum용
                    "pose": T.tolist(),                      # 4x4, 참고/평가용 (학습엔 직접 안 씀)
                })
                n_accepted += 1

            if detections:
                print(
                    f"[pvnet-label]   {frame_name}: 인스턴스 {len(detections)}개 검사 "
                    f"-> 누적 채택 {n_accepted}건"
                )

    with open(labels_out, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(
        f"[pvnet-label] 완료: 전체 인스턴스 {n_total_instances}개 중 "
        f"{n_accepted}개 채택 (fitness >= {fitness_min}) -> {labels_out}"
    )
    if n_accepted == 0:
        print(
            "[pvnet-label] ⚠ 채택된 라벨이 0건입니다. --fitness-min을 낮추거나, "
            "기존 ICP 파이프라인(icp_test_tab)에서 이 세션의 fitness가 실제로 "
            "얼마나 나오는지 먼저 확인하세요."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="세션 폴더명, 콤마로 여러 개 나열 가능")
    parser.add_argument("--cad", required=True, help="CAD 파일 경로 (.stl/.ply/.obj)")
    parser.add_argument("--checkpoint", required=True, help="RTMDet-Ins 체크포인트 (.pth)")
    parser.add_argument("--config", required=True, help="RTMDet-Ins config (.py)")
    parser.add_argument("--num-keypoints", type=int, default=DEFAULT_NUM_KEYPOINTS)
    parser.add_argument("--fitness-min", type=float, default=DEFAULT_FITNESS_MIN)
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--visibility-tol-mm", type=float, default=DEFAULT_VISIBILITY_TOL_MM)
    parser.add_argument("--mask-out-dir", default=str(DEFAULT_MASK_OUT_DIR))
    parser.add_argument("--labels-out", default=str(DEFAULT_LABELS_OUT))
    parser.add_argument("--keypoints-3d-out", default=str(DEFAULT_KEYPOINTS_3D_OUT))
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    args = parser.parse_args()

    dataset_names = [s.strip() for s in args.dataset.split(",") if s.strip()]
    generate_labels(
        dataset_names=dataset_names,
        cad_path=args.cad,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        fitness_min=args.fitness_min,
        mask_out_dir=Path(args.mask_out_dir),
        labels_out=Path(args.labels_out),
        keypoints_3d_out=Path(args.keypoints_3d_out),
        num_keypoints=args.num_keypoints,
        score_threshold=args.score_threshold,
        visibility_tol_mm=args.visibility_tol_mm,
        device=args.device,
    )


if __name__ == "__main__":
    main()