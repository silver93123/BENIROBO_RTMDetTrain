"""GaussianUpsampler 체크포인트를 UI 없이 빠르게 눈으로 테스트하는 스크립트.

VS Code에서 바로 실행: 아래 "설정" 블록의 상수만 채워넣고 그냥 실행(F5/▶)하면
된다. 인자를 안 줘도 아래 상수값으로 동작한다 - 터미널에서 다르게 돌리고
싶으면 CLI 인자로 override 가능 (예: python scripts/test_pc_upsampler.py
--checkpoint 다른경로.pth).

아직 UI에 연결 안 됐다 - 이 스크립트로 먼저 결과가 쓸 만한지 확인하고 UI
통합 여부를 결정하는 게 순서다 (도메인 갭/환각 위험 우려 때문).

두 가지 입력 모드 (둘 중 하나만 채우면 됨):
    (A) SPARSE_POINTS_PATH만 채우기
        - 이미 (N,3) mm float32 npy 파일이 있는 경우. 가장 간단함.
        - 아직 이런 파일이 없다면 icp_test_tab.py로 세션을 열어 검출까지
          돌린 뒤 파이썬 콘솔에서 바로 만들 수 있다:
              pts = icp_runner.extract_instance_points_mm(det.mask, pcd_organized, valid_mask)
              np.save("instance0_sparse.npy", pts)
    (B) SESSION_PATH/FRAME_NAME/DETECTOR_CHECKPOINT/DETECTOR_CONFIG 채우기
        - 세션+프레임에서 검출부터 이 스크립트가 대신 해준다.
        - 이 경우 SPARSE_POINTS_PATH는 None으로 둘 것.

결과: sparse(빨강)/upsampled(파랑) 포인트를 한 PLY로 합쳐서 저장하고,
VIEW_RESULT=True면 app.core.icp_viewer로 바로 띄운다 (기존 ICP 결과
뷰어와 동일한 방식 재사용).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.upsampling import GaussianUpsampler, sample_gaussians  # noqa: E402

# =============================================================================
# 설정 - VS Code에서 그냥 실행하려면 이 블록만 채우면 된다
# =============================================================================
CHECKPOINT_PATH = "checkpoints/pc_upsampler/pc_upsampler_epoch100.pth"

# --- 입력 모드 (A): sparse 포인트 npy가 이미 있는 경우 ---
SPARSE_POINTS_PATH: str | None = None  # 예: "data/instance0_sparse.npy"

# --- 입력 모드 (B): 세션+프레임에서 직접 검출부터 하고 싶은 경우 ---
# SPARSE_POINTS_PATH가 None일 때만 이 블록이 쓰인다.
SESSION_PATH: str | None = 'data/dataset/20260720_112039'         # 예: "data/dataset/20260521_114500"
FRAME_NAME: str | None = 'frame_0007'            # 예: "frame_0000"
DETECTOR_CHECKPOINT: str | None = 'work_dirs/rtmdet-ins_bolt_m10_80_v1/best_coco_bbox_mAP_epoch_40.pth'   # 예: "checkpoints/rtmdet_ins_bracket/best.pth"
DETECTOR_CONFIG: str | None = 'work_dirs/rtmdet-ins_bolt_m10_80_v1/rtmdet-ins_bolt_m10_80_used.py'        # 예: "configs/rtmdet-ins_bracket_used.py"
INSTANCE_INDEX = 0

SAMPLES_PER_POINT = 8
OUTPUT_PLY_PATH: str | None = None  # None이면 임시폴더에 저장
VIEW_RESULT = True                   # 저장 후 바로 3D 뷰어로 띄울지
DEVICE = "cuda:0"                    # CUDA 없으면 자동으로 cpu로 전환됨

RED = (0.85, 0.2, 0.2)    # sparse 입력 색
BLUE = (0.2, 0.4, 0.9)    # 업샘플링 결과 색


# =============================================================================
# 아래는 로직 - 보통 안 건드려도 됨
# =============================================================================
def load_model(checkpoint_path: str, device: str) -> GaussianUpsampler:
    state = torch.load(checkpoint_path, map_location=device)
    model = GaussianUpsampler(
        feature_dim=state.get("feature_dim", 64),
        k_neighbors=state.get("k_neighbors", 16),
    )
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_sparse_points(
    sparse_points_path: str | None, session_path: str | None, frame_name: str | None,
    detector_checkpoint: str | None, detector_config: str | None, instance_index: int,
) -> np.ndarray:
    if sparse_points_path:
        pts = np.load(sparse_points_path).astype(np.float32)
        print(f"sparse 포인트 로드: {sparse_points_path} ({len(pts)}개)")
        return pts

    if not (session_path and frame_name and detector_checkpoint and detector_config):
        raise SystemExit(
            "SPARSE_POINTS_PATH를 안 주면 SESSION_PATH/FRAME_NAME/"
            "DETECTOR_CHECKPOINT/DETECTOR_CONFIG를 모두 지정해야 합니다."
        )

    from app.core.detector import Detector
    from app.core import icp_runner

    session_dir = Path(session_path)
    pcd_organized = np.load(session_dir / "pointcloud_organized" / f"{frame_name}.npy")
    valid_mask = np.load(session_dir / "valid_mask" / f"{frame_name}.npy")
    image_path = str(session_dir / "intensity" / f"{frame_name}.png")

    detector = Detector(
        checkpoint_path=detector_checkpoint, config_path=detector_config,
        backend="rtmdet_ins",
    )
    detections = [d for d in detector.predict(image_path) if d.mask is not None]
    if not detections:
        raise SystemExit("검출 결과가 없습니다.")
    if instance_index >= len(detections):
        raise SystemExit(f"instance_index={instance_index}인데 검출은 {len(detections)}개뿐입니다.")

    det = detections[instance_index]
    pts = icp_runner.extract_instance_points_mm(det.mask, pcd_organized, valid_mask, erode_px=3)
    print(f"세션에서 인스턴스 {instance_index} 추출: {len(pts)}개 포인트")
    return pts.astype(np.float32)


def save_comparison_ply(sparse_mm: np.ndarray, upsampled_mm: np.ndarray, out_path: str) -> None:
    import open3d as o3d

    sparse_pcd = o3d.geometry.PointCloud()
    sparse_pcd.points = o3d.utility.Vector3dVector(sparse_mm.astype(np.float64) / 1000.0)
    sparse_pcd.colors = o3d.utility.Vector3dVector(np.tile(RED, (len(sparse_mm), 1)))

    up_pcd = o3d.geometry.PointCloud()
    up_pcd.points = o3d.utility.Vector3dVector(upsampled_mm.astype(np.float64) / 1000.0)
    up_pcd.colors = o3d.utility.Vector3dVector(np.tile(BLUE, (len(upsampled_mm), 1)))

    combined = sparse_pcd + up_pcd
    o3d.io.write_point_cloud(out_path, combined, write_ascii=False)


def run(
    checkpoint: str, sparse_points: str | None, session: str | None, frame: str | None,
    detector_checkpoint: str | None, detector_config: str | None, instance_index: int,
    samples_per_point: int, output: str | None, view: bool, device: str,
) -> None:
    resolved_device = device if torch.cuda.is_available() else "cpu"
    if resolved_device != device:
        print(f"[경고] CUDA를 쓸 수 없어 device를 '{device}' -> '{resolved_device}'로 변경합니다.")

    sparse_mm = get_sparse_points(
        sparse_points, session, frame, detector_checkpoint, detector_config, instance_index,
    )
    model = load_model(checkpoint, resolved_device)

    with torch.no_grad():
        sparse_t = torch.from_numpy(sparse_mm).unsqueeze(0).to(resolved_device)
        mean, scale, R = model(sparse_t)
        sampled = sample_gaussians(mean, scale, R, r=samples_per_point)
    upsampled_mm = sampled.squeeze(0).cpu().numpy()

    print(f"입력 {len(sparse_mm)}개 -> 업샘플링 {len(upsampled_mm)}개 (배수 {samples_per_point}x)")

    out_path = output or str(Path(tempfile.gettempdir()) / "pc_upsample_test.ply")
    save_comparison_ply(sparse_mm, upsampled_mm, out_path)
    print(f"비교 PLY 저장: {out_path} (빨강=입력, 파랑=업샘플링 결과)")

    if view:
        subprocess.Popen([
            sys.executable, "-m", "app.core.icp_viewer", out_path,
            "--title", "PC 업샘플링 테스트 (빨강=입력, 파랑=결과)",
        ])


def main() -> None:
    # 위 "설정" 블록의 값을 기본값으로 쓰고, CLI 인자를 주면 그것으로 override.
    # VS Code에서 그냥 실행(인자 없음)하면 전부 위 상수값 그대로 쓰인다.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--sparse-points", default=SPARSE_POINTS_PATH)
    parser.add_argument("--session", default=SESSION_PATH)
    parser.add_argument("--frame", default=FRAME_NAME)
    parser.add_argument("--detector-checkpoint", default=DETECTOR_CHECKPOINT)
    parser.add_argument("--detector-config", default=DETECTOR_CONFIG)
    parser.add_argument("--instance-index", type=int, default=INSTANCE_INDEX)
    parser.add_argument("--samples-per-point", type=int, default=SAMPLES_PER_POINT)
    parser.add_argument("--output", default=OUTPUT_PLY_PATH)
    parser.add_argument("--view", action=argparse.BooleanOptionalAction, default=VIEW_RESULT)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    run(
        checkpoint=args.checkpoint, sparse_points=args.sparse_points,
        session=args.session, frame=args.frame,
        detector_checkpoint=args.detector_checkpoint, detector_config=args.detector_config,
        instance_index=args.instance_index, samples_per_point=args.samples_per_point,
        output=args.output, view=args.view, device=args.device,
    )


if __name__ == "__main__":
    main()