"""RotHead(회전 회귀 헤드) 학습용 pseudo-label 자동 생성.

원리: RTMDet-Ins(기존, baseline) 검출 + ICP 정합을 세션의 모든 프레임에
돌리고, fitness가 높은(신뢰도 높은) 인스턴스만 골라 그 ICP 결과의 회전
(T[:3,:3])을 "정답"처럼 pseudo-label로 채택한다. fitness가 낮은
인스턴스는 ICP 자체가 불확실한 것이므로 학습 데이터로 안 쓰고 버린다 -
노이즈 낀 라벨을 섞으면 회전 헤드 학습이 오히려 나빠질 수 있기 때문.

실행 (프로젝트 루트에서):
    python scripts/generate_rotation_labels.py \\
        --dataset 20260521_114500 \\
        --cad data/cad/bracket.stl \\
        --checkpoint checkpoints/rtmdet_ins_bracket/best.pth \\
        --config configs/rtmdet-ins_bracket_used.py

--dataset는 콤마로 여러 세션을 한 번에 넘길 수 있다:
    --dataset 20260521_114500,20260522_090000

생성물:
    <mask-out-dir>/<session>_<frame>_obj<i>.npy   - 채택된 인스턴스의 마스크
    <labels-out>                                   - scripts/train_rotation_head.py가
                                                      바로 읽을 수 있는 라벨 JSON
    <preview-dir>/<session>_<frame>_obj<i>.jpg     - (--preview-dir 지정 시) 마스크
                                                      오버레이 + bbox + fitness를 그린
                                                      검토용 이미지. 학습에는 안 쓰이고
                                                      GUI 탭(label_generation_tab.py)이
                                                      실시간으로 폴링해서 보여준다.

주의: RTMDet-Ins baseline으로 검출하고 ICP(기존 registration_type=fgr_global
또는 open3d_multistage)로 정합한 결과를 라벨 소스로 쓴다. 즉 이 스크립트가
만드는 라벨의 품질 상한선은 결국 기존 ICP 파이프라인의 품질이다 -
fitness-min을 너무 낮게 잡으면 노이즈 낀 라벨이 섞여 들어간다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import icp_runner  # noqa: E402
from app.core.detector import Detector  # noqa: E402
from app.core.icp_runner import ICPParams  # noqa: E402

# --- 하드코딩 상수 (CLI로 override 가능) ---
DEFAULT_DATASET_ROOT = ROOT / "data" / "dataset"
DEFAULT_MASK_OUT_DIR = ROOT / "data" / "rotation_labels_masks"
DEFAULT_LABELS_OUT = ROOT / "data" / "rotation_labels.json"
DEFAULT_FITNESS_MIN = 0.85  # ICP 탭 기본 fitness_threshold(보통 0.6~0.7)보다 엄격하게 잡음 -
                            # 학습 라벨은 "정합 성공"보다 더 높은 확신이 필요하기 때문.
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_DEVICE = "cuda:0"
DEFAULT_PREVIEW_DIR = ROOT / "data" / "rotation_labels_preview"
DEFAULT_PREVIEW_MAX_WIDTH = 480  # 미리보기는 검토용이라 원본 해상도 그대로 저장할 필요 없음


def _load_frame(session_dir: Path, frame_name: str) -> tuple[np.ndarray, np.ndarray]:
    """icp_test_tab._on_frame_row_changed와 동일한 로딩 방식."""
    pcd_organized = np.load(session_dir / "pointcloud_organized" / f"{frame_name}.npy")
    valid_mask = np.load(session_dir / "valid_mask" / f"{frame_name}.npy")
    return pcd_organized, valid_mask


def _usable_frames(session_dir: Path) -> list[str]:
    intensity_dir = session_dir / "intensity"
    organized_dir = session_dir / "pointcloud_organized"
    mask_dir = session_dir / "valid_mask"
    stems = sorted(f.stem for f in intensity_dir.glob("*.png"))
    return [
        s for s in stems
        if (organized_dir / f"{s}.npy").is_file() and (mask_dir / f"{s}.npy").is_file()
    ]


def _save_preview(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: np.ndarray,
    fitness: float,
    out_path: Path,
    max_width: int = DEFAULT_PREVIEW_MAX_WIDTH,
) -> None:
    """검출 마스크 오버레이 + bbox + fitness 값을 이미지에 그려 검토용으로 저장.

    학습 라벨(mask .npy)과는 완전히 별개의 파일이다 - 이 이미지는 사람이 눈으로
    "이 인스턴스가 왜 채택됐는지" 바로 확인하기 위한 용도라, 지웠다 다시 생성해도
    라벨 자체에는 영향이 없다. 그래서 리사이즈해서 가볍게 jpg로 저장한다.
    """
    overlay = image_bgr.copy()
    color = np.array([60, 220, 60], dtype=np.uint8)  # BGR 초록 - 채택된 인스턴스 마스크
    overlay[mask] = (
        overlay[mask].astype(np.float32) * 0.5 + color.astype(np.float32) * 0.5
    ).astype(np.uint8)

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        overlay, f"fitness={fitness:.3f}", (x1, max(y1 - 8, 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
    )

    h, w = overlay.shape[:2]
    if w > max_width:
        scale = max_width / w
        overlay = cv2.resize(overlay, (max_width, int(h * scale)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def generate_labels(
    dataset_names: list[str],
    cad_path: str,
    checkpoint_path: str,
    config_path: str,
    fitness_min: float,
    mask_out_dir: Path,
    labels_out: Path,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    device: str = DEFAULT_DEVICE,
    preview_dir: Path | None = None,
) -> None:
    mask_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        print(f"[pseudo-label] 미리보기 이미지 저장: {preview_dir}")

    params = ICPParams()
    print(f"[pseudo-label] CAD 로드 중: {cad_path}")
    cad_pcd = icp_runner.load_cad_as_pcd(cad_path, params)
    cad_visible_normal, cad_visible_flipped = icp_runner.build_visible_cad_pair(cad_pcd, params)

    detector = Detector(
        checkpoint_path=checkpoint_path, config_path=config_path,
        device=device, score_threshold=score_threshold, backend="rtmdet_ins",
    )

    items: list[dict] = []
    n_total_instances = 0
    n_accepted = 0

    for dataset_name in dataset_names:
        session_dir = DEFAULT_DATASET_ROOT / dataset_name
        if not session_dir.is_dir():
            print(f"[pseudo-label] ⚠ 세션 폴더 없음, 건너뜀: {session_dir}")
            continue

        frames = _usable_frames(session_dir)
        print(f"[pseudo-label] 세션 {dataset_name}: 프레임 {len(frames)}개")

        for frame_name in frames:
            pcd_organized, valid_mask = _load_frame(session_dir, frame_name)
            image_path = session_dir / "intensity" / f"{frame_name}.png"

            detections = detector.predict(str(image_path), conf_threshold=score_threshold)
            detections = [d for d in detections if d.mask is not None]

            # 미리보기 저장할 때만 프레임 이미지를 BGR로 한 번 읽어둔다 (인스턴스마다
            # 다시 읽지 않도록) - Detector.predict()는 내부에서 자기 몫을 따로 읽으므로
            # 이 변수와는 독립적이다.
            frame_bgr = None
            if preview_dir is not None and detections:
                gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if gray is not None:
                    frame_bgr = np.stack([gray, gray, gray], axis=-1)

            for i, det in enumerate(detections):
                n_total_instances += 1
                pts_mm = icp_runner.extract_instance_points_mm(
                    det.mask, pcd_organized, valid_mask, erode_px=params.mask_erode_px
                )
                result = icp_runner.run_icp_for_instance(
                    i, pts_mm, cad_pcd, cad_visible_normal, cad_visible_flipped, params=params,
                )

                # GUI(label_generation_tab.py)가 이 한 줄짜리 고정 포맷을 정규식으로 파싱해서
                # icp_workbench_base.py와 동일한 스타일의 성공/실패 카드를 그린다. 한글 자유
                # 문장 대신 key=value 포맷을 쓰는 이유는 파싱을 프레임/CAD 이름 등에 한글이나
                # 공백이 섞여도 깨지지 않게 하기 위함이다.
                if not result.ok or result.fitness is None:
                    reason = (result.error or "ICP 실패").replace(" ", "_")
                    print(
                        f"[pseudo-label]   RESULT frame={frame_name} obj={i} "
                        f"status=REJECT fitness=none reason={reason}"
                    )
                    continue

                if result.fitness < fitness_min:
                    print(
                        f"[pseudo-label]   RESULT frame={frame_name} obj={i} "
                        f"status=REJECT fitness={result.fitness:.3f} reason=fitness<{fitness_min:.3f}"
                    )
                    continue

                mask_filename = f"{dataset_name}_{frame_name}_obj{i}.npy"
                mask_path = mask_out_dir / mask_filename
                np.save(mask_path, det.mask.astype(bool))

                items.append({
                    "image": str(image_path),
                    "mask": str(mask_path),
                    "bbox": [float(v) for v in det.bbox],
                    "rotation_matrix": result.T[:3, :3].tolist(),
                })
                n_accepted += 1

                if preview_dir is not None and frame_bgr is not None:
                    preview_path = preview_dir / mask_filename.replace(".npy", ".jpg")
                    _save_preview(frame_bgr, det.mask, det.bbox, result.fitness, preview_path)

                print(
                    f"[pseudo-label]   RESULT frame={frame_name} obj={i} "
                    f"status=ACCEPT fitness={result.fitness:.3f}"
                )

            if detections:
                print(
                    f"[pseudo-label]   {frame_name}: 인스턴스 {len(detections)}개 검사 "
                    f"-> 누적 채택 {n_accepted}건"
                )

    with open(labels_out, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(
        f"[pseudo-label] 완료: 전체 인스턴스 {n_total_instances}개 중 "
        f"{n_accepted}개 채택 (fitness >= {fitness_min}) -> {labels_out}"
    )
    if n_accepted == 0:
        print(
            "[pseudo-label] ⚠ 채택된 라벨이 0건입니다. --fitness-min을 낮추거나, "
            "기존 ICP 파이프라인(icp_test_tab)에서 이 세션의 fitness가 실제로 "
            "얼마나 나오는지 먼저 확인하세요."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="세션 폴더명, 콤마로 여러 개 나열 가능")
    parser.add_argument("--cad", required=True, help="CAD 파일 경로 (.stl/.ply/.obj)")
    parser.add_argument("--checkpoint", required=True, help="RTMDet-Ins 체크포인트 (.pth)")
    parser.add_argument("--config", required=True, help="RTMDet-Ins config (.py)")
    parser.add_argument("--fitness-min", type=float, default=DEFAULT_FITNESS_MIN)
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--mask-out-dir", default=str(DEFAULT_MASK_OUT_DIR))
    parser.add_argument("--labels-out", default=str(DEFAULT_LABELS_OUT))
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument(
        "--preview-dir", default=None,
        help="지정하면 채택된 인스턴스마다 마스크 오버레이 검토용 이미지를 저장 (선택)",
    )
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
        score_threshold=args.score_threshold,
        device=args.device,
        preview_dir=Path(args.preview_dir) if args.preview_dir else None,
    )


if __name__ == "__main__":
    main()