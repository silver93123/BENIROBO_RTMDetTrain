"""Stage 2-1: 빈 피킹 데이터셋 수집 스크립트 (organized PCD 지원).

v2 변경점:
    - pointcloud_organized/ 폴더 추가: (H, W, 3) shape의 npy 저장
    - valid_mask/ 폴더 추가: (H, W) bool npy 저장
    - 기존 pointcloud/*.ply는 그대로 유지 (Open3D 호환)

용도:
    Detection 마스크와 PCD의 픽셀 매칭이 필요한 단계용.
    Stage 5 (crop_by_mask)에서 organized PCD가 필요함.

실행 (이 프로젝트(BENIROBO_RTMDetTrain) 루트에서):
    python scripts/collect_dataset.py --out data/dataset/brackets_for_train --num 5 --start-index 11

저장 파일:
    {out_dir}/
    ├── intensity/frame_NNNN.png             ← 8-bit mono (RTMDet 입력)
    ├── pointcloud/frame_NNNN.ply            ← Open3D PCD (m 단위, valid만)
    ├── pointcloud_organized/frame_NNNN.npy  ← (H,W,3) mm 단위, NaN 포함
    ├── valid_mask/frame_NNNN.npy            ← (H,W) bool
    ├── metadata/frame_NNNN.json             ← 캡처 정보
    └── config_snapshot.yaml                 ← 카메라 설정 백업
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.camera import create_camera  # noqa: E402

num_capture = 300

def parse_args() -> argparse.Namespace:
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description="빈 피킹 데이터셋 수집 (organized PCD 포함)")
    # 이 프로젝트(BENIROBO_RTMDetTrain) 안의 configs/ 폴더를 기본값으로 쓴다.
    # (다른 프로젝트 경로에 의존하지 않음)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "camera_config.yaml")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "dataset" / current_time)
    p.add_argument("--num", type=int, default=num_capture, help="캡처할 프레임 수")
    p.add_argument("--warmup", type=int, default=3, help="시작 시 버리는 워밍업 수")
    p.add_argument("--start-index", type=int, default=1, help="시작 프레임 번호")
    p.add_argument(
        "--formats", type=str,
        default="intensity,pointcloud,organized,mask,metadata",
        help=(
            "저장할 파일 종류 (콤마 구분). "
            "선택지: intensity,pointcloud,organized,mask,metadata"
        ),
    )
    return p.parse_args()


VALID_FORMAT_KEYS = {"intensity", "pointcloud", "organized", "mask", "metadata", "rgb"}


def describe_camera_cfg(cfg_camera: dict) -> str:
    """카메라 타입에 맞는 요약 문자열을 만든다. (콘솔 출력 및 로그 파싱용)

    capture_runner.py의 정규식은 이 줄을 파싱하지 않으므로 자유롭게 확장 가능.
    """
    cam_type = cfg_camera.get("type", "unknown")
    if cam_type == "lucid_helios":
        return (
            f"lucid_helios | exposure={cfg_camera.get('exposure_time_selector')}, "
            f"mode={cfg_camera.get('operating_mode')}"
        )
    if cam_type == "femto_bolt":
        return (
            f"femto_bolt | "
            f"{cfg_camera.get('depth_width')}x{cfg_camera.get('depth_height')}"
            f"@{cfg_camera.get('fps')}fps"
        )
    return f"{cam_type} | {cfg_camera}"


def parse_formats(formats_str: str) -> set:
    formats = {f.strip() for f in formats_str.split(",") if f.strip()}
    unknown = formats - VALID_FORMAT_KEYS
    if unknown:
        raise ValueError(
            f"알 수 없는 --formats 값: {sorted(unknown)}. "
            f"선택 가능: {sorted(VALID_FORMAT_KEYS)}"
        )
    return formats


def setup_output_dirs(out_dir: Path, formats: set) -> dict:
    """출력 디렉토리 구조 생성. formats에 포함된 종류만 폴더를 만든다."""
    all_subdirs = {
        "intensity": out_dir / "intensity",
        "pointcloud": out_dir / "pointcloud",
        "organized": out_dir / "pointcloud_organized",
        "mask": out_dir / "valid_mask",
        "metadata": out_dir / "metadata",
        "rgb": out_dir / "color_rgb",
    }
    subdirs = {k: v for k, v in all_subdirs.items() if k in formats}
    for p in subdirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return subdirs


def describe_camera_cfg_dict(cfg_camera: dict) -> dict:
    """metadata/frame_NNNN.json에 넣을 카메라 정보. 타입별로 관련 키만 담는다."""
    cam_type = cfg_camera.get("type")
    if cam_type == "lucid_helios":
        return {
            "type": cam_type,
            "pixel_format": cfg_camera.get("pixel_format"),
            "exposure_time_selector": cfg_camera.get("exposure_time_selector"),
            "operating_mode": cfg_camera.get("operating_mode"),
        }
    if cam_type == "femto_bolt":
        return {
            "type": cam_type,
            "depth_width": cfg_camera.get("depth_width"),
            "depth_height": cfg_camera.get("depth_height"),
            "fps": cfg_camera.get("fps"),
        }
    return {"type": cam_type}


def save_frame(frame, dirs: dict, idx: int, cfg_camera: dict, formats: set) -> dict:
    """한 프레임을 저장. formats에 포함된 종류만 실제로 디스크에 쓴다."""
    name = f"frame_{idx:04d}"
    saved_files = {}

    # 1. Intensity PNG
    if "intensity" in formats:
        path = dirs["intensity"] / f"{name}.png"
        cv2.imwrite(str(path), frame.intensity)
        saved_files["intensity"] = f"intensity/{name}.png"

    # 2. Organized PCD (mm 단위, NaN 포함) - Stage 5용 핵심
    if "organized" in formats:
        path = dirs["organized"] / f"{name}.npy"
        np.save(path, frame.points_organized.astype(np.float32))
        saved_files["organized"] = f"pointcloud_organized/{name}.npy"

    # 3. Valid mask
    if "mask" in formats:
        path = dirs["mask"] / f"{name}.npy"
        np.save(path, frame.valid_mask.astype(bool))
        saved_files["mask"] = f"valid_mask/{name}.npy"

    # 4. Open3D PLY (m 단위, valid points만) - 기존 호환성 유지
    if "pointcloud" in formats:
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(frame.points / 1000.0)
            ok = o3d.io.write_point_cloud(
                str(dirs["pointcloud"] / f"{name}.ply"), pcd, write_ascii=False
            )
            if ok:
                saved_files["pointcloud"] = f"pointcloud/{name}.ply"
            else:
                print(f"  [WARN] PLY 쓰기 실패", flush=True)
        except ImportError:
            pass  # Open3D 없으면 npy만으로 OK

    # 5. RGB PNG (옵션, Femto Bolt에서 capture_rgb: true일 때만 존재)
    if "rgb" in formats:
        color_rgb = getattr(frame, "color_rgb", None)
        if color_rgb is not None:
            path = dirs["rgb"] / f"{name}.png"
            # cv2.imwrite는 BGR을 기대하므로 변환 (프레임은 RGB 순서로 들어옴)
            bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), bgr)
            saved_files["rgb"] = f"color_rgb/{name}.png"
        else:
            print(
                "  [WARN] RGB 저장이 요청됐지만 이 프레임엔 RGB 데이터가 없습니다 "
                "(config의 camera.capture_rgb: true 설정을 확인하세요).",
                flush=True,
            )

    # 통계는 저장 여부와 무관하게 항상 계산 (진행 상황 표시용)
    valid_count = int(frame.valid_mask.sum())
    total = frame.height * frame.width
    valid_pct = 100.0 * valid_count / total

    pts = frame.points
    if pts.size > 0:
        z_min = float(pts[:, 2].min())
        z_max = float(pts[:, 2].max())
        z_med = float(np.median(pts[:, 2]))
    else:
        z_min = z_max = z_med = float("nan")

    metadata = {
        "frame_index": idx,
        "frame_name": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image": {"width": int(frame.width), "height": int(frame.height)},
        "stats": {
            "valid_pixels": valid_count,
            "total_pixels": total,
            "valid_ratio": round(valid_pct, 2),
            "z_min_mm": round(z_min, 1) if not np.isnan(z_min) else None,
            "z_max_mm": round(z_max, 1) if not np.isnan(z_max) else None,
            "z_median_mm": round(z_med, 1) if not np.isnan(z_med) else None,
            "num_points": int(len(pts)),
        },
        "camera_config": describe_camera_cfg_dict(cfg_camera),
        "files": saved_files,
    }

    if "metadata" in formats:
        with (dirs["metadata"] / f"{name}.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    try:
        formats = parse_formats(args.formats)
    except ValueError as e:
        print(f"[ERROR] {e}", flush=True)
        return 1
    if not formats:
        print("[ERROR] --formats가 비어 있습니다. 최소 하나는 선택해야 합니다.", flush=True)
        return 1

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dirs = setup_output_dirs(args.out, formats)
    shutil.copy2(args.config, args.out / "config_snapshot.yaml")

    print("=" * 70, flush=True)
    print("  데이터셋 수집 (v2: organized PCD 포함)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Config:    {args.config}", flush=True)
    print(f"  Output:    {args.out}", flush=True)
    print(f"  Frames:    {args.start_index} ~ {args.start_index + args.num - 1}", flush=True)
    print(f"  Formats:   {', '.join(sorted(formats))}", flush=True)
    print(f"  Camera:    {describe_camera_cfg(cfg['camera'])}", flush=True)
    print("=" * 70, flush=True)
    print("", flush=True)
    print("  부품 배치를 매 프레임마다 바꿔주세요.", flush=True)
    print("  Stage 5 검증용: organized PCD와 valid_mask도 함께 저장됩니다.", flush=True)
    print("", flush=True)

    captured = []
    try:
        with create_camera(cfg["camera"]) as cam:
            print(f"카메라 워밍업 ({args.warmup} frames)...", flush=True)
            for i in range(args.warmup):
                _ = cam.capture()
                print(f"  {i + 1}/{args.warmup}", flush=True)
            print("", flush=True)

            for k in range(args.num):
                idx = args.start_index + k
                print("-" * 70, flush=True)
                print(f"[{k + 1}/{args.num}] 프레임 {idx:04d}", flush=True)
                print(f"  → 배치 후 Enter (q=종료, s=스킵)", flush=True)

                try:
                    user_input = input("  > ").strip().lower()
                except KeyboardInterrupt:
                    print("\n  중단됨.", flush=True)
                    break

                if user_input == "q":
                    break
                if user_input == "s":
                    continue

                t0 = time.perf_counter()
                frame = cam.capture()
                dt_ms = (time.perf_counter() - t0) * 1000.0

                meta = save_frame(frame, dirs, idx, cfg["camera"], formats)

                s = meta["stats"]
                print(f"  ✓ saved | {dt_ms:5.1f} ms | "
                      f"valid {s['valid_ratio']:.1f}% | "
                      f"Z {s['z_min_mm']}~{s['z_max_mm']} mm "
                      f"(median {s['z_median_mm']})", flush=True)
                captured.append(meta)

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 70, flush=True)
    print(f"  완료: {len(captured)} 프레임", flush=True)
    print("=" * 70, flush=True)
    print(f"  저장: {args.out}", flush=True)

    if captured:
        valid_ratios = [m["stats"]["valid_ratio"] for m in captured]
        print(f"\n  valid: mean={np.mean(valid_ratios):.1f}%, "
              f"min={min(valid_ratios):.1f}%, max={max(valid_ratios):.1f}%",
              flush=True)
        if "organized" in formats:
            print(f"\n  organized PCD: {args.out / 'pointcloud_organized'}", flush=True)
        if "mask" in formats:
            print(f"  valid mask:    {args.out / 'valid_mask'}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())