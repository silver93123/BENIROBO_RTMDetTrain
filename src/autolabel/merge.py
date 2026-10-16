"""검수 완료 -> 데이터셋 병합 + 품질 지표 산출 + COCO 내보내기.

미검수 파일의 처리가 이 파이프라인에서 가장 위험한 지점이다.
urgent/normal 버킷의 JSON은 모델의 초안 그대로일 수 있는데, 이걸 그대로 학습에 넣으면
모델의 오류를 정답으로 되먹이는 셈이다 (의사라벨 피드백 붕괴). 그래서:

  * 검수 판정 = (생성 당시와 내용이 달라졌다) 또는 (이미지 flag `reviewed` 가 켜졌다)
  * 판정이 안 된 파일은 기본적으로 데이터셋에서 제외하고 개수를 보고한다
  * auto 버킷(자동승인)만 검수 없이 포함된다

edit_iou 통계는 라운드마다 확인할 계기판이다. 평균 edit_iou가 올라가면 모델이 좋아지는 중이고,
정체되거나 특정 클래스만 낮으면 그 클래스는 자동 라벨을 꺼야 한다.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from .config import DEFAULT, AutoLabelConfig
from .labelme_io import is_marked_reviewed, labelme_to_masks, read_labelme
from .mask_utils import decode_rle, greedy_match
from .predict import load_cache

RESERVED_NAMES = {"classes.json", "merge_index.json", "annotations.json", "round_stats.json"}


def _compare_with_prediction(reviewed: list[dict], cache: dict) -> dict:
    """검수 결과와 원본 예측을 비교해 편집량을 측정."""
    pred_masks = [decode_rle(i["rle"]) for i in cache["instances"]]
    rev_masks = [r["mask"] for r in reviewed]

    matched, unmatched_pred, unmatched_rev = greedy_match(pred_masks, rev_masks, min_iou=0.30)
    ious = [iou for _, _, iou in matched]

    return {
        "n_pred": len(pred_masks),
        "n_final": len(rev_masks),
        "n_matched": len(matched),
        "n_deleted": len(unmatched_pred),  # 사람이 지운 오검출
        "n_added": len(unmatched_rev),  # 사람이 새로 그린 미검출
        "mean_edit_iou": round(float(np.mean(ious)), 5) if ious else None,
        "min_edit_iou": round(float(min(ious)), 5) if ious else None,
    }


def classify_review_state(bucket: str, raw: bytes, generated_sha256: str, payload: dict) -> str:
    """'auto' | 'edited' | 'approved' | 'unreviewed'."""
    changed = hashlib.sha256(raw).hexdigest() != generated_sha256
    if bucket == "auto":
        return "edited" if changed else "auto"
    if changed:
        return "edited"
    if is_marked_reviewed(payload):
        return "approved"
    return "unreviewed"


def merge_round(
    round_dir: Path,
    dataset_dir: Path,
    cfg: AutoLabelConfig = DEFAULT,
    include_auto: bool = True,
    include_unreviewed: bool = False,
    copy_images: bool = False,
    progress: Callable[[str], None] = print,
) -> dict:
    """검수된 labelme JSON을 데이터셋 디렉터리로 확정 반영한다.

    dataset_dir 에는 원본 파일명으로 <stem>.json 이 저장된다.
    copy_images=True면 이미지도 함께 복사한다 (기본은 원본 경로를 상대경로로 참조).
    """
    round_dir, dataset_dir = Path(round_dir), Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with open(round_dir / "manifest.json", encoding="utf-8") as fp:
        manifest = json.load(fp)

    # 이전 병합이 남긴 파일을 지운다 (이번에 제외된 이미지가 옛 JSON으로 남지 않게)
    index_path = dataset_dir / "merge_index.json"
    if index_path.exists():
        for name in json.loads(index_path.read_text(encoding="utf-8")).get("files", []):
            (dataset_dir / name).unlink(missing_ok=True)

    per_class: dict[str, int] = defaultdict(int)
    records: list[dict] = []
    heavy_edits: list[str] = []
    unreviewed: list[str] = []
    written: list[str] = []
    state_counts: dict[str, int] = defaultdict(int)
    n_auto_shapes = 0
    n_human_shapes = 0

    for review_name, info in manifest["items"].items():
        json_path = round_dir / info["bucket"] / f"{review_name}.json"
        if not json_path.exists():
            progress(f"[skip] 검수 파일 없음: {json_path.name}")
            continue

        raw = json_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        state = classify_review_state(info["bucket"], raw, info["generated_sha256"], payload)
        state_counts[state] += 1

        src_image = Path(info["original_image"])
        if state == "unreviewed" and not include_unreviewed:
            unreviewed.append(src_image.name)
            continue
        if state == "auto" and not include_auto:
            continue

        reviewed = labelme_to_masks(payload)
        cache = load_cache(Path(info["cache"]))

        stats = _compare_with_prediction(reviewed, cache)
        stats.update(
            {
                "image": src_image.name,
                "bucket": info["bucket"],
                "state": state,
                "priority": info["priority"],
            }
        )
        records.append(stats)

        if stats["mean_edit_iou"] is not None and stats["mean_edit_iou"] < cfg.heavy_edit_iou:
            heavy_edits.append(src_image.name)

        for item in reviewed:
            per_class[item["label"]] += 1
            # description 에 provenance 가 남아 있으면 모델이 만든 shape, 없으면 사람이 그린 shape
            if item["provenance"].get("auto"):
                n_auto_shapes += 1
            else:
                n_human_shapes += 1

        out_payload = dict(payload)
        out_payload["imageData"] = None
        if copy_images:
            target = dataset_dir / src_image.name
            if not target.exists():
                shutil.copy2(src_image, target)
            out_payload["imagePath"] = src_image.name
        else:
            # 원본을 상대경로로 참조: labelme 로 데이터셋 폴더를 열어도 이미지를 찾는다
            out_payload["imagePath"] = Path(
                _relpath(src_image, dataset_dir)
            ).as_posix()

        out_name = f"{src_image.stem}.json"
        (dataset_dir / out_name).write_text(
            json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written.append(out_name)

    (dataset_dir / "classes.json").write_text(
        json.dumps({"classes": manifest["classes"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    index_path.write_text(json.dumps({"files": written}, ensure_ascii=False), encoding="utf-8")

    ious = [r["mean_edit_iou"] for r in records if r["mean_edit_iou"] is not None]
    summary = {
        "round_dir": str(round_dir.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "merged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "images": len(records),
        "instances": int(sum(per_class.values())),
        "per_class": dict(sorted(per_class.items())),
        "review_states": dict(state_counts),
        "unreviewed_excluded": len(unreviewed),
        "unreviewed_images": unreviewed,
        "shapes_from_model": n_auto_shapes,
        "shapes_drawn_by_human": n_human_shapes,
        "mean_edit_iou": round(float(np.mean(ious)), 5) if ious else None,
        "total_deleted": sum(r["n_deleted"] for r in records),
        "total_added": sum(r["n_added"] for r in records),
        "heavy_edit_images": heavy_edits,
    }

    with open(round_dir / "round_stats.json", "w", encoding="utf-8") as fp:
        json.dump({"summary": summary, "per_image": records}, fp, ensure_ascii=False, indent=2)

    progress(
        f"[merge] {summary['images']}장 / {summary['instances']}개 인스턴스 | "
        f"평균 edit_iou {summary['mean_edit_iou']} | "
        f"삭제 {summary['total_deleted']} / 추가 {summary['total_added']}"
    )
    if unreviewed:
        progress(
            f"[merge] 미검수 {len(unreviewed)}장은 제외했습니다 "
            f"(수정했거나 labelme의 'reviewed' 체크를 켜야 포함됩니다)"
        )
    return summary


def _relpath(target: Path, start: Path) -> str:
    return os.path.relpath(Path(target).resolve(), Path(start).resolve())


# ----------------------------------------------------------------------
# COCO 내보내기
# ----------------------------------------------------------------------


def _flatten_segmentation(outer: list[np.ndarray]) -> list[list[float]]:
    return [np.asarray(p, dtype=float).reshape(-1).tolist() for p in outer if len(p) >= 3]


def _dataset_json_files(dataset_dir: Path) -> list[Path]:
    index_path = dataset_dir / "merge_index.json"
    if index_path.exists():
        names = json.loads(index_path.read_text(encoding="utf-8")).get("files", [])
        return [dataset_dir / n for n in sorted(names)]
    return sorted(p for p in dataset_dir.glob("*.json") if p.name not in RESERVED_NAMES)


def export_coco(
    dataset_dir: Path,
    classes: list[str] | None = None,
    out_path: Path | None = None,
    progress: Callable[[str], None] = print,
) -> dict:
    """labelme JSON 모음을 COCO instance segmentation 포맷으로 변환.

    classes 를 생략하면 merge 가 남긴 classes.json (추론 당시 모델의 class_names) 을 쓴다.
    구멍(hole)은 COCO 폴리곤 표현으로 정확히 담을 수 없으므로 외곽 폴리곤만 내보낸다.
    구멍이 중요하면 RLE 포맷 export를 따로 만들어야 한다.
    """
    dataset_dir = Path(dataset_dir)
    out_path = Path(out_path) if out_path else dataset_dir / "annotations.json"

    if classes is None:
        classes_file = dataset_dir / "classes.json"
        if not classes_file.exists():
            raise RuntimeError("classes 를 지정하거나 먼저 merge 를 실행하세요 (classes.json 없음)")
        classes = json.loads(classes_file.read_text(encoding="utf-8"))["classes"]
    cat_id = {name: i + 1 for i, name in enumerate(classes)}

    coco = {
        "info": {
            "description": "auto-label round export",
            "date_created": datetime.now().isoformat(timespec="seconds"),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": n, "supercategory": ""} for i, n in enumerate(classes)],
    }

    ann_id = 1
    skipped: list[str] = []
    for img_id, json_path in enumerate(_dataset_json_files(dataset_dir), 1):
        payload = read_labelme(json_path)

        coco["images"].append(
            {
                "id": img_id,
                "file_name": Path(payload["imagePath"]).name,
                "width": int(payload["imageWidth"]),
                "height": int(payload["imageHeight"]),
            }
        )

        for item in labelme_to_masks(payload):
            if item["label"] not in cat_id:
                skipped.append(item["label"])
                continue
            segmentation = _flatten_segmentation(item["outer"])
            if not segmentation:
                continue
            mask = item["mask"]
            ys, xs = np.nonzero(mask)
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id[item["label"]],
                    "segmentation": segmentation,
                    "area": float(np.count_nonzero(mask)),
                    "bbox": [
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max() - xs.min() + 1),
                        float(ys.max() - ys.min() + 1),
                    ],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(coco, fp, ensure_ascii=False)

    if skipped:
        progress(f"[warn] 클래스 목록에 없는 라벨 {len(skipped)}개 무시: {sorted(set(skipped))}")
    progress(f"[coco] {len(coco['images'])}장 / {len(coco['annotations'])}개 어노테이션 -> {out_path}")
    return coco
