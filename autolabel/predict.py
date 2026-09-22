"""배치 추론 -> 예측 캐시 저장.

핵심 설계: 예측 캐시와 최종 라벨 파일을 분리한다.
캐시에는 score / RLE / TTA 일치도 / 모델 버전을 전부 담고,
라벨 파일(labelme JSON)은 검수용으로 캐시에서 파생시킨다.
캐시가 있으면 검수 UI는 재추론 없이 즉시 열린다.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .config import DEFAULT, AutoLabelConfig
from .mask_utils import encode_rle, greedy_match, mask_bbox

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class RTMDetPredictor:
    """mmdet RTMDet-Ins 래퍼. 인스턴스 세그멘테이션 전용."""

    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda:0"):
        from mmdet.apis import init_detector  # 지연 임포트: 무거운 의존성

        self.config_path = str(config_path)
        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self.model = init_detector(self.config_path, self.checkpoint_path, device=device)
        self.classes = list(self.model.dataset_meta["classes"])

    @property
    def tag(self) -> str:
        """체크포인트 파일명을 모델 버전 태그로 사용."""
        return Path(self.checkpoint_path).stem

    def _raw_infer(self, img_bgr: np.ndarray, score_thr: float, max_instances: int) -> list[dict]:
        from mmdet.apis import inference_detector

        result = inference_detector(self.model, img_bgr)
        pred = result.pred_instances

        scores = pred.scores.detach().cpu().numpy()
        labels = pred.labels.detach().cpu().numpy()
        masks = pred.masks.detach().cpu().numpy().astype(bool)

        order = np.argsort(-scores)[:max_instances]
        out: list[dict] = []
        for idx in order:
            score = float(scores[idx])
            if score < score_thr:
                continue
            mask = masks[idx]
            bbox = mask_bbox(mask)
            if bbox is None:
                continue
            out.append(
                {
                    "label_id": int(labels[idx]),
                    "label": self.classes[int(labels[idx])],
                    "score": score,
                    "bbox": bbox,
                    "area": int(np.count_nonzero(mask)),
                    "mask": mask,
                }
            )
        return out

    def predict(self, img_bgr: np.ndarray, cfg: AutoLabelConfig = DEFAULT) -> list[dict]:
        """인스턴스 리스트 반환. TTA를 켜면 각 인스턴스에 tta_iou가 붙는다."""
        instances = self._raw_infer(img_bgr, cfg.score_thr, cfg.max_instances)

        if not cfg.use_tta_hflip:
            for inst in instances:
                inst["tta_iou"] = None
            return instances

        flipped = self._raw_infer(img_bgr[:, ::-1], cfg.score_thr, cfg.max_instances)
        for inst in flipped:
            inst["mask"] = inst["mask"][:, ::-1]  # 원본 좌표계로 되돌림

        matched, unmatched_a, _ = greedy_match(
            [i["mask"] for i in instances],
            [i["mask"] for i in flipped],
            min_iou=cfg.tta_match_iou,
        )
        for inst in instances:
            inst["tta_iou"] = None
        for i, _j, iou in matched:
            instances[i]["tta_iou"] = float(iou)
        for i in unmatched_a:
            # 반전 추론에서 대응 객체를 못 찾음 = 불안정한 검출
            instances[i]["tta_iou"] = 0.0
        return instances


# ----------------------------------------------------------------------
# 캐시 입출력
# ----------------------------------------------------------------------


def _cache_payload(
    image_path: Path,
    height: int,
    width: int,
    instances: list[dict],
    model_info: dict,
) -> dict:
    scores = [i["score"] for i in instances]
    tta = [i["tta_iou"] for i in instances if i.get("tta_iou") is not None]
    return {
        "image": image_path.name,
        "height": int(height),
        "width": int(width),
        "model": model_info,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "instances": [
            {
                "label": i["label"],
                "label_id": i["label_id"],
                "score": round(i["score"], 5),
                "bbox": i["bbox"],
                "area": i["area"],
                "tta_iou": (None if i.get("tta_iou") is None else round(i["tta_iou"], 5)),
                "rle": encode_rle(i["mask"]),
            }
            for i in instances
        ],
        "metrics": {
            "count": len(instances),
            "min_score": round(min(scores), 5) if scores else 0.0,
            "mean_score": round(float(np.mean(scores)), 5) if scores else 0.0,
            "mean_tta_iou": round(float(np.mean(tta)), 5) if tta else None,
        },
    }


def cache_path_for(cache_dir: Path, image_path: Path) -> Path:
    return cache_dir / f"{image_path.stem}.pred.json"


def load_cache(path: Path) -> dict:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def list_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def predict_folder(
    image_dir: Path,
    cache_dir: Path,
    predictor: RTMDetPredictor,
    cfg: AutoLabelConfig = DEFAULT,
    overwrite: bool = False,
    progress=print,
) -> list[Path]:
    """폴더 전체를 추론해서 예측 캐시를 남긴다."""
    image_dir = Path(image_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_info = {
        "config": predictor.config_path,
        "checkpoint": predictor.checkpoint_path,
        "tag": predictor.tag,
        "score_thr": cfg.score_thr,
        "tta_hflip": cfg.use_tta_hflip,
    }

    images = list_images(image_dir)
    written: list[Path] = []
    started = time.time()

    for n, image_path in enumerate(images, 1):
        out_path = cache_path_for(cache_dir, image_path)
        if out_path.exists() and not overwrite:
            written.append(out_path)
            continue

        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            progress(f"[skip] 읽기 실패: {image_path.name}")
            continue

        instances = predictor.predict(img, cfg)
        payload = _cache_payload(image_path, img.shape[0], img.shape[1], instances, model_info)
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False)
        written.append(out_path)

        if n % 20 == 0 or n == len(images):
            elapsed = time.time() - started
            progress(f"[predict] {n}/{len(images)}  ({elapsed:.1f}s)")

    return written