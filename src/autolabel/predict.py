"""배치 추론 -> 예측 캐시 저장.

추론은 앱의 다른 탭(app/core/detector.py)과 똑같은 경로를 탄다:
  - 이미지는 cv2.IMREAD_GRAYSCALE 로 읽어 3채널로 복제한다.
  - 검출기는 src.detection.create_detector() 로 만든다 (DetectorBase.infer).
그래서 탭 3에서 보이는 검출 결과와 오토라벨링이 만드는 초안이 같다.

핵심 설계: 예측 캐시와 최종 라벨 파일을 분리한다.
캐시에는 score / RLE / TTA 일치도 / 모델 버전 / 클래스 목록을 전부 담고,
라벨 파일(labelme JSON)은 검수용으로 캐시에서 파생시킨다.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .config import DEFAULT, AutoLabelConfig
from .mask_utils import encode_rle, greedy_match, mask_bbox

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}  # inference_test_tab.IMAGE_EXTS 와 동일


def build_detector(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda:0",
    score_threshold: float = DEFAULT.score_thr,
    backend: str = "rtmdet_ins",
):
    """앱의 검출기 팩토리로 RTMDet-Ins 검출기를 만든다 (mmdet은 여기서 처음 로드)."""
    from src.detection import create_detector  # 지연 임포트: 무거운 의존성

    return create_detector(
        {
            "type": backend,
            "params": {
                "config": str(config_path),
                "checkpoint": str(checkpoint_path),
                "device": device,
                "score_threshold": float(score_threshold),
            },
        }
    )


def read_image_like_app(path: Path) -> np.ndarray | None:
    """app/core/detector.py 의 predict() 와 같은 방식으로 이미지를 읽는다.

    그레이스케일로 읽은 뒤 3채널로 복제한다. 읽기 실패면 None.
    """
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    return np.stack([gray, gray, gray], axis=-1)


class DetectorPredictor:
    """DetectorBase 구현체를 감싸 오토라벨링용 인스턴스 dict를 만든다.

    detector 는 `infer(image) -> list[DetectionResult]` 와 `class_names` 만 있으면 된다
    (테스트에서는 가짜 검출기를 주입할 수 있다).
    """

    def __init__(self, detector: Any, config_path: str = "", checkpoint_path: str = "", tag: str = ""):
        self.detector = detector
        self.config_path = str(config_path)
        self.checkpoint_path = str(checkpoint_path)
        self.classes = [str(c) for c in detector.class_names]
        self.tag = tag or (Path(checkpoint_path).stem if checkpoint_path else "unknown")

    def _infer(self, img_bgr: np.ndarray, max_instances: int) -> list[dict]:
        results = self.detector.infer(np.ascontiguousarray(img_bgr))
        # DetectorBase 규약상 score 내림차순이지만 방어적으로 한 번 더 정렬한다.
        results = sorted(results, key=lambda r: -float(r.score))[:max_instances]

        out: list[dict] = []
        for r in results:
            mask = np.asarray(r.mask, dtype=bool)
            bbox = mask_bbox(mask)
            if bbox is None:
                continue
            out.append(
                {
                    "label_id": int(r.class_id),
                    "label": str(r.class_name),
                    "score": float(r.score),
                    "bbox": bbox,
                    "area": int(np.count_nonzero(mask)),
                    "mask": mask,
                }
            )
        return out

    def predict(self, img_bgr: np.ndarray, cfg: AutoLabelConfig = DEFAULT) -> list[dict]:
        """인스턴스 리스트 반환. TTA를 켜면 각 인스턴스에 tta_iou가 붙는다."""
        instances = self._infer(img_bgr, cfg.max_instances)

        if not cfg.use_tta_hflip:
            for inst in instances:
                inst["tta_iou"] = None
            return instances

        # 음수 stride 배열은 torch 변환에서 깨질 수 있으므로 연속 배열로 만든다.
        flipped = self._infer(np.ascontiguousarray(img_bgr[:, ::-1]), cfg.max_instances)
        for inst in flipped:
            inst["mask"] = np.ascontiguousarray(inst["mask"][:, ::-1])  # 원본 좌표계로 복원

        matched, unmatched, _ = greedy_match(
            [i["mask"] for i in instances],
            [i["mask"] for i in flipped],
            min_iou=cfg.tta_match_iou,
        )
        for inst in instances:
            inst["tta_iou"] = None
        for i, _j, iou in matched:
            instances[i]["tta_iou"] = float(iou)
        for i in unmatched:
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
    return sorted(
        p for p in Path(image_dir).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def predict_folder(
    image_dir: Path,
    cache_dir: Path,
    predictor: DetectorPredictor,
    cfg: AutoLabelConfig = DEFAULT,
    overwrite: bool = False,
    progress: Callable[[str], None] = print,
) -> list[Path]:
    """폴더 전체를 추론해서 예측 캐시를 남긴다."""
    image_dir = Path(image_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_info = {
        "config": predictor.config_path,
        "checkpoint": predictor.checkpoint_path,
        "tag": predictor.tag,
        "classes": predictor.classes,
        "score_thr": cfg.score_thr,
        "tta_hflip": cfg.use_tta_hflip,
    }

    images = list_images(image_dir)
    if not images:
        raise RuntimeError(f"이미지가 없습니다: {image_dir}")

    written: list[Path] = []
    started = time.time()

    for n, image_path in enumerate(images, 1):
        out_path = cache_path_for(cache_dir, image_path)
        if out_path.exists() and not overwrite:
            written.append(out_path)
        else:
            img = read_image_like_app(image_path)
            if img is None:
                progress(f"[skip] 읽기 실패: {image_path.name}")
            else:
                instances = predictor.predict(img, cfg)
                payload = _cache_payload(image_path, img.shape[0], img.shape[1], instances, model_info)
                with open(out_path, "w", encoding="utf-8") as fp:
                    json.dump(payload, fp, ensure_ascii=False)
                written.append(out_path)

        # 탭의 진행률 바가 이 형식을 파싱한다: "[predict] n/N"
        if n % 10 == 0 or n == len(images):
            progress(f"[predict] {n}/{len(images)}  ({time.time() - started:.1f}s)")

    return written
