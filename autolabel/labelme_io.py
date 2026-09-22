"""예측 캐시 <-> labelme JSON 변환.

labelme는 shapes[].flags 를 그대로 보존해준다.
여기에 provenance(auto / score / model)를 심어두면
나중에 "자동 라벨이 모델을 망쳤나"를 검증할 수 있다.
이 필드가 없으면 검증할 방법이 사실상 없다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import DEFAULT, AutoLabelConfig
from .mask_utils import decode_rle, mask_to_polygons, polygons_to_mask

LABELME_VERSION = "5.5.0"


def cache_to_labelme(
    cache: dict,
    image_filename: str,
    cfg: AutoLabelConfig = DEFAULT,
) -> dict:
    """예측 캐시 한 건을 labelme JSON 딕셔너리로 변환."""
    height = cache["height"]
    width = cache["width"]
    model_tag = cache.get("model", {}).get("tag", "unknown")

    shapes: list[dict] = []
    for group_id, inst in enumerate(cache["instances"]):
        mask = decode_rle(inst["rle"])
        polys = mask_to_polygons(
            mask,
            eps_ratio=cfg.approx_eps_ratio,
            min_area=cfg.min_mask_area,
            keep_holes=cfg.keep_holes,
            max_vertices=cfg.max_vertices,
        )
        for poly in polys:
            shapes.append(
                {
                    "label": inst["label"],
                    "points": [[float(x), float(y)] for x, y in poly["points"]],
                    "group_id": group_id if cfg.keep_holes else None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {
                        "auto": True,
                        "hole": bool(poly["is_hole"]),
                        "score": float(inst["score"]),
                        "model": model_tag,
                    },
                    "mask": None,
                }
            )

    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,  # --nodata 와 동일. 파일 크기를 수십 배 줄인다.
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def write_labelme(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def read_labelme(path: Path) -> dict:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def labelme_to_masks(payload: dict) -> list[dict]:
    """labelme JSON을 인스턴스 단위 마스크로 복원.

    group_id가 있으면 같은 group_id끼리 한 인스턴스로 묶고,
    flags.hole 이 True인 shape는 구멍으로 처리한다.
    """
    height = payload["imageHeight"]
    width = payload["imageWidth"]

    groups: dict[object, dict] = {}
    for idx, shape in enumerate(payload.get("shapes", [])):
        if shape.get("shape_type") != "polygon":
            continue
        pts = np.asarray(shape["points"], dtype=float)
        if len(pts) < 3:
            continue
        gid = shape.get("group_id")
        key = ("g", gid) if gid is not None else ("s", idx)
        entry = groups.setdefault(
            key,
            {
                "label": shape["label"],
                "outer": [],
                "holes": [],
                "flags": shape.get("flags", {}) or {},
            },
        )
        if (shape.get("flags") or {}).get("hole"):
            entry["holes"].append(pts)
        else:
            entry["outer"].append(pts)

    out: list[dict] = []
    for entry in groups.values():
        if not entry["outer"]:
            continue
        mask = polygons_to_mask(entry["outer"], height, width, entry["holes"])
        if not mask.any():
            continue
        out.append(
            {
                "label": entry["label"],
                "mask": mask,
                "outer": entry["outer"],
                "holes": entry["holes"],
                "flags": entry["flags"],
            }
        )
    return out


def write_labels_txt(path: Path, classes: list[str]) -> None:
    """labelme --labels 로 넘길 클래스 목록 파일."""
    lines = ["__ignore__", "_background_", *classes]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")