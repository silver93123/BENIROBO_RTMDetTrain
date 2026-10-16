"""예측 캐시 <-> labelme JSON 변환.

labelme 7.x 는 JSON을 읽을 때 엄격하게 검증한다 (labelme/_label_file.py 확인):
  - shape.flags / 최상위 flags 의 값은 전부 bool 이어야 한다 (float/str 이면 로드 실패)
  - group_id 는 int 또는 None
  - imageHeight / imageWidth 가 실제 이미지 크기와 다르면 로드 실패
그래서 provenance(score, 모델)는 flags가 아니라 shape.description(문자열)에 넣는다.
description 은 shape와 함께 편집/삭제/복제를 따라다니므로 출처 추적에 적합하다.

검수 완료 표시는 이미지 단위 flag `reviewed` (labelme --flags reviewed) 로 받는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import DEFAULT, AutoLabelConfig
from .mask_utils import decode_rle, mask_to_polygons, polygons_to_mask

LABELME_VERSION = "7.0.0"  # labelme는 읽을 때 이 값을 검사하지 않는다
REVIEWED_FLAG = "reviewed"
PROVENANCE_PREFIX = "auto"


# ----------------------------------------------------------------------
# provenance (shape.description 문자열)
# ----------------------------------------------------------------------


def make_provenance(score: float, model_tag: str) -> str:
    return f"{PROVENANCE_PREFIX}|score={score:.3f}|model={model_tag}"


def parse_provenance(description: str | None) -> dict:
    """'auto|score=0.870|model=tag' -> {'auto': True, 'score': 0.87, 'model': 'tag'}. 아니면 {}."""
    if not description or not description.startswith(PROVENANCE_PREFIX + "|"):
        return {}
    out: dict = {"auto": True}
    for part in description.split("|")[1:]:
        key, _, value = part.partition("=")
        if key == "score":
            try:
                out["score"] = float(value)
            except ValueError:
                pass
        elif key == "model":
            out["model"] = value
    return out


# ----------------------------------------------------------------------
# 캐시 -> labelme
# ----------------------------------------------------------------------


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
                    # keep_holes=False 면 인스턴스 하나가 shape 하나이므로 group_id 불필요
                    "group_id": group_id if cfg.keep_holes else None,
                    "description": make_provenance(float(inst["score"]), model_tag),
                    "shape_type": "polygon",
                    "flags": {"hole": bool(poly["is_hole"])},  # bool 만 허용
                    "mask": None,
                }
            )

    return {
        "version": LABELME_VERSION,
        "flags": {REVIEWED_FLAG: False},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,  # 이미지는 JSON 옆 파일(심볼릭 링크)에서 읽는다
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def write_labelme(path: Path, payload: dict) -> bytes:
    """JSON을 쓰고, 쓴 바이트를 반환한다 (미검수 판별용 해시 계산에 사용)."""
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    Path(path).write_bytes(data)
    return data


def read_labelme(path: Path) -> dict:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


# ----------------------------------------------------------------------
# labelme -> 마스크
# ----------------------------------------------------------------------


def is_marked_reviewed(payload: dict) -> bool:
    """이미지 단위 flag `reviewed` 가 켜져 있는가."""
    return bool((payload.get("flags") or {}).get(REVIEWED_FLAG, False))


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
                "provenance": parse_provenance(shape.get("description")),
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
                "provenance": entry["provenance"],
            }
        )
    return out


def write_labels_txt(path: Path, classes: list[str]) -> None:
    """labelme --labels 로 넘길 클래스 목록 (한 줄에 하나, 다른 항목을 섞지 않는다)."""
    Path(path).write_text("\n".join(classes) + "\n", encoding="utf-8")
