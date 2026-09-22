"""마스크 관련 순수 함수 모음.

외부 의존성은 numpy / opencv 뿐이다.
여기 있는 함수들은 labelme 경로와 CVAT 경로 양쪽에서 그대로 재사용된다.
나중에 CVAT으로 옮기더라도 이 파일은 손대지 않는다.
"""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

# ----------------------------------------------------------------------
# RLE (열 우선, COCO uncompressed 방식과 동일한 순서)
# ----------------------------------------------------------------------


def encode_rle(mask: np.ndarray) -> dict:
    """불리언 2D 배열을 JSON 직렬화 가능한 RLE 딕셔너리로 변환."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    flat = mask.transpose().reshape(-1)  # 열 우선

    # 0에서 시작하는 런 길이 시퀀스
    changes = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], changes, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat.size and flat[0]:
        counts = [0] + counts  # 첫 런이 1이면 길이 0짜리 0-런을 앞에 넣는다

    return {"size": [int(h), int(w)], "counts": [int(c) for c in counts]}


def decode_rle(rle: dict) -> np.ndarray:
    """encode_rle의 역변환."""
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    value = False
    for count in rle["counts"]:
        if value and count:
            flat[pos : pos + count] = True
        pos += count
        value = not value
    return flat.reshape(w, h).transpose()


# ----------------------------------------------------------------------
# 기하 유틸
# ----------------------------------------------------------------------


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = np.count_nonzero(a | b)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(a & b) / union)


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    """[x1, y1, x2, y2] (양 끝 포함). 빈 마스크면 None."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


# ----------------------------------------------------------------------
# 마스크 -> 폴리곤
# ----------------------------------------------------------------------


def _simplify(contour: np.ndarray, eps_ratio: float, max_vertices: int) -> np.ndarray:
    """정점 수가 max_vertices 이하가 될 때까지 epsilon을 키우며 단순화."""
    arc = cv2.arcLength(contour, True)
    ratio = eps_ratio
    for _ in range(6):
        approx = cv2.approxPolyDP(contour, ratio * arc, True).reshape(-1, 2)
        if len(approx) <= max_vertices:
            return approx
        ratio *= 1.8
    return approx


def mask_to_polygons(
    mask: np.ndarray,
    eps_ratio: float = 0.0025,
    min_area: int = 120,
    keep_holes: bool = False,
    max_vertices: int = 80,
) -> list[dict]:
    """불리언 마스크를 폴리곤 리스트로 변환.

    반환 형식: [{"points": (N,2) float ndarray, "is_hole": bool}, ...]
    keep_holes=True면 내부 컨투어(관통홀)도 is_hole=True로 포함한다.
    """
    m = np.asarray(mask, dtype=np.uint8)
    mode = cv2.RETR_CCOMP if keep_holes else cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(m, mode, cv2.CHAIN_APPROX_SIMPLE)

    out: list[dict] = []
    for idx, contour in enumerate(contours):
        if cv2.contourArea(contour) < min_area:
            continue
        approx = _simplify(contour, eps_ratio, max_vertices)
        if len(approx) < 3:
            continue
        is_hole = False
        if keep_holes and hierarchy is not None:
            # RETR_CCOMP: hierarchy[0][i][3] != -1 이면 내부 컨투어
            is_hole = bool(hierarchy[0][idx][3] != -1)
        out.append({"points": approx.astype(float), "is_hole": is_hole})
    return out


def polygons_to_mask(
    polygons: Iterable[np.ndarray], height: int, width: int, holes: Iterable[np.ndarray] = ()
) -> np.ndarray:
    """폴리곤 리스트를 불리언 마스크로 래스터화 (검수 전후 IoU 비교용)."""
    canvas = np.zeros((height, width), dtype=np.uint8)
    outer = [np.round(np.asarray(p)).astype(np.int32) for p in polygons if len(p) >= 3]
    if outer:
        cv2.fillPoly(canvas, outer, 1)
    inner = [np.round(np.asarray(p)).astype(np.int32) for p in holes if len(p) >= 3]
    if inner:
        cv2.fillPoly(canvas, inner, 0)
    return canvas.astype(bool)


def greedy_match(
    masks_a: list[np.ndarray], masks_b: list[np.ndarray], min_iou: float = 0.3
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """두 인스턴스 집합을 IoU 기준 그리디 매칭.

    반환: (매칭쌍 [(i, j, iou)], A의 미매칭 인덱스, B의 미매칭 인덱스)
    """
    if not masks_a or not masks_b:
        return [], list(range(len(masks_a))), list(range(len(masks_b)))

    pairs: list[tuple[float, int, int]] = []
    for i, a in enumerate(masks_a):
        for j, b in enumerate(masks_b):
            iou = mask_iou(a, b)
            if iou >= min_iou:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)

    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for iou, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((i, j, iou))

    left_a = [i for i in range(len(masks_a)) if i not in used_a]
    left_b = [j for j in range(len(masks_b)) if j not in used_b]
    return matched, left_a, left_b