"""예측 캐시의 마스크를 CAD 투영 실루엣 + 깊이로 보정한 '보정 캐시'를 만든다.

predict(원본 캐시) -> refine(보정 캐시) -> queue 순서로 쓴다.
원본 캐시는 건드리지 않는다: 보정을 끄면 queue 가 원본 캐시를 그대로 읽으면 된다.

보정 캐시는 원본과 같은 형식이다 (rle / bbox / area 만 교체). 인스턴스에 "refine" 항목이
추가되며 label/score/tta_iou 는 그대로다. 보정에 실패한 인스턴스(깊이 파일 없음, score 낮음 등)는
원본 마스크를 유지한다. 보정 캐시가 검수 초안의 기준이 되므로 merge 의 edit_iou 도
'보정된 초안 대비 사람이 얼마나 고쳤나'가 된다.

점군은 이미지 폴더가 <세션>/intensity 일 때 <세션>/pointcloud_organized, <세션>/valid_mask 에서 읽는다.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from app.core.camera_intrinsics import estimate_intrinsics_from_organized_pcd
from src.detection.mask_refine import RefineParams, get_bank, load_session_arrays, refine_masks

from .mask_utils import decode_rle, encode_rle, mask_bbox
from .predict import cache_path_for, list_images, load_cache


def refine_cache_folder(
    image_dir: Path,
    cache_dir: Path,
    out_dir: Path,
    cad_path: Path,
    params: RefineParams | None = None,
    progress: Callable[[str], None] = print,
) -> dict:
    image_dir, cache_dir, out_dir = Path(image_dir), Path(cache_dir), Path(out_dir)
    if not Path(cad_path).is_file():
        raise FileNotFoundError(f"CAD 파일이 없습니다: {cad_path}")
    params = params or RefineParams()
    bank = get_bank(cad_path)
    if bank.aspect < params.elong_min:
        progress(f"[warn] CAD가 길쭉하지 않습니다(aspect {bank.aspect:.1f}). 이 보정은 길쭉한 축대칭 부품용입니다.")

    images = list_images(image_dir)
    if not images:
        raise RuntimeError(f"이미지가 없습니다: {image_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_inst = n_ok = n_no_data = n_missing_cache = 0
    reasons: dict[str, int] = {}
    started = time.time()

    for n, image_path in enumerate(images, 1):
        src = cache_path_for(cache_dir, image_path)
        if not src.exists():
            n_missing_cache += 1
        else:
            cache = load_cache(src)
            try:
                pcd, valid = load_session_arrays(image_path)
                intr = estimate_intrinsics_from_organized_pcd(pcd, valid)
                data_err = None
            except Exception as exc:  # noqa: BLE001 - 이미지 하나 때문에 전체가 멈추지 않게
                pcd = valid = intr = None
                data_err = str(exc).splitlines()[0]
                n_no_data += 1
                if n_no_data <= 3:
                    progress(f"[warn] {image_path.name}: 보정 생략 ({data_err})")

            decoded: list[tuple[dict, np.ndarray]] = []
            for inst in cache["instances"]:
                n_inst += 1
                inst["rle_raw"] = inst["rle"]  # 원본 보존 (디버깅/비교용)
                if data_err is not None:
                    inst["refine"] = {"ok": False, "reason": "no_session_data"}
                    continue
                mask = decode_rle(inst["rle"])
                if mask.shape != valid.shape:
                    inst["refine"] = {"ok": False, "reason": "shape_mismatch"}
                    continue
                decoded.append((inst, mask))
            # 한 이미지의 인스턴스를 함께 보정해야 마스크끼리 겹친 픽셀을 정리할 수 있다
            batch = refine_masks([m for _, m in decoded], pcd, valid, intr, bank, params) if decoded else []
            for (inst, mask), res in zip(decoded, batch):
                info = {"ok": bool(res.ok), "reason": res.reason, "score": round(float(res.score), 4),
                        "phi_deg": round(float(res.phi_deg), 1), "theta_deg": round(float(res.theta_deg), 1),
                        "area_before": int(res.n_before), "area_after": int(res.n_after)}
                inst["refine"] = info
                if res.ok:
                    box = mask_bbox(res.mask)
                    if box is None:  # 방어: 빈 결과면 원본 유지
                        info["ok"], info["reason"] = False, "empty_result"
                    else:
                        inst["rle"] = encode_rle(res.mask)
                        inst["bbox"] = box
                        inst["area"] = int(np.count_nonzero(res.mask))
                        n_ok += 1
                if not info["ok"]:
                    reasons[info["reason"]] = reasons.get(info["reason"], 0) + 1
            cache["model"] = {**cache.get("model", {}), "mask_refine": {
                "cad": str(cad_path), "depth_tol_mm": params.depth_tol_mm, "min_score": params.min_score}}
            with open(out_dir / src.name, "w", encoding="utf-8") as fp:
                json.dump(cache, fp, ensure_ascii=False)

        if n % 10 == 0 or n == len(images):
            progress(f"[refine] {n}/{len(images)}  ({time.time() - started:.1f}s)")

    summary = {"images": len(images), "instances": n_inst, "refined": n_ok, "no_session_data": n_no_data,
               "missing_cache": n_missing_cache, "fail_reasons": reasons}
    progress(f"[refine] 인스턴스 {n_inst}개 중 {n_ok}개 보정 | 세션 데이터 없는 이미지 {n_no_data}장 | 포기 사유 {reasons}")
    if n_inst and n_ok == 0:
        progress("[warn] 보정된 인스턴스가 0개입니다. 이미지 폴더가 <세션>/intensity 인지, CAD가 맞는지 확인하세요.")
    return summary