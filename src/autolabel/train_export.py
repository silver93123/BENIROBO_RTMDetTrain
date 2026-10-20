"""오토라벨 export(COCO) -> 기존 학습 파이프라인용 데이터셋 폴더 변환.

기존 학습(scripts/2_Train_rtmdet_model.py)이 기대하는 구조:
    data/dataset/<name>/intensity/<이미지들>
    data/dataset/<name>/annotations/instances_Train.json
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable


def export_training_dataset(
    autolabel_dataset: Path,
    out_dir: Path,
    overwrite: bool = False,
    progress: Callable[[str], None] = print,
) -> tuple[int, int]:
    """annotations.json 에 들어 있는 이미지만 out_dir/intensity 로 모으고
    out_dir/annotations/instances_Train.json 을 만든다. (이미지 수, 인스턴스 수) 반환.

    overwrite=True 면 out_dir 의 intensity/, annotations/ 만 지운다 (그 밖의 파일은 건드리지 않음).
    """
    src, dst = Path(autolabel_dataset), Path(out_dir)
    ann_path = src / "annotations.json"
    if not ann_path.is_file():
        raise RuntimeError(f"{ann_path} 없음 - 먼저 '병합하고 COCO 내보내기'를 실행하세요.")
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    if not coco["images"]:
        raise RuntimeError("annotations.json 에 이미지가 없습니다 (검수 완료된 이미지가 0장).")

    img_dir, ann_dir = dst / "intensity", dst / "annotations"
    if img_dir.exists() or ann_dir.exists():
        if not overwrite:
            raise RuntimeError(f"이미 있는 데이터셋입니다 (덮어쓰지 않음): {dst}")
        shutil.rmtree(img_dir, ignore_errors=True)
        shutil.rmtree(ann_dir, ignore_errors=True)

    # 이미지를 먼저 전부 찾아 둔다 (하나라도 없으면 아무것도 만들지 않고 실패)
    plan: list[tuple[Path, str]] = []
    for im in coco["images"]:
        name = im["file_name"]
        cands = [src / name]
        lj = src / f"{Path(name).stem}.json"
        if lj.is_file():  # labelme 의 imagePath(상대경로) 기준으로도 찾는다
            cands.append((src / json.loads(lj.read_text(encoding="utf-8"))["imagePath"]).resolve())
        found = next((c for c in cands if c.is_file()), None)
        if found is None:
            raise RuntimeError(f"이미지를 찾을 수 없습니다: {name}")
        plan.append((found, name))

    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    for found, name in plan:
        shutil.copy2(found, img_dir / name)
    (ann_dir / "instances_Train.json").write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    progress(f"[train] {len(coco['images'])}장 / {len(coco['annotations'])}개 인스턴스 -> {dst}")
    return len(coco["images"]), len(coco["annotations"])
