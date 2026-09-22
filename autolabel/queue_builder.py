"""검수 대기열 구성.

labelme는 폴더 안 파일을 이름순으로만 보여준다.
그래서 우선순위 순서를 파일명 접두사(r0001_, r0002_ ...)로 인코딩해서
심볼릭 링크 폴더를 만든다. 원본 이미지는 복사하지 않는다.

산출물:
  <round_dir>/
    urgent/        <- 우선 검수 (심볼릭 링크 + labelme json)
    normal/        <- 나중 검수
    auto/          <- 사람 검수 생략, 바로 학습 투입
    labels.txt
    manifest.json  <- 검수용 파일명 -> 원본 매핑
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .config import DEFAULT, AutoLabelConfig
from .labelme_io import cache_to_labelme, write_labelme, write_labels_txt
from .predict import list_images, load_cache
from .uncertainty import rank

BUCKETS = ("urgent", "normal", "auto")


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(os.path.relpath(src, dst.parent), dst)
    except OSError:
        shutil.copy2(src, dst)


def build_queue(
    image_dir: Path,
    cache_dir: Path,
    round_dir: Path,
    classes: list[str],
    cfg: AutoLabelConfig = DEFAULT,
    progress=print,
) -> dict:
    """예측 캐시를 읽어 우선순위 대기열을 만든다."""
    image_dir, cache_dir, round_dir = Path(image_dir), Path(cache_dir), Path(round_dir)

    images = {p.stem: p for p in list_images(image_dir)}
    caches = []
    for cache_file in sorted(cache_dir.glob("*.pred.json")):
        cache = load_cache(cache_file)
        stem = Path(cache["image"]).stem
        if stem not in images:
            progress(f"[skip] 원본 이미지 없음: {cache['image']}")
            continue
        cache["_stem"] = stem
        caches.append(cache)

    if not caches:
        raise RuntimeError(f"예측 캐시가 없습니다: {cache_dir}")

    ranked = rank(caches, cfg)

    review = [r for r in ranked if not r["auto_accept"]]
    auto = [r for r in ranked if r["auto_accept"]]
    n_urgent = max(1, int(round(len(review) * cfg.urgent_ratio))) if review else 0

    # 버킷을 미리 배정해 둔다 (리스트 탐색 반복 방지)
    for entry in auto:
        entry["bucket"] = "auto"
    for pos, entry in enumerate(review):
        entry["bucket"] = "urgent" if pos < n_urgent else "normal"

    for name in BUCKETS:
        (round_dir / name).mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    for order, entry in enumerate(ranked):
        cache = entry["cache"]
        stem = cache["_stem"]
        src_img = images[stem]

        bucket = entry["bucket"]

        review_name = f"r{order:05d}_{stem}"
        dst_img = round_dir / bucket / f"{review_name}{src_img.suffix}"
        dst_json = round_dir / bucket / f"{review_name}.json"

        _link_or_copy(src_img, dst_img)
        payload = cache_to_labelme(cache, dst_img.name, cfg)
        write_labelme(dst_json, payload)

        manifest[review_name] = {
            "bucket": bucket,
            "original_image": str(src_img.resolve()),
            "original_stem": stem,
            "cache": str((cache_dir / f"{stem}.pred.json").resolve()),
            "priority": round(entry["priority"], 4),
            "auto_accept": entry["auto_accept"],
            "n_instances": len(cache["instances"]),
            "min_score": cache["metrics"].get("min_score"),
        }

    write_labels_txt(round_dir / "labels.txt", classes)
    with open(round_dir / "manifest.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "image_dir": str(image_dir.resolve()),
                "cache_dir": str(cache_dir.resolve()),
                "classes": classes,
                "counts": {
                    "total": len(ranked),
                    "urgent": n_urgent,
                    "normal": len(review) - n_urgent,
                    "auto": len(auto),
                },
                "items": manifest,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )

    summary = {
        "total": len(ranked),
        "urgent": n_urgent,
        "normal": len(review) - n_urgent,
        "auto": len(auto),
        "auto_ratio": round(len(auto) / len(ranked), 4),
    }
    progress(
        f"[queue] 총 {summary['total']}장 | urgent {summary['urgent']} | "
        f"normal {summary['normal']} | auto {summary['auto']} "
        f"(자동승인 {summary['auto_ratio'] * 100:.1f}%)"
    )
    return summary


def launch_labelme(round_dir: Path, bucket: str = "urgent") -> subprocess.Popen:
    """labelme를 별도 프로세스로 띄운다.

    Qt 이벤트 루프 충돌을 피하려고 PyQt6 앱에 임베드하지 않고
    별도 프로세스로 분리한다 (Open3D 뷰어와 같은 패턴).
    PyQt6 앱 안에서는 subprocess 대신 QProcess.startDetached 를 쓰면 된다.
    """
    round_dir = Path(round_dir)
    target = round_dir / bucket
    if not target.exists():
        raise FileNotFoundError(target)

    cmd = [
        "labelme",
        str(target),
        "--labels",
        str(round_dir / "labels.txt"),
        "--nodata",
        "--autosave",
        "--nosortlabels",
    ]
    return subprocess.Popen(cmd)