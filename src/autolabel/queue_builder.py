"""검수 대기열 구성 + labelme 실행.

labelme는 폴더 안 파일을 이름순으로만 보여준다.
그래서 우선순위 순서를 파일명 접두사(r00000_, r00001_ ...)로 인코딩해서
심볼릭 링크 폴더를 만든다. 원본 이미지는 복사하지 않는다.

산출물:
  <round_dir>/
    urgent/        <- 우선 검수 (이미지 링크 + labelme json)
    normal/        <- 나중 검수
    auto/          <- 사람 검수 생략, 바로 학습 투입
    labels.txt
    manifest.json  <- 검수 파일명 -> 원본 매핑, 생성 당시 JSON 해시
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from .config import DEFAULT, AutoLabelConfig
from .labelme_io import REVIEWED_FLAG, cache_to_labelme, write_labelme, write_labels_txt
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
    classes: list[str] | None = None,
    cfg: AutoLabelConfig = DEFAULT,
    overwrite: bool = False,
    progress: Callable[[str], None] = print,
) -> dict:
    """예측 캐시를 읽어 우선순위 대기열을 만든다.

    이미 대기열이 있으면 덮어쓰지 않는다: 다시 만들면 사람이 다듬은 JSON이 초안으로 돌아가
    작업이 사라진다. 정말 새로 만들 때만 overwrite=True.
    """
    image_dir, cache_dir, round_dir = Path(image_dir), Path(cache_dir), Path(round_dir)

    if (round_dir / "manifest.json").exists() and not overwrite:
        raise FileExistsError(
            f"이미 대기열이 있습니다: {round_dir} (다시 만들면 검수한 내용이 사라집니다. "
            f"새 라운드 이름을 쓰거나 overwrite 를 지정하세요)"
        )

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

    # 클래스 목록: 인자 > 캐시(추론 당시 모델의 class_names)
    if classes is None:
        classes = caches[0].get("model", {}).get("classes")
        if not classes:
            raise RuntimeError("캐시에 클래스 목록이 없습니다. --classes 로 직접 지정하세요.")
    classes = list(classes)

    # 라벨 검증: 캐시의 라벨이 클래스 목록에 없으면 labelme에서 라벨이 어긋난다
    seen = {i["label"] for c in caches for i in c["instances"]}
    unknown = sorted(seen - set(classes))
    if unknown:
        progress(f"[warn] 클래스 목록에 없는 라벨: {unknown}")

    ranked = rank(caches, cfg)
    review = [r for r in ranked if not r["auto_accept"]]
    auto = [r for r in ranked if r["auto_accept"]]
    n_urgent = max(1, int(round(len(review) * cfg.urgent_ratio))) if review else 0

    for entry in auto:
        entry["bucket"] = "auto"
    for pos, entry in enumerate(review):
        entry["bucket"] = "urgent" if pos < n_urgent else "normal"

    # overwrite 시 이전 산출물을 비운다 (남은 파일이 섞이지 않게)
    for name in BUCKETS:
        bucket_dir = round_dir / name
        if bucket_dir.exists():
            shutil.rmtree(bucket_dir)
        bucket_dir.mkdir(parents=True, exist_ok=True)

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
        raw = write_labelme(dst_json, payload)

        manifest[review_name] = {
            "bucket": bucket,
            "original_image": str(src_img.resolve()),
            "original_stem": stem,
            "cache": str((cache_dir / f"{stem}.pred.json").resolve()),
            "priority": round(entry["priority"], 4),
            "auto_accept": entry["auto_accept"],
            "n_instances": len(cache["instances"]),
            "min_score": cache["metrics"].get("min_score"),
            # 사람이 손댔는지 판별하기 위한 생성 당시 내용 해시
            "generated_sha256": hashlib.sha256(raw).hexdigest(),
        }

    write_labels_txt(round_dir / "labels.txt", classes)
    counts = {
        "total": len(ranked),
        "urgent": n_urgent,
        "normal": len(review) - n_urgent,
        "auto": len(auto),
    }
    with open(round_dir / "manifest.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "image_dir": str(image_dir.resolve()),
                "cache_dir": str(cache_dir.resolve()),
                "classes": classes,
                "counts": counts,
                "items": manifest,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )

    summary = {**counts, "auto_ratio": round(len(auto) / len(ranked), 4)}
    progress(
        f"[queue] 총 {summary['total']}장 | urgent {summary['urgent']} | "
        f"normal {summary['normal']} | auto {summary['auto']} "
        f"(자동승인 {summary['auto_ratio'] * 100:.1f}%)"
    )
    return summary


# ----------------------------------------------------------------------
# labelme 실행
# ----------------------------------------------------------------------

# labelme는 PySide6, 이 앱은 PyQt6 를 쓴다. 앱 프로세스가 Qt 플러그인 경로를 환경변수로
# 지정해 두었다면(opencv-python 이 import 시 QT_QPA_PLATFORM_PLUGIN_PATH 를 설정하는 것이
# 흔한 Qt 충돌 원인으로 알려져 있다) 자식 프로세스인 labelme 가 엉뚱한 플러그인을 로드하다
# 죽을 수 있다. 그래서 자식에게는 이 변수들을 넘기지 않는다.
_QT_ENV_KEYS = ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH", "QT_QPA_FONTDIR")


def clean_qt_env(env: dict[str, str] | None = None) -> dict[str, str]:
    base = dict(os.environ if env is None else env)
    for key in _QT_ENV_KEYS:
        base.pop(key, None)
    return base


def labelme_command(
    round_dir: Path,
    bucket: str = "urgent",
    executable: str | Sequence[str] = "labelme",
) -> list[str]:
    """labelme 실행 인자. 버전별로 이름이 바뀐 옵션(--nodata/--autosave 등)은 쓰지 않는다.

    labelme 7.x 기본값이 이미 자동저장 켜짐 / 이미지 데이터 미포함이다.
    --flags reviewed 로 이미지별 '검수완료' 체크박스를 띄운다.
    """
    round_dir = Path(round_dir)
    target = round_dir / bucket
    if not target.is_dir():
        raise FileNotFoundError(target)
    exe = [executable] if isinstance(executable, str) else list(executable)
    return [
        *exe,
        str(target),
        "--labels",
        str(round_dir / "labels.txt"),
        "--flags",
        REVIEWED_FLAG,
    ]


def launch_labelme(
    round_dir: Path,
    bucket: str = "urgent",
    executable: str | Sequence[str] = "labelme",
    log_path: Path | None = None,
) -> subprocess.Popen:
    """labelme를 앱과 분리된 프로세스로 띄운다 (창을 몇 시간 열어둬도 앱에 영향 없음)."""
    cmd = labelme_command(round_dir, bucket, executable)
    log_file = None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")
    try:
        return subprocess.Popen(
            cmd,
            env=clean_qt_env(),
            stdout=log_file or subprocess.DEVNULL,
            stderr=subprocess.STDOUT if log_file else subprocess.DEVNULL,
            start_new_session=True,
        )
    finally:
        if log_file is not None:
            log_file.close()  # 자식이 자기 핸들을 이미 갖고 있다
