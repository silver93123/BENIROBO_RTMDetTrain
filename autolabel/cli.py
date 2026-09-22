"""오토라벨링 CLI.

전형적인 한 라운드:

  python -m autolabel predict  --images data/new_2026w38 --cache cache/round03 \\
                               --config configs/rtmdet-ins.py --checkpoint work/best.pth
  python -m autolabel queue    --images data/new_2026w38 --cache cache/round03 \\
                               --round rounds/round03 --classes part_a,part_b
  python -m autolabel edit     --round rounds/round03 --bucket urgent
  python -m autolabel edit     --round rounds/round03 --bucket normal
  python -m autolabel merge    --round rounds/round03 --dataset dataset/round03
  python -m autolabel export   --dataset dataset/round03 --classes part_a,part_b \\
                               --out dataset/round03/annotations.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import DEFAULT
from .merge import export_coco, merge_round
from .queue_builder import build_queue, launch_labelme


def _classes(value: str) -> list[str]:
    items = [c.strip() for c in value.split(",") if c.strip()]
    if not items:
        raise argparse.ArgumentTypeError("클래스 목록이 비어 있습니다")
    return items


def _cfg_from_args(args) -> object:
    overrides = {}
    if getattr(args, "score_thr", None) is not None:
        overrides["score_thr"] = args.score_thr
    if getattr(args, "no_tta", False):
        overrides["use_tta_hflip"] = False
    return replace(DEFAULT, **overrides) if overrides else DEFAULT


def cmd_predict(args) -> int:
    from .predict import RTMDetPredictor, predict_folder

    cfg = _cfg_from_args(args)
    predictor = RTMDetPredictor(args.config, args.checkpoint, args.device)
    print(f"[model] {predictor.tag} | classes={predictor.classes}")
    predict_folder(
        Path(args.images),
        Path(args.cache),
        predictor,
        cfg,
        overwrite=args.overwrite,
    )
    return 0


def cmd_queue(args) -> int:
    cfg = _cfg_from_args(args)
    build_queue(
        Path(args.images),
        Path(args.cache),
        Path(args.round),
        args.classes,
        cfg,
    )
    return 0


def cmd_edit(args) -> int:
    proc = launch_labelme(Path(args.round), args.bucket)
    print(f"[edit] labelme 실행 (pid={proc.pid}). 창을 닫으면 종료됩니다.")
    return proc.wait()


def cmd_merge(args) -> int:
    cfg = _cfg_from_args(args)
    merge_round(
        Path(args.round),
        Path(args.dataset),
        cfg,
        include_auto=not args.exclude_auto,
        copy_images=args.copy_images,
    )
    return 0


def cmd_export(args) -> int:
    export_coco(Path(args.dataset), args.classes, Path(args.out))
    return 0


def cmd_stats(args) -> int:
    path = Path(args.round) / "round_stats.json"
    if not path.exists():
        print(f"통계 파일이 없습니다: {path}", file=sys.stderr)
        return 1
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    summary = data["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autolabel", description="RTMDet 오토라벨링 파이프라인")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="배치 추론 -> 예측 캐시")
    p.add_argument("--images", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--config", required=True, help="mmdet config .py")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--score-thr", type=float, default=None)
    p.add_argument("--no-tta", action="store_true", help="좌우반전 TTA 생략 (추론 2배 빠름)")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("queue", help="우선순위 검수 대기열 생성")
    p.add_argument("--images", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--round", required=True)
    p.add_argument("--classes", required=True, type=_classes)
    p.set_defaults(func=cmd_queue)

    p = sub.add_parser("edit", help="labelme 실행")
    p.add_argument("--round", required=True)
    p.add_argument("--bucket", default="urgent", choices=["urgent", "normal", "auto"])
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("merge", help="검수 결과를 데이터셋으로 확정")
    p.add_argument("--round", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--exclude-auto", action="store_true", help="자동승인분 제외")
    p.add_argument("--copy-images", action="store_true")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("export", help="labelme -> COCO 변환")
    p.add_argument("--dataset", required=True)
    p.add_argument("--classes", required=True, type=_classes)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("stats", help="라운드 품질 지표 출력")
    p.add_argument("--round", required=True)
    p.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())