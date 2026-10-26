"""오토라벨링 CLI (레포 루트에서 실행: python -m src.autolabel ...).

전형적인 한 라운드:

  python -m src.autolabel predict --images data/new_images --cache data/autolabel/cache/round03 \\
                                  --config work_dirs/xxx/cfg.py --checkpoint work_dirs/xxx/best.pth
  python -m src.autolabel refine  --images data/new_images --cache data/autolabel/cache/round03 \\
                                  --out data/autolabel/cache_refined/round03 --cad data/cad/bolt.stl   # 선택
  python -m src.autolabel queue   --images data/new_images --cache data/autolabel/cache/round03 \\
                                  --round data/autolabel/rounds/round03
  python -m src.autolabel edit    --round data/autolabel/rounds/round03 --bucket urgent
  python -m src.autolabel merge   --round data/autolabel/rounds/round03 \\
                                  --dataset data/autolabel/dataset/round03
  python -m src.autolabel export  --dataset data/autolabel/dataset/round03
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import DEFAULT
from .merge import export_coco, merge_round
from .train_export import export_training_dataset
from .queue_builder import build_queue, launch_labelme


def _classes(value: str) -> list[str]:
    items = [c.strip() for c in value.split(",") if c.strip()]
    if not items:
        raise argparse.ArgumentTypeError("클래스 목록이 비어 있습니다")
    return items


def _cfg_from_args(args):
    overrides = {}
    if getattr(args, "score_thr", None) is not None:
        overrides["score_thr"] = args.score_thr
    if getattr(args, "no_tta", False):
        overrides["use_tta_hflip"] = False
    if getattr(args, "max_vertices", None) is not None:
        overrides["max_vertices"] = args.max_vertices
    if getattr(args, "eps_ratio", None) is not None:
        overrides["approx_eps_ratio"] = args.eps_ratio
    return replace(DEFAULT, **overrides) if overrides else DEFAULT


def cmd_predict(args) -> int:
    from .predict import DetectorPredictor, build_detector, predict_folder

    cfg = _cfg_from_args(args)
    detector = build_detector(args.config, args.checkpoint, args.device, cfg.score_thr)
    predictor = DetectorPredictor(detector, args.config, args.checkpoint)
    print(f"[model] {predictor.tag} | classes={predictor.classes}", flush=True)
    predict_folder(
        Path(args.images),
        Path(args.cache),
        predictor,
        cfg,
        overwrite=args.overwrite,
        progress=lambda msg: print(msg, flush=True),
    )
    return 0


def cmd_refine(args) -> int:
    from src.detection.mask_refine import RefineParams

    from .refine_cache import refine_cache_folder

    params = replace(RefineParams(), depth_tol_mm=args.depth_tol, min_score=args.min_score)
    try:
        refine_cache_folder(
            Path(args.images), Path(args.cache), Path(args.out), Path(args.cad), params,
            progress=lambda msg: print(msg, flush=True),
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    return 0


def cmd_queue(args) -> int:
    cfg = _cfg_from_args(args)
    try:
        build_queue(
            Path(args.images),
            Path(args.cache),
            Path(args.round),
            args.classes,
            cfg,
            overwrite=args.overwrite,
        )
    except FileExistsError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    return 0


def cmd_edit(args) -> int:
    proc = launch_labelme(
        Path(args.round),
        args.bucket,
        args.labelme,
        log_path=Path(args.round) / "labelme.log",
    )
    print(f"[edit] labelme 실행 (pid={proc.pid}) — 로그: {Path(args.round) / 'labelme.log'}")
    return 0


def cmd_merge(args) -> int:
    cfg = _cfg_from_args(args)
    merge_round(
        Path(args.round),
        Path(args.dataset),
        cfg,
        include_auto=not args.exclude_auto,
        include_unreviewed=args.include_unreviewed,
        copy_images=args.copy_images,
    )
    return 0


def cmd_export(args) -> int:
    export_coco(
        Path(args.dataset),
        args.classes,
        Path(args.out) if args.out else None,
    )
    return 0


def cmd_to_train(args) -> int:
    export_training_dataset(Path(args.dataset), Path(args.out), overwrite=args.overwrite)
    return 0


def cmd_stats(args) -> int:
    path = Path(args.round) / "round_stats.json"
    if not path.exists():
        print(f"통계 파일이 없습니다: {path}", file=sys.stderr)
        return 1
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    print(json.dumps(data["summary"], ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.autolabel", description="RTMDet 오토라벨링 파이프라인")
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

    p = sub.add_parser("refine", help="예측 캐시 마스크를 CAD 투영 실루엣+깊이로 보정 -> 보정 캐시 (선택 단계)")
    p.add_argument("--images", required=True, help="<세션>/intensity 폴더 (같은 세션의 점군이 필요)")
    p.add_argument("--cache", required=True, help="원본 예측 캐시 (수정하지 않음)")
    p.add_argument("--out", required=True, help="보정 캐시 출력 폴더 -> queue 의 --cache 로 사용")
    p.add_argument("--cad", required=True, help="볼트 CAD (.stl/.ply/.obj)")
    p.add_argument("--depth-tol", type=float, default=8.0, help="물체 평면 허용 깊이(mm)")
    p.add_argument("--min-score", type=float, default=0.5, help="템플릿 적합 최소 점수 (미만이면 원본 유지)")
    p.set_defaults(func=cmd_refine)

    p = sub.add_parser("queue", help="우선순위 검수 대기열 생성")
    p.add_argument("--images", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--round", required=True)
    p.add_argument("--classes", type=_classes, default=None, help="생략하면 캐시(추론 당시 모델)의 클래스 목록")
    p.add_argument("--overwrite", action="store_true", help="기존 대기열을 지우고 새로 만든다 (검수 내용 소실)")
    p.add_argument("--max-vertices", type=int, default=None, help="다각형 최대 점 수 (기본: config.py 값)")
    p.add_argument("--eps-ratio", type=float, default=None, help="다각형 단순화 강도 (클수록 점이 줄어듦, 기본: config.py 값)")
    p.set_defaults(func=cmd_queue)

    p = sub.add_parser("edit", help="labelme 실행")
    p.add_argument("--round", required=True)
    p.add_argument("--bucket", default="urgent", choices=["urgent", "normal", "auto"])
    p.add_argument("--labelme", default="labelme", help="labelme 실행 파일 (별도 env에 설치했다면 절대경로)")
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("merge", help="검수 결과를 데이터셋으로 확정")
    p.add_argument("--round", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--exclude-auto", action="store_true", help="자동승인분 제외")
    p.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="수정도 reviewed 체크도 안 된 초안까지 포함 (권장하지 않음)",
    )
    p.add_argument("--copy-images", action="store_true")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("export", help="labelme -> COCO 변환")
    p.add_argument("--dataset", required=True)
    p.add_argument("--classes", type=_classes, default=None, help="생략하면 dataset/classes.json")
    p.add_argument("--out", default=None, help="기본: <dataset>/annotations.json")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("to-train", help="COCO export -> 학습용 데이터셋 폴더(intensity/ + annotations/instances_Train.json)")
    p.add_argument("--dataset", required=True, help="오토라벨 데이터셋 폴더 (annotations.json 이 있는 곳)")
    p.add_argument("--out", required=True, help="만들 학습 데이터셋 폴더 (예: data/dataset/autolabel_round04)")
    p.add_argument("--overwrite", action="store_true", help="기존 intensity/, annotations/ 를 지우고 다시 만든다")
    p.set_defaults(func=cmd_to_train)

    p = sub.add_parser("stats", help="라운드 품질 지표 출력")
    p.add_argument("--round", required=True)
    p.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
