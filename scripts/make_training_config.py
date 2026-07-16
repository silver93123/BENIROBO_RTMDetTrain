"""COCO 어노테이션 파일에서 클래스 정보를 읽어 RTMDet-Ins 학습 config를 자동 생성한다.

라벨링 툴(CVAT 등)에서 export한 COCO 1.0 형식 annotations/instances_Train.json의
"categories" 목록을 그대로 읽어서, configs/_template_rtmdet.py 템플릿에 채워 넣는다.
클래스 이름/개수/팔레트/work_dir을 매번 손으로 옮겨 적을 필요가 없다.

실행 (프로젝트 루트에서):
    python scripts/make_training_config.py --dataset 20260716_165411
    python scripts/make_training_config.py --dataset 20260716_165411 --out configs/rtmdet-ins_snacks.py

동작:
    1. data/dataset/<dataset>/annotations/instances_Train.json 을 읽는다.
    2. "categories" 목록(id 순 정렬)을 그대로 classes 튜플로 만든다.
    3. 클래스 수만큼 팔레트 색상을 자동 배정한다 (기본 팔레트 순환).
    4. work_dir을 클래스 이름 기반으로 자동 생성한다
       (기존 work_dir을 절대 재사용하지 않음 - 다른 객체 학습을 이어받는 사고 방지).
    5. configs/_template_rtmdet.py 를 채워서 --out 경로에 저장한다.

주의:
    _base_ (mmdet config 상속 경로)와 load_from (COCO pretrained 체크포인트)은
    이 환경(conda 환경 이름 등)에 따라 달라지므로, --base-config-path /
    --load-from 인자로 다르면 직접 지정하세요. 기본값은 기존
    configs/rtmdet-ins_bracket.py에서 쓰던 경로를 그대로 따릅니다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "configs" / "_template_rtmdet.py"

# rtmdet-ins_bracket.py에서 쓰던 기본값 (환경마다 다를 수 있어 인자로 override 가능)
DEFAULT_BASE_CONFIG_PATH = (
    "/home/silver/miniconda3/envs/vision_env/lib/python3.10/site-packages/"
    "mmdet/.mim/configs/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py"
)
DEFAULT_LOAD_FROM = "models/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth"

# 클래스 수가 이보다 많으면 색상을 순환해서 재사용한다.
DEFAULT_PALETTE = [
    (220, 20, 60), (60, 220, 20), (20, 60, 220), (220, 160, 20),
    (160, 20, 220), (20, 220, 160), (220, 20, 160), (160, 220, 20),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="COCO 어노테이션으로 RTMDet-Ins config 자동 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python scripts/make_training_config.py --dataset 20260716_165411
  python scripts/make_training_config.py --dataset 20260716_165411 --epochs 100 --batch-size 4
        """,
    )
    p.add_argument("--dataset", type=str, required=True, metavar="FOLDER_NAME",
                    help="data/dataset/ 아래 폴더명 (예: 20260716_165411)")
    p.add_argument("--ann-file", type=str, default=None,
                    help="COCO json 경로 직접 지정 (기본: <dataset>/annotations/instances_Train.json)")
    p.add_argument("--out", type=str, default=None,
                    help="생성할 config 경로 (기본: configs/rtmdet-ins_<클래스이름조합>.py)")
    p.add_argument("--work-dir-name", type=str, default=None,
                    help="work_dir 폴더 이름 (기본: 클래스 이름 기반 자동 생성)")
    p.add_argument("--base-config-path", type=str, default=DEFAULT_BASE_CONFIG_PATH,
                    help="mmdet _base_ config의 절대경로 (환경별로 다를 수 있음)")
    p.add_argument("--load-from", type=str, default=DEFAULT_LOAD_FROM,
                    help="COCO pretrained 체크포인트 경로")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.0001)
    return p.parse_args()


def load_categories(ann_file: Path) -> list[dict]:
    with ann_file.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    categories = coco.get("categories", [])
    if not categories:
        print(f"ERROR: '{ann_file}'에 categories가 없습니다.", flush=True)
        sys.exit(1)
    return sorted(categories, key=lambda c: c["id"])


def slugify(name: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()
    return slug or "class"


def main() -> int:
    args = parse_args()

    dataset_dir = ROOT / "data" / "dataset" / args.dataset
    ann_file = (
        Path(args.ann_file) if args.ann_file
        else dataset_dir / "annotations" / "instances_Train.json"
    )
    if not ann_file.is_file():
        print(f"ERROR: 어노테이션 파일을 찾을 수 없습니다: {ann_file}", flush=True)
        return 1

    categories = load_categories(ann_file)
    class_names = [c["name"] for c in categories]
    num_classes = len(class_names)
    palette = [DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)] for i in range(num_classes)]

    print("=" * 70, flush=True)
    print("  RTMDet-Ins config 자동 생성", flush=True)
    print("=" * 70, flush=True)
    print(f"  Annotation:   {ann_file}", flush=True)
    print(f"  Classes({num_classes}): {class_names}", flush=True)

    slug = args.work_dir_name or "_".join(slugify(n) for n in class_names)
    work_dir = f"work_dirs/rtmdet-ins_{slug}_v1"

    out_path = Path(args.out) if args.out else ROOT / "configs" / f"rtmdet-ins_{slug}.py"
    if out_path.exists():
        print(f"ERROR: 이미 존재하는 파일입니다 (덮어쓰지 않습니다): {out_path}", flush=True)
        print("       다른 --out 경로를 지정하거나, 기존 파일을 지우고 다시 실행하세요.", flush=True)
        return 1

    if not TEMPLATE_PATH.is_file():
        print(f"ERROR: 템플릿 파일이 없습니다: {TEMPLATE_PATH}", flush=True)
        return 1

    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    # 클래스가 1개면 트레일링 콤마를 붙여야 진짜 튜플이 된다 ('a') != ('a',)
    classes_tuple_str = (
        "(" + ", ".join(f"'{n}'" for n in class_names)
        + ("," if num_classes == 1 else "")
        + ")"
    )
    palette_str = "[" + ", ".join(str(p) for p in palette) + "]"

    rendered = (
        template
        .replace("__BASE_CONFIG_PATH__", args.base_config_path)
        .replace("__NUM_CLASSES__", str(num_classes))
        .replace("__DATA_ROOT__", f"data/dataset/{args.dataset}/")
        .replace("__CLASSES_TUPLE__", classes_tuple_str)
        .replace("__PALETTE_LIST__", palette_str)
        .replace("__WORK_DIR__", work_dir)
        .replace("__MAX_EPOCHS__", str(args.epochs))
        .replace("__BASE_LR__", str(args.lr))
        .replace("__BATCH_SIZE__", str(args.batch_size))
        .replace("__LOAD_FROM__", args.load_from)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")

    print(f"  Work dir:     {work_dir}", flush=True)
    print(f"  생성됨:       {out_path}", flush=True)
    print("=" * 70, flush=True)
    print("", flush=True)
    print("  '2. 모델 학습' 탭에서 --config로 이 파일을 선택하고,", flush=True)
    print(f"  --dataset 값은 '{args.dataset}' 로 입력하세요.", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())