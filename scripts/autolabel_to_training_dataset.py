"""오토라벨 export -> 학습용 데이터셋 폴더 변환 (CLI 래퍼).
앱에서는 '11. 2D 오토라벨링' 탭의 버튼이 같은 기능을 쓴다.

    python scripts/autolabel_to_training_dataset.py --autolabel-dataset data/autolabel/dataset/round04 --name autolabel_round04
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.autolabel.train_export import export_training_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--autolabel-dataset", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()
try:
    export_training_dataset(Path(a.autolabel_dataset), ROOT / "data" / "dataset" / a.name, a.overwrite)
except RuntimeError as e:
    sys.exit(f"ERROR: {e}")
