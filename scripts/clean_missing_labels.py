"""rotation_labels.json에서 image/mask 파일이 실제로 존재하는 항목만 남긴다.

LiveLabelGenerationTab이 한동안 촬영 이미지를 임시 경로(/tmp/tcp_*.png)로만
참조하던 버그가 있었다 (라벨을 저장했다면 이제는 고쳐져서 data/ 아래로 영구
복사되지만, 그 전에 저장된 항목들은 여전히 죽은 /tmp 경로를 갖고 있다).
학습 스크립트는 이런 항목을 만나면 DataLoader worker가 죽으면서 전체 학습이
멈춘다 - 한 개만 없어도 전부 못 돈다.

사용법: 아래 LABELS_JSON 경로만 확인하고 실행.
    python scripts/clean_missing_labels.py

원본은 <파일명>.bak으로 백업해두고, 살아있는 항목만 남긴 결과를 원래
경로에 덮어쓴다.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# =============================================================================
# 설정값
# =============================================================================
LABELS_JSON = ROOT / "data" / "rotation_labels.json"
# =============================================================================


def main() -> None:
    if not LABELS_JSON.is_file():
        print(f"라벨 파일이 없습니다: {LABELS_JSON}")
        return

    with open(LABELS_JSON, "r", encoding="utf-8") as f:
        items = json.load(f)

    kept, dropped = [], []
    for item in items:
        image_ok = os.path.isfile(item.get("image", ""))
        mask_ok = os.path.isfile(item.get("mask", ""))
        if image_ok and mask_ok:
            kept.append(item)
        else:
            dropped.append((item, image_ok, mask_ok))

    print(f"전체 {len(items)}건 중 유효 {len(kept)}건 / 제거 대상 {len(dropped)}건")
    for item, image_ok, mask_ok in dropped:
        reason = []
        if not image_ok:
            reason.append(f"image 없음: {item.get('image')}")
        if not mask_ok:
            reason.append(f"mask 없음: {item.get('mask')}")
        print(f"  - 제거: {' / '.join(reason)}")

    if not dropped:
        print("제거할 항목이 없습니다 - 그대로 둡니다.")
        return

    backup_path = LABELS_JSON.with_suffix(LABELS_JSON.suffix + ".bak")
    shutil.copy2(LABELS_JSON, backup_path)
    print(f"\n원본 백업: {backup_path}")

    with open(LABELS_JSON, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    print(f"정리 완료: {LABELS_JSON} ({len(kept)}건만 남음)")


if __name__ == "__main__":
    main()