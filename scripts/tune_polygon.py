"""폴리곤 단순화 설정을 실제 예측 캐시로 비교한다 (재추론/대기열 재생성 불필요, 읽기 전용).

사용 (레포 루트에서):
    python scripts/tune_polygon.py data/autolabel/cache/round01
    python scripts/tune_polygon.py data/autolabel/cache/round01 --candidates 0.0025:80 0.005:20 0.01:20

각 후보(eps_ratio:max_vertices)마다 인스턴스당 평균/최대 정점 수와, 폴리곤을 다시 마스크로 그렸을 때
원본 마스크와의 IoU(평균/최소)를 출력한다. 정점은 적게, IoU는 높게 나오는 설정을 고르면 된다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.autolabel.mask_utils import decode_rle, mask_iou, mask_to_polygons, polygons_to_mask  # noqa: E402
from src.autolabel.predict import load_cache  # noqa: E402

DEFAULT_CANDIDATES = [
    "0.0025:80", "0.005:80", "0.005:20", "0.01:20",
    "0.0025:80:2", "0.005:80:2", "0.005:20:2", "0.0075:20:2",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir")
    ap.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES, help="eps_ratio:max_vertices[:smooth_sigma]")
    ap.add_argument("--min-area", type=int, default=120)
    ap.add_argument("--limit", type=int, default=50, help="사용할 이미지 수 상한")
    args = ap.parse_args()

    files = sorted(Path(args.cache_dir).glob("*.pred.json"))[: args.limit]
    if not files:
        sys.exit(f"예측 캐시가 없습니다: {args.cache_dir}")
    masks = [decode_rle(i["rle"]) for f in files for i in load_cache(f)["instances"]]
    if not masks:
        sys.exit("인스턴스가 하나도 없습니다.")
    print(f"이미지 {len(files)}장 / 인스턴스 {len(masks)}개\n")
    print(f"{'eps:max[:sigma]':<15}{'정점 평균':>9}{'정점 최대':>9}{'20초과(%)':>10}{'IoU 평균':>10}{'IoU 최소':>9}")

    for cand in args.candidates:
        parts = cand.split(":")
        eps, mv = float(parts[0]), int(parts[1])
        sigma = float(parts[2]) if len(parts) > 2 else 0.0
        counts, ious = [], []
        for m in masks:
            polys = mask_to_polygons(m, eps_ratio=eps, min_area=args.min_area, keep_holes=False,
                                    max_vertices=mv, smooth_sigma=sigma)
            if not polys:
                continue
            counts.append(sum(len(p["points"]) for p in polys))
            rec = polygons_to_mask([p["points"] for p in polys], *m.shape)
            ious.append(mask_iou(m, rec))
        over = 100.0 * np.mean([c > 20 for c in counts])
        print(f"{cand:<15}{np.mean(counts):>9.1f}{max(counts):>9d}{over:>10.1f}{np.mean(ious):>10.3f}{min(ious):>9.3f}")


if __name__ == "__main__":
    main()
