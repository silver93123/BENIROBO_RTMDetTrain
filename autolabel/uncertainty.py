"""능동학습: 어떤 이미지를 사람이 먼저 봐야 하는가.

전부 검수하면 효율이 안 난다. 불확실한 것부터 보여주고,
확실한 것은 사람을 거치지 않고 통과시킨다.
라운드가 진행될수록 auto-accept 비율이 올라가는 것이
모델이 좋아지고 있다는 가장 직접적인 신호다.
"""

from __future__ import annotations

import numpy as np

from .config import DEFAULT, AutoLabelConfig

# 우선순위 점수 가중치 (합 1.0)
W_SCORE = 0.40
W_TTA = 0.40
W_COUNT = 0.20


def priority_score(cache: dict, cfg: AutoLabelConfig = DEFAULT) -> float:
    """0(안심) ~ 1(반드시 사람이 봐야 함). 높을수록 먼저 검수."""
    instances = cache.get("instances", [])
    metrics = cache.get("metrics", {})

    # 검출이 아예 없으면 최우선. 모델이 완전히 놓친 케이스다.
    if not instances:
        return 1.0

    # 1) 가장 자신 없는 인스턴스 기준
    min_score = float(metrics.get("min_score", 0.0))
    score_term = 1.0 - min_score

    # 2) 좌우반전 TTA 불일치
    mean_tta = metrics.get("mean_tta_iou")
    if mean_tta is None:
        tta_term = 0.5  # TTA를 안 돌렸으면 중립값
    else:
        tta_term = 1.0 - float(mean_tta)

    # 3) 검출 개수 이상치
    count = len(instances)
    if cfg.expected_count_min <= count <= cfg.expected_count_max:
        count_term = 0.0
    else:
        count_term = 1.0

    return float(
        np.clip(W_SCORE * score_term + W_TTA * tta_term + W_COUNT * count_term, 0.0, 1.0)
    )


def is_auto_acceptable(cache: dict, cfg: AutoLabelConfig = DEFAULT) -> bool:
    """사람 검수를 생략해도 되는 이미지인지 판정."""
    instances = cache.get("instances", [])
    if not instances:
        return False

    count = len(instances)
    if not (cfg.expected_count_min <= count <= cfg.expected_count_max):
        return False

    for inst in instances:
        if inst["score"] < cfg.auto_accept_score:
            return False
        tta = inst.get("tta_iou")
        if tta is not None and tta < cfg.auto_accept_tta_iou:
            return False
    return True


def rank(caches: list[dict], cfg: AutoLabelConfig = DEFAULT) -> list[dict]:
    """캐시 리스트에 priority / auto_accept를 붙이고 우선순위 내림차순 정렬."""
    scored = []
    for cache in caches:
        scored.append(
            {
                "cache": cache,
                "priority": priority_score(cache, cfg),
                "auto_accept": is_auto_acceptable(cache, cfg),
            }
        )
    scored.sort(key=lambda r: (-r["priority"], r["cache"]["image"]))
    return scored