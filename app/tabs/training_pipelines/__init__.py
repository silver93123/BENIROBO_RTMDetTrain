"""'2. 모델 학습' 섹션의 하위 파이프라인 레지스트리.

app/tabs/icp_pipelines와 동일한 목적: main_window.py가 이 목록으로 트리의
하위 항목 + QStackedWidget 페이지를 자동 구성한다. 새 학습 파이프라인을
추가할 때는 이 폴더에 탭 클래스를 새로 작성하고 아래 목록에 한 줄만
추가하면 된다 - main_window.py는 손댈 필요 없다.
"""
from __future__ import annotations

from app.tabs.training_pipelines.rothead_training_tab import RotHeadTrainingTab
from app.tabs.training_pipelines.rtmdet_training_tab import RTMDetTrainingTab

AVAILABLE_TRAINING_PIPELINES = [
    ("RTMDet 학습", RTMDetTrainingTab),
    ("RotHead 학습", RotHeadTrainingTab),
]

__all__ = ["RTMDetTrainingTab", "RotHeadTrainingTab", "AVAILABLE_TRAINING_PIPELINES"]