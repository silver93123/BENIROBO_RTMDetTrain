"""RTMDet-Ins model-in-the-loop 오토라벨링 코어 (GUI 의존성 없음).

GUI는 app/tabs/autolabel_tab.py, CLI는 `python -m src.autolabel`.
"""

from .config import DEFAULT, AutoLabelConfig

__all__ = ["AutoLabelConfig", "DEFAULT"]
