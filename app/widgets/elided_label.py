"""폭이 모자라면 글자를 '…'로 줄여 보여주는 QLabel.

문구가 바뀔 때마다 위젯의 최소 폭이 늘어나 상단 줄/창 크기가 같이 커지는 것을 막는다 (최소 폭 0).
전체 문구는 툴팁으로 본다.
"""
from __future__ import annotations

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QLabel, QSizePolicy


class ElidedLabel(QLabel):
    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:   # noqa: N802 (Qt 이름)
        self._full = text
        self.setToolTip(text)
        super().setText("")                 # 실제 그리기는 paintEvent에서
        self.update()

    def text(self) -> str:
        return self._full

    def minimumSizeHint(self) -> QSize:     # noqa: N802
        return QSize(0, super().minimumSizeHint().height())

    def sizeHint(self) -> QSize:            # noqa: N802
        fm = self.fontMetrics()
        return QSize(fm.horizontalAdvance(self._full) + 2, fm.height() + 2)

    def paintEvent(self, _event) -> None:   # noqa: N802
        p = QPainter(self)
        p.setPen(self.palette().color(self.foregroundRole()))
        shown = self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, self.width())
        p.drawText(self.rect(), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), shown)
