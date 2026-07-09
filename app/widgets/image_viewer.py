"""이미지 위에 검출 박스를 오버레이해서 보여주는 위젯."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from PyQt6.QtWidgets import QLabel, QSizePolicy

from app.core.detector import Detection


class ImageViewer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #f2f1ec; border-radius: 8px;")
        self.setMinimumHeight(280)
        self._base_pixmap: QPixmap | None = None
        self._detections: list[Detection] = []
        self.setText("이미지를 불러오세요")

    def load_image(self, path: str) -> None:
        self._base_pixmap = QPixmap(path)
        self._detections = []
        self._refresh()

    def set_detections(self, detections: list[Detection]) -> None:
        self._detections = detections
        self._refresh()

    def _refresh(self) -> None:
        if self._base_pixmap is None or self._base_pixmap.isNull():
            self.setText("이미지를 불러오세요")
            return

        canvas = QPixmap(self._base_pixmap)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont()
        font.setPointSize(11)
        painter.setFont(font)

        colors = [QColor("#1D9E75"), QColor("#D85A30"), QColor("#378ADD"), QColor("#D4537E")]
        for i, det in enumerate(self._detections):
            color = colors[i % len(colors)]
            pen = QPen(color, 3)
            painter.setPen(pen)
            x1, y1, x2, y2 = det.bbox
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

            label_text = f"{det.label} {det.confidence:.2f}"
            painter.fillRect(QRectF(x1, y1 - 20, 8 * len(label_text), 20), color)
            painter.setPen(QPen(QColor("white")))
            painter.drawText(int(x1) + 4, int(y1) - 5, label_text)
            painter.setPen(pen)

        painter.end()

        scaled = canvas.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()
