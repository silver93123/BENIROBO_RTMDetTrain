"""이미지 위에 검출 박스 + 마스크 오버레이를 보여주는 위젯."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QImage
from PyQt6.QtWidgets import QLabel, QSizePolicy

from app.core.detector import Detection

MASK_ALPHA = 100  # 0~255, 마스크 반투명도 (낮을수록 더 투명)


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

        # 1단계: 마스크 영역을 반투명 색으로 먼저 채운다 (bbox/라벨보다 아래에 깔림)
        for i, det in enumerate(self._detections):
            if det.mask is None:
                continue
            color = colors[i % len(colors)]
            mask_image = self._mask_to_qimage(det.mask, color)
            if mask_image is not None:
                painter.drawImage(0, 0, mask_image)

        # 2단계: bbox + 라벨
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

    @staticmethod
    def _mask_to_qimage(mask: np.ndarray, color: QColor) -> QImage | None:
        """(H, W) bool 마스크를 반투명 RGBA QImage로 변환한다 (원본 이미지 크기와 동일해야 함)."""
        if mask is None or mask.ndim != 2:
            return None

        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = color.red()
        rgba[..., 1] = color.green()
        rgba[..., 2] = color.blue()
        rgba[..., 3] = np.where(mask, MASK_ALPHA, 0).astype(np.uint8)

        data = rgba.tobytes()
        qimage = QImage(data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        return qimage.copy()  # 자체 버퍼를 소유하도록 깊은 복사 (data가 GC돼도 안전)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()