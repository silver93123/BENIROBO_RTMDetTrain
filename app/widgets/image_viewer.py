"""이미지 위에 검출 박스 + 마스크 오버레이 + (선택) pose 미리보기 오버레이를 보여주는 위젯."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QImage
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QSlider, QWidget

from app.core.detector import Detection

MASK_ALPHA = 100  # 0~255, 마스크 반투명도 (낮을수록 더 투명)
POSE_OVERLAY_ALPHA = 100  # 0~255, pose 미리보기 윤곽선 반투명도 (낮을수록 더 투명)
POSE_OVERLAY_RADIUS = 2.5  # px
POSE_OVERLAY_LINE_WIDTH = 1.2  # px, 윤곽선 두께 (채워진 원이 아니라 속이 빈 원)
POSE_OVERLAY_COLOR = QColor(0, 220, 255, POSE_OVERLAY_ALPHA)  # 하늘색 - 노란 볼트 실물과 안 겹치게


ROI_MIN_SIZE = 16  # px, 드래그로 이보다 작게 그리면 클릭으로 보고 무시


def make_label_opacity_bar(viewer: "ImageViewer", initial_percent: int = 70) -> QWidget:
    """'라벨 투명도' 슬라이더. 박스/마스크 정보 라벨과 픽 포인트 좌표 라벨의 불투명도를 조절한다 (0=숨김, 100=불투명)."""
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(QLabel("라벨 투명도"))
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 100)
    slider.setFixedWidth(110)
    slider.setToolTip("박스/라벨/좌표 글자의 불투명도. 0이면 숨김, 100이면 불투명")
    value = QLabel("")
    value.setFixedWidth(34)

    def _apply(v: int) -> None:
        value.setText(f"{v}%")
        viewer.set_box_opacity(v / 100.0)

    slider.valueChanged.connect(_apply)
    slider.setValue(int(initial_percent))
    lay.addWidget(slider)
    lay.addWidget(value)
    box.slider = slider
    return box


class ImageViewer(QLabel):
    #: 사용자가 마우스로 ROI를 그리면 (x1, y1, x2, y2) 이미지 좌표(x2/y2 미포함)로 알린다.
    roi_changed = pyqtSignal(int, int, int, int)
    #: 클릭 선택 모드(set_click_pick)에서 이미지 위를 왼쪽 클릭하면 이미지 좌표 (x, y)를 알린다.
    point_clicked = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #f2f1ec; border-radius: 8px;")
        self.setMinimumHeight(280)
        self._base_pixmap: QPixmap | None = None
        self._detections: list[Detection] = []
        self._pose_overlays: dict[int, np.ndarray] = {}  # obj index -> (N,2) 투영된 2D 점
        self._pick_overlays: list[dict] = []  # 픽 포인트 + 좌표축 (set_pick_overlays 참고)
        self._box_opacity = 1.0  # 0~1, bbox 테두리/라벨 불투명도 (기본 1 = 기존 동작)
        self._roi: tuple[int, int, int, int] | None = None  # 표시할 ROI (이미지 좌표)
        self._roi_drawing = False        # True면 마우스 드래그로 ROI를 그릴 수 있다
        self._click_pick = False         # True면 왼쪽 클릭 한 점을 point_clicked로 알린다
        self._roi_drag_start: tuple[float, float] | None = None
        self._roi_before_drag: tuple[int, int, int, int] | None = None
        self.setText("이미지를 불러오세요")

    def load_image(self, path: str) -> None:
        self._base_pixmap = QPixmap(path)
        self._detections = []
        self._pose_overlays = {}
        self._pick_overlays = []
        self._refresh()

    def set_pick_overlays(self, overlays: list[dict] | None) -> None:
        """픽 포인트와 회전 좌표축(X 빨강 / Y 초록 / Z 파랑)을 이미지 위에 그린다.

        overlays 항목: {"origin": (u, v), "axes": [(u, v) x3: X/Y/Z 축 끝점],
                        "label": "obj0\nX.. Y.. Z..\nR.. P.. Y..", "color": QColor | None}
        None 또는 빈 리스트면 지운다.
        """
        self._pick_overlays = list(overlays or [])
        self._refresh()

    def set_detections(self, detections: list[Detection]) -> None:
        self._detections = detections
        self._refresh()

    def image_size(self) -> tuple[int, int] | None:
        """(width, height). 이미지가 없으면 None."""
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return None
        return self._base_pixmap.width(), self._base_pixmap.height()

    def set_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        """ROI 사각형을 표시한다 (밖은 어둡게). None이면 숨김."""
        self._roi = None if roi is None else tuple(int(v) for v in roi)
        self._refresh()

    def set_roi_drawing(self, enabled: bool) -> None:
        """True면 마우스 왼쪽 드래그로 ROI를 그린다 (roi_changed 시그널)."""
        self._roi_drawing = bool(enabled)
        self._roi_drag_start = None
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def set_click_pick(self, enabled: bool) -> None:
        """True면 이미지 위 왼쪽 클릭을 point_clicked(x, y)로 알린다 (이미지 밖 클릭은 무시)."""
        self._click_pick = bool(enabled)
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def _widget_to_image(self, pos, clamp: bool = True) -> tuple[float, float] | None:
        """위젯 좌표 -> 이미지 좌표. clamp=True면 표시된 이미지 밖은 가장자리로 clamp, False면 None."""
        pm = self.pixmap()
        size = self.image_size()
        if pm is None or pm.isNull() or size is None:
            return None
        pw, ph = pm.width(), pm.height()
        if pw <= 0 or ph <= 0:
            return None
        off_x = (self.width() - pw) / 2.0
        off_y = (self.height() - ph) / 2.0
        x = (pos.x() - off_x) * size[0] / pw
        y = (pos.y() - off_y) * size[1] / ph
        if not clamp and not (0.0 <= x < size[0] and 0.0 <= y < size[1]):
            return None
        return min(max(x, 0.0), size[0]), min(max(y, 0.0), size[1])

    def mousePressEvent(self, event) -> None:
        if self._click_pick and event.button() == Qt.MouseButton.LeftButton:
            p = self._widget_to_image(event.position(), clamp=False)
            if p is not None:
                self.point_clicked.emit(p[0], p[1])
            return
        if self._roi_drawing and event.button() == Qt.MouseButton.LeftButton:
            p = self._widget_to_image(event.position())
            if p is not None:
                self._roi_drag_start = p
                self._roi_before_drag = self._roi
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._roi_drag_start is not None:
            p = self._widget_to_image(event.position())
            if p is not None:
                x1, y1 = self._roi_drag_start
                self._roi = (int(min(x1, p[0])), int(min(y1, p[1])), int(max(x1, p[0])), int(max(y1, p[1])))
                self._refresh()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._roi_drag_start is not None and event.button() == Qt.MouseButton.LeftButton:
            p = self._widget_to_image(event.position())
            x1, y1 = self._roi_drag_start
            self._roi_drag_start = None
            if p is not None:
                rx1, ry1, rx2, ry2 = int(min(x1, p[0])), int(min(y1, p[1])), int(max(x1, p[0])), int(max(y1, p[1]))
                if rx2 - rx1 >= ROI_MIN_SIZE and ry2 - ry1 >= ROI_MIN_SIZE:
                    self._roi = (rx1, ry1, rx2, ry2)
                    self.roi_changed.emit(rx1, ry1, rx2, ry2)
                else:
                    self._roi = self._roi_before_drag  # 너무 작은 드래그 = 클릭으로 보고 이전 ROI 유지
                self._refresh()
            return
        super().mouseReleaseEvent(event)

    def set_box_opacity(self, opacity: float) -> None:
        """bbox 테두리/라벨과 픽 포인트 좌표 라벨의 불투명도(0=완전 투명 ~ 1=불투명). 라벨이 물체를 가릴 때 낮춘다."""
        self._box_opacity = min(1.0, max(0.0, float(opacity)))
        self._refresh()

    def set_pose_overlay(self, obj_index: int, points_2d: np.ndarray | None) -> None:
        """obj_index 인스턴스의 pose 미리보기 점(CAD를 현재 입력 각도로 투영한 것)을
        설정/갱신한다. points_2d가 None이면 그 인스턴스의 오버레이를 지운다.

        수동 라벨링 탭에서 각도 스핀박스를 조정할 때마다 호출되어, "지금 입력한
        각도가 실제 사진 속 물체와 얼마나 맞는지"를 즉시 눈으로 확인할 수 있게 한다.
        """
        if points_2d is None:
            self._pose_overlays.pop(obj_index, None)
        else:
            self._pose_overlays[obj_index] = points_2d
        self._refresh()

    def clear_pose_overlays(self) -> None:
        self._pose_overlays = {}
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
        painter.setOpacity(self._box_opacity)
        if self._box_opacity > 0.0:
            for i, det in enumerate(self._detections):
                color = colors[i % len(colors)]
                pen = QPen(color, 3)
                painter.setPen(pen)
                x1, y1, x2, y2 = det.bbox
                painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

                label_text = f"obj{i}: {det.label} {det.confidence:.2f}"
                painter.fillRect(QRectF(x1, y1 - 20, 8 * len(label_text), 20), color)
                painter.setPen(QPen(QColor("white")))
                painter.drawText(int(x1) + 4, int(y1) - 5, label_text)
                painter.setPen(pen)
        painter.setOpacity(1.0)  # 3단계(pose 오버레이)는 영향받지 않게 복원

        # ROI: 밖을 어둡게 + 노란 점선 테두리
        if self._roi is not None:
            rx1, ry1, rx2, ry2 = self._roi
            cw, ch = canvas.width(), canvas.height()
            shade = QColor(0, 0, 0, 120)
            painter.fillRect(QRectF(0, 0, cw, ry1), shade)
            painter.fillRect(QRectF(0, ry2, cw, ch - ry2), shade)
            painter.fillRect(QRectF(0, ry1, rx1, ry2 - ry1), shade)
            painter.fillRect(QRectF(rx2, ry1, cw - rx2, ry2 - ry1), shade)
            roi_pen = QPen(QColor(255, 210, 0), 3, Qt.PenStyle.DashLine)
            painter.setPen(roi_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(rx1, ry1, rx2 - rx1, ry2 - ry1))

        # 3단계: pose 미리보기 오버레이 (반투명 노란 "윤곽선" 원 - CAD를 현재
        # 입력 각도로 투영한 것. 속이 빈 원이라 실제 물체 사진이 안 가려지고,
        # 실루엣과 겹치면 입력한 각도가 잘 맞다는 뜻).
        overlay_pen = QPen(POSE_OVERLAY_COLOR, POSE_OVERLAY_LINE_WIDTH)
        painter.setPen(overlay_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for points_2d in self._pose_overlays.values():
            for x, y in points_2d:
                painter.drawEllipse(QRectF(x - POSE_OVERLAY_RADIUS, y - POSE_OVERLAY_RADIUS,
                                            POSE_OVERLAY_RADIUS * 2, POSE_OVERLAY_RADIUS * 2))

        self._draw_pick_overlays(painter, canvas.width())

        painter.end()

        scaled = canvas.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def _draw_pick_overlays(self, painter: QPainter, image_w: int) -> None:
        """픽 포인트(원) + 축 3개(X/Y/Z) + 좌표 텍스트. 굵기/글자는 이미지 폭에 비례."""
        if not self._pick_overlays:
            return
        # 화면에 표시되는 크기가 이미지 해상도와 무관하게 거의 일정하도록 축소 비율만큼 키운다.
        unit = max(1.0, image_w / max(self.width(), 1))
        axis_colors = [QColor(230, 40, 40), QColor(40, 190, 60), QColor(50, 100, 255)]
        axis_names = ["X", "Y", "Z"]
        font = QFont()
        font.setPixelSize(int(11 * unit))
        font.setBold(True)
        painter.setFont(font)
        fm = painter.fontMetrics()
        for ov in self._pick_overlays:
            ox, oy = ov["origin"]
            for (ex, ey), col, name in zip(ov["axes"], axis_colors, axis_names):
                painter.setPen(QPen(col, 2.5 * unit, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(QPointF(ox, oy), QPointF(ex, ey))
                painter.drawText(QPointF(ex + 3 * unit, ey - 3 * unit), name)
            painter.setPen(QPen(QColor("white"), 2 * unit))
            painter.setBrush(ov.get("dot") or QColor(255, 220, 0))
            r = 4 * unit
            painter.drawEllipse(QRectF(ox - r, oy - r, 2 * r, 2 * r))
            painter.setBrush(Qt.BrushStyle.NoBrush)

            lines = str(ov.get("label", "")).split("\n")
            painter.setOpacity(self._box_opacity)   # 라벨이 이미지를 가리면 슬라이더로 옅게/숨김
            if lines and lines[0] and self._box_opacity > 0.0:
                w = max(fm.horizontalAdvance(t) for t in lines) + int(10 * unit)
                h = fm.height() * len(lines) + int(6 * unit)
                bx = ox + 12 * unit
                by = oy - 12 * unit - h if ov.get("label_above") else oy + 12 * unit
                painter.fillRect(QRectF(bx, by, w, h), QColor(0, 0, 0, 170))
                painter.setPen(QPen(QColor("white")))
                for k, t in enumerate(lines):
                    painter.drawText(QPointF(bx + 5 * unit, by + fm.ascent() + k * fm.height() + 3 * unit), t)
            painter.setOpacity(1.0)

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