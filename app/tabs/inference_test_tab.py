"""탭 3: 오프라인 2D 검출 테스트 (ICP/3D pose 없음).

실제 파이프라인(3_Detect_and_PickPoint.py)과 동일하게 체크포인트 + config
두 가지를 모두 지정해야 RTMDetInferencer를 만들 수 있다.

이미지는 한 장씩 파일 다이얼로그로 불러오는 대신, 폴더를 지정하면 그 안의
이미지들을 목록으로 펼쳐두고 목록에서 골라 미리보기 -> 추론 실행하는 방식.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QFileDialog, QSlider, QMessageBox, QLineEdit,
)

from app.core.detector import Detector, Detection
from app.core.config_patcher import find_latest_best_checkpoint
from app.core.paths import DEFAULT_CONFIG_PATH
from app.widgets.image_viewer import ImageViewer

DEFAULT_SCORE_THRESHOLD = 0.3  # 실제 파이프라인 스크립트의 SCORE_THRESHOLD와 동일
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


class InferenceTestTab(QWidget):
    log_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checkpoint_path: str | None = None
        self._config_path: str | None = None
        self._image_path: str | None = None
        self._image_paths: list[str] = []  # 현재 폴더의 이미지 전체 경로 (image_list와 index 일치)
        self._last_detections: list[Detection] = []
        self._build_ui()
        self._prefill_latest_checkpoint()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        # ------------------------------------------------------- 좌측: 폴더 + 이미지 목록
        left = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(220)

        left.addWidget(QLabel("이미지 폴더"))
        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("폴더를 선택하세요")
        self.folder_edit.setReadOnly(True)
        folder_row.addWidget(self.folder_edit, stretch=1)
        btn_browse_folder = QPushButton("폴더 선택")
        btn_browse_folder.clicked.connect(self._on_browse_image_folder)
        folder_row.addWidget(btn_browse_folder)
        left.addLayout(folder_row)

        self.image_count_label = QLabel("이미지 목록 (0장)")
        self.image_count_label.setStyleSheet("color: #666; font-size: 11px; margin-top: 6px;")
        left.addWidget(self.image_count_label)

        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._on_image_row_changed)
        left.addWidget(self.image_list, stretch=1)

        root.addWidget(left_widget)

        # ------------------------------------------------------- 중앙: 설정 + 미리보기
        center = QVBoxLayout()

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("체크포인트"))
        self.checkpoint_edit = QLineEdit()
        path_row.addWidget(self.checkpoint_edit, stretch=1)
        btn_browse_ckpt = QPushButton("선택")
        btn_browse_ckpt.clicked.connect(self._on_browse_checkpoint)
        path_row.addWidget(btn_browse_ckpt)
        center.addLayout(path_row)

        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("config"))
        self.config_edit = QLineEdit()
        self.config_edit.setPlaceholderText("보통 체크포인트와 같은 work_dir 안의 .py 파일")
        cfg_row.addWidget(self.config_edit, stretch=1)
        btn_browse_cfg = QPushButton("선택")
        btn_browse_cfg.clicked.connect(self._on_browse_config)
        cfg_row.addWidget(btn_browse_cfg)
        center.addLayout(cfg_row)

        run_row = QHBoxLayout()
        self.btn_run = QPushButton("추론 실행")
        self.btn_run.clicked.connect(self._on_run_inference)
        run_row.addWidget(self.btn_run)
        run_row.addStretch(1)
        center.addLayout(run_row)

        self.image_viewer = ImageViewer()
        center.addWidget(self.image_viewer, stretch=1)
        root.addLayout(center, stretch=2)

        # ------------------------------------------------------- 우측: 검출 결과
        right = QVBoxLayout()
        right.addWidget(QLabel("검출 결과"))
        self.result_list = QListWidget()
        right.addWidget(self.result_list, stretch=1)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("conf threshold"))
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(int(DEFAULT_SCORE_THRESHOLD * 100))
        self.thresh_label = QLabel(f"{DEFAULT_SCORE_THRESHOLD:.2f}")
        self.thresh_slider.valueChanged.connect(self._on_threshold_changed)
        thresh_row.addWidget(self.thresh_slider, stretch=1)
        thresh_row.addWidget(self.thresh_label)
        right.addLayout(thresh_row)

        self.btn_export = QPushButton("검출 결과 내보내기 (json)")
        self.btn_export.clicked.connect(self._on_export)
        right.addWidget(self.btn_export)

        root.addLayout(right, stretch=1)

    def _prefill_latest_checkpoint(self) -> None:
        """탭이 열릴 때, 기본 config의 work_dir에서 가장 최근 best 체크포인트를 자동으로 채운다."""
        if not DEFAULT_CONFIG_PATH.is_file():
            return
        cfg_path = str(DEFAULT_CONFIG_PATH)
        self.config_edit.setText(cfg_path)

        best = find_latest_best_checkpoint(cfg_path)
        if best:
            self.checkpoint_edit.setText(best)
            self.log_message.emit(f"최신 best 체크포인트 자동 설정: {best}")
        else:
            self.log_message.emit(
                "work_dir에 best 체크포인트가 아직 없어 자동 설정을 건너뜁니다. "
                "직접 선택해주세요."
            )

    # ----------------------------------------------------------- actions
    def _on_browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "체크포인트 선택", "", "PyTorch (*.pth)")
        if not path:
            return
        self.checkpoint_edit.setText(path)
        self.log_message.emit(f"체크포인트 설정: {path}")

        # 체크포인트와 같은 폴더에 config .py가 하나뿐이면 자동으로 제안
        if not self.config_edit.text():
            candidates = list(Path(path).parent.glob("*.py"))
            if len(candidates) == 1:
                self.config_edit.setText(str(candidates[0]))
                self.log_message.emit(f"같은 폴더에서 config 자동 제안: {candidates[0]}")

    def _on_browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "config 파일 선택", "", "Python (*.py)")
        if path:
            self.config_edit.setText(path)

    def _on_browse_image_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택")
        if not folder:
            return
        self.folder_edit.setText(folder)
        self._load_image_folder(folder)

    def _load_image_folder(self, folder: str) -> None:
        folder_path = Path(folder)
        files = sorted(
            f for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )

        self._image_paths = [str(f) for f in files]
        self.image_list.clear()
        for f in files:
            self.image_list.addItem(QListWidgetItem(f.name))
        self.image_count_label.setText(f"이미지 목록 ({len(files)}장)")

        self._image_path = None
        self._last_detections = []
        self.result_list.clear()
        self.image_viewer.load_image("")  # 미리보기 초기화

        if not files:
            QMessageBox.information(self, "알림", "선택한 폴더에 이미지가 없습니다.")
            self.log_message.emit(f"이미지 폴더 스캔: {folder} (0장)")
            return

        self.log_message.emit(f"이미지 폴더 스캔: {folder} ({len(files)}장)")
        self.image_list.setCurrentRow(0)  # 첫 이미지를 자동으로 미리보기

    def _on_image_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._image_paths):
            return
        self._image_path = self._image_paths[row]
        self.image_viewer.load_image(self._image_path)
        self._last_detections = []
        self.result_list.clear()
        self.log_message.emit(f"이미지 선택: {self._image_path}")

    def _on_run_inference(self) -> None:
        if not self._image_path:
            QMessageBox.warning(self, "알림", "먼저 이미지 폴더를 선택하고 목록에서 이미지를 고르세요.")
            return
        checkpoint = self.checkpoint_edit.text().strip()
        config = self.config_edit.text().strip()
        if not checkpoint or not config:
            QMessageBox.warning(self, "알림", "체크포인트와 config를 모두 지정하세요.")
            return

        threshold = self.thresh_slider.value() / 100.0

        detector = Detector(
            checkpoint_path=checkpoint,
            config_path=config,
            score_threshold=threshold,
        )
        try:
            detections = detector.predict(self._image_path, conf_threshold=threshold)
        except ImportError as exc:
            QMessageBox.critical(
                self, "추론 엔진 없음",
                f"{exc}\n\n"
                "이 오류는 실제 모델을 로드하지 못했다는 뜻입니다 - 더미 결과가 아닙니다."
            )
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "추론 오류", str(exc))
            return

        self._last_detections = detections
        self.image_viewer.set_detections(detections)
        self._refresh_result_list()
        self.log_message.emit(f"검출 완료: {len(detections)}건 (threshold={threshold:.2f})")

    def _on_threshold_changed(self, value: int) -> None:
        self.thresh_label.setText(f"{value / 100:.2f}")
        if self._last_detections:
            filtered = [d for d in self._last_detections if d.confidence >= value / 100]
            self.image_viewer.set_detections(filtered)
            self._refresh_result_list(filtered)

    def _refresh_result_list(self, detections=None) -> None:
        detections = detections if detections is not None else self._last_detections
        self.result_list.clear()
        for i, det in enumerate(detections):
            item = QListWidgetItem(f"object_{i}  |  class: {det.label}  |  conf {det.confidence:.2f}")
            self.result_list.addItem(item)

    def _on_export(self) -> None:
        if not self._last_detections:
            QMessageBox.warning(self, "알림", "먼저 추론을 실행하세요.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "결과 저장", "detections.json", "JSON (*.json)"
        )
        if not path:
            return
        import json

        data = [
            {"label": d.label, "confidence": d.confidence, "bbox": d.bbox}
            for d in self._last_detections
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.log_message.emit(f"검출 결과 저장: {path}")