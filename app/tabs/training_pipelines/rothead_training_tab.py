"""'2. 모델 학습' 섹션의 RotHead(회전 회귀 헤드) 학습 파이프라인.

실제 학습 스크립트(scripts/train_rotation_head.py) 기준:
  python scripts/train_rotation_head.py --labels-json <경로> --output-dir <경로>
      [--epochs ...] [--batch-size ...] [--lr ...] [--crop-size ...]
      [--backbone ...] [--symmetry-group ...]

RTMDet 학습(RTMDetTrainingTab)과는 완전히 다른 워크플로우:
  - 학습 스크립트가 다름, 로그 포맷이 다름 (mmdet의 "Epoch(train)/coco/bbox_mAP"가
    아니라 "[epoch N/M] geodesic loss = ... rad")
  - 지표도 다름 (mAP가 아니라 geodesic loss, 대칭군 하나를 골라야 함)
  - 데이터 입력도 다름 (--dataset 폴더명이 아니라 --labels-json 파일 경로)
그래서 TrainRunner(범용 프로세스 실행기)는 그대로 재사용하되, progress 파싱은
이 탭이 직접 한다 (TrainRunner.progress는 mmdet 전용 정규식이라 여기선 안 씀 -
그냥 안 쓰이는 신호로 남겨두고, log_line만 직접 파싱한다).
"""
from __future__ import annotations

import os
import re

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QMessageBox, QFrame, QComboBox, QSpinBox, QDoubleSpinBox,
)

from app.core.train_runner import TrainRunner
from app.core.paths import PROJECT_ROOT
from app.widgets.log_console import LogConsole

# scripts/train_rotation_head.py의 SYMMETRY_GROUPS 키와 반드시 동기화할 것.
SYMMETRY_GROUP_CHOICES = ["none", "z180"]

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-4
DEFAULT_CROP_SIZE = 128
DEFAULT_BACKBONE = "resnet18"

# scripts/train_rotation_head.py의 출력: "[epoch 3/50] geodesic loss = 1.4823 rad (84.9 deg)"
PROGRESS_LINE_RE = re.compile(
    r"\[epoch\s*(?P<epoch>\d+)/(?P<total>\d+)\]\s*geodesic loss\s*=\s*"
    r"(?P<loss_rad>[\d.]+)\s*rad\s*\((?P<loss_deg>[\d.]+)\s*deg\)"
)


class MetricCard(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "QFrame { background-color: #f2f1ec; border-radius: 8px; padding: 6px; }"
        )
        layout = QVBoxLayout(self)
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #666; font-size: 11px;")
        self.value_label = QLabel("-")
        self.value_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, text: str) -> None:
        self.value_label.setText(text)


class RotHeadTrainingTab(QWidget):
    log_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loss_history_deg: list[float] = []
        self.runner: TrainRunner | None = None
        self._build_ui()
        self._working_dir = str(PROJECT_ROOT)

    # ---------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        labels_row = QHBoxLayout()
        labels_row.addWidget(QLabel("--labels-json"))
        self.labels_json_edit = QLineEdit()
        self.labels_json_edit.setPlaceholderText(
            "이미지+마스크+bbox+rotation_matrix 라벨 목록 (data/rotation_labels_*.json)"
        )
        labels_row.addWidget(self.labels_json_edit, stretch=1)
        btn_browse_labels = QPushButton("선택")
        btn_browse_labels.clicked.connect(self._on_browse_labels_json)
        labels_row.addWidget(btn_browse_labels)
        layout.addLayout(labels_row)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("--output-dir"))
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("체크포인트 저장 경로 (예: checkpoints/rotation_head_bracket)")
        output_row.addWidget(self.output_dir_edit, stretch=1)
        btn_browse_output = QPushButton("선택")
        btn_browse_output.clicked.connect(self._on_browse_output_dir)
        output_row.addWidget(btn_browse_output)
        layout.addLayout(output_row)

        # 대칭군: 이 부품의 CAD 형상을 보고 골라야 함. RTMDet 파이프라인과 마찬가지로
        # "부품 하나당 모델 하나" 학습이라 배치 전체에 공통으로 적용되는 상수 하나.
        sym_row = QHBoxLayout()
        sym_row.addWidget(QLabel("대칭군 (symmetry-group)"))
        self.combo_symmetry = QComboBox()
        self.combo_symmetry.addItems(SYMMETRY_GROUP_CHOICES)
        sym_row.addWidget(self.combo_symmetry)
        sym_hint = QLabel("이 부품 CAD가 회전 대칭이면(예: 양끝이 같은 막대형) z180 선택.\n"
                           "잘못 고르면 학습이 대칭 방향에서 진동하며 수렴 안 될 수 있음.")
        sym_hint.setStyleSheet("color: #888; font-size: 10px;")
        sym_hint.setWordWrap(True)
        sym_row.addWidget(sym_hint, stretch=1)
        layout.addLayout(sym_row)

        hp_row = QHBoxLayout()
        hp_row.addWidget(QLabel("epochs"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 100000)
        self.spin_epochs.setValue(DEFAULT_EPOCHS)
        hp_row.addWidget(self.spin_epochs)

        hp_row.addWidget(QLabel("batch size"))
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 4096)
        self.spin_batch.setValue(DEFAULT_BATCH_SIZE)
        hp_row.addWidget(self.spin_batch)

        hp_row.addWidget(QLabel("lr"))
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(6)
        self.spin_lr.setRange(1e-6, 1.0)
        self.spin_lr.setSingleStep(1e-5)
        self.spin_lr.setValue(DEFAULT_LR)
        hp_row.addWidget(self.spin_lr)

        hp_row.addWidget(QLabel("crop size"))
        self.spin_crop = QSpinBox()
        self.spin_crop.setRange(32, 1024)
        self.spin_crop.setValue(DEFAULT_CROP_SIZE)
        hp_row.addWidget(self.spin_crop)

        hp_row.addWidget(QLabel("backbone"))
        self.backbone_edit = QLineEdit(DEFAULT_BACKBONE)
        self.backbone_edit.setFixedWidth(90)
        hp_row.addWidget(self.backbone_edit)
        hp_row.addStretch(1)
        layout.addLayout(hp_row)

        cards_row = QHBoxLayout()
        self.card_epoch = MetricCard("Epoch")
        self.card_loss_rad = MetricCard("geodesic loss (rad)")
        self.card_loss_deg = MetricCard("geodesic loss (deg)")
        for c in (self.card_epoch, self.card_loss_rad, self.card_loss_deg):
            cards_row.addWidget(c)
        layout.addLayout(cards_row)

        self.figure = Figure(figsize=(5, 2))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)

        layout.addWidget(QLabel("학습 스크립트 출력 로그"))
        self.train_log = LogConsole()
        self.train_log.setMaximumHeight(160)
        layout.addWidget(self.train_log, stretch=1)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("학습 시작")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop = QPushButton("중단")
        self.btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        layout.addLayout(btn_row)

    # ----------------------------------------------------------- actions
    def _on_browse_labels_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "라벨 JSON 선택", "", "JSON (*.json)")
        if path:
            self.labels_json_edit.setText(path)

    def _on_browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "체크포인트 저장 폴더 선택")
        if path:
            self.output_dir_edit.setText(path)

    def _on_start(self) -> None:
        labels_json = self.labels_json_edit.text().strip()
        if not labels_json or not os.path.isfile(labels_json):
            QMessageBox.warning(self, "설정 확인 필요", "유효한 --labels-json 파일을 지정하세요.")
            return

        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "설정 확인 필요", "--output-dir을 지정하세요.")
            return

        command = (
            "python scripts/train_rotation_head.py "
            f"--labels-json {labels_json!r} --output-dir {output_dir!r} "
            f"--epochs {self.spin_epochs.value()} --batch-size {self.spin_batch.value()} "
            f"--lr {self.spin_lr.value()} --crop-size {self.spin_crop.value()} "
            f"--backbone {self.backbone_edit.text().strip() or DEFAULT_BACKBONE} "
            f"--symmetry-group {self.combo_symmetry.currentText()}"
        )

        self._loss_history_deg.clear()
        self.train_log.clear()

        self.runner = TrainRunner(self)
        self.runner.log_line.connect(self.train_log.append_log)
        self.runner.log_line.connect(self._parse_progress_line)
        self.runner.finished.connect(self._on_finished)
        self.runner.error.connect(self._on_error)
        self.runner.start(command, working_dir=self._working_dir)
        self.log_message.emit(
            f"RotHead 학습 시작 (labels={labels_json}, symmetry={self.combo_symmetry.currentText()})"
        )
        self.btn_start.setEnabled(False)

    def _on_stop(self) -> None:
        if self.runner:
            self.runner.stop()

    def _parse_progress_line(self, line: str) -> None:
        """TrainRunner.progress(mmdet 전용 정규식)는 이 스크립트의 로그와
        형식이 달라 매칭이 안 되므로, log_line을 직접 받아 이 탭이 파싱한다."""
        m = PROGRESS_LINE_RE.search(line)
        if not m:
            return

        epoch = int(m.group("epoch"))
        loss_deg = float(m.group("loss_deg"))
        self.card_epoch.set_value(str(epoch))
        self.card_loss_rad.set_value(f"{float(m.group('loss_rad')):.4f}")
        self.card_loss_deg.set_value(f"{loss_deg:.1f}")
        self._loss_history_deg.append(loss_deg)

        self.ax.clear()
        self.ax.plot(self._loss_history_deg, label="geodesic loss (deg)", color="#D85A30")
        self.ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw()

    def _on_finished(self, exit_code: int) -> None:
        self.log_message.emit(f"RotHead 학습 프로세스 종료 (exit code {exit_code})")
        self.btn_start.setEnabled(True)

    def _on_error(self, message: str) -> None:
        QMessageBox.critical(self, "학습 오류", message)
        self.btn_start.setEnabled(True)