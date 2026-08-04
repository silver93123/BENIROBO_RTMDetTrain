"""'6. 회전 라벨 생성' 탭 - generate_rotation_labels.py를 GUI에서 실행.

기존 RTMDetTrainingTab과 완전히 동일한 패턴을 쓴다: 실제 로직은 전부
scripts/generate_rotation_labels.py 안에 있고, 이 탭은 그 스크립트를
app.core.train_runner.TrainRunner(QProcess 래퍼)로 실행하면서 인자를
채워주고 stdout을 로그 콘솔에 흘려보내는 얇은 껍데기다. TrainRunner의
progress 시그널(mmengine 로그 정규식 파싱)은 이 스크립트 출력과 안 맞아
그냥 발동하지 않을 뿐이라 문제 없음 - log_line/finished/error만 쓴다.

실제 실행 커맨드:
    python scripts/generate_rotation_labels.py --dataset ... --cad ...
        --checkpoint ... --config ... --fitness-min ... --score-threshold ...
        --mask-out-dir ... --labels-out ... --device ...
"""
from __future__ import annotations

import os
import re
import shlex
from collections import deque
from pathlib import Path

from PyQt6.QtCore import pyqtSignal, QTimer, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from app.core.paths import DEFAULT_CAD_DIR, DEFAULT_DATASET_ROOT, PROJECT_ROOT
from app.core.train_runner import TrainRunner
from app.widgets.log_console import LogConsole

DEFAULT_SCRIPT = PROJECT_ROOT / "scripts" / "generate_rotation_labels.py"
DEFAULT_MASK_OUT_DIR = PROJECT_ROOT / "data" / "rotation_labels_masks"
DEFAULT_LABELS_OUT = PROJECT_ROOT / "data" / "rotation_labels.json"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "data" / "rotation_labels_preview"

# --- 미리보기 갤러리 설정 ---
PREVIEW_POLL_MS = 1000       # preview_dir을 이 주기로 스캔해서 새 파일을 찾음
PREVIEW_THUMB_SIZE = 160     # 썸네일 한 변 크기 (px)
PREVIEW_COLUMNS = 6          # 그리드 열 수
PREVIEW_MAX_ITEMS = 60       # 오래된 썸네일부터 버려서 갤러리가 무거워지지 않게 함

# generate_rotation_labels.py가 완료 시 찍는 요약 줄에서 채택 건수를 뽑기 위한 정규식.
# "완료: 전체 인스턴스 123개 중 87개 채택 (fitness >= 0.85) -> data/rotation_labels.json"
SUMMARY_RE = re.compile(r"전체 인스턴스\s*(?P<total>\d+)개 중\s*(?P<accepted>\d+)개 채택")

# 인스턴스별 accept/reject 카드용 정규식. 스크립트가 찍는 고정 포맷과 1:1 대응:
# "RESULT frame=frame_0006 obj=3 status=ACCEPT fitness=0.912"
# "RESULT frame=frame_0006 obj=4 status=REJECT fitness=0.565 reason=fitness<0.850"
RESULT_RE = re.compile(
    r"RESULT frame=(?P<frame>\S+) obj=(?P<obj>\d+) status=(?P<status>ACCEPT|REJECT) "
    r"fitness=(?P<fitness>[\d.]+|none)(?:\s+reason=(?P<reason>\S+))?"
)
RESULT_MAX_CARDS = 300  # 세션이 크면 인스턴스가 수백~수천 개일 수 있어 오래된 카드부터 제거


class LabelGenerationTab(QWidget):
    log_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.runner: TrainRunner | None = None
        self._preview_seen: set[str] = set()          # 이미 갤러리에 올린 파일명
        self._preview_widgets: deque[QLabel] = deque()  # FIFO - PREVIEW_MAX_ITEMS 넘으면 오래된 것부터 제거
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(PREVIEW_POLL_MS)
        self._preview_timer.timeout.connect(self._poll_preview_dir)
        self._result_cards: deque[QFrame] = deque()   # RESULT_MAX_CARDS 넘으면 오래된 카드부터 제거
        self._n_accept = 0
        self._n_reject = 0
        self._build_ui()

        if DEFAULT_CAD_DIR.is_dir():
            self.cad_edit.setPlaceholderText(f"예: {DEFAULT_CAD_DIR / 'bracket.stl'}")
        self.mask_out_edit.setText(str(DEFAULT_MASK_OUT_DIR))
        self.labels_out_edit.setText(str(DEFAULT_LABELS_OUT))
        self.preview_dir_edit.setText(str(DEFAULT_PREVIEW_DIR))

    # ------------------------------------------------------------ session
    def set_session_path(self, path: str) -> None:
        """데이터 수집/세션 탭에서 세션이 선택되면 --dataset 값에 이어붙인다.

        RTMDetTrainingTab.set_session_path와 달리 여기서는 값을 덮어쓰지 않고
        콤마로 이어붙인다 - 여러 세션을 누적해서 한 번에 라벨링하는 게
        자연스러운 사용 패턴이기 때문 (이미 들어있는 세션명이면 중복 추가 안 함).
        """
        dataset_name = os.path.basename(os.path.normpath(path))
        existing = [s.strip() for s in self.dataset_edit.text().split(",") if s.strip()]
        if dataset_name not in existing:
            existing.append(dataset_name)
        self.dataset_edit.setText(",".join(existing))
        self.log_message.emit(f"라벨 생성 탭 --dataset에 세션 추가: {dataset_name}")

    # ---------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        dataset_row = QHBoxLayout()
        dataset_row.addWidget(QLabel("--dataset (세션 폴더명, 콤마로 여러 개)"))
        self.dataset_edit = QLineEdit()
        self.dataset_edit.setPlaceholderText("예: 20260521_114500,20260522_090000")
        dataset_row.addWidget(self.dataset_edit, stretch=1)
        btn_add_session = QPushButton("세션 폴더 추가")
        btn_add_session.clicked.connect(self._on_add_session_dir)
        dataset_row.addWidget(btn_add_session)
        layout.addLayout(dataset_row)

        cad_row = QHBoxLayout()
        cad_row.addWidget(QLabel("--cad 파일"))
        self.cad_edit = QLineEdit()
        cad_row.addWidget(self.cad_edit, stretch=1)
        btn_browse_cad = QPushButton("선택")
        btn_browse_cad.clicked.connect(self._on_browse_cad)
        cad_row.addWidget(btn_browse_cad)
        layout.addLayout(cad_row)

        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(QLabel("--checkpoint (.pth)"))
        self.checkpoint_edit = QLineEdit()
        ckpt_row.addWidget(self.checkpoint_edit, stretch=1)
        btn_browse_ckpt = QPushButton("선택")
        btn_browse_ckpt.clicked.connect(self._on_browse_checkpoint)
        ckpt_row.addWidget(btn_browse_ckpt)
        layout.addLayout(ckpt_row)

        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("--config (.py)"))
        self.config_edit = QLineEdit()
        cfg_row.addWidget(self.config_edit, stretch=1)
        btn_browse_cfg = QPushButton("선택")
        btn_browse_cfg.clicked.connect(self._on_browse_config)
        cfg_row.addWidget(btn_browse_cfg)
        layout.addLayout(cfg_row)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("--fitness-min"))
        self.fitness_spin = QDoubleSpinBox()
        self.fitness_spin.setRange(0.0, 1.0)
        self.fitness_spin.setSingleStep(0.01)
        self.fitness_spin.setValue(0.85)
        self.fitness_spin.setToolTip(
            "ICP fitness가 이 값 이상인 인스턴스만 학습 라벨로 채택합니다. "
            "채택 건수가 0이면 이 값을 낮춰보세요 (ICP 탭 기본 threshold는 보통 0.6~0.7)."
        )
        thresh_row.addWidget(self.fitness_spin)

        thresh_row.addWidget(QLabel("--score-threshold"))
        self.score_spin = QDoubleSpinBox()
        self.score_spin.setRange(0.0, 1.0)
        self.score_spin.setSingleStep(0.05)
        self.score_spin.setValue(0.5)
        thresh_row.addWidget(self.score_spin)

        thresh_row.addWidget(QLabel("--device"))
        self.device_edit = QLineEdit("cuda:0")
        self.device_edit.setFixedWidth(90)
        thresh_row.addWidget(self.device_edit)
        thresh_row.addStretch(1)
        layout.addLayout(thresh_row)

        mask_out_row = QHBoxLayout()
        mask_out_row.addWidget(QLabel("--mask-out-dir"))
        self.mask_out_edit = QLineEdit()
        mask_out_row.addWidget(self.mask_out_edit, stretch=1)
        btn_browse_mask_out = QPushButton("선택")
        btn_browse_mask_out.clicked.connect(self._on_browse_mask_out_dir)
        mask_out_row.addWidget(btn_browse_mask_out)
        layout.addLayout(mask_out_row)

        labels_out_row = QHBoxLayout()
        labels_out_row.addWidget(QLabel("--labels-out"))
        self.labels_out_edit = QLineEdit()
        labels_out_row.addWidget(self.labels_out_edit, stretch=1)
        btn_browse_labels_out = QPushButton("선택")
        btn_browse_labels_out.clicked.connect(self._on_browse_labels_out)
        labels_out_row.addWidget(btn_browse_labels_out)
        layout.addLayout(labels_out_row)

        preview_row = QHBoxLayout()
        preview_row.addWidget(QLabel("--preview-dir (마스크 오버레이 검토 이미지, 비우면 저장 안 함)"))
        self.preview_dir_edit = QLineEdit()
        preview_row.addWidget(self.preview_dir_edit, stretch=1)
        btn_browse_preview = QPushButton("선택")
        btn_browse_preview.clicked.connect(self._on_browse_preview_dir)
        preview_row.addWidget(btn_browse_preview)
        layout.addLayout(preview_row)

        summary_row = QHBoxLayout()
        summary_row.addWidget(QLabel("결과 요약"))
        self.summary_label = QLabel("-")
        self.summary_label.setStyleSheet("font-weight: 600;")
        summary_row.addWidget(self.summary_label, stretch=1)
        layout.addLayout(summary_row)

        layout.addWidget(QLabel("미리보기 (채택된 인스턴스 - 초록: 마스크, 빨강: bbox·fitness)"))
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setMinimumHeight(2 * PREVIEW_THUMB_SIZE + 40)
        self.preview_container = QWidget()
        self.preview_grid = QGridLayout(self.preview_container)
        self.preview_grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.preview_scroll.setWidget(self.preview_container)
        layout.addWidget(self.preview_scroll)

        layout.addWidget(QLabel("스크립트 출력 로그"))
        self.script_log = LogConsole()
        self.script_log.setMaximumHeight(160)
        layout.addWidget(self.script_log, stretch=1)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("라벨 생성 시작")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop = QPushButton("중단")
        self.btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        layout.addLayout(btn_row)

        root.addWidget(left_widget, stretch=2)

        # ------------------------------------------------- 우측: 인스턴스별 결과
        # icp_workbench_base.py의 "ICP 결과" 카드 패널과 동일한 스타일(초록=성공,
        # 빨강=실패)을 그대로 재사용해서, 검출 탭에서 보던 것과 같은 방식으로
        # 검토할 수 있게 했다.
        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right_widget.setFixedWidth(300)

        self.result_count_label = QLabel("성공 0 / 실패 0")
        self.result_count_label.setStyleSheet("font-weight: 600;")
        right.addWidget(QLabel("인스턴스별 결과"))
        right.addWidget(self.result_count_label)

        self.result_scroll = QScrollArea()
        self.result_scroll.setWidgetResizable(True)
        self.result_container = QWidget()
        self.result_layout = QVBoxLayout(self.result_container)
        self.result_layout.addStretch(1)
        self.result_scroll.setWidget(self.result_container)
        right.addWidget(self.result_scroll, stretch=1)

        root.addWidget(right_widget)

    # ----------------------------------------------------------- browse
    def _on_add_session_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "세션 폴더 선택", str(DEFAULT_DATASET_ROOT) if DEFAULT_DATASET_ROOT.is_dir() else ""
        )
        if path:
            self.set_session_path(path)

    def _on_browse_cad(self) -> None:
        start_dir = str(DEFAULT_CAD_DIR) if DEFAULT_CAD_DIR.is_dir() else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "CAD 파일 선택", start_dir, "CAD (*.stl *.ply *.obj)"
        )
        if path:
            self.cad_edit.setText(path)

    def _on_browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "체크포인트 선택", "", "PyTorch (*.pth)")
        if path:
            self.checkpoint_edit.setText(path)

    def _on_browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "config 파일 선택", "", "Python (*.py)")
        if path:
            self.config_edit.setText(path)

    def _on_browse_mask_out_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "마스크 저장 폴더 선택", self.mask_out_edit.text())
        if path:
            self.mask_out_edit.setText(path)

    def _on_browse_labels_out(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "라벨 JSON 저장 위치", self.labels_out_edit.text(), "JSON (*.json)"
        )
        if path:
            self.labels_out_edit.setText(path)

    def _on_browse_preview_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "미리보기 저장 폴더 선택", self.preview_dir_edit.text())
        if path:
            self.preview_dir_edit.setText(path)

    # ----------------------------------------------------------- actions
    def _on_start(self) -> None:
        dataset = self.dataset_edit.text().strip()
        if not dataset:
            QMessageBox.warning(self, "설정 확인 필요", "--dataset 세션 폴더명을 입력하세요.")
            return

        cad_path = self.cad_edit.text().strip()
        if not cad_path or not os.path.isfile(cad_path):
            QMessageBox.warning(self, "설정 확인 필요", "유효한 CAD 파일을 선택하세요.")
            return

        checkpoint_path = self.checkpoint_edit.text().strip()
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            QMessageBox.warning(self, "설정 확인 필요", "유효한 체크포인트(.pth)를 선택하세요.")
            return

        config_path = self.config_edit.text().strip()
        if not config_path or not os.path.isfile(config_path):
            QMessageBox.warning(self, "설정 확인 필요", "유효한 config(.py)를 선택하세요.")
            return

        if not DEFAULT_SCRIPT.is_file():
            QMessageBox.warning(
                self, "스크립트 없음",
                f"{DEFAULT_SCRIPT} 를 찾을 수 없습니다. "
                "scripts/generate_rotation_labels.py 위치를 확인하세요.",
            )
            return

        mask_out_dir = self.mask_out_edit.text().strip() or str(DEFAULT_MASK_OUT_DIR)
        labels_out = self.labels_out_edit.text().strip() or str(DEFAULT_LABELS_OUT)
        device = self.device_edit.text().strip() or "cuda:0"
        preview_dir = self.preview_dir_edit.text().strip()

        # 경로에 공백이 섞여도 깨지지 않도록 각 인자를 개별적으로 quote한다
        # (TrainRunner.start()가 내부에서 shlex.split()으로 이 문자열을 다시 쪼갠다).
        args = [
            "python", str(DEFAULT_SCRIPT),
            "--dataset", dataset,
            "--cad", cad_path,
            "--checkpoint", checkpoint_path,
            "--config", config_path,
            "--fitness-min", str(self.fitness_spin.value()),
            "--score-threshold", str(self.score_spin.value()),
            "--mask-out-dir", mask_out_dir,
            "--labels-out", labels_out,
            "--device", device,
        ]
        if preview_dir:
            args += ["--preview-dir", preview_dir]
        command = " ".join(shlex.quote(a) for a in args)

        self.summary_label.setText("-")
        self.script_log.clear()
        self._clear_preview_gallery()
        self._clear_result_panel()

        self.runner = TrainRunner(self)
        self.runner.log_line.connect(self.script_log.append_log)
        self.runner.log_line.connect(self._on_log_line)
        self.runner.finished.connect(self._on_finished)
        self.runner.error.connect(self._on_error)
        self.runner.start(command, working_dir=str(PROJECT_ROOT))
        self.log_message.emit(f"라벨 생성 시작 (dataset={dataset}, fitness-min={self.fitness_spin.value()})")
        self.btn_start.setEnabled(False)

        if preview_dir:
            self._preview_dir_path = Path(preview_dir)
            self._preview_timer.start()

    def _on_stop(self) -> None:
        if self.runner:
            self.runner.stop()

    def _on_log_line(self, line: str) -> None:
        m = SUMMARY_RE.search(line)
        if m:
            total, accepted = m.group("total"), m.group("accepted")
            self.summary_label.setText(f"전체 {total}개 중 {accepted}개 채택")
            return

        m = RESULT_RE.search(line)
        if m:
            self._add_result_card(
                frame=m.group("frame"),
                obj=m.group("obj"),
                status=m.group("status"),
                fitness_str=m.group("fitness"),
                reason=m.group("reason"),
            )

    def _on_finished(self, exit_code: int) -> None:
        self.log_message.emit(f"라벨 생성 프로세스 종료 (exit code {exit_code})")
        self.btn_start.setEnabled(True)
        if exit_code == 0 and self.summary_label.text() == "-":
            self.summary_label.setText("완료 (요약 줄을 로그에서 못 찾음 - 로그 콘솔 확인)")
        if self._preview_timer.isActive():
            self._poll_preview_dir()  # 프로세스 종료 직전에 저장된 마지막 파일들까지 마저 반영
            self._preview_timer.stop()

    def _on_error(self, message: str) -> None:
        QMessageBox.critical(self, "라벨 생성 오류", message)
        self.btn_start.setEnabled(True)

    # ------------------------------------------------------- result cards
    def _clear_result_panel(self) -> None:
        self._n_accept = 0
        self._n_reject = 0
        self.result_count_label.setText("성공 0 / 실패 0")
        while self._result_cards:
            w = self._result_cards.popleft()
            self.result_layout.removeWidget(w)
            w.deleteLater()

    def _add_result_card(
        self, frame: str, obj: str, status: str, fitness_str: str, reason: str | None
    ) -> None:
        """icp_workbench_base._render_result_panel과 동일한 카드 스타일 재사용.

        기존 ICP 탭은 in-process로 검출/정합을 돌려서 결과 객체(ICPResult)를 직접
        갖고 있지만, 여기는 generate_rotation_labels.py가 별도 subprocess로 떠서
        stdout에 찍은 RESULT 줄을 파싱해 들어온 값이라 필드가 문자열 그대로다.
        """
        fitness = None if fitness_str == "none" else float(fitness_str)

        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setStyleSheet(
            "QFrame { border: 1px solid #ddd; border-radius: 6px; padding: 4px; margin-bottom: 4px; }"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 6, 8, 6)

        title = QLabel(f"{frame} · obj{obj}")
        title.setStyleSheet("font-weight: 600;")
        card_layout.addWidget(title)

        if status == "ACCEPT":
            self._n_accept += 1
            status_label = QLabel(f"✓ 채택 · fitness {fitness:.3f}")
            status_label.setStyleSheet("color: #2a8a2a;")
        else:
            self._n_reject += 1
            if fitness is not None:
                text = f"✗ 거부 · fitness {fitness:.3f} ({reason or 'threshold 미달'})"
            else:
                text = f"✗ 거부 · {reason or 'ICP 실패'}"
            status_label = QLabel(text)
            status_label.setStyleSheet("color: #c0392b;")
            status_label.setWordWrap(True)
        card_layout.addWidget(status_label)

        self.result_layout.insertWidget(self.result_layout.count() - 1, card)
        self._result_cards.append(card)
        self.result_count_label.setText(f"성공 {self._n_accept} / 실패 {self._n_reject}")

        if len(self._result_cards) > RESULT_MAX_CARDS:
            oldest = self._result_cards.popleft()
            self.result_layout.removeWidget(oldest)
            oldest.deleteLater()

    # ----------------------------------------------------------- preview
    def _clear_preview_gallery(self) -> None:
        self._preview_seen.clear()
        while self._preview_widgets:
            w = self._preview_widgets.popleft()
            self.preview_grid.removeWidget(w)
            w.deleteLater()

    def _poll_preview_dir(self) -> None:
        """preview_dir을 스캔해서 아직 갤러리에 없는 파일만 새로 추가한다.

        QTimer로 주기 폴링하는 방식을 쓴 이유: 미리보기 이미지는 별도
        프로세스(subprocess로 뜬 generate_rotation_labels.py)가 파일시스템에
        쓰는 것이라, 프로세스 간 직접 콜백을 걸 수 없다. 폴링이 가장 단순하고
        견고하다 (파일이 생기는 시점과 다 써지는 시점 사이 레이스는 mtime
        변화가 멈춘 뒤 다음 tick에 읽으면 되므로 1초 주기면 실질적으로 문제없음).
        """
        preview_dir = getattr(self, "_preview_dir_path", None)
        if preview_dir is None or not preview_dir.is_dir():
            return

        new_files = sorted(
            f for f in preview_dir.glob("*.jpg") if f.name not in self._preview_seen
        )
        for f in new_files:
            self._preview_seen.add(f.name)
            self._add_preview_thumbnail(f)

    def _add_preview_thumbnail(self, image_path: Path) -> None:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return  # 쓰는 도중인 파일을 읽었을 가능성 - 다음 폴링에서 파일명이 바뀌므로 그냥 건너뜀
        pixmap = pixmap.scaled(
            PREVIEW_THUMB_SIZE, PREVIEW_THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        )

        label = QLabel()
        label.setPixmap(pixmap)
        label.setToolTip(image_path.name)
        label.setFixedSize(PREVIEW_THUMB_SIZE, PREVIEW_THUMB_SIZE)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        index = len(self._preview_widgets)
        self.preview_grid.addWidget(label, index // PREVIEW_COLUMNS, index % PREVIEW_COLUMNS)
        self._preview_widgets.append(label)

        if len(self._preview_widgets) > PREVIEW_MAX_ITEMS:
            oldest = self._preview_widgets.popleft()
            self.preview_grid.removeWidget(oldest)
            oldest.deleteLater()
            self._regrid_preview_widgets()

    def _regrid_preview_widgets(self) -> None:
        """오래된 썸네일 제거 후 남은 위젯들을 빈틈없이 다시 배치."""
        for index, w in enumerate(self._preview_widgets):
            self.preview_grid.addWidget(w, index // PREVIEW_COLUMNS, index % PREVIEW_COLUMNS)