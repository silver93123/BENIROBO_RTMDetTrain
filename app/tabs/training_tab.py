"""탭 2: 모델 학습.

실제 학습 스크립트(Train_rtmdet_model.py) 기준:
  python scripts/2_Train_rtmdet_model.py --dataset <폴더명> [--config ...] [--epochs ...]

- --dataset 은 폴더명만 받는다 (스크립트 안에 하드코딩된 DATASET_BASE와 합쳐짐).
  DATASET_BASE는 스크립트 파일 안의 상수라서 이 앱에서 바꿀 수 없다.
  새 프로젝트 폴더로 옮겼다면 스크립트 안에서 그 상수부터 고쳐야 한다.
- 하이퍼파라미터는 --config로 지정한 config 파일 안에서 관리한다 (epoch만 --epochs로 override 가능).
- 이어서 학습(resume) 로직은 스크립트 안에 이미 있다 (work_dir에 best_*.pth 있으면 자동 이어서).
  이 탭의 "체크포인트 override"는 work_dir에 best 파일이 아직 하나도 없는
  최초 학습에만 실질적으로 적용된다 (best가 있으면 스크립트가 그걸 무조건 우선함).
"""
from __future__ import annotations

import os
import subprocess

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import pyqtSignal, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QMessageBox, QFrame,
)

from app.core.train_runner import TrainRunner
from app.core.config_patcher import (
    make_config_with_override, find_latest_best_checkpoint, find_work_dir,
)
from app.core.paths import PROJECT_ROOT, DEFAULT_CONFIG_PATH
from app.widgets.log_console import LogConsole

DEFAULT_COMMAND_TEMPLATE = "python scripts/2_Train_rtmdet_model.py --dataset {dataset} --config {config}"


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


class TrainingTab(QWidget):
    log_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loss_history: list[float] = []
        self._map_history: list[float] = []
        self.runner: TrainRunner | None = None
        self._build_ui()

        # 학습 스크립트는 항상 프로젝트 루트를 작업 디렉토리로 사용한다 (내부 처리, UI 노출 안 함).
        self._working_dir = str(PROJECT_ROOT)
        if DEFAULT_CONFIG_PATH.is_file():
            self.config_edit.setText(str(DEFAULT_CONFIG_PATH))

    # ------------------------------------------------------------ session
    def set_session_path(self, path: str) -> None:
        """데이터 세션 탭에서 세션을 선택하면 폴더명만 --dataset 값으로 채운다."""
        dataset_name = os.path.basename(os.path.normpath(path))
        self.dataset_edit.setText(dataset_name)
        self.log_message.emit(
            f"데이터 세션 탭에서 --dataset 값 자동 연동: {dataset_name} "
            "(스크립트 안 DATASET_BASE 하위 폴더명과 일치해야 합니다)"
        )

    # ---------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("--dataset (폴더명만)"))
        self.dataset_edit = QLineEdit()
        self.dataset_edit.setPlaceholderText("예: 20260521_114500")
        row1.addWidget(self.dataset_edit, stretch=1)
        row1.addWidget(QLabel("--epochs (선택)"))
        self.epochs_edit = QLineEdit()
        self.epochs_edit.setPlaceholderText("비우면 config 값 사용")
        self.epochs_edit.setFixedWidth(120)
        row1.addWidget(self.epochs_edit)
        layout.addLayout(row1)

        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("--config 파일 (.py)"))
        self.config_edit = QLineEdit()
        cfg_row.addWidget(self.config_edit, stretch=1)
        btn_browse_cfg = QPushButton("선택")
        btn_browse_cfg.clicked.connect(self._on_browse_config)
        cfg_row.addWidget(btn_browse_cfg)
        btn_open_editor = QPushButton("에디터로 열기")
        btn_open_editor.clicked.connect(self._on_open_config_editor)
        cfg_row.addWidget(btn_open_editor)
        btn_generate_cfg = QPushButton("COCO에서 config 자동 생성")
        btn_generate_cfg.setToolTip(
            "위 --dataset 폴더의 annotations/instances_Train.json에서 "
            "클래스 이름/개수를 읽어 새 config를 자동으로 만듭니다."
        )
        btn_generate_cfg.clicked.connect(self._on_generate_config)
        cfg_row.addWidget(btn_generate_cfg)
        layout.addLayout(cfg_row)

        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(QLabel("최초 학습 시작 체크포인트 override (선택)"))
        self.checkpoint_edit = QLineEdit()
        self.checkpoint_edit.setPlaceholderText(
            "work_dir에 best_*.pth가 이미 있으면 스크립트가 그걸 우선 사용 - 이 값은 무시됨"
        )
        ckpt_row.addWidget(self.checkpoint_edit, stretch=1)
        btn_browse_ckpt = QPushButton("찾아보기")
        btn_browse_ckpt.clicked.connect(self._on_browse_checkpoint)
        ckpt_row.addWidget(btn_browse_ckpt)
        btn_check_best = QPushButton("현재 시작점 확인")
        btn_check_best.setToolTip("이 config의 work_dir에 best_*.pth가 있는지 확인만 합니다.")
        btn_check_best.clicked.connect(self._on_check_start_point)
        ckpt_row.addWidget(btn_check_best)
        layout.addLayout(ckpt_row)

        cards_row = QHBoxLayout()
        self.card_epoch = MetricCard("Epoch")
        self.card_loss = MetricCard("loss")
        self.card_bbox_map = MetricCard("bbox mAP")
        self.card_segm_map = MetricCard("segm mAP")
        for c in (self.card_epoch, self.card_loss, self.card_bbox_map, self.card_segm_map):
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
    def _on_browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "config 파일 선택", "", "Python (*.py)")
        if path:
            self.config_edit.setText(path)

    def _on_open_config_editor(self) -> None:
        path = self.config_edit.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "알림", "먼저 유효한 config 파일을 선택하세요.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _on_browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "체크포인트 선택", "", "PyTorch (*.pth)")
        if path:
            self.checkpoint_edit.setText(path)

    def _on_generate_config(self) -> None:
        dataset = self.dataset_edit.text().strip()
        if not dataset:
            QMessageBox.warning(
                self, "설정 확인 필요",
                "먼저 위 --dataset 칸에 폴더명을 입력하세요 "
                "(그 폴더의 annotations/instances_Train.json을 읽습니다).",
            )
            return

        script_path = PROJECT_ROOT / "scripts" / "make_training_config.py"
        if not script_path.is_file():
            QMessageBox.warning(
                self, "스크립트 없음",
                f"{script_path} 를 찾을 수 없습니다.\n"
                "scripts/make_training_config.py 위치에 스크립트를 넣어주세요.",
            )
            return

        self.log_message.emit(f"config 자동 생성 실행 중... (dataset={dataset})")
        try:
            result = subprocess.run(
                ["python", str(script_path), "--dataset", dataset],
                cwd=str(PROJECT_ROOT),
                capture_output=True, text=True, timeout=30,
            )
        except Exception as exc:
            QMessageBox.critical(self, "실행 실패", f"스크립트 실행 중 오류:\n{exc}")
            return

        output = (result.stdout or "") + (result.stderr or "")
        for line in output.splitlines():
            self.train_log.append_log(line)

        if result.returncode != 0:
            QMessageBox.critical(
                self, "config 생성 실패",
                "자동 생성에 실패했습니다. 아래 로그 콘솔에서 원인을 확인하세요.",
            )
            return

        generated_path = None
        for line in output.splitlines():
            if "생성됨:" in line:
                generated_path = line.split("생성됨:", 1)[1].strip()
                break

        if generated_path and os.path.isfile(generated_path):
            self.config_edit.setText(generated_path)
            self.log_message.emit(f"COCO 어노테이션에서 config 자동 생성 완료: {generated_path}")
        else:
            QMessageBox.information(
                self, "생성 완료",
                "config가 생성된 것 같지만 경로를 자동으로 못 찾았습니다. "
                "로그 콘솔에서 경로를 확인하고 '선택' 버튼으로 직접 지정하세요.",
            )

    def _on_check_start_point(self) -> None:
        cfg_path = self.config_edit.text().strip()
        if not cfg_path or not os.path.isfile(cfg_path):
            QMessageBox.warning(self, "알림", "먼저 유효한 config 파일을 선택하세요.")
            return
        work_dir = find_work_dir(cfg_path)
        best = find_latest_best_checkpoint(cfg_path)
        if best:
            QMessageBox.information(
                self, "예상 시작점",
                f"work_dir({work_dir})에 best 체크포인트가 있어 이어서 학습됩니다:\n{best}\n\n"
                "(체크포인트 override 필드는 무시됩니다)"
            )
        else:
            QMessageBox.information(
                self, "예상 시작점",
                f"work_dir({work_dir})에 best 체크포인트가 없어, "
                "config의 load_from(또는 override 값)에서 최초 학습을 시작합니다."
            )

    def _on_start(self) -> None:
        dataset = self.dataset_edit.text().strip()
        if not dataset:
            QMessageBox.warning(self, "설정 확인 필요", "--dataset 폴더명을 입력하세요.")
            return

        cfg_path = self.config_edit.text().strip()
        if not cfg_path or not os.path.isfile(cfg_path):
            QMessageBox.warning(self, "설정 확인 필요", "유효한 config 파일을 선택하세요.")
            return

        actual_cfg_path = cfg_path
        override = self.checkpoint_edit.text().strip()
        if override:
            if not os.path.isfile(override):
                QMessageBox.warning(
                    self, "설정 확인 필요", f"체크포인트 파일을 찾을 수 없습니다: {override}"
                )
                return
            if find_latest_best_checkpoint(cfg_path):
                self.log_message.emit(
                    "주의: work_dir에 이미 best 체크포인트가 있어 override 값은 "
                    "스크립트에 의해 무시될 가능성이 높습니다."
                )
            actual_cfg_path = make_config_with_override(cfg_path, override)
            self.log_message.emit(f"override 적용된 config 생성: {actual_cfg_path}")

        # 실행 커맨드/작업 디렉토리는 UI에 노출하지 않고 내부적으로 결정한다.
        command = DEFAULT_COMMAND_TEMPLATE.format(dataset=dataset, config=actual_cfg_path)

        epochs = self.epochs_edit.text().strip()
        if epochs:
            if not epochs.isdigit():
                QMessageBox.warning(self, "설정 확인 필요", "--epochs 값은 숫자여야 합니다.")
                return
            command += f" --epochs {epochs}"

        self._loss_history.clear()
        self._map_history.clear()
        self.train_log.clear()

        self.runner = TrainRunner(self)
        self.runner.log_line.connect(self.train_log.append_log)
        self.runner.progress.connect(self._on_progress)
        self.runner.finished.connect(self._on_finished)
        self.runner.error.connect(self._on_error)
        self.runner.start(command, working_dir=self._working_dir)
        self.log_message.emit(f"학습 시작 (dataset={dataset}, epochs={epochs or 'config 기본값'})")
        self.btn_start.setEnabled(False)

    def _on_stop(self) -> None:
        if self.runner:
            self.runner.stop()

    def _on_progress(self, data: dict) -> None:
        if "epoch" in data:
            self.card_epoch.set_value(str(data["epoch"]))
        if "loss" in data:
            self.card_loss.set_value(f"{data['loss']:.3f}")
            self._loss_history.append(data["loss"])
        if "bbox_map" in data:
            self.card_bbox_map.set_value(f"{data['bbox_map']:.2f}")
            self._map_history.append(data["bbox_map"])
        if "segm_map" in data:
            self.card_segm_map.set_value(f"{data['segm_map']:.2f}")

        self.ax.clear()
        if self._loss_history:
            self.ax.plot(self._loss_history, label="loss", color="#D85A30")
        if self._map_history:
            self.ax.plot(self._map_history, label="bbox_mAP", color="#378ADD")
        if self._loss_history or self._map_history:
            self.ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw()

    def _on_finished(self, exit_code: int) -> None:
        self.log_message.emit(f"학습 프로세스 종료 (exit code {exit_code})")
        self.btn_start.setEnabled(True)

    def _on_error(self, message: str) -> None:
        QMessageBox.critical(self, "학습 오류", message)
        self.btn_start.setEnabled(True)