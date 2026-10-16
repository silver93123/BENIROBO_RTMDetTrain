"""2D 오토라벨링 탭: 추론 -> 우선순위 대기열 -> labelme 검수 -> 데이터셋 병합/COCO 내보내기.

실제 작업은 전부 `python -m src.autolabel ...` (src/autolabel/cli.py) 가 한다.
이 탭은 인자를 채워 QProcess 로 실행하고 로그/진행률을 보여주는 얇은 껍데기다.
mmdet 같은 무거운 의존성이 UI 프로세스에 로드되지 않고, 추론 중에도 UI가 멈추지 않는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PyQt6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT

# labelme 는 PySide6 기반이라 이 앱(PyQt6)과 같은 env 에 두면 충돌할 수 있다.
# 별도 env 에 설치했다면 그 env 의 labelme 절대경로로 바꾼다.
LABELME_EXECUTABLE = "labelme"

AUTOLABEL_ROOT = PROJECT_ROOT / "data" / "autolabel"


def _default_checkpoint() -> str:
    """탭 3 과 같은 방식으로 최신 best 체크포인트를 미리 채워 둔다 (실패해도 무방)."""
    try:
        from app.core.config_patcher import find_latest_best_checkpoint

        for args in ((), (PROJECT_ROOT / "work_dirs",), (str(PROJECT_ROOT / "work_dirs"),)):
            try:
                found = find_latest_best_checkpoint(*args)
            except TypeError:
                continue
            if found:
                return str(found)
    except Exception:
        pass
    return ""


class AutoLabelTab(QWidget):
    log_message = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._steps: list[tuple[str, list[str]]] = []
        self._current_step = ""
        self._labelme_proc = None
        self._build_ui()
        self._refresh_summary()

    # ------------------------------------------------------------------ UI

    def _path_row(self, edit: QLineEdit, mode: str, filt: str = "") -> QWidget:
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(edit, 1)
        btn = QPushButton("찾기")
        btn.clicked.connect(lambda: self._browse(edit, mode, filt))
        lay.addWidget(btn)
        return row

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        box = QGroupBox("입력")
        form = QFormLayout(box)
        self.ed_images = QLineEdit()
        self.ed_config = QLineEdit(str(DEFAULT_CONFIG_PATH))
        self.ed_ckpt = QLineEdit(_default_checkpoint())
        self.ed_round = QLineEdit("round01")
        self.ed_round.textChanged.connect(lambda _t: self._refresh_summary())
        form.addRow("새 이미지 폴더", self._path_row(self.ed_images, "dir"))
        form.addRow("Config (.py)", self._path_row(self.ed_config, "file", "Python (*.py)"))
        form.addRow("체크포인트 (.pth)", self._path_row(self.ed_ckpt, "file", "Checkpoint (*.pth)"))
        form.addRow("라운드 이름", self.ed_round)
        self.chk_tta = QCheckBox("좌우반전 TTA 사용 (불확실도 정확, 추론 2배 느림)")
        self.chk_tta.setChecked(True)
        form.addRow("", self.chk_tta)
        root.addWidget(box)

        box1 = QGroupBox("1. 추론 + 대기열 생성")
        lay1 = QVBoxLayout(box1)
        self.btn_predict = QPushButton("추론하고 검수 대기열 만들기")
        self.btn_predict.clicked.connect(self._on_predict_queue)
        lay1.addWidget(self.btn_predict)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        lay1.addWidget(self.progress)
        root.addWidget(box1)

        box2 = QGroupBox("2. 검수 (labelme)")
        lay2 = QHBoxLayout(box2)
        self.cmb_bucket = QComboBox()
        self.cmb_bucket.addItems(["urgent", "normal", "auto"])
        lay2.addWidget(QLabel("대기열"))
        lay2.addWidget(self.cmb_bucket)
        self.btn_edit = QPushButton("labelme 열기")
        self.btn_edit.clicked.connect(self._on_edit)
        lay2.addWidget(self.btn_edit, 1)
        root.addWidget(box2)

        box3 = QGroupBox("3. 데이터셋 확정 + COCO 내보내기")
        lay3 = QVBoxLayout(box3)
        self.chk_unreviewed = QCheckBox("미검수 초안도 포함 (권장하지 않음: 모델 오류가 정답으로 들어갑니다)")
        lay3.addWidget(self.chk_unreviewed)
        self.btn_merge = QPushButton("병합하고 COCO 내보내기")
        self.btn_merge.clicked.connect(self._on_merge)
        lay3.addWidget(self.btn_merge)
        root.addWidget(box3)

        self.lbl_summary = QLabel()
        self.lbl_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_summary.setWordWrap(True)
        root.addWidget(self.lbl_summary)
        root.addStretch(1)

    # ------------------------------------------------------------- helpers

    def _browse(self, edit: QLineEdit, mode: str, filt: str) -> None:
        start = edit.text() or str(PROJECT_ROOT)
        if mode == "dir":
            path = QFileDialog.getExistingDirectory(self, "폴더 선택", start)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "파일 선택", start, filt)
        if path:
            edit.setText(path)

    def _round_name(self) -> str:
        return self.ed_round.text().strip()

    def _paths(self) -> dict[str, Path]:
        name = self._round_name()
        return {
            "cache": AUTOLABEL_ROOT / "cache" / name,
            "round": AUTOLABEL_ROOT / "rounds" / name,
            "dataset": AUTOLABEL_ROOT / "dataset" / name,
        }

    def _log(self, msg: str) -> None:
        self.log_message.emit(f"[오토라벨링] {msg}")

    def _set_busy(self, busy: bool) -> None:
        for b in (self.btn_predict, self.btn_merge):
            b.setEnabled(not busy)

    def _refresh_summary(self) -> None:
        if not self._round_name():
            self.lbl_summary.setText("")
            return
        paths = self._paths()
        lines = [f"라운드 폴더: {paths['round']}"]
        manifest = paths["round"] / "manifest.json"
        if manifest.exists():
            try:
                counts = json.loads(manifest.read_text(encoding="utf-8"))["counts"]
                total = max(counts["total"], 1)
                lines.append(
                    f"대기열: 총 {counts['total']}장 (urgent {counts['urgent']} / "
                    f"normal {counts['normal']} / auto {counts['auto']}, 자동승인 {counts['auto'] / total * 100:.0f}%)"
                )
            except Exception as exc:  # noqa: BLE001
                lines.append(f"manifest 읽기 실패: {exc}")
        stats = paths["round"] / "round_stats.json"
        if stats.exists():
            try:
                s = json.loads(stats.read_text(encoding="utf-8"))["summary"]
                lines.append(
                    f"병합: {s['images']}장 / {s['instances']}개 | 평균 edit_iou {s['mean_edit_iou']} | "
                    f"삭제 {s['total_deleted']} / 추가 {s['total_added']} | 미검수 제외 {s['unreviewed_excluded']}장"
                )
            except Exception as exc:  # noqa: BLE001
                lines.append(f"통계 읽기 실패: {exc}")
        self.lbl_summary.setText("\n".join(lines))

    # --------------------------------------------------------- step runner

    def _module_args(self, *args: str) -> list[str]:
        return ["-m", "src.autolabel", *args]

    def _run_steps(self, steps: list[tuple[str, list[str]]]) -> None:
        if self._proc is not None:
            return
        self._steps = list(steps)
        self.progress.setValue(0)
        self._set_busy(True)
        self._next_step()

    def _next_step(self) -> None:
        if not self._steps:
            self._set_busy(False)
            self._refresh_summary()
            return
        self._current_step, args = self._steps.pop(0)
        proc = QProcess(self)
        proc.setWorkingDirectory(str(PROJECT_ROOT))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        proc.setProcessEnvironment(env)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        self._proc = proc
        self._log(f"{self._current_step} 시작")
        proc.start(sys.executable, args)

    def _on_output(self) -> None:
        if self._proc is None:
            return
        text = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in text.splitlines():
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("[predict]"):
                self._parse_progress(line)
            self.log_message.emit(line)

    def _parse_progress(self, line: str) -> None:
        # "[predict] 30/120  (12.3s)"
        try:
            frac = line.split()[1]
            n, total = frac.split("/")
            self.progress.setValue(int(int(n) * 100 / max(int(total), 1)))
        except (IndexError, ValueError):
            pass

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._log(f"{self._current_step}: 프로세스를 시작하지 못했습니다 ({sys.executable})")
            self._abort()

    def _on_finished(self, code: int, _status) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.deleteLater()
        if code != 0:
            self._log(f"{self._current_step} 실패 (종료 코드 {code}) — 위 로그를 확인하세요")
            self._steps.clear()
            self._set_busy(False)
            self._refresh_summary()
            return
        self._log(f"{self._current_step} 완료")
        self._next_step()

    def _abort(self) -> None:
        self._proc = None
        self._steps.clear()
        self._set_busy(False)

    # ------------------------------------------------------------- actions

    def _on_predict_queue(self) -> None:
        images = self.ed_images.text().strip()
        config = self.ed_config.text().strip()
        ckpt = self.ed_ckpt.text().strip()
        if not self._round_name():
            QMessageBox.warning(self, "오토라벨링", "라운드 이름을 입력하세요.")
            return
        for label, value, is_dir in (("이미지 폴더", images, True), ("Config", config, False), ("체크포인트", ckpt, False)):
            ok = Path(value).is_dir() if is_dir else Path(value).is_file()
            if not value or not ok:
                QMessageBox.warning(self, "오토라벨링", f"{label} 경로를 확인하세요:\n{value}")
                return

        paths = self._paths()
        overwrite: list[str] = []
        if (paths["round"] / "manifest.json").exists():
            answer = QMessageBox.question(
                self,
                "대기열이 이미 있습니다",
                f"'{self._round_name()}' 라운드에 이미 대기열이 있습니다.\n"
                "다시 만들면 지금까지 검수·수정한 내용이 모두 초안으로 돌아갑니다.\n\n계속할까요?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            overwrite = ["--overwrite"]

        predict_args = [
            "--images", images,
            "--cache", str(paths["cache"]),
            "--config", config,
            "--checkpoint", ckpt,
        ]
        if not self.chk_tta.isChecked():
            predict_args.append("--no-tta")
        if overwrite:
            predict_args.append("--overwrite")  # 같은 이름 라운드를 다시 만들면 캐시도 새로 계산

        queue_args = [
            "--images", images,
            "--cache", str(paths["cache"]),
            "--round", str(paths["round"]),
            *overwrite,
        ]
        self._run_steps(
            [
                ("추론", self._module_args("predict", *predict_args)),
                ("대기열 생성", self._module_args("queue", *queue_args)),
            ]
        )

    def _on_edit(self) -> None:
        paths = self._paths()
        bucket = self.cmb_bucket.currentText()
        if not (paths["round"] / bucket).is_dir():
            QMessageBox.warning(self, "오토라벨링", f"대기열이 없습니다. 먼저 1단계를 실행하세요:\n{paths['round'] / bucket}")
            return
        try:
            from src.autolabel.queue_builder import launch_labelme

            log_path = paths["round"] / "labelme.log"
            self._labelme_proc = launch_labelme(paths["round"], bucket, LABELME_EXECUTABLE, log_path=log_path)
        except FileNotFoundError:
            QMessageBox.warning(
                self,
                "labelme 를 찾을 수 없습니다",
                f"'{LABELME_EXECUTABLE}' 실행 파일이 없습니다.\n"
                "labelme 를 설치하거나 autolabel_tab.py 의 LABELME_EXECUTABLE 을 절대경로로 바꾸세요.",
            )
            return
        self._log(f"labelme 실행 ({bucket}) — 이미지 하나를 다듬은 뒤 'reviewed' 를 체크하면 학습에 포함됩니다")
        self._log_path = log_path
        QTimer.singleShot(3000, self._check_labelme_alive)

    def _check_labelme_alive(self) -> None:
        proc = self._labelme_proc
        if proc is None or proc.poll() is None:
            return
        tail = ""
        try:
            tail = "\n".join(self._log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-10:])
        except OSError:
            pass
        self._log(f"labelme 가 바로 종료되었습니다 (코드 {proc.returncode})\n{tail}")

    def _on_merge(self) -> None:
        paths = self._paths()
        if not (paths["round"] / "manifest.json").exists():
            QMessageBox.warning(self, "오토라벨링", "대기열이 없습니다. 먼저 1단계를 실행하세요.")
            return
        merge_args = ["--round", str(paths["round"]), "--dataset", str(paths["dataset"])]
        if self.chk_unreviewed.isChecked():
            merge_args.append("--include-unreviewed")
        self._run_steps(
            [
                ("병합", self._module_args("merge", *merge_args)),
                ("COCO 내보내기", self._module_args("export", "--dataset", str(paths["dataset"]))),
            ]
        )

    # -------------------------------------------------------------- cleanup

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._proc is not None:
            self._proc.kill()
        super().closeEvent(event)
