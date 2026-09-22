"""오토라벨링 탭 (Tab 5 후보).

기존 탭들과 같은 패턴을 따른다:
  - 무거운 작업은 QProcess 서브프로세스로 분리 (CUDA/Qt 충돌 회피)
  - 경로/상수는 UI로 노출하지 않고 파일 상단에 내부화
  - labelme는 별도 프로세스로 detached 실행

메인 윈도우에 붙이는 법:
    from tab_autolabel import AutoLabelTab
    self.tabs.addTab(AutoLabelTab(self), "오토라벨링")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PyQt6.QtCore import QProcess, QProcessEnvironment, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# ======================================================================
# 내부 상수 — 배포 시 실제 경로로 수정할 것
# ======================================================================
REPO_ROOT = Path(__file__).resolve().parent
MMDET_CONFIG = REPO_ROOT / "configs" / "rtmdet-ins_s_binpicking.py"
CHECKPOINT = REPO_ROOT / "work_dirs" / "latest" / "best_coco_segm_mAP.pth"
CLASSES = ["part_a", "part_b"]  # mmdet config의 classes와 반드시 일치
ROUNDS_ROOT = REPO_ROOT / "autolabel_rounds"
CACHE_ROOT = REPO_ROOT / "autolabel_cache"
DATASET_ROOT = REPO_ROOT / "autolabel_dataset"
PYTHON = sys.executable


class AutoLabelTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # --- 입력 ---
        src = QGroupBox("대상")
        src_layout = QHBoxLayout(src)
        self.image_dir_edit = QLineEdit()
        self.image_dir_edit.setPlaceholderText("신규 이미지 폴더를 선택하세요")
        browse = QPushButton("폴더 선택")
        browse.clicked.connect(self._pick_folder)
        self.round_edit = QLineEdit("round01")
        self.round_edit.setMaximumWidth(140)
        src_layout.addWidget(QLabel("이미지"))
        src_layout.addWidget(self.image_dir_edit, 1)
        src_layout.addWidget(browse)
        src_layout.addWidget(QLabel("라운드"))
        src_layout.addWidget(self.round_edit)
        root.addWidget(src)

        # --- 단계 버튼 ---
        steps = QGroupBox("파이프라인")
        steps_layout = QHBoxLayout(steps)
        self.btn_predict = QPushButton("1. 배치 추론")
        self.btn_queue = QPushButton("2. 대기열 생성")
        self.btn_edit = QPushButton("3. 검수 (labelme)")
        self.bucket_combo = QComboBox()
        self.bucket_combo.addItems(["urgent", "normal", "auto"])
        self.bucket_combo.setMaximumWidth(110)
        self.btn_merge = QPushButton("4. 병합 + COCO")

        self.btn_predict.clicked.connect(self._run_predict)
        self.btn_queue.clicked.connect(self._run_queue)
        self.btn_edit.clicked.connect(self._run_edit)
        self.btn_merge.clicked.connect(self._run_merge)

        for widget in (self.btn_predict, self.btn_queue, self.btn_edit,
                       self.bucket_combo, self.btn_merge):
            steps_layout.addWidget(widget)
        root.addWidget(steps)

        # --- 요약 ---
        self.summary = QLabel("대기 중")
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.summary)

        # --- 로그 ---
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        root.addWidget(self.log, 1)

    # ------------------------------------------------------------------
    # 경로 헬퍼
    # ------------------------------------------------------------------
    def _image_dir(self) -> Path | None:
        text = self.image_dir_edit.text().strip()
        if not text:
            self._append("[오류] 이미지 폴더를 먼저 선택하세요.")
            return None
        path = Path(text)
        if not path.is_dir():
            self._append(f"[오류] 폴더가 없습니다: {path}")
            return None
        return path

    def _round_name(self) -> str:
        return self.round_edit.text().strip() or "round01"

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "신규 이미지 폴더 선택")
        if folder:
            self.image_dir_edit.setText(folder)

    # ------------------------------------------------------------------
    # 프로세스 실행
    # ------------------------------------------------------------------
    def _busy(self) -> bool:
        return self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning

    def _run(self, args: list[str], label: str) -> None:
        if self._busy():
            self._append("[대기] 이전 작업이 아직 실행 중입니다.")
            return

        self._append(f"\n=== {label} ===")
        self._append("$ " + " ".join(args))
        self._set_enabled(False)

        proc = QProcess(self)
        proc.setWorkingDirectory(str(REPO_ROOT))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        proc.setProcessEnvironment(env)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda: self._drain(proc))
        proc.finished.connect(lambda code, _st: self._on_finished(label, code))
        proc.start(args[0], args[1:])
        self._proc = proc

    def _drain(self, proc: QProcess) -> None:
        data = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            self._append(line)

    def _on_finished(self, label: str, code: int) -> None:
        self._append(f"[{label}] 종료 코드 {code}")
        self._set_enabled(True)
        self._proc = None
        self._refresh_summary()

    def _set_enabled(self, value: bool) -> None:
        for btn in (self.btn_predict, self.btn_queue, self.btn_merge):
            btn.setEnabled(value)

    def _append(self, text: str) -> None:
        self.log.appendPlainText(text)

    # ------------------------------------------------------------------
    # 각 단계
    # ------------------------------------------------------------------
    def _run_predict(self) -> None:
        image_dir = self._image_dir()
        if image_dir is None:
            return
        if not CHECKPOINT.exists():
            self._append(f"[오류] 체크포인트가 없습니다: {CHECKPOINT}")
            return
        self._run(
            [
                PYTHON, "-m", "autolabel", "predict",
                "--images", str(image_dir),
                "--cache", str(CACHE_ROOT / self._round_name()),
                "--config", str(MMDET_CONFIG),
                "--checkpoint", str(CHECKPOINT),
            ],
            "배치 추론",
        )

    def _run_queue(self) -> None:
        image_dir = self._image_dir()
        if image_dir is None:
            return
        self._run(
            [
                PYTHON, "-m", "autolabel", "queue",
                "--images", str(image_dir),
                "--cache", str(CACHE_ROOT / self._round_name()),
                "--round", str(ROUNDS_ROOT / self._round_name()),
                "--classes", ",".join(CLASSES),
            ],
            "대기열 생성",
        )

    def _run_edit(self) -> None:
        """labelme는 사용자가 오래 붙잡고 있으므로 detached로 띄운다."""
        round_dir = ROUNDS_ROOT / self._round_name()
        bucket = self.bucket_combo.currentText()
        target = round_dir / bucket
        if not target.is_dir():
            self._append(f"[오류] 대기열이 없습니다: {target}")
            return

        args = [
            str(target),
            "--labels", str(round_dir / "labels.txt"),
            "--nodata", "--autosave", "--nosortlabels",
        ]
        ok, pid = QProcess.startDetached("labelme", args, str(REPO_ROOT))
        if ok:
            self._append(f"[검수] labelme 실행 (pid={pid}) — {bucket} 버킷")
        else:
            self._append("[오류] labelme 실행 실패. `pip install labelme` 확인.")

    def _run_merge(self) -> None:
        round_dir = ROUNDS_ROOT / self._round_name()
        dataset_dir = DATASET_ROOT / self._round_name()
        if not (round_dir / "manifest.json").exists():
            self._append(f"[오류] manifest.json이 없습니다: {round_dir}")
            return
        # 병합 후 COCO 변환까지 한 번에
        script = (
            "import sys;"
            "from autolabel.merge import merge_round, export_coco;"
            "from pathlib import Path;"
            f"r=Path({str(round_dir)!r});d=Path({str(dataset_dir)!r});"
            "merge_round(r,d);"
            f"export_coco(d,{CLASSES!r},d/'annotations.json')"
        )
        self._run([PYTHON, "-c", script], "병합 + COCO")

    # ------------------------------------------------------------------
    def _refresh_summary(self) -> None:
        stats_path = ROUNDS_ROOT / self._round_name() / "round_stats.json"
        manifest_path = ROUNDS_ROOT / self._round_name() / "manifest.json"
        parts: list[str] = []

        if manifest_path.exists():
            counts = json.loads(manifest_path.read_text(encoding="utf-8")).get("counts", {})
            total = counts.get("total", 0)
            auto = counts.get("auto", 0)
            ratio = (auto / total * 100) if total else 0.0
            parts.append(
                f"대기열 {total}장 (urgent {counts.get('urgent', 0)} / "
                f"normal {counts.get('normal', 0)} / auto {auto} = {ratio:.1f}%)"
            )

        if stats_path.exists():
            summary = json.loads(stats_path.read_text(encoding="utf-8"))["summary"]
            parts.append(
                f"평균 edit_IoU {summary.get('mean_edit_iou')} | "
                f"삭제 {summary.get('total_deleted')} / 추가 {summary.get('total_added')}"
            )

        self.summary.setText("  |  ".join(parts) if parts else "대기 중")