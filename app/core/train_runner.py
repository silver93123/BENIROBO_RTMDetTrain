"""
실제 학습 스크립트를 QProcess로 실행하고, stdout을 그대로 로그로 흘려보낸다.

mmengine의 기본 로그 포맷(LoggerHook)에서 아래 두 종류의 라인을 정규식으로 파싱해서
진행 상황(epoch/loss/mAP)을 뽑아낸다. 실제 로그 포맷이 조금 다르면 정규식을 조정해야 한다.

  Epoch(train) [18][5/5]  ... loss: 0.2110 ...
  Epoch(val)   [18][5/5]  ... coco/bbox_mAP: 0.6400 coco/segm_mAP: 0.5800 ...
"""
from __future__ import annotations

import re
import shlex

from PyQt6.QtCore import QObject, QProcess, pyqtSignal

TRAIN_LINE_RE = re.compile(
    r"Epoch\(train\)\s*\[(?P<epoch>\d+)\]\[(?P<iter>\d+)/(?P<total>\d+)\].*?"
    r"loss:\s*(?P<loss>[\d.]+)"
)
VAL_LINE_RE = re.compile(
    r"Epoch\(val\)\s*\[(?P<epoch>\d+)\].*?"
    r"coco/bbox_mAP:\s*(?P<bbox_map>[\d.]+)"
    r"(?:.*?coco/segm_mAP:\s*(?P<segm_map>[\d.]+))?"
)


class TrainRunner(QObject):
    log_line = pyqtSignal(str)
    progress = pyqtSignal(dict)  # 누적된 {"epoch", "loss", "bbox_map", "segm_map"} 일부/전체
    finished = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: QProcess | None = None
        self._last_progress: dict = {}

    def start(self, command: str, working_dir: str | None = None) -> None:
        if self._process is not None:
            self.error.emit("이미 실행 중인 학습 프로세스가 있습니다.")
            return

        try:
            parts = shlex.split(command)
        except ValueError as exc:
            self.error.emit(f"실행 커맨드를 해석할 수 없습니다: {exc}")
            return
        if not parts:
            self.error.emit("실행 커맨드가 비어 있습니다.")
            return

        self._last_progress = {}
        self._process = QProcess(self)
        if working_dir:
            self._process.setWorkingDirectory(working_dir)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_output)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(
            lambda err: self.error.emit(f"프로세스 오류: {err}")
        )

        program, *args = parts
        self._process.start(program, args)

    def stop(self) -> None:
        if self._process is not None:
            self._process.kill()

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning

    def _on_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            if not line.strip():
                continue
            self.log_line.emit(line)
            self._parse_progress(line)

    def _parse_progress(self, line: str) -> None:
        m = TRAIN_LINE_RE.search(line)
        if m:
            self._last_progress["epoch"] = int(m.group("epoch"))
            self._last_progress["loss"] = float(m.group("loss"))
            self.progress.emit(dict(self._last_progress))
            return

        m = VAL_LINE_RE.search(line)
        if m:
            self._last_progress["epoch"] = int(m.group("epoch"))
            self._last_progress["bbox_map"] = float(m.group("bbox_map"))
            if m.group("segm_map"):
                self._last_progress["segm_map"] = float(m.group("segm_map"))
            self.progress.emit(dict(self._last_progress))

    def _on_finished(self, exit_code: int, exit_status) -> None:
        self.finished.emit(exit_code)
        self._process = None
