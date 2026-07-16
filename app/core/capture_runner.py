"""
0번 탭(데이터 수집)에서 collect_dataset.py 스크립트를 QProcess로 실행한다.

collect_dataset.py는 매 프레임마다 input()으로 아래 입력을 기다리는
대화형(interactive) 스크립트다. 학습 탭의 TrainRunner(단순 stdout 스트리밍)와
달리, stdin에 값을 직접 흘려보내는 기능이 필요해서 별도로 분리했다.

  Enter  -> 현재 프레임 캡처
  s      -> 현재 프레임 스킵
  q      -> 수집 종료 (스크립트 내부에서 정상 종료 처리됨)

로그 라인에서 아래 두 종류를 정규식으로 뽑아 진행 상황(progress)으로 흘려보낸다.
스크립트 쪽 print 포맷이 바뀌면 이 정규식도 같이 맞춰줘야 한다.

  [3/5] 프레임 0013
  ✓ saved |  12.3 ms | valid 87.4% | Z 512.0~890.2 mm (median 640.1)
  완료: 5 프레임
"""
from __future__ import annotations

import re
import shlex

from PyQt6.QtCore import QObject, QProcess, pyqtSignal

FRAME_LINE_RE = re.compile(
    r"\[(?P<k>\d+)/(?P<num>\d+)\]\s*프레임\s*(?P<idx>\d+)"
)
SAVED_LINE_RE = re.compile(
    r"valid\s+(?P<valid>[\d.]+)%.*?"
    r"Z\s+(?P<zmin>-?[\d.]+)~(?P<zmax>-?[\d.]+)\s*mm\s*"
    r"\(median\s*(?P<zmed>-?[\d.]+)\)"
)
DONE_LINE_RE = re.compile(r"완료:\s*(?P<count>\d+)\s*프레임")


class DataCaptureRunner(QObject):
    log_line = pyqtSignal(str)
    # 누적 dict 일부/전체: k, num, idx, valid, zmin, zmax, zmed, captured_total
    progress = pyqtSignal(dict)
    finished = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: QProcess | None = None
        self._last_progress: dict = {}

    # --------------------------------------------------------------- 실행
    def start(self, command: str, working_dir: str | None = None) -> None:
        if self._process is not None:
            self.error.emit("이미 실행 중인 데이터 수집 프로세스가 있습니다.")
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

    def is_running(self) -> bool:
        return (
            self._process is not None
            and self._process.state() != QProcess.ProcessState.NotRunning
        )

    # ---------------------------------------------------------- stdin 입력
    def send_capture(self) -> None:
        """현재 프레임 캡처 (스크립트의 Enter 입력에 해당)."""
        self._write_line("")

    def send_skip(self) -> None:
        """현재 프레임 스킵 ('s' 입력)."""
        self._write_line("s")

    def send_quit(self) -> None:
        """수집 정상 종료 ('q' 입력). 스크립트가 요약 출력 후 스스로 끝난다."""
        self._write_line("q")

    def stop(self) -> None:
        """정상 종료(q)가 먹히지 않을 때만 쓰는 강제 종료."""
        if self._process is not None:
            self._process.kill()

    def _write_line(self, text: str) -> None:
        if not self.is_running():
            self.error.emit("실행 중인 데이터 수집 프로세스가 없습니다.")
            return
        self._process.write((text + "\n").encode("utf-8"))

    # --------------------------------------------------------------- 출력
    def _on_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        for line in data.splitlines():
            if not line.strip():
                continue
            self.log_line.emit(line)
            self._parse_progress(line)

    def _parse_progress(self, line: str) -> None:
        m = FRAME_LINE_RE.search(line)
        if m:
            self._last_progress["k"] = int(m.group("k"))
            self._last_progress["num"] = int(m.group("num"))
            self._last_progress["idx"] = int(m.group("idx"))
            self.progress.emit(dict(self._last_progress))
            return

        m = SAVED_LINE_RE.search(line)
        if m:
            self._last_progress["valid"] = float(m.group("valid"))
            self._last_progress["zmin"] = float(m.group("zmin"))
            self._last_progress["zmax"] = float(m.group("zmax"))
            self._last_progress["zmed"] = float(m.group("zmed"))
            self.progress.emit(dict(self._last_progress))
            return

        m = DONE_LINE_RE.search(line)
        if m:
            self._last_progress["captured_total"] = int(m.group("count"))
            self.progress.emit(dict(self._last_progress))

    def _on_finished(self, exit_code: int, exit_status) -> None:
        self.finished.emit(exit_code)
        self._process = None