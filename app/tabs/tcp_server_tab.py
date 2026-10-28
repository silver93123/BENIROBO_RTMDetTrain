"""탭 5: ICP 정합 TCP 서버 (로봇 요청 대기형).

로봇이 TCP로 요청("PICK")을 보내면 이 앱이 그때 촬영 -> 2D 검출 -> ICP 정합을 돌려서
픽 좌표를 응답한다. 서버를 켜 두기만 하면 되고, 화면에는 마지막 요청의 검출 결과/정합 결과/요청 로그가 보인다.

    로봇  --"PICK"-->  [이 탭]  촬영 -> 검출 -> 정합  --"OK,1,x,y,z,roll,pitch,yaw"-->  로봇

프로토콜 상세는 app/core/pick_server.py 맨 위 설명 참고. 모든 값(ICP 파라미터, 모델/CAD, 카메라, 포트,
명령어, 응답 형식)은 '12. 설정' 탭에서 바꾸고 저장한다 - 이 탭은 저장된 값을 요청이 올 때마다 읽는다
(호스트/포트만 서버 재시작 필요).

수동 1회 실행: 로봇 없이 같은 파이프라인을 한 번 돌려 결과를 화면으로 확인한다 (응답은 보내지 않음).
기존 '5-1. 실시간 촬영 ICP 테스트' 탭은 파라미터를 바꿔 가며 눈으로 확인하는 용도로 그대로 남아 있다.
"""
from __future__ import annotations

import functools
import traceback
from datetime import datetime

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from app.core.app_settings import SettingsStore, get_settings
from app.core.pick_pipeline import PickRunResult
from app.core.pick_server import PickServer, PickWorker
from app.widgets.image_viewer import ImageViewer, make_label_opacity_bar, make_zoom_bar


def _guarded(func):
    """슬롯 예외를 삼키지 않고 로그 + 팝업으로 보여준다 (PyQt 슬롯 안 예외는 조용히 죽는다)."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self._log(f"오류 ({func.__name__}): {exc}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "오류", f"{func.__name__}\n\n{exc}")
            return None
    return wrapper


class TcpServerTab(QWidget):
    log_message = pyqtSignal(str)
    _manual_finished = pyqtSignal(object)    # 워커 스레드 -> GUI 스레드 (수동 실행 결과)

    LOG_PREFIX = "TCP 서버 탭"
    MODE = "icp"             # PickPipeline 모드 ("2d"는 5-2 탭: ICP 없이 형상 보정 템플릿으로 계산)
    USE_PORT_2D = False      # True면 설정의 port_2d를 쓴다
    RUN_DESC = "촬영 -> 검출 -> 정합"

    def __init__(self, store: SettingsStore | None = None, parent=None):
        super().__init__(parent)
        self._store = store or get_settings()
        self._worker = PickWorker(self._store, log=lambda m: self.log_message.emit(f"[{self.LOG_PREFIX}] {m}"),
                                  mode=self.MODE)
        self._server = PickServer(self._store, self._worker, parent=self, use_port_2d=self.USE_PORT_2D)
        self._manual_running = False
        self._build_ui()

        self._server.status_changed.connect(self._on_status)
        self._server.client_count_changed.connect(self._on_clients)
        self._server.request_received.connect(self._on_request)
        self._server.response_sent.connect(self._on_response)
        self._server.run_finished.connect(self._show_result)
        self._server.log_message.connect(self._log)
        self._manual_finished.connect(self._on_manual_finished)
        self._store.changed.connect(self._refresh_config_summary)
        self._refresh_config_summary()
        self._update_buttons()

    def _log(self, message: str) -> None:
        self.log_message.emit(f"[{self.LOG_PREFIX}] {message}")

    # =================================================================
    # UI
    # =================================================================
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.btn_start = QPushButton("서버 시작")
        self.btn_start.setStyleSheet("font-weight: 600; padding: 4px 18px;")
        self.btn_start.clicked.connect(lambda _=False: self._on_start())
        self.btn_stop = QPushButton("서버 정지")
        self.btn_stop.clicked.connect(lambda _=False: self._on_stop())
        self.btn_manual = QPushButton("수동 1회 실행")
        self.btn_manual.setToolTip("로봇 없이 같은 처리를 한 번 돌려 결과를 화면에 표시합니다 (응답은 전송되지 않음)")
        self.btn_manual.clicked.connect(lambda _=False: self._on_manual())
        top.addWidget(self.btn_start)
        top.addWidget(self.btn_stop)
        top.addWidget(self.btn_manual)
        top.addSpacing(16)
        self.status_label = QLabel("● 정지")
        top.addWidget(self.status_label)
        top.addStretch(1)
        self._top_layout = top
        root.addLayout(top)

        self.config_label = QLabel("")
        self.config_label.setStyleSheet("color: #666; font-size: 11px;")
        self.config_label.setWordWrap(True)
        root.addWidget(self.config_label)

        body = QHBoxLayout()
        root.addLayout(body, stretch=1)

        # 왼쪽: 마지막 프레임 + 검출 오버레이
        left = QVBoxLayout()
        self.image_viewer = ImageViewer()
        head = QHBoxLayout()
        head.addWidget(QLabel("마지막 요청의 이미지와 검출 (형상 보정까지 반영된 마스크)"))
        head.addStretch(1)
        head.addWidget(make_zoom_bar(self.image_viewer))   # 휠 확대/축소, 가운데/오른쪽 드래그 이동
        left.addLayout(head)
        self.image_viewer.setMinimumSize(480, 360)
        left.addWidget(self.image_viewer, stretch=1)
        self.opacity_bar = make_label_opacity_bar(self.image_viewer)   # 라벨이 마스크를 가릴 때 투명도 조절
        top.addWidget(self.opacity_bar)
        self.run_label = QLabel("아직 요청이 없습니다.")
        self.run_label.setWordWrap(True)
        left.addWidget(self.run_label)
        body.addLayout(left, stretch=3)

        # 오른쪽: 결과 카드 + 요청 로그
        right = QVBoxLayout()
        right.addWidget(QLabel("마지막 결과 (인스턴스별)"))
        self.result_scroll = QScrollArea()
        self.result_scroll.setWidgetResizable(True)
        self.result_container = QWidget()
        self.result_layout = QVBoxLayout(self.result_container)
        self.result_layout.addStretch(1)
        self.result_scroll.setWidget(self.result_container)
        right.addWidget(self.result_scroll, stretch=3)

        right.addWidget(QLabel("요청 / 응답 로그   ◀ 수신   ▶ 송신"))
        self.traffic = QPlainTextEdit()
        self.traffic.setReadOnly(True)
        self.traffic.setMaximumBlockCount(500)
        self.traffic.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)   # 응답 한 줄이 그대로 보이게
        self.traffic.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        right.addWidget(self.traffic, stretch=2)
        body.addLayout(right, stretch=2)

    # =================================================================
    # 서버 제어
    # =================================================================
    @_guarded
    def _on_start(self) -> None:
        try:
            self._server.start()
        except OSError as exc:
            tcp = self._store.snapshot().tcp
            QMessageBox.critical(
                self, "서버 시작 실패",
                f"{tcp.host}:{self._server.port()} 에서 대기할 수 없습니다.\n\n{exc}\n\n"
                "포트가 이미 사용 중이거나 호스트 주소가 이 PC의 것이 아닐 수 있습니다 ('12. 설정' > TCP 통신).")
            self._log(f"서버 시작 실패: {exc}")
        self._update_buttons()

    @_guarded
    def _on_stop(self) -> None:
        self._server.stop()
        self._worker.submit("release_camera")    # 켜둔 카메라 해제
        self._update_buttons()

    def shutdown(self) -> None:
        """앱 종료 시 호출: 서버 정지 + 워커/카메라 정리."""
        try:
            self._server.stop()
        finally:
            self._worker.stop()

    def _update_buttons(self) -> None:
        running = self._server.is_running
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_manual.setEnabled(not self._manual_running)

    # =================================================================
    # 서버 이벤트 (GUI 스레드에서 실행)
    # =================================================================
    def _on_status(self, status: str) -> None:
        if status.startswith("listening"):
            self.status_label.setText(f"● 대기 중  {status.split(' ', 1)[1]}   (클라이언트 0)")
            self.status_label.setStyleSheet("color: #2a8a2a; font-weight: 600;")
        else:
            self.status_label.setText("● 정지")
            self.status_label.setStyleSheet("color: #888;")
        self._update_buttons()

    def _on_clients(self, n: int) -> None:
        text = self.status_label.text()
        if "대기 중" in text:
            base = text.split("(클라이언트")[0].rstrip()
            self.status_label.setText(f"{base}   (클라이언트 {n})")

    def _append_traffic(self, line: str) -> None:
        self.traffic.appendPlainText(f"{datetime.now().strftime('%H:%M:%S')}  {line}")

    def _on_request(self, peer: str, text: str) -> None:
        self._append_traffic(f"◀ {peer}  {text}")
        self._log(f"요청 수신 {peer}: {text}")

    def _on_response(self, peer: str, text: str) -> None:
        self._append_traffic(f"▶ {peer}  {text}")
        self._log(f"응답 송신 {peer}: {text}")

    # =================================================================
    # 수동 실행
    # =================================================================
    @_guarded
    def _on_manual(self) -> None:
        if self._manual_running:
            return
        self._manual_running = True
        self._update_buttons()
        self.run_label.setText(f"수동 실행 중... ({self.RUN_DESC})")
        # 서버가 꺼져 있으면 1회용이니 끝나고 카메라를 닫는다. 켜져 있으면 설정을 따른다.
        keep = None if self._server.is_running else False
        fut = self._worker.submit("pick", keep_camera_open=keep)

        def _done(f) -> None:   # 워커 스레드에서 호출됨 -> 시그널로 GUI 스레드에 넘긴다
            exc = f.exception()
            if exc is not None:
                r = PickRunResult(error_code="INTERNAL_ERROR", error_message=f"{type(exc).__name__}: {exc}")
            else:
                r = f.result()
            self._manual_finished.emit(r)

        fut.add_done_callback(_done)
        self._log("수동 1회 실행 시작")

    def _on_manual_finished(self, result: PickRunResult) -> None:
        self._manual_running = False
        self._update_buttons()
        self._append_traffic(f"(수동 실행) {result.summary()}")
        self._show_result(result)

    # =================================================================
    # 결과 표시
    # =================================================================
    def _clear_cards(self) -> None:
        while self.result_layout.count() > 1:
            item = self.result_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)   # deleteLater만 쓰면 이벤트 루프 전까지 옛 카드가 남아 겹쳐 보인다
                w.deleteLater()

    def _add_card(self, title: str, lines: list[tuple[str, str]], border: str = "#ddd") -> None:
        card = QFrame()
        card.setObjectName("resultCard")
        card.setStyleSheet(f"QFrame#resultCard {{ border: 1px solid {border}; border-radius: 6px; }}")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(8, 6, 8, 6)
        head = QLabel(title)
        head.setStyleSheet("font-weight: 600;")
        lay.addWidget(head)
        for text, color in lines:
            lab = QLabel(text)
            lab.setWordWrap(True)
            if color:
                lab.setStyleSheet(f"color: {color};")
            lay.addWidget(lab)
        self.result_layout.insertWidget(self.result_layout.count() - 1, card)

    def _show_result(self, res: PickRunResult) -> None:
        if res.image_path:
            self.image_viewer.load_image(res.image_path)
            self.image_viewer.set_detections(res.detections)
        ok_color = "#2a8a2a"
        self.run_label.setText(
            f"{res.started_at.replace('T', ' ')}   {res.summary()}\n"
            + "   ".join(f"{k} {v:.0f}ms" for k, v in res.timings_ms.items()))
        self.run_label.setStyleSheet(f"color: {ok_color};" if res.ok else "color: #c0392b;")

        self._clear_cards()
        max_picks = max(1, self._store.snapshot().tcp.max_picks)
        for rank, c in enumerate(res.candidates):
            sent = "  [응답 포함]" if rank < max_picks else ""
            self._add_card(
                f"#{rank + 1}  obj{c.instance_id}  {c.label} (conf {c.confidence:.2f}){sent}",
                [(f"pick  X{c.xyz_mm[0]:+.1f}  Y{c.xyz_mm[1]:+.1f}  Z{c.xyz_mm[2]:+.1f} mm", ""),
                 (f"R{c.rpy_deg[0]:+.1f}  P{c.rpy_deg[1]:+.1f}  Y{c.rpy_deg[2]:+.1f} deg", ""),
                 (f"fitness {c.fitness:.3f}   rmse {c.rmse_mm:.2f}mm" + ("   (뒤집힘 보정)" if c.was_flipped else ""), ok_color)],
                border="#9cc99c" if rank < max_picks else "#ddd")
        ok_ids = {c.instance_id for c in res.candidates}
        for r in res.icp_results:
            if r.instance_id in ok_ids:
                continue
            conf = res.detections[r.instance_id].confidence if r.instance_id < len(res.detections) else 0.0
            fit = f"   fitness {r.fitness:.3f}" if r.fitness is not None else ""
            self._add_card(f"obj{r.instance_id}  (conf {conf:.2f})  ✗ 실패",
                           [(f"{r.error or '실패'}{fit}", "#c0392b")], border="#e0b4ae")
        for orig_idx, why, conf in res.suspect_excluded:
            self._add_card(f"obj{orig_idx}  (conf {conf:.2f})  의심 마스크로 제외",
                           [(why, "#a0702a")], border="#e3cfa0")
        if not res.ok and not res.icp_results:
            self._add_card(f"[{res.error_code}]", [(res.error_message, "#c0392b")], border="#e0b4ae")

    # =================================================================
    # 설정 요약
    # =================================================================
    def _refresh_config_summary(self) -> None:
        s = self._store.snapshot()
        ckpt = s.detection.checkpoint_path.replace("\\", "/").split("/")[-1] or "(미지정)"
        cad = s.cad.cad_path.replace("\\", "/").split("/")[-1] or "(미지정)"
        self.config_label.setText(
            f"카메라 {s.camera.camera_type} ×{s.camera.num_frames}프레임  |  모델 {ckpt}  |  CAD {cad}  |  "
            f"형상 보정 {'ON' if s.refine.enabled else 'OFF'}  |  fitness ≥ {s.icp.fitness_threshold:.2f}  |  "
            f"요청 '{s.tcp.trigger_cmd}' -> 응답 최대 {s.tcp.max_picks}개  |  값 변경은 '12. 설정' 탭 (저장 후 다음 요청부터 적용)")
