"""로봇 요청 대기 TCP 서버 + 촬영/정합 워커.

흐름 (로봇이 요청할 때마다 한 번):

    로봇 --"PICK\\n"--> [PickServer 소켓 스레드] --작업 제출--> [PickWorker 스레드]
                                                         카메라 촬영 -> PickPipeline.run()
    로봇 <--"OK,1,x,y,z,roll,pitch,yaw\\n"-- 응답 문자열 생성 <-- PickRunResult

프로토콜 (줄 단위 텍스트, 명령/응답 형식은 설정 탭 > TCP 에서 바꾼다. 기본값):

    요청   PICK           촬영 -> 검출 -> 정합 -> 픽 좌표 응답
           PING           살아있는지 확인 (카메라/모델 안 건드림)
    응답   OK,<n>,x,y,z[,roll,pitch,yaw] x n          성공 (n = 응답에 담긴 후보 수, 정렬된 순서)
           NG,<코드>                                  실패
           PONG                                       PING 응답
    NG 코드: NO_DETECTION, NO_VALID_POSE, CAMERA_ERROR, CONFIG_ERROR, DETECTION_ERROR,
             INTERNAL_ERROR, BUSY (이전 요청 처리 중), UNKNOWN_CMD, TIMEOUT
    단위: 카메라 좌표계 mm / deg (핸드-아이 변환은 로봇 쪽에서)

요청 끝 문자는 \\n, \\r\\n, \\r 모두 받는다. 끝 문자 없이 "PICK"만 보내는 클라이언트도
알려진 명령과 정확히 일치하면 바로 처리한다.

스레드 모델: 클라이언트마다 처리 스레드가 하나씩 있고, 카메라/모델/CAD는 PickWorker 스레드
하나가 독점한다 (Arena/Orbbec SDK와 Detector를 한 스레드에서만 쓰기 위함). 요청이 겹치면
먼저 온 것을 처리하는 동안 나머지는 BUSY로 즉시 거절한다.
"""
from __future__ import annotations

import os
import queue
import re
import socket
import threading
import time
import traceback
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Callable

import yaml
from PyQt6.QtCore import QObject, pyqtSignal

from app.core.app_settings import CAMERA_CONFIG_PATHS, CameraSettings, SettingsStore, TcpSettings
from app.core.pick_pipeline import PickCandidate, PickPipeline, PickRunResult, PipelineError

REQUEST_TIMEOUT_S = 120.0     # 한 요청의 파이프라인 최대 대기 시간
MAX_PENDING_BYTES = 4096      # 끝 문자 없이 이만큼 쌓이면 쓰레기로 보고 버린다


# =============================================================================
# 프로토콜 (순수 함수 - 소켓 없이 테스트 가능)
# =============================================================================
def _fmt(v: float, decimals: int) -> str:
    s = f"{v:.{decimals}f}"
    if s.startswith("-") and float(s) == 0.0:   # "-0.00" 방지
        s = s[1:]
    return s


def format_ok(cands: list[PickCandidate], tcp: TcpSettings) -> str:
    picks = cands[:max(1, tcp.max_picks)]
    fields = [tcp.ok_prefix, str(len(picks))]
    for c in picks:
        fields += [_fmt(v, tcp.decimals) for v in c.xyz_mm]
        if tcp.include_rpy:
            fields += [_fmt(v, tcp.decimals) for v in c.rpy_deg]
    return tcp.field_sep.join(fields)


def format_ng(code: str, tcp: TcpSettings) -> str:
    return tcp.field_sep.join([tcp.ng_prefix, code])


def classify_request(text: str, tcp: TcpSettings) -> str:
    """'trigger' | 'ping' | 'unknown'. 첫 토큰(공백/쉼표/세미콜론 구분)만 보고 대소문자는 무시한다."""
    tokens = re.split(r"[\s,;]+", text.strip())
    first = tokens[0].upper() if tokens and tokens[0] else ""
    if first and first == tcp.trigger_cmd.strip().upper():
        return "trigger"
    if first and first == tcp.ping_cmd.strip().upper():
        return "ping"
    return "unknown"


def split_messages(buf: bytes, tcp: TcpSettings) -> tuple[list[str], bytes]:
    """수신 버퍼에서 완성된 요청을 꺼낸다. (메시지 목록, 남은 버퍼)를 반환한다."""
    messages: list[str] = []
    # 줄바꿈(\r, \n) 기준으로 자른다. 마지막 조각은 아직 덜 온 것일 수 있다.
    parts = re.split(rb"[\r\n]+", buf)
    rest = parts.pop() if parts else b""
    for p in parts:
        text = p.decode("utf-8", errors="replace").strip()
        if text:
            messages.append(text)
    # 끝 문자를 안 붙이는 클라이언트: 남은 조각이 알려진 명령과 정확히 같으면 완성으로 본다.
    pending = rest.decode("utf-8", errors="replace").strip()
    if pending and classify_request(pending, tcp) != "unknown" and \
            re.fullmatch(r"[A-Za-z0-9_]+", pending):
        messages.append(pending)
        rest = b""
    return messages, rest


# =============================================================================
# 워커: 카메라 + 파이프라인 소유 스레드
# =============================================================================
def build_camera_cfg(cam: CameraSettings) -> dict:
    """camera_config_*.yaml의 camera 섹션을 읽고 평균화 값만 설정으로 덮어쓴다."""
    path = CAMERA_CONFIG_PATHS.get(cam.camera_type)
    if path is None:
        raise PipelineError("CONFIG_ERROR", f"지원하지 않는 카메라 타입: {cam.camera_type}")
    if not path.is_file():
        raise PipelineError("CONFIG_ERROR", f"카메라 설정 파일이 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = dict(yaml.safe_load(f)["camera"])
    avg = dict(cfg.get("averaging") or {})
    avg["num_frames"] = int(cam.num_frames)
    avg["min_valid_ratio"] = float(cam.min_valid_ratio)
    cfg["averaging"] = avg
    return cfg


def _default_camera_factory(cfg: dict):
    from src.camera import create_camera   # 카메라 SDK는 실제로 열 때만 import
    return create_camera(cfg)


class PickWorker:
    """카메라와 PickPipeline을 한 스레드에서 독점하고, 작업을 순서대로 처리한다."""

    def __init__(self, store: SettingsStore,
                 camera_factory: Callable[[dict], object] | None = None,
                 log: Callable[[str], None] | None = None):
        self._store = store
        self._factory = camera_factory or _default_camera_factory
        self._log = log or (lambda _m: None)
        self._jobs: "queue.Queue[tuple | None]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._pending = 0
        self._pending_lock = threading.Lock()
        # 아래는 워커 스레드에서만 건드린다
        self._cam = None
        self._cam_key: tuple | None = None
        self._pipeline = PickPipeline(log=self._log)

    # ---- 외부 API (아무 스레드에서나) ----
    def is_busy(self) -> bool:
        """촬영/정합 작업(pick)이 진행 중이거나 대기 중인가. 선로딩(warmup)은 세지 않는다 -
        선로딩 중에 온 요청은 BUSY로 거절하지 않고 선로딩 뒤에 이어서 처리한다."""
        with self._pending_lock:
            return self._pending > 0

    def submit(self, kind: str, keep_camera_open: bool | None = None) -> "Future":
        """kind: 'pick' (촬영+파이프라인 -> PickRunResult) | 'warmup' | 'release_camera'."""
        self._ensure_thread()
        fut: Future = Future()
        if kind == "pick":
            with self._pending_lock:
                self._pending += 1
        self._jobs.put((kind, keep_camera_open, fut))
        return fut

    def stop(self, timeout: float = 10.0) -> None:
        t = self._thread
        if t is None:
            return
        self._jobs.put(None)
        t.join(timeout)
        self._thread = None

    # ---- 워커 스레드 ----
    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, name="PickWorker", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        try:
            while True:
                job = self._jobs.get()
                if job is None:
                    break
                kind, keep_open, fut = job
                try:
                    if kind == "pick":
                        fut.set_result(self._do_pick(keep_open))
                    elif kind == "warmup":
                        fut.set_result(self._do_warmup())
                    elif kind == "release_camera":
                        self._close_camera()
                        fut.set_result(None)
                    else:
                        fut.set_exception(ValueError(f"알 수 없는 작업: {kind}"))
                except BaseException as exc:  # noqa: BLE001 - 워커가 죽으면 서버 전체가 멈춘다
                    self._log(f"워커 작업 예외({kind}): {exc}\n{traceback.format_exc()}")
                    if not fut.done():
                        fut.set_exception(exc)
                finally:
                    if kind == "pick":
                        with self._pending_lock:
                            self._pending -= 1
        finally:
            self._close_camera()

    def _do_warmup(self) -> str | None:
        """모델/CAD 선로딩. 실패 사유 문자열을 반환 (성공이면 None)."""
        try:
            self._pipeline.warmup(self._store.snapshot())
            return None
        except PipelineError as exc:
            return f"[{exc.code}] {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"[INTERNAL_ERROR] {type(exc).__name__}: {exc}"

    def _do_pick(self, keep_open: bool | None) -> PickRunResult:
        snap = self._store.snapshot()
        keep = snap.tcp.keep_camera_open if keep_open is None else keep_open
        t0 = time.perf_counter()
        try:
            frame = self._capture(snap.camera)
        except PipelineError as exc:
            return PickRunResult(error_code=exc.code, error_message=str(exc))
        except Exception as exc:  # noqa: BLE001
            self._close_camera()   # 끊긴 카메라일 수 있으니 다음 요청에서 다시 연다
            return PickRunResult(error_code="CAMERA_ERROR", error_message=f"{type(exc).__name__}: {exc}")
        capture_ms = (time.perf_counter() - t0) * 1000.0
        try:
            result = self._pipeline.run(frame, snap)
        finally:
            if not keep:
                self._close_camera()
        result.timings_ms["capture"] = capture_ms
        return result

    def _capture(self, cam_settings: CameraSettings):
        key = (cam_settings.camera_type, cam_settings.num_frames, cam_settings.min_valid_ratio)
        if self._cam is not None and self._cam_key != key:
            self._close_camera()    # 카메라 설정이 바뀌었으면 새 값으로 다시 연다
        if self._cam is None:
            cfg = build_camera_cfg(cam_settings)
            cam = self._factory(cfg)
            try:
                cam.open()
            except Exception as exc:  # noqa: BLE001
                raise PipelineError("CAMERA_ERROR", f"카메라 연결 실패: {type(exc).__name__}: {exc}") from exc
            self._cam, self._cam_key = cam, key
            self._log(f"카메라 연결: {cam_settings.camera_type} (평균화 {cam_settings.num_frames}프레임)")
        return self._cam.capture()

    def _close_camera(self) -> None:
        cam, self._cam, self._cam_key = self._cam, None, None
        if cam is not None:
            try:
                cam.close()
                self._log("카메라 해제")
            except Exception as exc:  # noqa: BLE001
                self._log(f"카메라 해제 중 오류(무시): {exc}")


# =============================================================================
# 서버
# =============================================================================
class PickServer(QObject):
    status_changed = pyqtSignal(str)            # "stopped" | "listening host:port"
    client_count_changed = pyqtSignal(int)
    request_received = pyqtSignal(str, str)     # peer, 요청 텍스트
    response_sent = pyqtSignal(str, str)        # peer, 응답 텍스트
    run_finished = pyqtSignal(object)           # PickRunResult (TCP 요청으로 돈 것)
    log_message = pyqtSignal(str)

    def __init__(self, store: SettingsStore, worker: PickWorker, parent: QObject | None = None):
        super().__init__(parent)
        self._store = store
        self._worker = worker
        self._sock: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._clients: dict[socket.socket, str] = {}
        self._clients_lock = threading.Lock()
        self._handlers: list[threading.Thread] = []

    @property
    def is_running(self) -> bool:
        return self._sock is not None

    def start(self) -> None:
        """리슨 시작. 포트가 이미 쓰이는 중이면 OSError."""
        if self._sock is not None:
            return
        tcp = self._store.snapshot().tcp
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if os.name == "nt":   # Windows의 REUSEADDR은 이중 바인드를 허용해서 쓰지 않는다
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((tcp.host, int(tcp.port)))
            sock.listen(5)
            sock.settimeout(0.5)
        except OSError:
            sock.close()
            raise
        self._sock = sock
        self._stop_evt.clear()
        self._accept_thread = threading.Thread(target=self._accept_loop, name="PickAccept", daemon=True)
        self._accept_thread.start()
        self.status_changed.emit(f"listening {tcp.host}:{tcp.port}")
        self.log_message.emit(f"TCP 서버 시작: {tcp.host}:{tcp.port}  "
                              f"(요청 '{tcp.trigger_cmd}' / '{tcp.ping_cmd}')")

        if tcp.warmup_on_start:
            fut = self._worker.submit("warmup")
            fut.add_done_callback(self._on_warmup_done)

    def stop(self) -> None:
        if self._sock is None:
            return
        self._stop_evt.set()
        try:
            self._sock.close()
        except OSError:
            pass
        self._sock = None
        with self._clients_lock:
            conns = list(self._clients)
        for c in conns:
            try:
                c.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                c.close()
            except OSError:
                pass
        if self._accept_thread is not None:
            self._accept_thread.join(2.0)
        for t in list(self._handlers):
            t.join(1.0)
        self._handlers.clear()
        self._accept_thread = None
        self.status_changed.emit("stopped")
        self.client_count_changed.emit(0)
        self.log_message.emit("TCP 서버 정지")

    # ---- 내부 ----
    def _on_warmup_done(self, fut: "Future") -> None:
        try:
            err = fut.result()
        except BaseException as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        if err:
            self.log_message.emit(f"선로딩 실패 (요청이 오면 같은 오류로 NG 응답): {err}")
        else:
            self.log_message.emit("선로딩 완료: 모델/CAD 준비됨")

    def _accept_loop(self) -> None:
        while not self._stop_evt.is_set():
            sock = self._sock
            if sock is None:
                break
            try:
                conn, addr = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            peer = f"{addr[0]}:{addr[1]}"
            t = threading.Thread(target=self._serve_client, args=(conn, peer), name=f"PickClient-{peer}",
                                 daemon=True)
            self._handlers = [h for h in self._handlers if h.is_alive()]
            self._handlers.append(t)
            t.start()

    def _serve_client(self, conn: socket.socket, peer: str) -> None:
        with self._clients_lock:
            self._clients[conn] = peer
            n = len(self._clients)
        self.client_count_changed.emit(n)
        self.log_message.emit(f"클라이언트 연결: {peer}")
        try:
            conn.settimeout(0.5)
            try:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
            buf = b""
            while not self._stop_evt.is_set():
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                tcp = self._store.snapshot().tcp
                messages, buf = split_messages(buf, tcp)
                if len(buf) > MAX_PENDING_BYTES:
                    self.log_message.emit(f"{peer}: 끝 문자 없는 데이터가 너무 길어 버립니다 ({len(buf)}B)")
                    buf = b""
                for text in messages:
                    reply = self._handle(peer, text)
                    try:
                        conn.sendall(reply.encode("utf-8") + self._store.snapshot().tcp.terminator_bytes())
                    except OSError:
                        return
                    self.response_sent.emit(peer, reply)
        finally:
            with self._clients_lock:
                self._clients.pop(conn, None)
                n = len(self._clients)
            try:
                conn.close()
            except OSError:
                pass
            if not self._stop_evt.is_set():
                self.client_count_changed.emit(n)
                self.log_message.emit(f"클라이언트 종료: {peer}")

    def _handle(self, peer: str, text: str) -> str:
        tcp = self._store.snapshot().tcp
        self.request_received.emit(peer, text)
        kind = classify_request(text, tcp)
        if kind == "ping":
            return "PONG"
        if kind != "trigger":
            return format_ng("UNKNOWN_CMD", tcp)
        if self._worker.is_busy():
            return format_ng("BUSY", tcp)
        fut = self._worker.submit("pick")
        try:
            result: PickRunResult = fut.result(timeout=REQUEST_TIMEOUT_S)
        except FutureTimeout:
            return format_ng("TIMEOUT", tcp)
        except Exception as exc:  # noqa: BLE001
            self.log_message.emit(f"요청 처리 예외: {exc}")
            return format_ng("INTERNAL_ERROR", tcp)
        self.run_finished.emit(result)
        tcp = self._store.snapshot().tcp   # 처리 중에 응답 형식이 저장으로 바뀌었을 수 있어 다시 읽는다
        if result.ok:
            return format_ok(result.candidates, tcp)
        return format_ng(result.error_code or "INTERNAL_ERROR", tcp)
