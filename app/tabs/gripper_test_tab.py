"""탭 13: 그리퍼 동작 테스트.

로봇 그리퍼가 내가 원하는 좌표로 제대로 이동하는지 확인하기 위한 탭이다. 검출/정합 없이 사람이 위치를 고른다.

    1. [촬영]            카메라로 한 프레임 촬영 -> 이미지 표시 (5번 탭과 같은 크기)
    2. 이미지 클릭        클릭한 픽셀 주변의 점군에서 XYZ(mm, 카메라 좌표)를 읽어 표시
    3. TCP 서버           로봇이 좌표를 요청("PICK")하면 마지막으로 클릭한 좌표를 응답

    로봇  --"PICK"-->  [이 탭]  --"OK,1,x,y,z,roll,pitch,yaw"-->  로봇     (roll/pitch/yaw는 0, 5번 탭과 같은 응답 형식)

좌표는 픽셀 위치를 핀홀 모델로 환산한 값이 아니라 점군을 직접 읽은 값이다 (렌즈 왜곡 보정이 끝난 좌표).
호스트/포트/명령어/응답 형식은 '12. 설정' 탭 > TCP 통신에서 바꾼다 (이 탭의 포트는 '13번 탭: 그리퍼 테스트').
새로 촬영하면 이전 클릭 좌표는 지운다 - 다른 프레임의 좌표를 로봇에 보내는 실수를 막기 위해서다.
"""
from __future__ import annotations

import functools
import math
import traceback
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
import cv2
import numpy as np
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QFileDialog, QGridLayout, QHeaderView, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from app.core.app_settings import SettingsStore, get_settings
from app.core.detect_test_common import SETTINGS_HINT, busy_cursor, capture_frame, new_capture_name, save_intensity_png
from app.core.gripper_test import (
    DEFAULT_WINDOW, ClickedPoint, ClickedPointStore, GripperPointWorker, PlaneReport, analyze_plane, point_from_cloud,
)
from app.core import transform_matrix as tm
from app.core.pick_server import PickServer, format_ng, format_ok
from app.core.point_pair_calib import (
    CALIB_PATH, DEFAULT_RMS_LIMIT_MM, CalibError, CalibPair, OutputConfig, RigidFit, load_calibration, save_calibration,
    save_manual_transform, solve,
)
from app.widgets.elided_label import ElidedLabel
from app.widgets.image_viewer import ImageViewer


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


class GripperTestTab(QWidget):
    log_message = pyqtSignal(str)

    LOG_PREFIX = "그리퍼 테스트 탭"

    def __init__(self, store: SettingsStore | None = None, parent=None):
        super().__init__(parent)
        self._store = store or get_settings()
        self._frame = None                       # 마지막 촬영 FrameData
        self._point_store = ClickedPointStore()
        self._output = OutputConfig()            # PICK 응답 좌표계 (카메라 / 로봇 변환 적용)
        self._worker = GripperPointWorker(self._point_store, self._output)
        self._fit: RigidFit | None = None        # 마지막으로 계산한 변환
        self._calib_stale = False                # 표를 고친 뒤 아직 다시 계산하지 않았다
        self._fit_manual = False                 # 현재 변환이 '행렬 직접 입력'으로 적용한 값이다 (점 짝으로 푼 값이 아님)
        self._server = PickServer(self._store, self._worker, parent=self, port_field="port_gripper")
        self._build_ui()

        self._server.status_changed.connect(self._on_status)
        self._server.client_count_changed.connect(self._on_clients)
        self._server.request_received.connect(self._on_request)
        self._server.response_sent.connect(self._on_response)
        self._server.log_message.connect(self._log)
        self._store.changed.connect(self._refresh_config_summary)
        self.image_viewer.point_clicked.connect(self._on_image_clicked)
        self.image_viewer.roi_changed.connect(self._on_plane_roi)
        self.image_viewer.zoom_changed.connect(self._on_zoom_changed)
        self.image_viewer.pixel_hovered.connect(self._on_pixel_hovered)
        self.btn_zoom_in.clicked.connect(lambda _=False: self.image_viewer.zoom_by(1.5))
        self.btn_zoom_out.clicked.connect(lambda _=False: self.image_viewer.zoom_by(1 / 1.5))
        self.btn_zoom_fit.clicked.connect(lambda _=False: self.image_viewer.reset_view())
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
        self.btn_capture = QPushButton("촬영")
        self.btn_capture.setStyleSheet("font-weight: 600; padding: 4px 18px;")
        self.btn_capture.setToolTip("'12. 설정' 탭의 카메라 값으로 한 프레임 촬영해 이미지를 표시합니다")
        self.btn_capture.clicked.connect(lambda _=False: self._on_capture())
        top.addWidget(self.btn_capture)
        top.addSpacing(16)
        self.btn_start = QPushButton("서버 시작")
        self.btn_start.clicked.connect(lambda _=False: self._on_start())
        self.btn_stop = QPushButton("서버 정지")
        self.btn_stop.clicked.connect(lambda _=False: self._on_stop())
        top.addWidget(self.btn_start)
        top.addWidget(self.btn_stop)
        top.addSpacing(16)
        self.status_label = QLabel("● 정지")
        top.addWidget(self.status_label)
        top.addStretch(1)
        root.addLayout(top)

        self.config_label = QLabel("")
        self.config_label.setStyleSheet("color: #666; font-size: 11px;")
        self.config_label.setWordWrap(True)
        root.addWidget(self.config_label)

        body = QHBoxLayout()
        root.addLayout(body, stretch=1)

        # 왼쪽: 촬영 이미지 (클릭으로 위치 선택) - 5번 탭과 같은 최소 크기
        left = QVBoxLayout()
        zrow = QHBoxLayout()
        zrow.addWidget(QLabel("촬영 이미지 (클릭 = 점 선택)"))
        zrow.addStretch(1)
        self.zoom_label = QLabel("")
        self.zoom_label.setStyleSheet("color: #555;")
        self.zoom_label.setFixedWidth(64)      # 글자 길이가 바뀌어도 왼쪽 열 폭(=이미지 크기)이 흔들리지 않게
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        zrow.addWidget(self.zoom_label)
        self.btn_zoom_out = QPushButton("－")
        self.btn_zoom_in = QPushButton("＋")
        self.btn_zoom_fit = QPushButton("맞춤")
        for b, tip in ((self.btn_zoom_out, "축소"), (self.btn_zoom_in, "확대"), (self.btn_zoom_fit, "화면에 맞춤")):
            b.setFixedWidth(46)
            b.setToolTip(tip)
            zrow.addWidget(b)
        left.addLayout(zrow)
        self.image_viewer = ImageViewer()
        self.image_viewer.set_zoom_enabled(True)
        self.image_viewer.setMinimumSize(480, 360)
        self.image_viewer.setText("[촬영]을 눌러 이미지를 가져오세요")
        self.image_viewer.set_click_pick(True)
        left.addWidget(self.image_viewer, stretch=1)
        self.hover_label = QLabel("마우스 휠: 확대/축소 (커서 위치 기준)  ·  가운데/오른쪽 버튼 드래그: 이동  ·  왼쪽 클릭: 점 선택")
        self.hover_label.setStyleSheet("color: #555; font-size: 11px; font-family: Consolas, monospace;")
        self.hover_label.setWordWrap(True)
        left.addWidget(self.hover_label)
        self.frame_label = QLabel("아직 촬영하지 않았습니다.")
        self.frame_label.setWordWrap(True)
        left.addWidget(self.frame_label)
        body.addLayout(left, stretch=1)

        # 오른쪽: 클릭 좌표 + 요청 로그
        right = QVBoxLayout()
        page_coord = QWidget()
        pc = QVBoxLayout(page_coord)
        pc.setContentsMargins(0, 6, 0, 0)
        self.coord_box = QGroupBox("클릭한 위치의 점군 좌표 (카메라 좌표계, mm)")
        box = self.coord_box
        bl = QVBoxLayout(box)
        self.chk_robot_click = QCheckBox("로봇 좌표로 변환")
        self.chk_robot_click.setToolTip(
            "켜면 클릭한 점을 '좌표 변환' 탭에서 계산한 변환식으로 로봇 좌표로 바꿔 표시하고, 로봇 PICK 응답도 로봇 좌표로 보냅니다\n"
            "('좌표 변환' 탭의 PICK 응답 좌표계와 같은 설정). 샘플 점을 찍어 로봇을 그 좌표로 보내 변환이 맞는지 확인하는 용도입니다.\n"
            "변환을 아직 계산하지 않았거나 잔차 RMS가 한계를 넘으면 켤 수 없습니다. 앱을 다시 켜면 꺼진 상태로 시작합니다.")
        self.chk_robot_click.toggled.connect(self._on_robot_click_toggled)
        bl.addWidget(self.chk_robot_click)
        self.coord_label = QLabel("-")
        self.coord_label.setStyleSheet("font-family: Consolas, monospace; font-size: 18px; font-weight: 600;")
        self.coord_label.setWordWrap(True)
        bl.addWidget(self.coord_label)
        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet("color: #555;")
        self.detail_label.setWordWrap(True)
        bl.addWidget(self.detail_label)
        form = QFormLayout()
        self.spin_window = QSpinBox()
        self.spin_window.setRange(1, 31)
        self.spin_window.setSingleStep(2)
        self.spin_window.setValue(DEFAULT_WINDOW)
        self.spin_window.setSuffix(" px")
        self.spin_window.setToolTip("클릭 픽셀 주변 이 크기의 정사각 창에서 유효한 점들의 중앙값을 씁니다. 1이면 그 픽셀 한 점만.\n"
                                    "깊이가 비어 있는 픽셀(반사/그림자)을 클릭했을 때 주변 점으로 메워 줍니다.")
        self.spin_window.valueChanged.connect(lambda _v: self._recompute_last_click())
        form.addRow("읽을 창 크기 (홀수)", self.spin_window)
        bl.addLayout(form)
        row = QHBoxLayout()
        self.btn_copy = QPushButton("좌표 복사")
        self.btn_copy.clicked.connect(lambda _=False: self._on_copy())
        self.btn_clear = QPushButton("선택 해제")
        self.btn_clear.setToolTip("선택을 지웁니다. 해제 후 로봇이 요청하면 NG,NO_POINT 로 응답합니다.")
        self.btn_clear.clicked.connect(lambda _=False: self._clear_point())
        row.addWidget(self.btn_copy)
        row.addWidget(self.btn_clear)
        row.addStretch(1)
        bl.addLayout(row)
        self.sent_label = QLabel("로봇에 보낼 좌표: 없음")
        self.sent_label.setWordWrap(True)
        bl.addWidget(self.sent_label)
        pc.addWidget(box)

        pbox = QGroupBox("평면 검증 (평평한 판이 기울거나 휘어 보이는지)")
        pl = QVBoxLayout(pbox)
        self.btn_plane = QPushButton("평면 검증 모드 (이미지에서 판 영역을 드래그)")
        self.btn_plane.setCheckable(True)
        self.btn_plane.setToolTip("켜면 클릭 대신 드래그로 영역을 지정합니다. 영역 안 점군에 평면을 맞춰 기울기와 휘어짐을 보여줍니다.\n"
                                  "평평한 판(정반/유리판 등)이 영역을 꽉 채우도록 지정하세요. 끄면 다시 클릭으로 좌표를 고릅니다.")
        self.btn_plane.toggled.connect(self._on_plane_mode)
        pl.addWidget(self.btn_plane)
        self.plane_label = QLabel("")
        self.plane_label.setWordWrap(True)
        self.plane_label.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        pl.addWidget(self.plane_label)
        self.plane_map = QLabel("")
        self.plane_map.setFixedHeight(0)
        pl.addWidget(self.plane_map)
        pc.addWidget(pbox)
        pc.addStretch(1)
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(page_coord, "클릭 좌표 / 평면 검증")
        self.right_tabs.addTab(self._build_calib_page(), "좌표 변환 (카메라→로봇)")
        self.right_tabs.addTab(self._build_matrix_page(), "행렬 직접 입력")
        right.addWidget(self.right_tabs, stretch=3)

        right.addWidget(QLabel("요청 / 응답 로그   ◀ 수신   ▶ 송신"))
        self.traffic = QPlainTextEdit()
        self.traffic.setReadOnly(True)
        self.traffic.setMaximumBlockCount(500)
        self.traffic.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.traffic.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        right.addWidget(self.traffic, stretch=1)
        self._load_calib_file()
        body.addLayout(right, stretch=1)

    # =================================================================
    # 촬영 / 클릭
    # =================================================================
    @_guarded
    def _on_capture(self) -> None:
        self.btn_capture.setEnabled(False)
        try:
            with busy_cursor():
                frame, cam = capture_frame()
        except Exception as exc:  # noqa: BLE001 - 카메라 연결 실패 등은 안내 문구와 함께 보여준다
            self._log(f"촬영 실패: {type(exc).__name__}: {exc}")
            QMessageBox.warning(
                self, "촬영 실패",
                f"카메라 촬영에 실패했습니다.\n\n{type(exc).__name__}: {exc}\n\n카메라 종류/평균화 프레임은 {SETTINGS_HINT}")
            return
        finally:
            self.btn_capture.setEnabled(True)
        if frame.points_organized is None:
            QMessageBox.warning(self, "점군 없음", "이 프레임에는 정렬된 점군(points_organized)이 없어 좌표를 읽을 수 없습니다.")
            return
        self._clear_point(log=False)    # 다른 프레임의 좌표가 로봇에 나가지 않게
        self.image_viewer.set_roi(None)
        self.plane_label.setText("")
        self.plane_map.setFixedHeight(0)
        self.plane_map.clear()
        self._frame = frame
        name = new_capture_name()
        self.image_viewer.load_image(save_intensity_png(frame.intensity, name))
        h, w = frame.intensity.shape[:2]
        n_valid = int(frame.valid_mask.sum()) if getattr(frame, "valid_mask", None) is not None else 0
        self.frame_label.setText(
            f"{datetime.now().strftime('%H:%M:%S')} 촬영  {w}x{h}  |  유효 깊이 {n_valid / max(h * w, 1) * 100:.0f}%  |  "
            f"카메라 {cam.camera_type} ×{cam.num_frames}프레임")
        self._log(f"촬영: {w}x{h}, 유효 깊이 {n_valid / max(h * w, 1) * 100:.0f}%")

    @_guarded
    def _on_image_clicked(self, x: float, y: float) -> None:
        if self._frame is None:
            return
        u, v = math.floor(x), math.floor(y)      # 커서가 가리키는 픽셀 (반올림하면 최대 반 픽셀 어긋난다)
        self._last_click = (u, v)
        self._apply_click(u, v)

    def _on_zoom_changed(self, scale: float) -> None:
        self.zoom_label.setText(f"{scale * 100:.0f}%")
        self.zoom_label.setToolTip(f"화면 배율 {scale * 100:.0f}% (이미지 픽셀 1개 = 화면 {scale:.2f}px)")

    def _on_pixel_hovered(self, u: int, v: int) -> None:
        """마우스가 가리키는 픽셀의 점군 값을 보여준다 - 깊이가 비었거나 가장자리인 곳을 피해 고를 수 있게."""
        if u < 0 or self._frame is None:
            return
        pt = point_from_cloud(self._frame.points_organized, getattr(self._frame, "valid_mask", None), u, v, 1)
        if pt is None:
            self.hover_label.setText(f"커서 ({u}, {v})  깊이 없음 (이 픽셀은 점군이 비어 있음)")
        else:
            X, Y, Z = pt.xyz_mm
            self.hover_label.setText(f"커서 ({u}, {v})  X{X:+8.2f}  Y{Y:+8.2f}  Z{Z:+8.2f} mm  (이 픽셀 한 점)")

    def _recompute_last_click(self) -> None:
        click = getattr(self, "_last_click", None)
        if self._frame is not None and click is not None and self._point_store.get() is not None:
            self._apply_click(*click)

    def _apply_click(self, x: float, y: float) -> None:
        window = self.spin_window.value()
        pt = point_from_cloud(self._frame.points_organized, getattr(self._frame, "valid_mask", None), x, y, window)
        if pt is None:
            self._point_store.set(None)
            self.image_viewer.set_pick_overlays(None)
            self.coord_label.setText("유효한 점군 없음")
            self.coord_label.setStyleSheet("font-family: Consolas, monospace; font-size: 18px; font-weight: 600; color: #c0392b;")
            self.detail_label.setText(f"픽셀 ({int(round(x))}, {int(round(y))}) 주변 {window}x{window} 창에 유효한 깊이가 없습니다. "
                                      "다른 위치를 클릭하거나 창 크기를 키워 보세요.")
            self.sent_label.setText("로봇에 보낼 좌표: 없음 (요청 시 NG,NO_POINT)")
            self._log(f"클릭 ({int(round(x))}, {int(round(y))}): 유효한 점군 없음")
            return
        pt.stamp = datetime.now().strftime("%H:%M:%S")
        self._point_store.set(pt)
        X, Y, Z = pt.xyz_mm
        self.detail_label.setText(
            f"픽셀 (u={pt.u}, v={pt.v})  |  창 {pt.window}x{pt.window}에서 유효 점 {pt.n_points}개의 중앙값  |  "
            f"창 안 Z 표준편차 {pt.spread_mm:.2f} mm" + ("  (값이 크면 경계/기울어진 면 - 위치를 확인)" if pt.spread_mm > 3.0 else ""))
        self._refresh_sent_label(pt)
        self._show_point(pt)
        rx, ry, rz = self._display_xyz(pt)[0]
        robot = self._display_xyz(pt)[1]
        self._log(f"클릭 ({pt.u}, {pt.v}) -> 카메라 X{X:+.2f} Y{Y:+.2f} Z{Z:+.2f} mm"
                  + (f" -> 로봇 X{rx:+.2f} Y{ry:+.2f} Z{rz:+.2f} mm" if robot else "") + f" (유효 점 {pt.n_points}개)")

    def _display_xyz(self, pt: ClickedPoint) -> tuple[tuple[float, float, float], bool]:
        """(표시할 좌표, 로봇 좌표인가). 로봇 변환이 켜져 있고 쓸 수 있을 때만 변환한다."""
        if self._output.snapshot()[0] == "robot":
            _f, R, t, rms, limit = self._output.snapshot()
            if R is not None and t is not None and not (rms is not None and rms > limit):
                v = R @ np.asarray(pt.xyz_mm, dtype=float) + t
                return (float(v[0]), float(v[1]), float(v[2])), True
        return pt.xyz_mm, False

    def _show_point(self, pt: ClickedPoint) -> None:
        """좌표 큰 글씨 / 이미지 라벨을 현재 좌표계(카메라|로봇)로 그린다."""
        (X, Y, Z), robot = self._display_xyz(pt)
        self.coord_box.setTitle("클릭한 위치의 좌표 (로봇 좌표계, mm - 변환 적용)" if robot
                                else "클릭한 위치의 점군 좌표 (카메라 좌표계, mm)")
        color = "color: #1a6fb0;" if robot else ""
        self.coord_label.setText(f"X {X:+9.2f}\nY {Y:+9.2f}\nZ {Z:+9.2f}")
        self.coord_label.setStyleSheet(f"font-family: Consolas, monospace; font-size: 18px; font-weight: 600; {color}")
        tag = "[로봇] " if robot else ""
        self.image_viewer.set_pick_overlays([{
            "origin": (pt.u + 0.5, pt.v + 0.5), "axes": [], "dot": QColor(255, 60, 60),
            "label": f"{tag}X{X:+.1f} Y{Y:+.1f} Z{Z:+.1f}", "label_above": True}])

    @_guarded
    def _on_robot_click_toggled(self, on: bool) -> None:
        if on:
            fit, limit = self._fit, self.spin_limit.value()
            if fit is None:
                why = "아직 좌표 변환을 계산하지 않았습니다. '좌표 변환' 탭에서 점 짝을 모아 [변환 계산]을 먼저 하세요."
            elif not self._fit_manual and fit.rms_mm > limit:
                why = f"변환 잔차 RMS {fit.rms_mm:.2f} mm 가 한계 {limit:.2f} mm 를 넘습니다. 점 짝을 보강하거나 한계를 조정하세요."
            else:
                why = ""
            if why:
                QMessageBox.warning(self, "로봇 좌표로 변환할 수 없음", why)
                self.chk_robot_click.blockSignals(True)
                self.chk_robot_click.setChecked(False)
                self.chk_robot_click.blockSignals(False)
                return
        self.cmb_frame.setCurrentIndex(1 if on else 0)   # 응답 좌표계와 한 설정 (-> _apply_output_config)
        self._log("클릭 좌표 표시/응답 좌표계: " + ("로봇 (변환 적용)" if on else "카메라"))

    def _refresh_sent_label(self, pt: ClickedPoint | None) -> None:
        """로봇이 지금 PICK을 보내면 받을 응답 문자열을 그대로 보여준다 (좌표 변환 설정 반영)."""
        if pt is None:
            self.sent_label.setText("로봇에 보낼 좌표: 없음 (요청 시 NG,NO_POINT)")
            return
        res = self._worker.submit("pick").result()
        tcp = self._store.snapshot().tcp
        text = format_ok(res.candidates, tcp) if res.ok else format_ng(res.error_code, tcp)
        frame = "로봇 좌표(변환 적용)" if self._output.snapshot()[0] == "robot" else "카메라 좌표"
        self.sent_label.setText(f"로봇에 보낼 응답 [{frame}] ({pt.stamp} 선택): {text}")

    def _clear_point(self, log: bool = True) -> None:
        self._point_store.set(None)
        self._last_click = None
        self.image_viewer.set_pick_overlays(None)
        self.coord_label.setText("-")
        self.coord_label.setStyleSheet("font-family: Consolas, monospace; font-size: 18px; font-weight: 600;")
        self.detail_label.setText("")
        self._refresh_sent_label(None)
        if log:
            self._log("선택 해제")

    # =================================================================
    # 좌표 변환 (카메라 -> 로봇) 추정
    # =================================================================
    CALIB_COLS = ["사용", "카메라X", "카메라Y", "카메라Z", "로봇 X", "로봇 Y", "로봇 Z", "오차mm"]

    def _build_calib_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 6, 0, 0)
        info = QLabel("같은 점을 ① 이미지에서 클릭(카메라 좌표)하고 ② 로봇 TCP로 그 점에 대고 티치 펜던트의 X, Y, Z(mm)를 "
                      "'로봇' 칸에 입력해 점 짝을 쌓습니다. 작업 영역 전체, 높이 2~3단으로 10~20개 권장.\n"
                      "로봇 손에 달린 카메라는 촬영 자세가 바뀌면 변환도 달라집니다. 점을 모을 때와 실제 피킹 때 같은 촬영 자세에서 찍으세요.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #555; font-size: 11px;")
        lay.addWidget(info)

        form = QFormLayout()
        self.ed_pose_memo = QLineEdit()
        self.ed_pose_memo.setPlaceholderText("예: 촬영 위치 P1 (티치 포인트 번호/관절값)")
        form.addRow("촬영 자세 메모", self.ed_pose_memo)
        lay.addLayout(form)

        self.calib_table = QTableWidget(0, len(self.CALIB_COLS))
        self.calib_table.setHorizontalHeaderLabels(self.CALIB_COLS)
        self.calib_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.calib_table.verticalHeader().setDefaultSectionSize(24)
        self.calib_table.setMinimumHeight(150)
        hh = self.calib_table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)         # 좁은 오른쪽 열에서도 8칸이 다 보이게
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.calib_table.setColumnWidth(0, 40)
        self.calib_table.setStyleSheet("QTableWidget { font-size: 11px; }")
        self.calib_table.itemChanged.connect(lambda _i: self._mark_calib_stale())
        lay.addWidget(self.calib_table, stretch=1)

        row = QHBoxLayout()
        self.btn_calib_add = QPushButton("현재 클릭 점 추가")
        self.btn_calib_add.setToolTip("지금 클릭해 둔 카메라 좌표를 새 행으로 넣습니다. 로봇 X, Y, Z 칸은 직접 입력하세요.")
        self.btn_calib_add.clicked.connect(lambda _=False: self._on_calib_add())
        self.btn_calib_del = QPushButton("선택 행 삭제")
        self.btn_calib_del.clicked.connect(lambda _=False: self._on_calib_delete())
        self.btn_calib_solve = QPushButton("변환 계산")
        self.btn_calib_solve.setStyleSheet("font-weight: 600;")
        self.btn_calib_solve.clicked.connect(lambda _=False: self._on_calib_solve())
        self.btn_calib_save = QPushButton("저장")
        self.btn_calib_save.setToolTip(f"{CALIB_PATH} 에 점 짝과 변환을 저장합니다 (다음에 열 때 점 짝을 불러옵니다)")
        self.btn_calib_save.clicked.connect(lambda _=False: self._on_calib_save())
        for b in (self.btn_calib_add, self.btn_calib_del, self.btn_calib_solve, self.btn_calib_save):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        self.calib_result = QLabel("")
        self.calib_result.setWordWrap(True)
        self.calib_result.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        lay.addWidget(self.calib_result)

        out = QFormLayout()
        self.cmb_frame = QComboBox()
        self.cmb_frame.addItems(["카메라 좌표 (변환 안 함)", "로봇 좌표 (변환 적용)"])
        self.cmb_frame.setToolTip("로봇의 PICK 요청에 어느 좌표계로 응답할지. 로봇 좌표는 위 변환을 적용한 X, Y, Z입니다.\n"
                                  "앱을 다시 켜면 항상 '카메라 좌표'로 시작합니다 (실수로 변환이 적용된 채 움직이지 않게).")
        self.cmb_frame.currentIndexChanged.connect(lambda _i: self._apply_output_config())
        out.addRow("PICK 응답 좌표계", self.cmb_frame)
        self.spin_limit = QDoubleSpinBox()
        self.spin_limit.setRange(0.1, 50.0)
        self.spin_limit.setSingleStep(0.5)
        self.spin_limit.setValue(DEFAULT_RMS_LIMIT_MM)
        self.spin_limit.setSuffix(" mm")
        self.spin_limit.setToolTip("변환 잔차 RMS가 이 값보다 크면 로봇 좌표로 응답하지 않고 NG,CALIB_BAD를 보냅니다 (보정이 나쁜 채 로봇이 움직이는 것을 막음).")
        self.spin_limit.valueChanged.connect(lambda _v: self._apply_output_config())
        out.addRow("잔차 RMS 한계", self.spin_limit)
        lay.addLayout(out)
        return page

    def _mark_calib_stale(self) -> None:
        if getattr(self, "_calib_loading", False) or self._fit_manual:
            return           # 수동 입력 행렬은 표가 바뀌어도 [변환 계산] 전까지 유지된다 (오래됐다는 경고 대상이 아님)
        self._calib_stale = True
        if self._fit is not None:
            self.calib_result.setText(self.calib_result.text().split("\n⚠")[0]
                                      + "\n⚠ 표가 바뀌었습니다. 아직 이전 계산 결과가 적용 중입니다 - [변환 계산]을 다시 누르세요.")

    def _add_calib_row(self, pair: CalibPair) -> None:
        t = self.calib_table
        self._calib_loading = True
        try:
            r = t.rowCount()
            t.insertRow(r)
            use = QTableWidgetItem()
            use.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            use.setCheckState(Qt.CheckState.Checked if pair.use else Qt.CheckState.Unchecked)
            use.setToolTip("체크 해제하면 변환 계산에서 빼고, 그 점의 오차만 검증용으로 봅니다")
            t.setItem(r, 0, use)
            for k in range(3):
                it = QTableWidgetItem(f"{pair.cam[k]:.1f}")
                it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                it.setData(Qt.ItemDataRole.UserRole, float(pair.cam[k]))     # 표시는 반올림해도 계산은 원값으로
                t.setItem(r, 1 + k, it)
            for k in range(3):
                t.setItem(r, 4 + k, QTableWidgetItem("" if pair.robot is None else f"{pair.robot[k]:.3f}".rstrip("0").rstrip(".")))
            err = QTableWidgetItem("")
            err.setFlags(Qt.ItemFlag.ItemIsEnabled)
            t.setItem(r, 7, err)
        finally:
            self._calib_loading = False

    def _calib_pairs(self) -> list[CalibPair]:
        """표 -> CalibPair 목록. 로봇 칸이 일부만 비었거나 숫자가 아니면 ValueError(행 번호 포함)."""
        pairs = []
        t = self.calib_table
        for r in range(t.rowCount()):
            cam = tuple(float(t.item(r, 1 + k).data(Qt.ItemDataRole.UserRole)) for k in range(3))
            texts = [t.item(r, 4 + k).text().strip().replace(",", ".") for k in range(3)]
            if all(x == "" for x in texts):
                robot = None
            elif any(x == "" for x in texts):
                raise ValueError(f"{r + 1}번 행: 로봇 X, Y, Z 세 칸을 모두 입력하세요.")
            else:
                try:
                    robot = tuple(float(x) for x in texts)
                except ValueError:
                    raise ValueError(f"{r + 1}번 행: 로봇 좌표가 숫자가 아닙니다 ({', '.join(texts)}).") from None
            pairs.append(CalibPair(cam, robot, t.item(r, 0).checkState() == Qt.CheckState.Checked))
        return pairs

    @_guarded
    def _on_calib_add(self) -> None:
        pt = self._point_store.get()
        if pt is None:
            QMessageBox.information(self, "클릭한 점 없음", "먼저 [촬영] 후 이미지에서 점을 클릭하세요 (유효한 점군이 읽혀야 합니다).")
            return
        self._add_calib_row(CalibPair(pt.xyz_mm))
        self._mark_calib_stale()
        r = self.calib_table.rowCount() - 1
        self.calib_table.setCurrentCell(r, 4)
        self.calib_table.editItem(self.calib_table.item(r, 4))
        self._log(f"보정점 {r + 1} 추가: 카메라 X{pt.xyz_mm[0]:+.2f} Y{pt.xyz_mm[1]:+.2f} Z{pt.xyz_mm[2]:+.2f} - 로봇 좌표를 입력하세요")

    @_guarded
    def _on_calib_delete(self) -> None:
        rows = sorted({i.row() for i in self.calib_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.calib_table.removeRow(r)
        if rows:
            self._mark_calib_stale()

    @_guarded
    def _on_calib_solve(self) -> None:
        try:
            pairs = self._calib_pairs()
            fit = solve(pairs)
        except (ValueError, CalibError) as exc:      # CalibError도 ValueError 하위
            QMessageBox.warning(self, "변환 계산 불가", str(exc))
            return
        self._fit = fit
        self._fit_manual = False
        self._calib_stale = False
        self._calib_loading = True
        try:
            for r, e in enumerate(fit.errors_mm):
                it = self.calib_table.item(r, 7)
                it.setText("" if e is None else f"{e:.2f}")
                limit = self.spin_limit.value()
                bad = e is not None and (e > limit or e > max(3.0 * fit.rms_mm, 1.0))
                it.setData(Qt.ItemDataRole.ForegroundRole, QColor("#c0392b") if bad else None)   # None = 기본 글자색(테마 따름)
        finally:
            self._calib_loading = False
        rx, ry, rz = fit.euler_zyx_deg()
        loo = f"{fit.loo_rms_mm:.2f}" if fit.loo_rms_mm is not None else "-"
        lines = [
            f"사용 점 {fit.n_used}개  잔차 RMS {fit.rms_mm:.2f} mm  최대 {fit.max_mm:.2f} mm  leave-one-out RMS {loo} mm",
            f"이동 t = ({fit.t[0]:+.2f}, {fit.t[1]:+.2f}, {fit.t[2]:+.2f}) mm   회전(Rz·Ry·Rx, 참고용) = ({rx:+.2f}, {ry:+.2f}, {rz:+.2f})°",
            "robot = R·camera + t    (오차 열: 체크 해제한 점은 검증용 오차)",
        ]
        limit = self.spin_limit.value()
        if fit.rms_mm > limit:
            lines.append(f"⚠ 잔차 RMS가 한계({limit:.1f} mm)보다 큽니다. 로봇 좌표 응답은 NG,CALIB_BAD로 막힙니다. 오차 큰 점(빨강)을 확인하세요.")
        lines += [f"⚠ {w}" for w in fit.warnings]
        self.calib_result.setText("\n".join(lines))
        self._apply_output_config()
        self._log(f"좌표 변환 계산: 점 {fit.n_used}개, 잔차 RMS {fit.rms_mm:.2f} mm, 최대 {fit.max_mm:.2f} mm, LOO {loo} mm")

    def _apply_output_config(self) -> None:
        frame = "robot" if self.cmb_frame.currentIndex() == 1 else "camera"
        fit = self._fit
        rms = None if (fit is None or self._fit_manual) else fit.rms_mm     # 수동 입력 행렬은 잔차 검사 없음 (None)
        self._output.set(frame, None if fit is None else fit.R, None if fit is None else fit.t, rms, self.spin_limit.value())
        robot = frame == "robot"
        if self.chk_robot_click.isChecked() != robot:    # 콤보에서 바꿔도 체크박스가 따라가게
            self.chk_robot_click.blockSignals(True)
            self.chk_robot_click.setChecked(robot)
            self.chk_robot_click.blockSignals(False)
        pt = self._point_store.get()
        self._refresh_sent_label(pt)
        if pt is not None:
            self._show_point(pt)

    @_guarded
    def _on_calib_save(self) -> None:
        try:
            pairs = self._calib_pairs()
        except ValueError as exc:
            QMessageBox.warning(self, "저장 불가", str(exc))
            return
        manual = self._fit_manual and self._fit is not None     # 수동 입력 행렬은 점 짝으로 다시 풀지 않고 그대로 저장
        if not manual and (self._calib_stale or self._fit is None):
            try:
                self._on_calib_solve_silent(pairs)
            except CalibError:
                self._fit = None      # 점이 부족하면 점 짝만 저장한다
        save_calibration(CALIB_PATH, pairs, self._fit, self.ed_pose_memo.text().strip(), self.spin_limit.value(),
                         source="manual" if manual else "pairs")
        self._log(f"좌표 변환 저장: {CALIB_PATH}" + (" (수동 입력 행렬)" if manual else ""))
        QMessageBox.information(self, "저장", f"저장했습니다.\n{CALIB_PATH}")

    def _on_calib_solve_silent(self, pairs: list[CalibPair]) -> None:
        self._fit = solve(pairs)
        self._fit_manual = False
        self._calib_stale = False
        self._apply_output_config()

    def _load_calib_file(self) -> None:
        """저장된 점 짝을 불러온다. 응답 좌표계는 항상 '카메라'로 시작한다 (안전)."""
        try:
            pairs, transform, memo, limit = load_calibration(CALIB_PATH)
        except Exception as exc:  # noqa: BLE001 - 깨진 파일이 탭 생성을 막지 않게
            self._log(f"좌표 변환 파일을 읽지 못했습니다 ({CALIB_PATH}): {exc}")
            return
        for p in pairs:
            self._add_calib_row(p)
        self.ed_pose_memo.setText(memo)
        self.spin_limit.setValue(limit)
        if transform and transform.get("source") == "manual":
            try:       # 수동 입력 행렬은 점 짝으로 다시 풀지 않고 저장된 R, t를 그대로 쓴다
                R, t = np.asarray(transform["R"], dtype=float), np.asarray(transform["t"], dtype=float)
                if R.shape != (3, 3) or t.shape != (3,):
                    raise ValueError("행렬 모양이 올바르지 않습니다")
                self._set_manual_fit(R, t)
                self._log("저장된 수동 입력 행렬을 불러왔습니다 (응답 좌표계는 '카메라'로 시작)")
            except (ValueError, KeyError, TypeError) as exc:
                self._log(f"저장된 수동 입력 행렬을 읽지 못했습니다: {exc}")
        elif pairs:
            try:
                self._fit = solve(pairs)
                self._calib_stale = False
                self._on_calib_solve()
            except (ValueError, CalibError):
                pass
            self._log(f"저장된 보정점 {len(pairs)}개를 불러왔습니다 (응답 좌표계는 '카메라'로 시작)")

    # =================================================================
    # 행렬 직접 입력 / 저장된 변환과 비교
    # =================================================================
    def _build_matrix_page(self) -> QWidget:
        self._mat_ref: dict | None = None          # 비교 기준 B: {"R", "t", "src"}
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 6, 0, 0)
        info = QLabel("이동 t(mm)와 회전(deg)을 넣으면 카메라→로봇 변환 행렬 (robot = R·camera + t)을 만듭니다. "
                      "저장된 변환을 불러오면 입력 칸이 그 값으로 채워지니, 값을 고쳐 가며 기준과의 차이를 비교하세요. "
                      "[적용]을 누르기 전에는 PICK 응답이나 저장 파일의 변환을 바꾸지 않습니다.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #555; font-size: 11px;")
        lay.addWidget(info)

        grid = QGridLayout()
        grid.addWidget(QLabel("이동 t (mm)"), 1, 0)
        grid.addWidget(QLabel("회전 (deg)"), 3, 0)
        self.mat_t, self.mat_r = [], []
        for k, name in enumerate("XYZ"):
            grid.addWidget(QLabel(name), 0, 1 + k, Qt.AlignmentFlag.AlignHCenter)
            sp = QDoubleSpinBox()
            sp.setRange(-9999.0, 9999.0)
            sp.setDecimals(3)
            sp.setSingleStep(1.0)
            sp.setToolTip(f"이동 t의 {name} (mm)")
            sp.valueChanged.connect(lambda _v: self._mat_refresh())
            self.mat_t.append(sp)
            grid.addWidget(sp, 1, 1 + k)
        for k, name in enumerate(("Rx", "Ry", "Rz")):
            grid.addWidget(QLabel(name), 2, 1 + k, Qt.AlignmentFlag.AlignHCenter)
            sp = QDoubleSpinBox()
            sp.setRange(-360.0, 360.0)
            sp.setDecimals(4)
            sp.setSingleStep(0.5)
            sp.setToolTip(f"{name} (deg)")
            sp.valueChanged.connect(lambda _v: self._mat_refresh())
            self.mat_r.append(sp)
            grid.addWidget(sp, 3, 1 + k)
        lay.addLayout(grid)

        form = QFormLayout()
        self.cmb_mat_order = QComboBox()
        self.cmb_mat_order.addItems(list(tm.EULER_ORDERS))
        self.cmb_mat_order.setToolTip("회전 칸의 오일러 각을 행렬로 바꾸는 순서. 앞의 것이 '좌표 변환' 탭 결과의 '회전(Rz·Ry·Rx)' 표시와 같습니다.\n"
                                      "순서를 바꾸면 칸의 숫자는 그대로 두고 해석만 바뀝니다 (로봇이 쓰는 순서를 찾는 데 씁니다).")
        self.cmb_mat_order.currentIndexChanged.connect(lambda _i: self._mat_refresh())
        form.addRow("오일러 순서", self.cmb_mat_order)
        lay.addLayout(form)

        row = QHBoxLayout()
        self.btn_mat_load = QPushButton("저장 파일")
        self.btn_mat_load.setToolTip(f"{CALIB_PATH} 의 저장된 변환(R, t)을 입력 칸과 비교 기준(B)으로 불러옵니다.")
        self.btn_mat_load.clicked.connect(lambda _=False: self._on_mat_load_saved())
        self.btn_mat_other = QPushButton("다른 파일…")
        self.btn_mat_other.clicked.connect(lambda _=False: self._on_mat_load_other())
        self.btn_mat_fit = QPushButton("현재 결과")
        self.btn_mat_fit.setToolTip("'좌표 변환' 탭에서 지금 계산해 둔 변환(저장 전 포함)을 입력 칸과 비교 기준(B)으로 가져옵니다.")
        self.btn_mat_fit.clicked.connect(lambda _=False: self._on_mat_use_fit())
        for b in (self.btn_mat_load, self.btn_mat_other, self.btn_mat_fit):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        self.mat_ref_label = ElidedLabel("비교 기준(B): 없음 - [저장 파일] 등 위 버튼으로 불러오세요")
        self.mat_ref_label.setStyleSheet("color: #555; font-size: 11px;")
        lay.addWidget(self.mat_ref_label)

        self.mat_out = QPlainTextEdit()
        self.mat_out.setReadOnly(True)
        self.mat_out.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.mat_out.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        self.mat_out.setMinimumHeight(60)
        lay.addWidget(self.mat_out, stretch=1)

        row2 = QHBoxLayout()
        self.btn_mat_copy = QPushButton("행렬 복사")
        self.btn_mat_copy.setToolTip("입력(A)으로 만든 4x4 행렬을 np.array 문자열로 클립보드에 복사합니다 (전체 정밀도).")
        self.btn_mat_copy.clicked.connect(lambda _=False: self._on_mat_copy())
        self.btn_mat_zero = QPushButton("입력 초기화")
        self.btn_mat_zero.clicked.connect(lambda _=False: self._mat_fill(np.eye(3), np.zeros(3)))
        row2.addWidget(self.btn_mat_copy)
        row2.addWidget(self.btn_mat_zero)
        row2.addStretch(1)
        lay.addLayout(row2)
        self.btn_mat_apply = QPushButton("입력 행렬을 현재 변환으로 적용")
        self.btn_mat_apply.setStyleSheet("font-weight: 600;")
        self.btn_mat_apply.setToolTip("입력(A) 행렬을 '좌표 변환' 탭의 현재 변환으로 씁니다 (확인 창이 뜹니다).\n"
                                      "'좌표 변환' 탭 표의 점 짝과는 무관하게 이 행렬을 그대로 쓰며, 잔차 RMS 한계 검사(CALIB_BAD)도 적용되지 않습니다.\n"
                                      "저장 파일(data/gripper_calib.json)에도 바로 기록되어 4-1/5-2 탭과 다음 실행에서도 이 행렬을 씁니다.")
        self.btn_mat_apply.clicked.connect(lambda _=False: self._on_mat_apply())
        lay.addWidget(self.btn_mat_apply)
        self._mat_refresh()
        return page

    def _mat_input(self) -> tuple[np.ndarray, np.ndarray]:
        t = np.array([sp.value() for sp in self.mat_t], dtype=float)
        rx, ry, rz = (sp.value() for sp in self.mat_r)
        return tm.euler_to_R(rx, ry, rz, self.cmb_mat_order.currentText()), t

    def _mat_fill(self, R, t) -> None:
        """입력 칸을 (R, t)로 채운다 (오일러는 지금 고른 순서로). 칸 신호는 막고 마지막에 한 번만 갱신."""
        rx, ry, rz = tm.R_to_euler(R, self.cmb_mat_order.currentText())
        for sp, v in zip(self.mat_t + self.mat_r, [*np.asarray(t, dtype=float), rx, ry, rz]):
            sp.blockSignals(True)
            sp.setValue(float(v))
            sp.blockSignals(False)
        self._mat_refresh()

    def _mat_probe_points(self) -> tuple[list, str]:
        """비교에 쓸 카메라 좌표: '좌표 변환' 탭 표의 보정점이 있으면 그것, 없으면 기본 점 5개."""
        pts = []
        for r in range(self.calib_table.rowCount()):
            try:
                pts.append(tuple(float(self.calib_table.item(r, 1 + k).data(Qt.ItemDataRole.UserRole)) for k in range(3)))
            except (AttributeError, TypeError, ValueError):
                continue
        if pts:
            return pts, f"보정점 {len(pts)}개"
        return list(tm.DEFAULT_PROBE_POINTS), "기본 점 5개 (Z=600 mm, 좌우/상하 ±200)"

    def _mat_refresh(self) -> None:
        R, t = self._mat_input()
        order = self.cmb_mat_order.currentText()
        rx, ry, rz = (sp.value() for sp in self.mat_r)
        lines = [tm.format_T(tm.to_T(R, t), f"A (입력)   {order} = ({rx:+.4f}, {ry:+.4f}, {rz:+.4f})°")]
        ref = self._mat_ref
        if ref is None:
            lines += ["", "B (기준): 없음 - [저장 파일 불러오기] 등으로 기준을 정하면 차이를 보여줍니다."]
        else:
            bx, by, bz = tm.R_to_euler(ref["R"], order)
            lines += ["", tm.format_T(tm.to_T(ref["R"], ref["t"]), f"B (기준)   {order} = ({bx:+.4f}, {by:+.4f}, {bz:+.4f})°")]
            pts, pts_name = self._mat_probe_points()
            c = tm.compare_transforms(R, t, ref["R"], ref["t"], pts)
            dt = c["dt"]
            verdict = ("사실상 같음" if c["pt_max"] < 0.2 else
                       "작은 차이 (픽 정밀도에 영향 가능)" if c["pt_max"] < 1.0 else
                       "차이가 큼 - 오일러 순서/부호/단위(mm)를 확인하세요")
            lines += ["", f"차이 A − B  →  {verdict}",
                      f"  (비교 점: {pts_name})",
                      f"  회전 차이     {c['rot_deg']:.4f}°",
                      f"  이동 차이     ({dt[0]:+.3f}, {dt[1]:+.3f}, {dt[2]:+.3f}) mm   |Δt| {c['dt_norm']:.3f} mm",
                      f"  점 위치 차이  RMS {c['pt_rms']:.3f} mm   최대 {c['pt_max']:.3f} mm"]
        self.mat_out.setPlainText("\n".join(lines))

    def _set_mat_ref(self, R, t, src: str) -> None:
        self._mat_ref = {"R": np.asarray(R, dtype=float), "t": np.asarray(t, dtype=float), "src": src}
        self.mat_ref_label.setText(f"비교 기준(B): {src}")
        self._mat_fill(R, t)

    def _mat_load_json(self, path) -> bool:
        path = Path(path)
        if not path.is_file():
            QMessageBox.information(self, "파일 없음", f"변환 파일이 없습니다.\n{path}")
            return False
        _pairs, tf, memo, _limit = load_calibration(path)
        if not tf or "R" not in tf or "t" not in tf:
            QMessageBox.information(self, "변환 없음", f"이 파일에는 저장된 변환(R, t)이 없습니다 (점 짝만 저장된 상태일 수 있음).\n{path}")
            return False
        R, t = np.asarray(tf["R"], dtype=float), np.asarray(tf["t"], dtype=float)
        if R.shape != (3, 3) or t.shape != (3,) or not (np.isfinite(R).all() and np.isfinite(t).all()):
            QMessageBox.warning(self, "형식 오류", f"R은 3x3, t는 길이 3의 숫자여야 합니다.\n{path}")
            return False
        extra = []
        if tf.get("rms_mm") is not None:
            extra.append(f"RMS {float(tf['rms_mm']):.2f} mm")
        if tf.get("n_used"):
            extra.append(f"점 {int(tf['n_used'])}개")
        if memo:
            extra.append(memo)
        self._set_mat_ref(R, t, f"{path.name}" + (f"  ({' · '.join(extra)})" if extra else ""))
        self._log(f"행렬 직접 입력: 변환을 불러왔습니다 - {path}")
        return True

    @_guarded
    def _on_mat_load_saved(self) -> None:
        self._mat_load_json(CALIB_PATH)

    @_guarded
    def _on_mat_load_other(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "변환 JSON 선택", str(Path(CALIB_PATH).parent), "JSON (*.json)")
        if path:
            self._mat_load_json(path)

    @_guarded
    def _on_mat_use_fit(self) -> None:
        if self._fit is None:
            QMessageBox.information(self, "계산 결과 없음", "'좌표 변환' 탭에서 먼저 [변환 계산]을 하세요.")
            return
        if self._fit_manual:
            self._set_mat_ref(self._fit.R, self._fit.t, "현재 변환 (수동 입력 행렬 - 잔차 검사 없음)")
            return
        suffix = "  ⚠ 표가 바뀐 뒤 다시 계산하지 않음" if self._calib_stale else ""
        self._set_mat_ref(self._fit.R, self._fit.t,
                          f"현재 계산 결과 (RMS {self._fit.rms_mm:.2f} mm · 점 {self._fit.n_used}개){suffix}")

    def _clear_calib_errors(self) -> None:
        """변환 계산 탭 표의 '오차mm' 열을 비운다 (수동 입력 행렬은 표의 점 짝과 무관하다)."""
        self._calib_loading = True
        try:
            for r in range(self.calib_table.rowCount()):
                it = self.calib_table.item(r, 7)
                if it is not None:
                    it.setText("")
                    it.setData(Qt.ItemDataRole.ForegroundRole, None)
        finally:
            self._calib_loading = False

    def _set_manual_fit(self, R, t) -> None:
        """'행렬 직접 입력'으로 정한 변환을 현재 변환으로 쓴다. 점 짝(표)으로 푼 값이 아니고, 표의 점과 비교하지도 않는다
        (표의 점은 다른 촬영 자세에서 모은 것일 수 있다). 그래서 잔차 RMS 한계 검사도 적용하지 않는다."""
        R, t = np.asarray(R, dtype=float), np.asarray(t, dtype=float)
        self._fit = RigidFit(R=R, t=t, rms_mm=0.0, max_mm=0.0, loo_rms_mm=None, n_used=0, errors_mm=[],
                             extent_mm=0.0, scale_ratio=1.0, warnings=[])
        self._fit_manual = True
        self._calib_stale = False
        self._clear_calib_errors()
        rx, ry, rz = self._fit.euler_zyx_deg()
        self.calib_result.setText("\n".join([
            "■ 수동 입력 행렬 적용 중 - 표의 점 짝과 무관하게 이 행렬을 그대로 씁니다 (잔차 검사 없음)",
            f"이동 t = ({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}) mm   회전(Rz·Ry·Rx, 참고용) = ({rx:+.2f}, {ry:+.2f}, {rz:+.2f})°",
            "robot = R·camera + t    (표를 고쳐도 이 행렬은 [변환 계산]을 누르기 전까지 유지됩니다)"]))
        self._apply_output_config()

    @_guarded
    def _on_mat_apply(self) -> None:
        R, t = self._mat_input()
        ans = QMessageBox.question(
            self, "행렬 적용",
            "입력한 행렬을 현재 좌표 변환으로 적용하고, 저장 파일에도 바로 기록합니다.\n"
            f"({CALIB_PATH})\n\n"
            "• 4-1 / 5-2 탭과 다음 실행에서도 이 행렬을 씁니다.\n"
            "• 기존 파일의 변환이 점 짝으로 푼 것이면 같은 폴더에 백업(.bak-날짜시각)해 둡니다.\n"
            "• '좌표 변환' 탭의 점 짝(표)과는 무관하게 이 행렬을 그대로 쓰며, 잔차 RMS 한계 검사는 적용되지 않습니다.\n\n"
            "PICK 응답 좌표계가 '로봇 좌표'이면 이 행렬로 로봇이 움직입니다. 계속할까요?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if ans != QMessageBox.StandardButton.Yes:
            return
        self._set_manual_fit(R, t)
        try:
            backup = save_manual_transform(CALIB_PATH, R, t, self.ed_pose_memo.text().strip(), self.spin_limit.value())
        except OSError as exc:
            self._log(f"행렬 직접 입력: 적용은 됐지만 저장 파일에 기록하지 못했습니다 ({CALIB_PATH}): {exc}")
            QMessageBox.warning(self, "저장 실패", f"이 앱에서는 적용됐지만 저장 파일에 쓰지 못했습니다.\n"
                                                  f"4-1/5-2 탭은 이전 저장 파일을 계속 씁니다.\n{exc}")
            return
        self.calib_result.setText(self.calib_result.text() + f"\n저장 파일에 기록했습니다: {Path(CALIB_PATH).name} (4-1/5-2 탭도 이 행렬을 씁니다)"
                                  + (f"\n기존 파일 백업: {backup.name}" if backup else ""))
        self._log("행렬 직접 입력: 입력 행렬을 현재 변환으로 적용하고 저장 파일에 기록했습니다 (점 짝과 무관, 잔차 검사 없음)"
                  + (f" - 기존 파일 백업: {backup}" if backup else ""))

    @_guarded
    def _on_mat_copy(self) -> None:
        R, t = self._mat_input()
        QApplication.clipboard().setText(tm.format_numpy(tm.to_T(R, t)))
        self._log("행렬 직접 입력: 입력 행렬(A)을 클립보드에 복사했습니다")


    # =================================================================
    # 평면 검증
    # =================================================================
    def _on_plane_mode(self, on: bool) -> None:
        if on:
            self.image_viewer.set_click_pick(False)
            self.image_viewer.set_roi_drawing(True)
        else:
            self.image_viewer.set_roi_drawing(False)
            self.image_viewer.set_roi(None)
            self.image_viewer.set_click_pick(True)

    @_guarded
    def _on_plane_roi(self, x1: int, y1: int, x2: int, y2: int) -> None:
        if self._frame is None:
            return
        rep = analyze_plane(self._frame.points_organized, getattr(self._frame, "valid_mask", None), (x1, y1, x2, y2))
        if rep is None:
            self.plane_label.setText("유효한 점이 부족합니다 (영역을 더 크게, 판 위로 지정하세요).")
            self.plane_map.setFixedHeight(0)
            return
        self.plane_label.setText(self._plane_text(rep))
        self._show_residual_map(rep)
        self._log(f"평면 검증 ROI ({x1},{y1})-({x2},{y2}): 기울기 {rep.tilt_deg:.2f}° (X축 {rep.tilt_x_deg:+.2f}° / Y축 {rep.tilt_y_deg:+.2f}°), "
                  f"잔차 RMS {rep.rms_mm:.2f} mm, 휘어짐 {rep.bow_mm:+.2f} mm")

    @staticmethod
    def _plane_text(r: PlaneReport) -> str:
        lines = [
            f"점 {r.n_points}개 (이상치 {r.n_outliers}개 제외)  판 크기 약 {r.extent_mm:.0f} mm  거리 Z {r.centroid_mm[2]:.0f} mm",
            f"광축 대비 기울기 {r.tilt_deg:.2f}°   (X축 둘레 {r.tilt_x_deg:+.2f}° / Y축 둘레 {r.tilt_y_deg:+.2f}°)",
            f"평면 잔차 RMS {r.rms_mm:.2f} mm   최대 {r.max_abs_mm:.1f} mm",
            f"곡면(2차) 보정 후 RMS {r.rms_after_quad_mm:.2f} mm   휘어짐 {r.bow_mm:+.2f} mm "
            f"({'가운데 볼록' if r.bow_mm > 0 else '가운데 움푹'})",
        ]
        explained = 1.0 - r.rms_after_quad_mm / max(r.rms_mm, 1e-9)
        if r.rms_mm < 0.8:
            verdict = "판독: 점군이 평면에 가깝습니다. 기울기는 카메라 설치 각도일 가능성이 큽니다 (좌표 변환/수평 보정으로 해결)."
        elif explained > 0.4 and abs(r.bow_mm) > 1.0:
            verdict = ("판독: 완만한 휘어짐이 오차의 대부분입니다. 렌즈 왜곡/깊이 보정 문제로 보입니다 "
                       "(워밍업 후 재측정 -> 남으면 거리별/위치별 보정).")
        else:
            verdict = "판독: 휘어짐으로 설명되지 않는 잡음/요철이 큽니다. 평균화 프레임 수, 표면 반사, 노출을 확인하세요."
        lines.append(verdict)
        return "\n".join(lines)

    def _show_residual_map(self, r: PlaneReport) -> None:
        m = r.residual_map
        lim = max(1.0, 3.0 * r.rms_mm)
        norm = np.clip((m + lim) / (2 * lim), 0.0, 1.0)
        img = cv2.applyColorMap((np.nan_to_num(norm, nan=0.5) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img[~np.isfinite(m)] = (200, 200, 200)
        h, w = img.shape[:2]
        scale = min(170.0 / max(h, 1), 300.0 / max(w, 1))
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_NEAREST)
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.plane_map.setPixmap(QPixmap.fromImage(qimg))
        self.plane_map.setFixedHeight(qimg.height())
        self.plane_map.setToolTip(f"평면 잔차 맵: 파랑 -{lim:.1f} mm ~ 빨강 +{lim:.1f} mm (+는 카메라 쪽). 한쪽이 붉고 반대쪽이 푸르면 기울어짐, 가운데/가장자리가 다르면 휘어짐")

    def _on_copy(self) -> None:
        pt = self._point_store.get()
        if pt is None:
            return
        QApplication.clipboard().setText(f"{pt.xyz_mm[0]:.2f}, {pt.xyz_mm[1]:.2f}, {pt.xyz_mm[2]:.2f}")

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
        self._update_buttons()

    def shutdown(self) -> None:
        """앱 종료 시 호출."""
        self._server.stop()

    def _update_buttons(self) -> None:
        running = self._server.is_running
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)

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

    def _on_response(self, peer: str, text: str) -> None:
        self._append_traffic(f"▶ {peer}  {text}")
        self._log(f"응답 송신 {peer}: {text}")

    def _refresh_config_summary(self) -> None:
        s = self._store.snapshot()
        self.config_label.setText(
            f"포트 {s.tcp.port_gripper}  |  요청 '{s.tcp.trigger_cmd}' -> 클릭한 좌표 응답  |  카메라 {s.camera.camera_type} "
            f"×{s.camera.num_frames}프레임  |  값 변경은 '12. 설정' 탭 (호스트/포트는 서버 재시작 필요)")
        self._refresh_sent_label(self._point_store.get())
