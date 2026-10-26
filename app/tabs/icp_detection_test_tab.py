"""탭 4: ICP 검출 테스트 (세션 기반, 독립 탭).

data/dataset/<세션>/ 의 프레임 하나를 불러와 "2D 검출 -> 형상 보정 -> 3D 정합(ICP)"을
단계별로 돌려 보고, 단계마다 결과를 눈으로 확인하는 테스트 탭이다.

    입력: 세션 프레임 (intensity.png + pointcloud_organized.npy + valid_mask.npy)

    2D 검출
        ROI 크롭                드래그/숫자 지정, 영역 안만 잘라 추론
        RTMDet-Ins              그레이 -> 3채널, conf 필터, 원본 좌표 복원
        형상 보정 (선택)        CAD 투영 템플릿 + 깊이로 마스크 정리 (src/detection/mask_refine.py)
    3D 정합 (ICP)  - 현재 구현분만 연결 (PCA 축 추정 / roll 후보 탐색은 아직 없음)
        점군 추출               마스크 침식 -> 마스크 안 유효 점
        이상치 제거             통계적 outlier
        CAD 준비                로드 + 축 보정, 가시면 계산(HPR)
        초기 pose (기존)        중심 맞춤 + 고정 roll/pitch/yaw, 또는 FGR
        다단계 ICP              voxel 6mm -> 3mm -> 원본밀도
        뒤집힘 보정             R[2,2] < 0 이면 뒤집힌 가시면 CAD로 재정합
        게이트                  fitness, xyz 범위, 회전 구속
    출력: 인스턴스별 pose(T, roll/pitch/yaw), 픽 포인트(mm), fitness, 실패 사유, 3D 뷰어

기존 ICPWorkbenchTab(app/tabs/icp_workbench_base.py)을 상속하지 않는 독립 탭이다 -
5/7/9번 탭이 그 베이스를 계속 쓰므로 베이스는 건드리지 않았고, 계산 로직은 전부
app/core/icp_runner.py, src/detection/mask_refine.py의 기존 함수를 그대로 호출한다
(이 파일은 UI와 단계 연결만 담당).
"""
from __future__ import annotations

import dataclasses
import functools
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QProcess, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from app.core import icp_runner
from app.core.app_settings import get_settings
from app.core.camera_intrinsics import estimate_intrinsics_from_organized_pcd
from app.core.config_patcher import find_latest_best_checkpoint
from app.core.detector import Detection, Detector
from app.core.icp_param_specs import B_SECTIONS, FGR_SPECS
from app.core.icp_runner import ICPParams, ICPResult
from app.core.paths import DEFAULT_CAD_DIR, DEFAULT_CONFIG_PATH, DEFAULT_DATASET_ROOT
from app.core.registration import AVAILABLE_REGISTRATION_TYPES
from app.widgets.image_viewer import ImageViewer
from src.detection.mask_refine import RefineParams, RefineResult, get_bank, refine_masks

DEFAULT_SCORE_THRESHOLD = 0.3
CAD_EXTS = {".stl", ".ply", ".obj"}
ROI_MIN_SIZE = 16  # px, ImageViewer.ROI_MIN_SIZE와 동일 기준

# 3D 정합 파라미터 UI 명세는 설정 탭과 공용이다 (app/core/icp_param_specs.py) -
# 위젯 생성 / 값 읽기 / 기본값 복원이 전부 이 표 하나로 돌아간다 (필드 추가 = 한 줄).
_B_SECTIONS = B_SECTIONS
_FGR_SPECS = FGR_SPECS


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


class ICPDetectionTestTab(QWidget):
    log_message = pyqtSignal(str)

    LOG_PREFIX = "ICP 검출 탭"

    def __init__(self, parent=None):
        super().__init__(parent)
        # 프레임
        self._session_path: str | None = None
        self._frame_names: list[str] = []
        self._current_frame: str | None = None
        self._image_path: str | None = None
        self._pcd: np.ndarray | None = None        # (H,W,3) mm
        self._valid: np.ndarray | None = None      # (H,W) bool
        # 2D 검출 결과
        self._raw_detections: list[Detection] = []    # 모델 원본 출력
        self._detections: list[Detection] = []        # ICP에 쓰이는 검출 (보정 켜면 보정본)
        self._refine_results: list[RefineResult] = []  # _detections와 index 일치 (보정 안 했으면 빈 리스트)
        self._refine_applied = False
        self._detector: Detector | None = None
        self._detector_key: tuple[str, str] | None = None
        # 3D 정합 결과 / CAD 캐시
        self._icp_results: list[ICPResult] = []
        self._cad_pcd = None
        self._cad_visible_normal = None
        self._cad_visible_flipped = None
        self._cad_path_loaded: str | None = None
        self._cad_axis_loaded: tuple[float, float, float] | None = None
        self._cad_init_rot_loaded: tuple[float, float, float] | None = None
        self._cad_ref_dist_loaded: float | None = None
        self._viewer_process: QProcess | None = None
        # UI
        self._param_widgets: dict[str, QWidget] = {}
        self._build_ui()
        self._prefill_latest_checkpoint()
        self._refresh_cad_list()
        self._load_params_from_settings(announce=False)

    # =================================================================
    # 공통 유틸
    # =================================================================
    def _log(self, message: str) -> None:
        self.log_message.emit(f"[{self.LOG_PREFIX}] {message}")

    # =================================================================
    # UI
    # =================================================================
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.addWidget(self._build_left_panel())

        center = QVBoxLayout()
        top_row = QHBoxLayout()
        self.btn_run_all = QPushButton("검출 + 정합 전체 실행")
        self.btn_run_all.setToolTip("2D 검출(+형상 보정) 후 바로 ICP 정합까지 실행")
        self.btn_run_all.clicked.connect(lambda _=False: self._on_run_all())
        top_row.addWidget(self.btn_run_all)
        top_row.addStretch(1)
        center.addLayout(top_row)

        controls = QWidget()
        controls_lay = QVBoxLayout(controls)
        controls_lay.setContentsMargins(0, 0, 0, 0)
        controls_lay.addWidget(self._build_group_a())
        controls_lay.addWidget(self._build_group_b())
        controls_lay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMaximumHeight(360)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        center.addWidget(scroll)

        self.image_viewer = ImageViewer()
        self.image_viewer.roi_changed.connect(self._on_roi_drawn)
        center.addWidget(self.image_viewer, stretch=1)
        root.addLayout(center, stretch=2)

        root.addWidget(self._build_right_panel())

    # ------------------------------------------------------------ 좌측
    def _build_left_panel(self) -> QWidget:
        left_widget = QWidget()
        left_widget.setFixedWidth(320)
        left = QVBoxLayout(left_widget)

        left.addWidget(QLabel("세션 폴더"))
        session_row = QHBoxLayout()
        self.session_edit = QLineEdit()
        self.session_edit.setReadOnly(True)
        self.session_edit.setPlaceholderText("세션 폴더를 선택하세요")
        session_row.addWidget(self.session_edit, stretch=1)
        btn_browse_session = QPushButton("선택")
        btn_browse_session.clicked.connect(lambda _=False: self._on_browse_session())
        session_row.addWidget(btn_browse_session)
        left.addLayout(session_row)

        self.frame_count_label = QLabel("프레임 목록 (0장)")
        self.frame_count_label.setStyleSheet("color: #666; font-size: 11px; margin-top: 8px;")
        left.addWidget(self.frame_count_label)
        self.frame_list = QListWidget()
        self.frame_list.currentRowChanged.connect(self._on_frame_row_changed)
        left.addWidget(self.frame_list, stretch=1)

        left.addWidget(QLabel("체크포인트"))
        self.checkpoint_edit = QLineEdit()
        left.addWidget(self.checkpoint_edit)
        btn_ckpt = QPushButton("선택")
        btn_ckpt.clicked.connect(lambda _=False: self._on_browse_checkpoint())
        left.addWidget(btn_ckpt)

        left.addWidget(QLabel("config"))
        self.config_edit = QLineEdit()
        left.addWidget(self.config_edit)
        btn_cfg = QPushButton("선택")
        btn_cfg.clicked.connect(lambda _=False: self._on_browse_config())
        left.addWidget(btn_cfg)

        left.addWidget(QLabel("CAD 모델 (형상 보정 + 정합 공용)"))
        cad_row = QHBoxLayout()
        self.cad_combo = QComboBox()
        cad_row.addWidget(self.cad_combo, stretch=1)
        btn_refresh_cad = QPushButton("↻")
        btn_refresh_cad.setFixedWidth(28)
        btn_refresh_cad.setToolTip("data/cad/ 폴더 다시 스캔")
        btn_refresh_cad.clicked.connect(lambda _=False: self._refresh_cad_list())
        cad_row.addWidget(btn_refresh_cad)
        left.addLayout(cad_row)
        cad_hint = QLabel(f"{DEFAULT_CAD_DIR} 폴더 스캔")
        cad_hint.setStyleSheet("color: #888; font-size: 10px;")
        cad_hint.setWordWrap(True)
        left.addWidget(cad_hint)
        return left_widget

    # --------------------------------------------------------- 2D 검출 그룹
    def _build_group_a(self) -> QGroupBox:
        box = QGroupBox("2D 검출")
        lay = QVBoxLayout(box)

        # ROI
        roi_row = QHBoxLayout()
        self.chk_roi = QCheckBox("ROI 크롭")
        self.chk_roi.setToolTip(
            "켜면 이미지 위에서 마우스로 드래그해 사각형을 그립니다 (숫자로 미세 조정 가능).\n"
            "추론은 이 영역만 잘라서 수행하고, 밖의 물체는 검출되지 않습니다.\n"
            "ROI는 프레임을 바꿔도 유지됩니다 (같은 카메라/빈 고정 가정)."
        )
        self.chk_roi.toggled.connect(self._on_roi_toggled)
        roi_row.addWidget(self.chk_roi)
        self.roi_spins: dict[str, QSpinBox] = {}
        for key in ("x1", "y1", "x2", "y2"):
            roi_row.addWidget(QLabel(key))
            sp = QSpinBox()
            sp.setRange(0, 99999)
            sp.setEnabled(False)
            sp.valueChanged.connect(lambda _=None: self._on_roi_spin_changed())
            self.roi_spins[key] = sp
            roi_row.addWidget(sp)
        self.btn_roi_reset = QPushButton("전체로")
        self.btn_roi_reset.setEnabled(False)
        self.btn_roi_reset.clicked.connect(lambda _=False: self._on_roi_reset())
        roi_row.addWidget(self.btn_roi_reset)
        roi_row.addStretch(1)
        lay.addLayout(roi_row)

        # 검출
        det_row = QHBoxLayout()
        self.btn_run_detect = QPushButton("RTMDet 2D 검출 실행")
        self.btn_run_detect.clicked.connect(lambda _=False: self._on_run_detection())
        det_row.addWidget(self.btn_run_detect)
        det_row.addWidget(QLabel("conf"))
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(int(DEFAULT_SCORE_THRESHOLD * 100))
        self.thresh_slider.setFixedWidth(100)
        self.thresh_label = QLabel(f"{DEFAULT_SCORE_THRESHOLD:.2f}")
        self.thresh_slider.valueChanged.connect(lambda v: self.thresh_label.setText(f"{v / 100:.2f}"))
        det_row.addWidget(self.thresh_slider)
        det_row.addWidget(self.thresh_label)
        det_row.addStretch(1)
        lay.addLayout(det_row)

        # 형상 보정
        refine_row = QHBoxLayout()
        self.chk_refine = QCheckBox("형상 보정 (CAD 투영 + 깊이)")
        self.chk_refine.setChecked(True)
        self.chk_refine.setToolTip(
            "2D 검출 직후 마스크를 선택한 CAD의 투영 실루엣과 깊이에 맞춰 다듬은 뒤 ICP에 넘깁니다.\n"
            "축대칭에 가까운 길쭉한 부품(볼트 등)용. 끄면 모델 원본 마스크를 씁니다."
        )
        self.chk_refine.toggled.connect(lambda _=None: self._on_refine_option_changed())
        refine_row.addWidget(self.chk_refine)
        refine_row.addWidget(QLabel("깊이 허용(mm)"))
        self.spin_depth_tol = QDoubleSpinBox()
        self.spin_depth_tol.setRange(1.0, 50.0)
        self.spin_depth_tol.setSingleStep(1.0)
        self.spin_depth_tol.setValue(RefineParams().depth_tol_mm)
        self.spin_depth_tol.setToolTip("물체 평면에서 이 거리 이내의 점을 물체로 인정. 센서 노이즈가 크면 키운다.")
        refine_row.addWidget(self.spin_depth_tol)
        refine_row.addWidget(QLabel("최소 score"))
        self.spin_min_score = QDoubleSpinBox()
        self.spin_min_score.setRange(0.0, 1.0)
        self.spin_min_score.setSingleStep(0.05)
        self.spin_min_score.setValue(RefineParams().min_score)
        self.spin_min_score.setToolTip("템플릿 적합 점수가 이 값 미만이면 보정을 포기하고 원본 마스크를 유지.")
        refine_row.addWidget(self.spin_min_score)
        refine_row.addStretch(1)
        lay.addLayout(refine_row)

        refine_row2 = QHBoxLayout()
        self.chk_depth_phi = QCheckBox("깊이로 φ 제한")
        self.chk_depth_phi.setChecked(RefineParams().depth_phi)
        self.chk_depth_phi.setToolTip(
            "축이 카메라 쪽으로 기울어진 방향(어느 끝이 가까운지)을 깊이 기울기로 정한다.\n"
            "보정이 자주 포기되면 꺼서 비교해 보세요 (끄면 실루엣만으로 추정)."
        )
        refine_row2.addWidget(self.chk_depth_phi)
        self.chk_overlap = QCheckBox("겹침 정리")
        self.chk_overlap.setChecked(RefineParams().resolve_overlap)
        self.chk_overlap.setToolTip(
            "보정된 마스크끼리 겹친 픽셀을 한쪽에만 남긴다.\n같은 물체의 중복 검출(IoU 0.8 이상)은 건드리지 않는다."
        )
        refine_row2.addWidget(self.chk_overlap)
        self.btn_refine_apply = QPushButton("보정 재적용")
        self.btn_refine_apply.setToolTip("재추론 없이 현재 검출 결과에 보정만 다시 적용 (파라미터 튜닝/CAD 변경용)")
        self.btn_refine_apply.clicked.connect(lambda _=False: self._on_refine_option_changed(force=True))
        refine_row2.addWidget(self.btn_refine_apply)
        refine_row2.addStretch(1)
        lay.addLayout(refine_row2)

        self.refine_status = QLabel("")
        self.refine_status.setWordWrap(True)
        self.refine_status.setStyleSheet("color: #666; font-size: 11px;")
        lay.addWidget(self.refine_status)
        return box

    # --------------------------------------------------------- 3D 정합 그룹
    def _build_group_b(self) -> QGroupBox:
        box = QGroupBox("3D 정합 (ICP)")
        outer = QVBoxLayout(box)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        row = 0
        for title, specs in _B_SECTIONS:
            header = QLabel(title)
            header.setStyleSheet("font-weight: 600; margin-top: 4px;")
            grid.addWidget(header, row, 0, 1, 4)
            row += 1
            for i, spec in enumerate(specs):
                col = (i % 2) * 2
                self._add_param_widget(grid, row, col, spec)
                if i % 2 == 1:
                    row += 1
            if len(specs) % 2 == 1:
                row += 1
            if title.startswith("초기 pose"):
                grid.addWidget(QLabel("정합 알고리즘"), row, 0)
                self.combo_registration_type = QComboBox()
                self.combo_registration_type.addItems(AVAILABLE_REGISTRATION_TYPES)
                default_idx = self.combo_registration_type.findText(ICPParams().registration_type)
                self.combo_registration_type.setCurrentIndex(max(0, default_idx))
                self.combo_registration_type.setToolTip(
                    "open3d_multistage: 위 초기 roll/pitch/yaw로 시작하는 다단계 ICP (기본)\n"
                    "fgr_global: 초기 자세 없이 FPFH 전역 정합 후 ICP 정밀화"
                )
                self.combo_registration_type.currentTextChanged.connect(self._on_registration_type_changed)
                grid.addWidget(self.combo_registration_type, row, 1)
                row += 1
                info = QLabel("다단계 ICP 단계 / normal 반경 / 업샘플링은 '12. 설정' 탭의 저장값을 따릅니다 (open3d_multistage일 때)\n"
                              "뒤집힘 보정: R[2,2] < 0 이면 뒤집힌 가시면 CAD로 재정합 (브라켓 전제)")
                info.setStyleSheet("color: #888; font-size: 10px;")
                info.setWordWrap(True)
                grid.addWidget(info, row, 0, 1, 4)
                row += 1
        outer.addLayout(grid)

        self.fgr_box = QGroupBox("FGR 파라미터 (정합 알고리즘 = fgr_global)")
        fgr_grid = QGridLayout(self.fgr_box)
        fgr_grid.setHorizontalSpacing(10)
        fgr_grid.setVerticalSpacing(4)
        frow = 0
        for i, spec in enumerate(_FGR_SPECS):
            col = (i % 2) * 2
            self._add_param_widget(fgr_grid, frow, col, spec)
            if i % 2 == 1:
                frow += 1
        self.fgr_box.setVisible(self.combo_registration_type.currentText() == "fgr_global")
        outer.addWidget(self.fgr_box)

        btn_row = QHBoxLayout()
        self.btn_run_icp = QPushButton("ICP 정합 실행")
        self.btn_run_icp.clicked.connect(lambda _=False: self._on_run_icp())
        btn_row.addWidget(self.btn_run_icp)
        btn_reset = QPushButton("기본값으로")
        btn_reset.clicked.connect(lambda _=False: self._reset_params())
        btn_row.addWidget(btn_reset)
        btn_load = QPushButton("설정값 불러오기")
        btn_load.setToolTip("'12. 설정' 탭에 저장된 ICP 값으로 아래 입력칸을 채웁니다")
        btn_load.clicked.connect(lambda _=False: self._load_params_from_settings())
        btn_row.addWidget(btn_load)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)
        return box

    def _add_param_widget(self, grid: QGridLayout, row: int, col: int, spec: tuple) -> None:
        field, label, kind, lo, hi, step, decimals, tip = spec
        default = getattr(ICPParams(), field)
        if kind == "bool":
            widget: QWidget = QCheckBox(label)
            widget.setChecked(bool(default))
            if tip:
                widget.setToolTip(tip)
            grid.addWidget(widget, row, col, 1, 2)
        else:
            grid.addWidget(QLabel(label), row, col)
            if kind == "int":
                widget = QSpinBox()
                widget.setRange(int(lo), int(hi))
                widget.setSingleStep(int(step))
                widget.setValue(int(default))
            else:
                widget = QDoubleSpinBox()
                widget.setRange(float(lo), float(hi))
                widget.setSingleStep(float(step))
                widget.setDecimals(int(decimals))
                widget.setValue(float(default))
            widget.setFixedWidth(90)
            if tip:
                widget.setToolTip(tip)
            grid.addWidget(widget, row, col + 1)
        self._param_widgets[field] = widget

    def _on_registration_type_changed(self, algo_type: str) -> None:
        self.fgr_box.setVisible(algo_type == "fgr_global")

    def _collect_params(self) -> ICPParams:
        values: dict = {}
        for field, widget in self._param_widgets.items():
            if isinstance(widget, QCheckBox):
                values[field] = widget.isChecked()
            else:
                values[field] = widget.value()
        values["registration_type"] = self.combo_registration_type.currentText()
        # 이 탭 입력칸에 없는 항목(ICP 단계, normal 반경, 업샘플링 등)은 '12. 설정' 탭에 저장된 값을 따른다.
        return dataclasses.replace(get_settings().snapshot().icp, **values)

    def _load_params_from_settings(self, announce: bool = True) -> None:
        saved = get_settings().snapshot().icp
        for field, widget in self._param_widgets.items():
            value = getattr(saved, field)
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            else:
                widget.setValue(value)
        idx = self.combo_registration_type.findText(saved.registration_type)
        self.combo_registration_type.setCurrentIndex(max(0, idx))
        if announce:
            self._log("입력칸을 '12. 설정' 탭에 저장된 ICP 값으로 채움")

    def _reset_params(self) -> None:
        defaults = ICPParams()
        for field, widget in self._param_widgets.items():
            value = getattr(defaults, field)
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            else:
                widget.setValue(value)
        idx = self.combo_registration_type.findText(defaults.registration_type)
        self.combo_registration_type.setCurrentIndex(max(0, idx))

    # ------------------------------------------------------------ 우측
    def _build_right_panel(self) -> QWidget:
        right_widget = QWidget()
        right_widget.setFixedWidth(280)
        right = QVBoxLayout(right_widget)

        right.addWidget(QLabel("결과 (인스턴스별)"))
        self.result_scroll = QScrollArea()
        self.result_scroll.setWidgetResizable(True)
        self.result_container = QWidget()
        self.result_layout = QVBoxLayout(self.result_container)
        self.result_layout.addStretch(1)
        self.result_scroll.setWidget(self.result_container)
        right.addWidget(self.result_scroll, stretch=1)

        self.btn_open_viewer = QPushButton("3D 뷰어 열기")
        self.btn_open_viewer.clicked.connect(lambda _=False: self._on_open_viewer())
        self.btn_open_viewer.setEnabled(False)
        right.addWidget(self.btn_open_viewer)
        return right_widget

    # =================================================================
    # 체크포인트 / config / CAD
    # =================================================================
    def _prefill_latest_checkpoint(self) -> None:
        if not DEFAULT_CONFIG_PATH.is_file():
            return
        cfg_path = str(DEFAULT_CONFIG_PATH)
        self.config_edit.setText(cfg_path)
        best = find_latest_best_checkpoint(cfg_path)
        if best:
            self.checkpoint_edit.setText(best)
            self._log(f"최신 best 체크포인트 자동 설정: {best}")

    @_guarded
    def _on_browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "체크포인트 선택", "", "PyTorch (*.pth)")
        if path:
            self.checkpoint_edit.setText(path)
            self.checkpoint_edit.setToolTip(path)

    @_guarded
    def _on_browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "config 파일 선택", "", "Python (*.py)")
        if path:
            self.config_edit.setText(path)
            self.config_edit.setToolTip(path)

    def _refresh_cad_list(self) -> None:
        self.cad_combo.clear()
        if not DEFAULT_CAD_DIR.is_dir():
            self._log(f"CAD 폴더 없음: {DEFAULT_CAD_DIR}")
            return
        files = sorted(f for f in DEFAULT_CAD_DIR.iterdir()
                       if f.is_file() and f.suffix.lower() in CAD_EXTS)
        for f in files:
            self.cad_combo.addItem(f.name, str(f))
        # 형상 보정은 길쭉한 부품용이라, 'bolt'가 들어간 CAD가 정확히 하나면 그걸 기본 선택한다.
        bolts = [i for i, f in enumerate(files) if "bolt" in f.name.lower()]
        if len(bolts) == 1:
            self.cad_combo.setCurrentIndex(bolts[0])
        self._log(f"CAD 폴더 스캔: {len(files)}개")

    # =================================================================
    # 세션 / 프레임
    # =================================================================
    def set_session_path(self, session_path: str) -> None:
        """탭0/탭1에서 세션이 정해지면(시그널) 여기도 같이 갱신."""
        self.session_edit.setText(session_path)
        self._load_session(session_path)

    @_guarded
    def _on_browse_session(self) -> None:
        start_dir = str(DEFAULT_DATASET_ROOT) if DEFAULT_DATASET_ROOT.is_dir() else ""
        folder = QFileDialog.getExistingDirectory(self, "세션 폴더 선택", start_dir)
        if not folder:
            return
        self.session_edit.setText(folder)
        self._load_session(folder)

    @_guarded
    def _load_session(self, session_path: str) -> None:
        base = Path(session_path)
        intensity_dir = base / "intensity"
        organized_dir = base / "pointcloud_organized"
        mask_dir = base / "valid_mask"

        if not intensity_dir.is_dir():
            QMessageBox.warning(self, "알림", "이 폴더에는 intensity/ 폴더가 없습니다.")
            return
        if not organized_dir.is_dir() or not mask_dir.is_dir():
            QMessageBox.warning(
                self, "알림",
                "이 세션에는 pointcloud_organized/ 또는 valid_mask/ 폴더가 없습니다.\n"
                "ICP 정합에는 두 폴더가 모두 필요합니다 (collect_dataset.py로 수집한 세션인지 확인하세요)."
            )
            return

        self._session_path = str(base)
        stems = sorted(f.stem for f in intensity_dir.glob("*.png"))
        usable = [s for s in stems if (organized_dir / f"{s}.npy").is_file()
                  and (mask_dir / f"{s}.npy").is_file()]

        self._frame_names = usable
        self.frame_list.blockSignals(True)
        self.frame_list.clear()
        for s in usable:
            self.frame_list.addItem(QListWidgetItem(s))
        self.frame_list.blockSignals(False)
        self.frame_count_label.setText(f"프레임 목록 ({len(usable)}장)")
        self._reset_frame_state()

        if not usable:
            QMessageBox.information(self, "알림", "3D 데이터(pointcloud_organized+valid_mask)가 있는 프레임이 없습니다.")
            self._log(f"세션 로드: {base} (사용 가능 프레임 0장)")
            return
        self._log(f"세션 로드: {base} (사용 가능 프레임 {len(usable)}장)")
        self.frame_list.setCurrentRow(0)

    @_guarded
    def _on_frame_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._frame_names) or not self._session_path:
            return
        name = self._frame_names[row]
        base = Path(self._session_path)
        self._pcd = np.load(base / "pointcloud_organized" / f"{name}.npy")
        self._valid = np.load(base / "valid_mask" / f"{name}.npy").astype(bool)
        self._image_path = str(base / "intensity" / f"{name}.png")
        self._current_frame = name

        self.image_viewer.load_image(self._image_path)
        self._sync_roi_to_image()
        self._clear_detection_state()
        self._log(f"프레임 준비: {name}")

    def _reset_frame_state(self) -> None:
        self._current_frame = None
        self._image_path = None
        self._pcd = None
        self._valid = None
        self._clear_detection_state()

    def _clear_detection_state(self) -> None:
        self._raw_detections = []
        self._detections = []
        self._refine_results = []
        self._refine_applied = False
        self._icp_results = []
        self.refine_status.setText("")
        self._clear_result_panel()
        self.btn_open_viewer.setEnabled(False)

    # =================================================================
    # ROI
    # =================================================================
    def _current_roi(self) -> tuple[int, int, int, int] | None:
        if not self.chk_roi.isChecked():
            return None
        r = tuple(self.roi_spins[k].value() for k in ("x1", "y1", "x2", "y2"))
        return r if (r[2] - r[0] >= ROI_MIN_SIZE and r[3] - r[1] >= ROI_MIN_SIZE) else None

    def _set_roi_spins(self, roi: tuple[int, int, int, int]) -> None:
        for k, v in zip(("x1", "y1", "x2", "y2"), roi):
            sp = self.roi_spins[k]
            sp.blockSignals(True)
            sp.setValue(int(v))
            sp.blockSignals(False)

    def _sync_roi_to_image(self) -> None:
        """프레임이 바뀌면 스핀박스 범위를 이미지 크기에 맞추고, ROI를 (크기가 맞으면 유지) 다시 표시한다."""
        size = self.image_viewer.image_size()
        if size is None:
            return
        w, h = size
        for k, mx in (("x1", w - 1), ("y1", h - 1), ("x2", w), ("y2", h)):
            sp = self.roi_spins[k]
            sp.blockSignals(True)
            sp.setRange(0, mx)
            sp.blockSignals(False)
        if self.chk_roi.isChecked():
            roi = tuple(self.roi_spins[k].value() for k in ("x1", "y1", "x2", "y2"))
            if roi[2] <= roi[0] or roi[3] <= roi[1] or roi[2] > w or roi[3] > h:
                roi = (0, 0, w, h)
                self._set_roi_spins(roi)
            self.image_viewer.set_roi(roi)

    @_guarded
    def _on_roi_toggled(self, on: bool) -> None:
        for sp in self.roi_spins.values():
            sp.setEnabled(on)
        self.btn_roi_reset.setEnabled(on)
        self.image_viewer.set_roi_drawing(on)
        if on:
            size = self.image_viewer.image_size()
            cur = tuple(self.roi_spins[k].value() for k in ("x1", "y1", "x2", "y2"))
            if size is not None and (cur[2] <= cur[0] or cur[3] <= cur[1]):
                self._set_roi_spins((0, 0, size[0], size[1]))
            self._sync_roi_to_image()
            self._log("ROI 사용: 이미지 위에서 드래그해 영역을 지정하세요")
        else:
            self.image_viewer.set_roi(None)
            self._log("ROI 해제: 전체 이미지로 검출합니다")

    def _on_roi_drawn(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self._set_roi_spins((x1, y1, x2, y2))
        self._log(f"ROI 설정: ({x1}, {y1}) ~ ({x2}, {y2}) [{x2 - x1}x{y2 - y1}px]")

    def _on_roi_spin_changed(self) -> None:
        r = tuple(self.roi_spins[k].value() for k in ("x1", "y1", "x2", "y2"))
        if r[2] > r[0] and r[3] > r[1]:
            self.image_viewer.set_roi(r)

    def _on_roi_reset(self) -> None:
        size = self.image_viewer.image_size()
        if size is None:
            return
        self._set_roi_spins((0, 0, size[0], size[1]))
        self.image_viewer.set_roi((0, 0, size[0], size[1]))

    # =================================================================
    # RTMDet 2D 검출
    # =================================================================
    @_guarded
    def _on_run_detection(self) -> bool:
        return self._run_detection()

    def _run_detection(self) -> bool:
        """검출(+형상 보정) 실행. 검출이 1건 이상이면 True."""
        if not self._image_path:
            QMessageBox.warning(self, "알림", "먼저 세션을 선택하고 프레임을 고르세요.")
            return False
        checkpoint = self.checkpoint_edit.text().strip()
        config = self.config_edit.text().strip()
        if not checkpoint or not config:
            QMessageBox.warning(self, "알림", "체크포인트와 config를 모두 지정하세요.")
            return False

        threshold = self.thresh_slider.value() / 100.0
        roi = self._current_roi()
        key = (checkpoint, config)
        if self._detector is None or self._detector_key != key:
            # 모델 로드는 첫 predict 때 한 번만 일어난다 - 같은 체크포인트면 재사용해서 재로딩을 피한다.
            self._detector = Detector(checkpoint_path=checkpoint, config_path=config,
                                      score_threshold=threshold, backend="rtmdet_ins")
            self._detector_key = key

        try:
            detections = self._detector.predict(self._image_path, conf_threshold=threshold, roi=roi)
        except ImportError as exc:
            QMessageBox.critical(self, "추론 엔진 없음", str(exc))
            return False
        except Exception as exc:  # noqa: BLE001
            self._detector = None  # 모델 로드 자체가 실패했을 수 있으니 다음엔 새로 만든다
            QMessageBox.critical(self, "추론 오류", str(exc))
            return False

        detections = [d for d in detections if d.mask is not None]
        self._raw_detections = detections
        roi_txt = f", ROI={roi}" if roi else ""
        self._log(f"2D 검출 완료: {len(detections)}건 (conf≥{threshold:.2f}{roi_txt})")

        self._apply_refine_and_show()
        if not detections:
            QMessageBox.information(self, "알림", "마스크가 있는 검출 결과가 없습니다.")
        return bool(detections)

    # =================================================================
    # 형상 보정
    # =================================================================
    def _refine_enabled(self) -> bool:
        return self.chk_refine.isChecked()

    def _apply_refine_and_show(self) -> None:
        """_raw_detections에서 _detections(+보정 결과)를 새로 만들고 화면/결과 패널을 초기화한다."""
        self._icp_results = []
        self._clear_result_panel()
        self.btn_open_viewer.setEnabled(False)

        self._detections, self._refine_results = self._maybe_refine(self._raw_detections)
        self._refine_applied = bool(self._refine_results)
        self.image_viewer.set_detections(self._detections)

    def _maybe_refine(self, detections: list[Detection]) -> tuple[list[Detection], list[RefineResult]]:
        """보정이 켜져 있으면 (보정된 검출, 보정 결과)를, 아니면 (입력 그대로, [])를 반환한다.

        어떤 이유로든 보정이 불가능하면(CAD 미선택, 점군 없음 등) 팝업으로 알리고 원본을 반환한다
        -> ICP는 항상 돌릴 수 있다. 원본 리스트(_raw_detections)는 건드리지 않는다.
        """
        self.refine_status.setText("")
        if not self._refine_enabled() or not detections:
            return detections, []
        try:
            cad_index = self.cad_combo.currentIndex()
            if cad_index < 0:
                raise ValueError("형상 보정에 쓸 CAD 모델을 선택하세요 (좌측 'CAD 모델').")
            if self._pcd is None or self._valid is None:
                raise ValueError("이 프레임에 점군/유효마스크가 없습니다.")
            bank = get_bank(self.cad_combo.itemData(cad_index))
            intr = estimate_intrinsics_from_organized_pcd(self._pcd, self._valid)
            params = dataclasses.replace(
                RefineParams(),
                depth_tol_mm=float(self.spin_depth_tol.value()),
                min_score=float(self.spin_min_score.value()),
                depth_phi=self.chk_depth_phi.isChecked(),
                resolve_overlap=self.chk_overlap.isChecked(),
            )
            results = refine_masks([d.mask for d in detections], self._pcd, self._valid, intr, bank, params)

            out: list[Detection] = []
            for det, res in zip(detections, results):
                if res.ok:
                    ys, xs = np.nonzero(res.mask)
                    bbox = (float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1)
                    out.append(dataclasses.replace(det, mask=res.mask, bbox=bbox))
                else:
                    out.append(det)  # 포기 -> 원본 그대로

            n_ok = sum(r.ok for r in results)
            self._log(f"형상 보정: {n_ok}/{len(results)}건 적용 (CAD aspect {bank.aspect:.1f}, "
                      f"depth_tol={params.depth_tol_mm:.1f}mm, min_score={params.min_score:.2f}, "
                      f"depth_phi={params.depth_phi})")
            reasons: dict[str, int] = {}
            for i, r in enumerate(results):
                self._log(f"  obj{i}: {r.summary()}")
                if not r.ok:
                    reasons[r.reason] = reasons.get(r.reason, 0) + 1
            status = f"보정 {n_ok}/{len(results)}건 적용"
            if reasons:
                status += " | 원본 유지: " + ", ".join(f"{k}×{v}" for k, v in reasons.items())
            self.refine_status.setText(status)
            if bank.aspect < params.elong_min:
                self._log("경고: CAD가 길쭉하지 않아(aspect 낮음) 형상 보정 방식에 적합하지 않을 수 있습니다.")
            return out, list(results)
        except Exception as exc:  # noqa: BLE001
            self._log(f"형상 보정 실패: {exc}")
            self.refine_status.setText(f"보정 실패: {exc}")
            QMessageBox.warning(self, "형상 보정 실패", f"{exc}\n\n원본 마스크를 사용합니다.")
            return detections, []

    @_guarded
    def _on_refine_option_changed(self, force: bool = False) -> None:
        """보정 체크 토글/재적용: 원본 검출에서 다시 만들고, 낡은 ICP 결과는 지운다."""
        if not self._raw_detections:
            return
        self._apply_refine_and_show()
        state = "보정 마스크" if self._refine_applied else "원본 마스크"
        self._log(f"{state}로 갱신 - ICP를 다시 실행하세요")

    # =================================================================
    # 3D 정합 (ICP)
    # =================================================================
    @_guarded
    def _on_run_icp(self) -> None:
        self._run_icp()

    def _run_icp(self) -> None:
        if not self._detections:
            QMessageBox.warning(self, "알림", "먼저 2D 검출을 실행하세요.")
            return
        if self._pcd is None or self._valid is None:
            QMessageBox.warning(self, "알림", "이 프레임에 점군/유효마스크가 없습니다.")
            return
        cad_index = self.cad_combo.currentIndex()
        if cad_index < 0:
            QMessageBox.warning(self, "알림", "CAD 모델을 선택하세요 (data/cad/ 폴더가 비어있지 않은지 확인).")
            return

        cad_path = self.cad_combo.itemData(cad_index)
        params = self._collect_params()
        try:
            self._ensure_cad_loaded(cad_path, params)  # CAD 준비
            icp_runner._build_default_estimator(params)  # 알고리즘 설정 오류를 인스턴스 루프 전에 한 번만 확인
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "정합 준비 오류", str(exc))
            self._log(f"정합 준비 오류: {exc}")
            return

        self._log(
            f"3D 정합 시작: 인스턴스 {len(self._detections)}개, 알고리즘={params.registration_type} "
            f"(fitness≥{params.fitness_threshold:.2f}, 회전구속 R±{params.roll_limit_deg:.0f} "
            f"P±{params.pitch_limit_deg:.0f} Y±{params.yaw_limit_deg:.0f}deg)"
        )

        results: list[ICPResult] = []
        for i, det in enumerate(self._detections):
            try:
                # 점군 추출 (마스크 침식 포함)
                pts_mm = icp_runner.extract_instance_points_mm(
                    det.mask, self._pcd, self._valid, erode_px=params.mask_erode_px)
                # 이상치 제거 ~ 게이트
                result = icp_runner.run_icp_for_instance(
                    i, pts_mm, self._cad_pcd, self._cad_visible_normal, self._cad_visible_flipped,
                    params=params)
            except Exception as exc:  # noqa: BLE001
                self._log(f"  obj{i} 예외: {exc}\n{traceback.format_exc()}")
                result = ICPResult(instance_id=i, ok=False, error=f"예외: {exc}")
            results.append(result)
            self._log_instance(i, result)

        self._icp_results = results
        self._render_result_panel(results)
        self.btn_open_viewer.setEnabled(len(results) > 0)
        n_ok = sum(r.ok for r in results)
        self._log(f"3D 정합 완료: 성공 {n_ok}/{len(results)}")

    def _log_instance(self, i: int, r: ICPResult) -> None:
        pts = f"점 {r.num_points_scene}→{r.num_points_after_outlier}"
        if r.ok:
            self._log(f"  obj{i} ✓ fitness={r.fitness:.3f} rmse={r.rmse_m * 1000:.2f}mm ({pts}) "
                      f"pick={tuple(round(v, 1) for v in r.pick_point_mm)} mm"
                      + (" [뒤집힘 보정]" if r.was_flipped else ""))
            return
        self._log(f"  obj{i} ✗ {r.error} ({pts})")
        for sl in (r.stage_logs or []):
            voxel_str = f"{sl['voxel']}" if sl["voxel"] is not None else "원본밀도"
            self._log(f"     stage{sl['stage']}({sl['method']}, voxel={voxel_str}): "
                      f"src={sl['n_src']}pt tgt={sl['n_tgt']}pt "
                      f"fitness={sl['fitness']:.3f} rmse={sl['rmse'] * 1000:.2f}mm")

    def _ensure_cad_loaded(self, cad_path: str, params: ICPParams) -> None:
        """CAD 준비: CAD 로드(+축 보정)와 가시면 계산. 값이 바뀔 때만 다시 계산한다."""
        axis = params.cad_axis_correction_deg
        if self._cad_pcd is None or self._cad_path_loaded != cad_path or self._cad_axis_loaded != axis:
            self._log(f"CAD 로드 중: {cad_path} (축보정 R{axis[0]:.0f} P{axis[1]:.0f} Y{axis[2]:.0f}deg)")
            self._cad_pcd = icp_runner.load_cad_as_pcd(cad_path, params)
            self._cad_path_loaded = cad_path
            self._cad_axis_loaded = axis
            self._cad_visible_normal = None

        init_rot = params.init_rotation_deg
        ref_dist = params.cad_hpr_ref_distance_m
        if (self._cad_visible_normal is None
                or self._cad_init_rot_loaded != init_rot
                or self._cad_ref_dist_loaded != ref_dist):
            visible_normal, visible_flipped = icp_runner.build_visible_cad_pair(self._cad_pcd, params)
            total = len(self._cad_pcd.points)
            vis = len(visible_normal.points)
            if total == 0 or vis / total < 0.05:
                self._log(f"⚠ CAD 가시면이 비정상적으로 적음({vis}/{total}점) - "
                          f"'카메라~부품 거리(m)' 값을 확인하세요. 일단 CAD 전체로 폴백합니다.")
                visible_normal = self._cad_pcd
                visible_flipped = self._cad_pcd
            self._cad_visible_normal = visible_normal
            self._cad_visible_flipped = visible_flipped
            self._cad_init_rot_loaded = init_rot
            self._cad_ref_dist_loaded = ref_dist
            self._log(f"CAD 가시면 준비 완료: 전체 {total}점 -> 가시 {vis}점 "
                      f"({100 * vis / max(total, 1):.1f}%, 기준거리={ref_dist:.2f}m)")

    # =================================================================
    # 검출 + 정합 전체 실행
    # =================================================================
    @_guarded
    def _on_run_all(self) -> None:
        if self._run_detection():
            self._run_icp()

    # =================================================================
    # 결과 패널
    # =================================================================
    def _clear_result_panel(self) -> None:
        while self.result_layout.count() > 1:
            item = self.result_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # deleteLater만 쓰면 이벤트 루프가 돌기 전까지 옛 카드가 화면에
                # 남아 새 카드와 겹쳐 보인다 -> 부모에서 즉시 분리 후 삭제 예약.
                widget.setParent(None)
                widget.deleteLater()

    def _a3_line(self, i: int) -> str:
        if not self._refine_applied:
            return "형상 보정 꺼짐 (원본 마스크)"
        if i >= len(self._refine_results):
            return ""
        r = self._refine_results[i]
        if r.ok:
            return (f"형상 보정 ✓ score {r.score:.2f}  φ{r.phi_deg:+.0f}° θ{r.theta_deg:.0f}°  "
                    f"면적 {r.n_before}→{r.n_after}")
        return f"형상 보정 포기: {r.reason}"

    def _render_result_panel(self, results: list[ICPResult]) -> None:
        self._clear_result_panel()
        for r in results:
            card = QFrame()
            card.setObjectName("resultCard")
            card.setFrameShape(QFrame.Shape.StyledPanel)
            # 주의: 셀렉터를 'QFrame'으로 쓰면 자식 QLabel(QFrame 하위 클래스)에도
            # 테두리/padding이 적용돼 글자가 겹쳐 보인다 -> objectName으로 카드만 한정.
            card.setStyleSheet(
                "QFrame#resultCard { border: 1px solid #ddd; border-radius: 6px; }"
            )
            layout = QVBoxLayout(card)
            layout.setContentsMargins(8, 6, 8, 6)

            conf = ""
            if 0 <= r.instance_id < len(self._detections):
                conf = f"  (conf {self._detections[r.instance_id].confidence:.2f})"
            title = QLabel(f"obj{r.instance_id}{conf}")
            title.setStyleSheet("font-weight: 600;")
            layout.addWidget(title)

            a3 = QLabel(self._a3_line(r.instance_id))
            a3.setStyleSheet("color: #888; font-size: 10px;")
            a3.setWordWrap(True)
            layout.addWidget(a3)

            pts = QLabel(f"점 {r.num_points_scene} → outlier 제거 후 {r.num_points_after_outlier}")
            pts.setStyleSheet("color: #888; font-size: 10px;")
            layout.addWidget(pts)

            if r.ok:
                pose = r.pose["euler_deg"]
                pos = r.pick_point_mm
                status = QLabel(f"fitness {r.fitness:.3f}  rmse {r.rmse_m * 1000:.2f}mm")
                status.setStyleSheet("color: #2a8a2a;")
                layout.addWidget(status)
                layout.addWidget(QLabel(f"pick X{pos[0]:+.1f} Y{pos[1]:+.1f} Z{pos[2]:+.1f} mm"))
                layout.addWidget(QLabel(
                    f"R{pose['roll_deg']:+.1f} P{pose['pitch_deg']:+.1f} Y{pose['yaw_deg']:+.1f} deg"))
                if r.was_flipped:
                    flip_label = QLabel("뒤집힘 보정됨")
                    flip_label.setStyleSheet("color: #888; font-size: 10px;")
                    layout.addWidget(flip_label)
            else:
                status = QLabel(r.error or "실패")
                status.setStyleSheet("color: #c0392b;")
                status.setWordWrap(True)
                layout.addWidget(status)
                if r.fitness is not None:
                    layout.addWidget(QLabel(f"fitness {r.fitness:.3f}"))

            self.result_layout.insertWidget(self.result_layout.count() - 1, card)

    # =================================================================
    # 3D 뷰어
    # =================================================================
    @_guarded
    def _on_open_viewer(self) -> None:
        if not self._icp_results or self._cad_pcd is None:
            return

        exclude_mask = None
        if self._detections and self._valid is not None:
            exclude_mask = np.zeros_like(self._valid, dtype=bool)
            for det in self._detections:
                if det.mask is not None:
                    exclude_mask |= det.mask.astype(bool)
        background_pcd = None
        if self._pcd is not None and self._valid is not None:
            background_pcd = icp_runner.build_background_pcd(
                self._pcd, self._valid, exclude_mask=exclude_mask, color_mode="height")

        components = icp_runner.build_scene_components(self._icp_results, self._cad_pcd, background_pcd)
        if not components:
            QMessageBox.information(self, "알림", "표시할 포인트클라우드가 없습니다 (배경/검출 결과 모두 비어있음).")
            return

        import open3d as o3d

        view_dir = Path(tempfile.gettempdir()) / f"icp_view_{self._current_frame or 'frame'}"
        view_dir.mkdir(parents=True, exist_ok=True)

        layers = []
        for i, (name, pcd) in enumerate(components.items()):
            filename = f"layer_{i}.ply"
            o3d.io.write_point_cloud(str(view_dir / filename), pcd, write_ascii=False)
            layers.append({"name": name, "file": filename, "visible": True})

        manifest_path = view_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"layers": layers}, f, ensure_ascii=False, indent=2)

        if self._viewer_process is not None and self._viewer_process.state() != QProcess.ProcessState.NotRunning:
            self._viewer_process.kill()

        self._viewer_process = QProcess(self)
        self._viewer_process.start(
            sys.executable,
            ["-m", "app.core.icp_viewer", str(manifest_path), "--title", f"ICP 결과 - {self._current_frame}"],
        )
        self._log(f"3D 뷰어 실행: {manifest_path} ({len(layers)}개 레이어)")