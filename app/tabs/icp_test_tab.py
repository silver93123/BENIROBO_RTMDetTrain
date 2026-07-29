"""탭 4: ICP 정합 테스트.

세션/CAD/프레임 선택, ICP 파라미터, 3D 뷰어는 모든 파이프라인이 공유하는
"헤더" 역할만 한다. 실제 검출/정합 로직은 app/tabs/icp_pipelines/의
파이프라인 탭(RTMDet, RotHead, ...)에 위임한다 - 이 파일은 어떤 detector를
쓰는지, init을 어디서 가져오는지 전혀 모른다. 새 파이프라인이 추가되면
app/tabs/icp_pipelines/AVAILABLE_ICP_PIPELINES에 한 줄만 추가하면 되고,
이 파일은 손댈 필요가 없다.

3D 결과 확인은 별도 open3d 프로세스로 띄운다 (app/core/icp_viewer.py,
QProcess로 실행 - 학습 탭과 동일한 subprocess 패턴).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PyQt6.QtCore import pyqtSignal, Qt, QProcess
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QFileDialog, QSlider, QMessageBox,
    QLineEdit, QComboBox, QScrollArea, QFrame, QGroupBox, QGridLayout,
    QDoubleSpinBox, QSpinBox, QCheckBox, QTabWidget,
)

from app.core.detector import Detection
from app.core.config_patcher import find_latest_best_checkpoint
from app.core.paths import DEFAULT_CONFIG_PATH, DEFAULT_CAD_DIR, DEFAULT_DATASET_ROOT
from app.core.pipeline_context import FrameContext
from app.widgets.image_viewer import ImageViewer
from app.core import icp_runner
from app.core.icp_runner import ICPResult, ICPParams
from app.core.registration import AVAILABLE_REGISTRATION_TYPES
from app.tabs.icp_pipelines import AVAILABLE_ICP_PIPELINES

DEFAULT_SCORE_THRESHOLD = 0.3
CAD_EXTS = {".stl", ".ply", ".obj"}


class ICPTestTab(QWidget):
    log_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checkpoint_path: str | None = None
        self._config_path: str | None = None
        self._session_path: str | None = None
        self._frame_names: list[str] = []          # intensity 파일명(stem) 목록
        self._current_frame: str | None = None
        self._pcd_organized: np.ndarray | None = None  # (H,W,3) mm
        self._valid_mask: np.ndarray | None = None      # (H,W) bool
        self._last_detections: list[Detection] = []
        self._last_icp_results: list[ICPResult] = []
        self._cad_pcd = None
        self._cad_visible_normal = None
        self._cad_visible_flipped = None
        self._cad_path_loaded: str | None = None
        self._cad_axis_loaded: tuple[float, float, float] | None = None
        self._cad_init_rot_loaded: tuple[float, float, float] | None = None
        self._cad_ref_dist_loaded: float | None = None
        self._viewer_process: QProcess | None = None
        self._build_ui()
        self._prefill_latest_checkpoint()
        self._refresh_cad_list()

    # ----------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        # ------------------------------------------------------- 좌측: 세션 / CAD / 프레임
        left = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(230)

        left.addWidget(QLabel("세션 폴더"))
        session_row = QHBoxLayout()
        self.session_edit = QLineEdit()
        self.session_edit.setReadOnly(True)
        self.session_edit.setPlaceholderText("세션 폴더를 선택하세요")
        session_row.addWidget(self.session_edit, stretch=1)
        btn_browse_session = QPushButton("선택")
        btn_browse_session.clicked.connect(self._on_browse_session)
        session_row.addWidget(btn_browse_session)
        left.addLayout(session_row)

        left.addWidget(QLabel("CAD 모델"))
        cad_row = QHBoxLayout()
        self.cad_combo = QComboBox()
        cad_row.addWidget(self.cad_combo, stretch=1)
        btn_refresh_cad = QPushButton("↻")
        btn_refresh_cad.setFixedWidth(28)
        btn_refresh_cad.setToolTip("data/cad/ 폴더 다시 스캔")
        btn_refresh_cad.clicked.connect(self._refresh_cad_list)
        cad_row.addWidget(btn_refresh_cad)
        left.addLayout(cad_row)
        cad_hint = QLabel(f"{DEFAULT_CAD_DIR} 폴더 스캔")
        cad_hint.setStyleSheet("color: #888; font-size: 10px;")
        cad_hint.setWordWrap(True)
        left.addWidget(cad_hint)

        self.frame_count_label = QLabel("프레임 목록 (0장)")
        self.frame_count_label.setStyleSheet("color: #666; font-size: 11px; margin-top: 8px;")
        left.addWidget(self.frame_count_label)

        self.frame_list = QListWidget()
        self.frame_list.currentRowChanged.connect(self._on_frame_row_changed)
        left.addWidget(self.frame_list, stretch=1)

        root.addWidget(left_widget)

        # ------------------------------------------------------- 중앙: 실행 + 미리보기
        center = QVBoxLayout()

        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(QLabel("체크포인트"))
        self.checkpoint_edit = QLineEdit()
        ckpt_row.addWidget(self.checkpoint_edit, stretch=1)
        btn_browse_ckpt = QPushButton("선택")
        btn_browse_ckpt.clicked.connect(self._on_browse_checkpoint)
        ckpt_row.addWidget(btn_browse_ckpt)
        center.addLayout(ckpt_row)

        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("config"))
        self.config_edit = QLineEdit()
        cfg_row.addWidget(self.config_edit, stretch=1)
        btn_browse_cfg = QPushButton("선택")
        btn_browse_cfg.clicked.connect(self._on_browse_config)
        cfg_row.addWidget(btn_browse_cfg)
        center.addLayout(cfg_row)

        # 파이프라인 선택: 콤보박스가 아니라 내부 탭으로 분리한다. RTMDet과
        # RotHead는 detect() 자체가 구조적으로 다른 파이프라인이라(어떤
        # Detector backend를 쓰는지, init을 어디서 가져오는지) 콤보박스 하나로
        # 값만 바꾸는 방식이 안 맞는다 - AVAILABLE_ICP_PIPELINES 참고.
        self.pipeline_tabs = QTabWidget()
        for name, cls in AVAILABLE_ICP_PIPELINES:
            tab_instance = cls()
            self.pipeline_tabs.addTab(tab_instance, name)
        center.addWidget(self.pipeline_tabs)

        run_row = QHBoxLayout()
        self.btn_run_detect = QPushButton("2D 검출 실행")
        self.btn_run_detect.clicked.connect(self._on_run_detection)
        run_row.addWidget(self.btn_run_detect)

        self.btn_run_icp = QPushButton("ICP 정합 실행")
        self.btn_run_icp.clicked.connect(self._on_run_icp)
        run_row.addWidget(self.btn_run_icp)

        run_row.addWidget(QLabel("conf"))
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(int(DEFAULT_SCORE_THRESHOLD * 100))
        self.thresh_slider.setFixedWidth(90)
        self.thresh_label = QLabel(f"{DEFAULT_SCORE_THRESHOLD:.2f}")
        self.thresh_slider.valueChanged.connect(
            lambda v: self.thresh_label.setText(f"{v / 100:.2f}")
        )
        run_row.addWidget(self.thresh_slider)
        run_row.addWidget(self.thresh_label)
        run_row.addStretch(1)
        center.addLayout(run_row)

        # 2026-07 추가: 파라미터 박스(ICP + FGR)가 늘어나면서 화면 안에
        # 다 안 들어오는 문제 - 이 둘만 따로 스크롤 영역에 담아서 높이를
        # 제한한다. 이미지 뷰어는 스크롤 밖에 그대로 둬서 항상 보이게 유지.
        # 이 두 박스는 모든 파이프라인이 공유하는 값이다 (RTMDet 파이프라인은
        # 항상 여기서 init을 가져오고, RotHead 파이프라인은 실패한 인스턴스의
        # fallback으로 가져온다 - app/tabs/icp_pipelines/base.py 참고).
        params_container = QWidget()
        params_layout = QVBoxLayout(params_container)
        params_layout.setContentsMargins(0, 0, 0, 0)
        self.fgr_box = self._build_fgr_params_box()
        params_layout.addWidget(self._build_icp_params_box())
        params_layout.addWidget(self.fgr_box)
        params_layout.addStretch(1)

        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setWidget(params_container)
        params_scroll.setFrameShape(QFrame.Shape.NoFrame)
        params_scroll.setMaximumHeight(320)  # 화면이 더 넓으면 이 값을 늘려도 됨
        params_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        center.addWidget(params_scroll)

        self.image_viewer = ImageViewer()
        center.addWidget(self.image_viewer, stretch=1)
        root.addLayout(center, stretch=2)

        # ------------------------------------------------------- 우측: ICP 결과
        right = QVBoxLayout()
        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setFixedWidth(280)

        right.addWidget(QLabel("ICP 결과"))
        self.result_scroll = QScrollArea()
        self.result_scroll.setWidgetResizable(True)
        self.result_container = QWidget()
        self.result_layout = QVBoxLayout(self.result_container)
        self.result_layout.addStretch(1)
        self.result_scroll.setWidget(self.result_container)
        right.addWidget(self.result_scroll, stretch=1)

        self.btn_open_viewer = QPushButton("3D 뷰어 열기")
        self.btn_open_viewer.clicked.connect(self._on_open_viewer)
        self.btn_open_viewer.setEnabled(False)
        right.addWidget(self.btn_open_viewer)

        root.addWidget(right_widget)

    def _build_icp_params_box(self) -> QGroupBox:
        """voxel/outlier/rotation constraint/xyz max/initial pose를 전부 스핀박스로 노출한다.
        기본값은 icp_runner.ICPParams()와 동일하다."""
        defaults = ICPParams()
        box = QGroupBox("ICP 파라미터 (모든 파이프라인 공유)")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)

        def add_double(row, col, label, value, minimum, maximum, step, decimals=3):
            grid.addWidget(QLabel(label), row, col * 2)
            spin = QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            spin.setValue(value)
            spin.setFixedWidth(80)
            grid.addWidget(spin, row, col * 2 + 1)
            return spin

        def add_int(row, col, label, value, minimum, maximum, step=1):
            grid.addWidget(QLabel(label), row, col * 2)
            spin = QSpinBox()
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setValue(value)
            spin.setFixedWidth(80)
            grid.addWidget(spin, row, col * 2 + 1)
            return spin

        self.spin_mask_erode = add_int(0, 0, "마스크 침식 px", defaults.mask_erode_px, 0, 10)
        self.spin_cad_ref_dist = add_double(0, 1, "카메라~부품 거리(m)", defaults.cad_hpr_ref_distance_m, 0.05, 5.0, 0.01, 3)
        erode_hint = QLabel("마스크 침식: depth 경계 노이즈 완충용 (0=끔).\n"
                             "카메라~부품 거리: CAD 가시면(보이는 면만 정합) 계산 기준값.")
        erode_hint.setStyleSheet("color: #888; font-size: 10px;")
        erode_hint.setWordWrap(True)
        grid.addWidget(erode_hint, 10, 0, 1, 4)

        self.spin_outlier_n = add_int(1, 0, "outlier n", defaults.outlier_nb_neighbors, 1, 200)
        self.spin_outlier_std = add_double(1, 1, "outlier σ", defaults.outlier_std_ratio, 0.1, 10.0, 0.1, 2)

        self.spin_fitness = add_double(2, 0, "fitness ≥", defaults.fitness_threshold, 0.0, 1.0, 0.01, 2)
        self.spin_xyz_max = add_double(2, 1, "XYZ max (m)", defaults.xyz_max_m, 0.1, 10.0, 0.1, 2)

        self.spin_roll_limit = add_double(3, 0, "roll ± deg", defaults.roll_limit_deg, 0.0, 180.0, 1.0, 1)
        self.spin_pitch_limit = add_double(3, 1, "pitch ± deg", defaults.pitch_limit_deg, 0.0, 180.0, 1.0, 1)
        self.spin_yaw_limit = add_double(4, 0, "yaw ± deg", defaults.yaw_limit_deg, 0.0, 180.0, 1.0, 1)

        self.spin_init_roll = add_double(5, 0, "초기 roll deg", defaults.init_roll_deg, -180.0, 180.0, 1.0, 1)
        self.spin_init_pitch = add_double(5, 1, "초기 pitch deg", defaults.init_pitch_deg, -180.0, 180.0, 1.0, 1)
        self.spin_init_yaw = add_double(6, 0, "초기 yaw deg", defaults.init_yaw_deg, -180.0, 180.0, 1.0, 1)

        self.spin_axis_roll = add_double(7, 0, "CAD 축보정 roll", defaults.cad_axis_roll_deg, -180.0, 180.0, 1.0, 1)
        self.spin_axis_pitch = add_double(7, 1, "CAD 축보정 pitch", defaults.cad_axis_pitch_deg, -180.0, 180.0, 1.0, 1)
        self.spin_axis_yaw = add_double(8, 0, "CAD 축보정 yaw", defaults.cad_axis_yaw_deg, -180.0, 180.0, 1.0, 1)
        axis_hint = QLabel("ICP는 회전 없이 중심만 맞추고 시작합니다 - CAD가 실제 물체 방향과\n안 맞으면 여기부터 조정하세요 (CAD 바뀔 때마다 다시 맞춰야 함).")
        axis_hint.setStyleSheet("color: #888; font-size: 10px;")
        axis_hint.setWordWrap(True)
        grid.addWidget(axis_hint, 9, 0, 1, 4)

        btn_reset = QPushButton("기본값으로")
        btn_reset.clicked.connect(lambda: self._reset_icp_params(defaults))
        grid.addWidget(btn_reset, 8, 2, 1, 2)

        # 2026-07 패치: 정합 알고리즘 선택. 목록은 registration 패키지의
        # AVAILABLE_REGISTRATION_TYPES를 그대로 쓴다 - 새 알고리즘이
        # 추가되면 여기 손댈 필요 없이 자동으로 콤보박스에 나타난다.
        # RTMDet 파이프라인은 이 값을 항상 쓰고, RotHead 파이프라인은
        # initial_pose를 못 낸 인스턴스의 fallback으로만 쓴다.
        grid.addWidget(QLabel("정합 알고리즘 (fallback)"), 11, 0)
        self.combo_registration_type = QComboBox()
        self.combo_registration_type.addItems(AVAILABLE_REGISTRATION_TYPES)
        default_idx = self.combo_registration_type.findText(defaults.registration_type)
        self.combo_registration_type.setCurrentIndex(max(0, default_idx))
        grid.addWidget(self.combo_registration_type, 11, 1)
        self.combo_registration_type.currentTextChanged.connect(self._on_registration_type_changed)
        algo_hint = QLabel("파이프라인 탭이 initial pose를 직접 못 내는 경우 여기로 fallback합니다.\n"
                            "RTMDet 탭은 항상, RotHead 탭은 crop 실패 등 예외 상황에서만 씁니다.\n"
                            "알고리즘별 세부 파라미터는 아래(open3d_multistage는 이 박스,\n"
                            "fgr_global은 바로 아래 'FGR 파라미터' 박스)에서 조정합니다.")
        algo_hint.setStyleSheet("color: #888; font-size: 10px;")
        algo_hint.setWordWrap(True)
        grid.addWidget(algo_hint, 12, 0, 1, 4)

        return box

    def _on_registration_type_changed(self, algo_type: str) -> None:
        """FGR 전용 파라미터 박스는 fgr_global 선택시에만 보여준다 -
        open3d_multistage를 쓸 땐 어차피 안 쓰이는 파라미터라 숨겨서 헷갈리지
        않게 한다."""
        self.fgr_box.setVisible(algo_type == "fgr_global")

    def _build_fgr_params_box(self) -> QGroupBox:
        """FGR(fgr_global) 전용 파라미터. registration/fgr_global.py의
        FGRParams 기본값과 1:1로 대응한다. registration_type이
        open3d_multistage일 때는 이 박스 자체가 숨겨진다
        (_on_registration_type_changed 참고)."""
        defaults = ICPParams()
        box = QGroupBox("FGR 파라미터 (registration_type=fgr_global)")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)

        def add_double(row, col, label, value, minimum, maximum, step, decimals=4):
            grid.addWidget(QLabel(label), row, col * 2)
            spin = QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            spin.setValue(value)
            spin.setFixedWidth(90)
            grid.addWidget(spin, row, col * 2 + 1)
            return spin

        self.spin_fgr_voxel = add_double(0, 0, "voxel size (m)", defaults.fgr_voxel_size_m,
                                          0.0005, 0.05, 0.0005, 4)
        self.spin_fgr_normal_factor = add_double(0, 1, "normal 반경 배수", defaults.fgr_normal_radius_factor,
                                                  0.5, 10.0, 0.5, 2)
        self.spin_fgr_fpfh_factor = add_double(1, 0, "FPFH 반경 배수", defaults.fgr_fpfh_radius_factor,
                                                1.0, 20.0, 0.5, 2)
        self.spin_fgr_dist_factor = add_double(1, 1, "대응거리 배수", defaults.fgr_distance_threshold_factor,
                                                0.5, 10.0, 0.5, 2)

        self.check_fgr_refine = QCheckBox("ICP로 정밀화 (refine_with_icp)")
        self.check_fgr_refine.setChecked(defaults.fgr_refine_with_icp)
        grid.addWidget(self.check_fgr_refine, 2, 0, 1, 2)
        self.spin_fgr_refine_dist = add_double(3, 0, "정밀화 max_dist (m)", defaults.fgr_refine_max_dist_m,
                                                0.0005, 0.02, 0.0005, 4)

        self.check_fgr_rotation_prior = QCheckBox("회전 prior 검증 (use_rotation_prior)")
        self.check_fgr_rotation_prior.setChecked(defaults.fgr_use_rotation_prior)
        grid.addWidget(self.check_fgr_rotation_prior, 4, 0, 1, 2)
        self.spin_fgr_max_dev = add_double(5, 0, "최대 허용 편차 (deg)", defaults.fgr_max_rotation_deviation_deg,
                                            0.0, 180.0, 5.0, 1)

        hint = QLabel("voxel size: 부품 크기에 맞춰 조정 (작은 부품은 3mm 이하 권장).\n"
                      "대응거리 배수: 이 값 * voxel size가 FGR이 대응점으로 인정하는 최대 거리.\n"
                      "회전 prior 검증: 결과 회전이 '초기 roll/pitch/yaw'와 너무 다르면\n"
                      "대칭/반복 형상 오탐으로 보고 초기값 기반 ICP로 대체합니다.")
        hint.setStyleSheet("color: #888; font-size: 10px;")
        hint.setWordWrap(True)
        grid.addWidget(hint, 6, 0, 1, 4)

        box.setVisible(defaults.registration_type == "fgr_global")
        return box

    def _reset_icp_params(self, defaults: ICPParams) -> None:
        self.spin_mask_erode.setValue(defaults.mask_erode_px)
        self.spin_cad_ref_dist.setValue(defaults.cad_hpr_ref_distance_m)
        self.spin_outlier_n.setValue(defaults.outlier_nb_neighbors)
        self.spin_outlier_std.setValue(defaults.outlier_std_ratio)
        self.spin_fitness.setValue(defaults.fitness_threshold)
        self.spin_xyz_max.setValue(defaults.xyz_max_m)
        self.spin_roll_limit.setValue(defaults.roll_limit_deg)
        self.spin_pitch_limit.setValue(defaults.pitch_limit_deg)
        self.spin_yaw_limit.setValue(defaults.yaw_limit_deg)
        self.spin_init_roll.setValue(defaults.init_roll_deg)
        self.spin_init_pitch.setValue(defaults.init_pitch_deg)
        self.spin_init_yaw.setValue(defaults.init_yaw_deg)
        self.spin_axis_roll.setValue(defaults.cad_axis_roll_deg)
        self.spin_axis_pitch.setValue(defaults.cad_axis_pitch_deg)
        self.spin_axis_yaw.setValue(defaults.cad_axis_yaw_deg)
        idx = self.combo_registration_type.findText(defaults.registration_type)
        self.combo_registration_type.setCurrentIndex(max(0, idx))
        self.spin_fgr_voxel.setValue(defaults.fgr_voxel_size_m)
        self.spin_fgr_normal_factor.setValue(defaults.fgr_normal_radius_factor)
        self.spin_fgr_fpfh_factor.setValue(defaults.fgr_fpfh_radius_factor)
        self.spin_fgr_dist_factor.setValue(defaults.fgr_distance_threshold_factor)
        self.check_fgr_refine.setChecked(defaults.fgr_refine_with_icp)
        self.spin_fgr_refine_dist.setValue(defaults.fgr_refine_max_dist_m)
        self.check_fgr_rotation_prior.setChecked(defaults.fgr_use_rotation_prior)
        self.spin_fgr_max_dev.setValue(defaults.fgr_max_rotation_deviation_deg)

    def _build_icp_params(self) -> ICPParams:
        """스핀박스 현재 값들로 ICPParams를 만든다 (icp_stages 다단계 리스트는 기본값 유지)."""
        return ICPParams(
            mask_erode_px=self.spin_mask_erode.value(),
            cad_hpr_ref_distance_m=self.spin_cad_ref_dist.value(),
            outlier_nb_neighbors=self.spin_outlier_n.value(),
            outlier_std_ratio=self.spin_outlier_std.value(),
            fitness_threshold=self.spin_fitness.value(),
            xyz_max_m=self.spin_xyz_max.value(),
            cad_axis_roll_deg=self.spin_axis_roll.value(),
            cad_axis_pitch_deg=self.spin_axis_pitch.value(),
            cad_axis_yaw_deg=self.spin_axis_yaw.value(),
            init_roll_deg=self.spin_init_roll.value(),
            init_pitch_deg=self.spin_init_pitch.value(),
            init_yaw_deg=self.spin_init_yaw.value(),
            roll_limit_deg=self.spin_roll_limit.value(),
            pitch_limit_deg=self.spin_pitch_limit.value(),
            yaw_limit_deg=self.spin_yaw_limit.value(),
            registration_type=self.combo_registration_type.currentText(),
            fgr_voxel_size_m=self.spin_fgr_voxel.value(),
            fgr_normal_radius_factor=self.spin_fgr_normal_factor.value(),
            fgr_fpfh_radius_factor=self.spin_fgr_fpfh_factor.value(),
            fgr_distance_threshold_factor=self.spin_fgr_dist_factor.value(),
            fgr_refine_with_icp=self.check_fgr_refine.isChecked(),
            fgr_refine_max_dist_m=self.spin_fgr_refine_dist.value(),
            fgr_use_rotation_prior=self.check_fgr_rotation_prior.isChecked(),
            fgr_max_rotation_deviation_deg=self.spin_fgr_max_dev.value(),
        )

    # ------------------------------------------------------ 체크포인트 prefill
    def _prefill_latest_checkpoint(self) -> None:
        if not DEFAULT_CONFIG_PATH.is_file():
            return
        cfg_path = str(DEFAULT_CONFIG_PATH)
        self.config_edit.setText(cfg_path)
        best = find_latest_best_checkpoint(cfg_path)
        if best:
            self.checkpoint_edit.setText(best)
            self.log_message.emit(f"[ICP 탭] 최신 best 체크포인트 자동 설정: {best}")

    def _on_browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "체크포인트 선택", "", "PyTorch (*.pth)")
        if path:
            self.checkpoint_edit.setText(path)

    def _on_browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "config 파일 선택", "", "Python (*.py)")
        if path:
            self.config_edit.setText(path)

    # ------------------------------------------------------------ CAD 목록
    def _refresh_cad_list(self) -> None:
        self.cad_combo.clear()
        if not DEFAULT_CAD_DIR.is_dir():
            self.log_message.emit(f"[ICP 탭] CAD 폴더 없음: {DEFAULT_CAD_DIR}")
            return
        files = sorted(
            f for f in DEFAULT_CAD_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in CAD_EXTS
        )
        for f in files:
            self.cad_combo.addItem(f.name, str(f))
        self.log_message.emit(f"[ICP 탭] CAD 폴더 스캔: {len(files)}개")

    # -------------------------------------------------------- 세션 / 프레임
    def set_session_path(self, session_path: str) -> None:
        """탭0/탭1에서 세션이 정해지면(시그널) 여기도 같이 갱신."""
        self.session_edit.setText(session_path)
        self._load_session(session_path)

    def _on_browse_session(self) -> None:
        start_dir = str(DEFAULT_DATASET_ROOT) if DEFAULT_DATASET_ROOT.is_dir() else ""
        folder = QFileDialog.getExistingDirectory(self, "세션 폴더 선택", start_dir)
        if not folder:
            return
        self.session_edit.setText(folder)
        self._load_session(folder)

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
        # pointcloud_organized/valid_mask 둘 다 있는 프레임만 사용 가능
        usable = [s for s in stems if (organized_dir / f"{s}.npy").is_file()
                  and (mask_dir / f"{s}.npy").is_file()]

        self._frame_names = usable
        self.frame_list.clear()
        for s in usable:
            self.frame_list.addItem(QListWidgetItem(s))
        self.frame_count_label.setText(f"프레임 목록 ({len(usable)}장)")

        self._reset_frame_state()
        if not usable:
            QMessageBox.information(self, "알림", "3D 데이터(pointcloud_organized+valid_mask)가 있는 프레임이 없습니다.")
            self.log_message.emit(f"[ICP 탭] 세션 로드: {base} (사용 가능 프레임 0장)")
            return

        self.log_message.emit(f"[ICP 탭] 세션 로드: {base} (사용 가능 프레임 {len(usable)}장)")
        self.frame_list.setCurrentRow(0)

    def _on_frame_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._frame_names) or not self._session_path:
            return
        name = self._frame_names[row]
        base = Path(self._session_path)

        self._current_frame = name
        self._pcd_organized = np.load(base / "pointcloud_organized" / f"{name}.npy")
        self._valid_mask = np.load(base / "valid_mask" / f"{name}.npy")

        self.image_viewer.load_image(str(base / "intensity" / f"{name}.png"))
        self._reset_frame_state(keep_frame=True)
        self.log_message.emit(f"[ICP 탭] 프레임 선택: {name}")

    def _reset_frame_state(self, keep_frame: bool = False) -> None:
        if not keep_frame:
            self._current_frame = None
            self._pcd_organized = None
            self._valid_mask = None
        self._last_detections = []
        self._last_icp_results = []
        self._clear_result_panel()
        self.btn_open_viewer.setEnabled(False)

    # ------------------------------------------------------ FrameContext 조립
    def _build_context(self, cad_loaded: bool) -> FrameContext:
        """활성 파이프라인 탭에 넘길 FrameContext를 조립한다.

        cad_loaded=False면 detect() 단계용 (CAD 필드는 아직 None이어도 무관 -
        어떤 파이프라인도 detect()에서 CAD를 쓰지 않는다).
        cad_loaded=True면 register() 직전 - CAD가 이미 로드되어 있어야 함."""
        return FrameContext(
            session_path=self._session_path,
            frame_name=self._current_frame,
            image_path=str(Path(self._session_path) / "intensity" / f"{self._current_frame}.png"),
            pcd_organized_mm=self._pcd_organized,
            valid_mask=self._valid_mask,
            cad_pcd=self._cad_pcd if cad_loaded else None,
            cad_visible_normal=self._cad_visible_normal if cad_loaded else None,
            cad_visible_flipped=self._cad_visible_flipped if cad_loaded else None,
            checkpoint_path=self.checkpoint_edit.text().strip(),
            config_path=self.config_edit.text().strip(),
            score_threshold=self.thresh_slider.value() / 100.0,
        )

    # -------------------------------------------------------------- 2D 검출
    def _on_run_detection(self) -> None:
        if not self._current_frame:
            QMessageBox.warning(self, "알림", "먼저 세션과 프레임을 선택하세요.")
            return
        if not self.checkpoint_edit.text().strip() or not self.config_edit.text().strip():
            QMessageBox.warning(self, "알림", "체크포인트와 config를 모두 지정하세요.")
            return

        active_tab = self.pipeline_tabs.currentWidget()
        ctx = self._build_context(cad_loaded=False)

        try:
            detections = active_tab.detect(ctx)
        except ImportError as exc:
            QMessageBox.critical(self, "추론 엔진 없음", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "추론 오류", str(exc))
            return

        n_with_pose = sum(1 for d in detections if getattr(d, "initial_pose", None) is not None)
        if n_with_pose:
            self.log_message.emit(
                f"[ICP 탭] {active_tab.pipeline_name}: initial_pose 제공 "
                f"{n_with_pose}/{len(detections)}건 (나머지는 fallback 정합 알고리즘 사용)"
            )

        self._last_detections = detections
        self.image_viewer.set_detections(self._last_detections)
        self._last_icp_results = []
        self._clear_result_panel()
        self.btn_open_viewer.setEnabled(False)

        self.log_message.emit(
            f"[ICP 탭] {active_tab.pipeline_name} 검출 완료: {len(detections)}건"
        )
        if not detections:
            QMessageBox.information(self, "알림", "마스크가 있는 검출 결과가 없습니다.")

    # -------------------------------------------------------------- ICP 실행
    def _on_run_icp(self) -> None:
        if not self._last_detections:
            QMessageBox.warning(self, "알림", "먼저 2D 검출을 실행하세요.")
            return
        cad_index = self.cad_combo.currentIndex()
        if cad_index < 0:
            QMessageBox.warning(self, "알림", "CAD 모델을 선택하세요 (data/cad/ 폴더가 비어있지 않은지 확인).")
            return

        cad_path = self.cad_combo.itemData(cad_index)
        params = self._build_icp_params()
        try:
            self._ensure_cad_loaded(cad_path, params)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "CAD 로드 오류", str(exc))
            return

        active_tab = self.pipeline_tabs.currentWidget()
        ctx = self._build_context(cad_loaded=True)

        self.log_message.emit(
            f"[ICP 탭] {active_tab.pipeline_name} 파이프라인 ICP 정합 시작: "
            f"인스턴스 {len(self._last_detections)}개 "
            f"(fitness≥{params.fitness_threshold:.2f}, 회전구속 R±{params.roll_limit_deg:.0f} "
            f"P±{params.pitch_limit_deg:.0f} Y±{params.yaw_limit_deg:.0f}deg)"
        )

        results = active_tab.register(self._last_detections, ctx, params)

        for i, (det, result) in enumerate(zip(self._last_detections, results)):
            if result.ok:
                init_src = "파이프라인 제공" if getattr(det, "initial_pose", None) is not None else "fallback"
                self.log_message.emit(
                    f"[ICP 탭]  obj{i} ✓ fitness={result.fitness:.3f} (init={init_src}) "
                    f"pick={tuple(round(v, 1) for v in result.pick_point_mm)} mm"
                )
            else:
                self.log_message.emit(f"[ICP 탭]  obj{i} ✗ {result.error}")
                if result.stage_logs:
                    for sl in result.stage_logs:
                        voxel_str = f"{sl['voxel']}" if sl["voxel"] is not None else "원본밀도"
                        self.log_message.emit(
                            f"[ICP 탭]     stage{sl['stage']}({sl['method']}, voxel={voxel_str}): "
                            f"src={sl['n_src']}pt tgt={sl['n_tgt']}pt "
                            f"fitness={sl['fitness']:.3f} rmse={sl['rmse']*1000:.2f}mm"
                        )

        self._last_icp_results = results
        self._render_result_panel(results)
        self.btn_open_viewer.setEnabled(any(r.ok for r in results))

        n_ok = sum(r.ok for r in results)
        self.log_message.emit(
            f"[ICP 탭] {active_tab.pipeline_name} ICP 완료: 성공 {n_ok}/{len(results)}"
        )

    def _ensure_cad_loaded(self, cad_path: str, params: ICPParams) -> None:
        axis = params.cad_axis_correction_deg
        if self._cad_pcd is None or self._cad_path_loaded != cad_path or self._cad_axis_loaded != axis:
            self.log_message.emit(
                f"[ICP 탭] CAD 로드 중: {cad_path} (축보정 R{axis[0]:.0f} P{axis[1]:.0f} Y{axis[2]:.0f}deg)"
            )
            self._cad_pcd = icp_runner.load_cad_as_pcd(cad_path, params)
            self._cad_path_loaded = cad_path
            self._cad_axis_loaded = axis
            self._cad_visible_normal = None  # 축보정이 바뀌었으니 가시면도 강제로 다시 계산

        init_rot = params.init_rotation_deg
        ref_dist = params.cad_hpr_ref_distance_m
        if (self._cad_visible_normal is None
                or self._cad_init_rot_loaded != init_rot
                or self._cad_ref_dist_loaded != ref_dist):
            visible_normal, visible_flipped = icp_runner.build_visible_cad_pair(
                self._cad_pcd, params
            )
            total = len(self._cad_pcd.points)
            vis = len(visible_normal.points)
            MIN_VISIBLE_RATIO = 0.05  # 전체의 5% 미만이면 가시면 계산이 잘못된 것으로 보고 폴백
            if total == 0 or vis / total < MIN_VISIBLE_RATIO:
                self.log_message.emit(
                    f"[ICP 탭] ⚠ CAD 가시면이 비정상적으로 적음({vis}/{total}점) - "
                    f"'카메라~부품 거리(m)' 값을 확인하세요. 일단 CAD 전체로 폴백합니다."
                )
                visible_normal = self._cad_pcd
                visible_flipped = self._cad_pcd
            self._cad_visible_normal = visible_normal
            self._cad_visible_flipped = visible_flipped
            self._cad_init_rot_loaded = init_rot
            self._cad_ref_dist_loaded = ref_dist
            self.log_message.emit(
                f"[ICP 탭] CAD 가시면 준비 완료: 전체 {total}점 -> 가시 {vis}점 "
                f"({100*vis/total:.1f}%, 기준거리={ref_dist:.2f}m)"
            )

    # -------------------------------------------------------------- 결과 패널
    def _clear_result_panel(self) -> None:
        while self.result_layout.count() > 1:
            item = self.result_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _render_result_panel(self, results: list[ICPResult]) -> None:
        self._clear_result_panel()
        for r in results:
            card = QFrame()
            card.setFrameShape(QFrame.Shape.StyledPanel)
            card.setStyleSheet(
                "QFrame { border: 1px solid #ddd; border-radius: 6px; padding: 4px; margin-bottom: 4px; }"
            )
            layout = QVBoxLayout(card)
            layout.setContentsMargins(8, 6, 8, 6)

            title = QLabel(f"obj{r.instance_id}")
            title.setStyleSheet("font-weight: 600;")
            layout.addWidget(title)

            if r.ok:
                pose = r.pose["euler_deg"]
                pos = r.pick_point_mm
                status = QLabel(f"fitness {r.fitness:.3f}")
                status.setStyleSheet("color: #2a8a2a;")
                layout.addWidget(status)
                layout.addWidget(QLabel(f"pick X{pos[0]:+.1f} Y{pos[1]:+.1f} Z{pos[2]:+.1f} mm"))
                layout.addWidget(QLabel(
                    f"R{pose['roll_deg']:+.1f} P{pose['pitch_deg']:+.1f} Y{pose['yaw_deg']:+.1f} deg"
                ))
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

    # -------------------------------------------------------------- 3D 뷰어
    def _on_open_viewer(self) -> None:
        if not self._last_icp_results or self._cad_pcd is None:
            return

        exclude_mask = None
        if self._last_detections and self._valid_mask is not None:
            exclude_mask = np.zeros_like(self._valid_mask, dtype=bool)
            for det in self._last_detections:
                if det.mask is not None:
                    exclude_mask |= det.mask.astype(bool)
        background_pcd = None
        if self._pcd_organized is not None and self._valid_mask is not None:
            background_pcd = icp_runner.build_background_pcd(
                self._pcd_organized, self._valid_mask, exclude_mask=exclude_mask
            )

        combined = icp_runner.build_scene_geometry(self._last_icp_results, self._cad_pcd, background_pcd)
        if len(combined.points) == 0:
            QMessageBox.information(self, "알림", "표시할 성공한 인스턴스가 없습니다.")
            return

        tmp_path = Path(tempfile.gettempdir()) / f"icp_view_{self._current_frame or 'frame'}.ply"
        import open3d as o3d
        o3d.io.write_point_cloud(str(tmp_path), combined, write_ascii=False)

        if self._viewer_process is not None and self._viewer_process.state() != QProcess.ProcessState.NotRunning:
            self._viewer_process.kill()

        self._viewer_process = QProcess(self)
        self._viewer_process.start(
            sys.executable,
            ["-m", "app.core.icp_viewer", str(tmp_path), "--title", f"ICP 결과 - {self._current_frame}"],
        )
        self.log_message.emit(f"[ICP 탭] 3D 뷰어 실행: {tmp_path}")