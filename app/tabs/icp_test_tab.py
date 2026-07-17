"""탭 4: ICP 정합 테스트.

탭3(오프라인 검출 테스트)과 같은 Detector로 2D 검출까지 수행한 뒤,
검출 마스크 x 세션의 pointcloud_organized/valid_mask로 인스턴스별 3D 포인트를
뽑아 CAD와 ICP 정합한다. 3D 결과 확인은 별도 open3d 프로세스로 띄운다
(app/core/icp_viewer.py, QProcess로 실행 - 학습 탭과 동일한 subprocess 패턴).
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
    QDoubleSpinBox, QSpinBox,
)

from app.core.detector import Detector, Detection
from app.core.config_patcher import find_latest_best_checkpoint
from app.core.paths import DEFAULT_CONFIG_PATH, DEFAULT_CAD_DIR, DEFAULT_DATASET_ROOT
from app.widgets.image_viewer import ImageViewer
from app.core import icp_runner
from app.core.icp_runner import ICPResult, ICPParams

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
        self._cad_down = None
        self._cad_path_loaded: str | None = None
        self._cad_voxel_loaded: float | None = None
        self._cad_axis_loaded: tuple[float, float, float] | None = None
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

        center.addWidget(self._build_icp_params_box())

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
        box = QGroupBox("ICP 파라미터")
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

        self.spin_voxel_scene = add_double(0, 0, "voxel(scene) m", defaults.voxel_size_scene, 0.0005, 0.05, 0.0005, 4)
        self.spin_voxel_cad = add_double(0, 1, "voxel(CAD) m", defaults.voxel_size_cad, 0.0005, 0.05, 0.0005, 4)

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

        return box

    def _reset_icp_params(self, defaults: ICPParams) -> None:
        self.spin_voxel_scene.setValue(defaults.voxel_size_scene)
        self.spin_voxel_cad.setValue(defaults.voxel_size_cad)
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

    def _build_icp_params(self) -> ICPParams:
        """스핀박스 현재 값들로 ICPParams를 만든다 (icp_stages 다단계 리스트는 기본값 유지)."""
        return ICPParams(
            voxel_size_cad=self.spin_voxel_cad.value(),
            voxel_size_scene=self.spin_voxel_scene.value(),
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

    # -------------------------------------------------------------- 2D 검출
    def _on_run_detection(self) -> None:
        if not self._current_frame:
            QMessageBox.warning(self, "알림", "먼저 세션과 프레임을 선택하세요.")
            return
        checkpoint = self.checkpoint_edit.text().strip()
        config = self.config_edit.text().strip()
        if not checkpoint or not config:
            QMessageBox.warning(self, "알림", "체크포인트와 config를 모두 지정하세요.")
            return

        threshold = self.thresh_slider.value() / 100.0
        image_path = str(Path(self._session_path) / "intensity" / f"{self._current_frame}.png")

        detector = Detector(checkpoint_path=checkpoint, config_path=config, score_threshold=threshold)
        try:
            detections = detector.predict(image_path, conf_threshold=threshold)
        except ImportError as exc:
            QMessageBox.critical(self, "추론 엔진 없음", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "추론 오류", str(exc))
            return

        self._last_detections = [d for d in detections if d.mask is not None]
        skipped = len(detections) - len(self._last_detections)
        self.image_viewer.set_detections(self._last_detections)
        self._last_icp_results = []
        self._clear_result_panel()
        self.btn_open_viewer.setEnabled(False)

        msg = f"검출 완료: {len(self._last_detections)}건 (mask 없음 {skipped}건 제외)"
        self.log_message.emit(f"[ICP 탭] {msg}")
        if not self._last_detections:
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

        self.log_message.emit(
            f"[ICP 탭] ICP 정합 시작: 인스턴스 {len(self._last_detections)}개 "
            f"(fitness≥{params.fitness_threshold:.2f}, 회전구속 R±{params.roll_limit_deg:.0f} "
            f"P±{params.pitch_limit_deg:.0f} Y±{params.yaw_limit_deg:.0f}deg)"
        )
        results: list[ICPResult] = []
        for i, det in enumerate(self._last_detections):
            pts_mm = icp_runner.extract_instance_points_mm(det.mask, self._pcd_organized, self._valid_mask)
            result = icp_runner.run_icp_for_instance(
                i, pts_mm, self._cad_pcd, self._cad_down, params=params
            )
            results.append(result)
            if result.ok:
                self.log_message.emit(
                    f"[ICP 탭]  obj{i} ✓ fitness={result.fitness:.3f} "
                    f"pick={tuple(round(v, 1) for v in result.pick_point_mm)} mm"
                )
            else:
                self.log_message.emit(f"[ICP 탭]  obj{i} ✗ {result.error}")

        self._last_icp_results = results
        self._render_result_panel(results)
        self.btn_open_viewer.setEnabled(any(r.ok for r in results))

        n_ok = sum(r.ok for r in results)
        self.log_message.emit(f"[ICP 탭] ICP 완료: 성공 {n_ok}/{len(results)}")

    def _ensure_cad_loaded(self, cad_path: str, params: ICPParams) -> None:
        axis = params.cad_axis_correction_deg
        if self._cad_pcd is None or self._cad_path_loaded != cad_path or self._cad_axis_loaded != axis:
            self.log_message.emit(
                f"[ICP 탭] CAD 로드 중: {cad_path} (축보정 R{axis[0]:.0f} P{axis[1]:.0f} Y{axis[2]:.0f}deg)"
            )
            self._cad_pcd = icp_runner.load_cad_as_pcd(cad_path, params)
            self._cad_path_loaded = cad_path
            self._cad_axis_loaded = axis
            self._cad_down = None  # voxel/축보정이 바뀌었을 수 있으니 강제로 다시 계산

        if self._cad_down is None or self._cad_voxel_loaded != params.voxel_size_cad:
            self._cad_down = icp_runner.downsample_cad(self._cad_pcd, params)
            self._cad_voxel_loaded = params.voxel_size_cad
            self.log_message.emit(
                f"[ICP 탭] CAD 준비 완료: {len(self._cad_pcd.points)}점 "
                f"(다운샘플 {len(self._cad_down.points)}점, voxel={params.voxel_size_cad})"
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