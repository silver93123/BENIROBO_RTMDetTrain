"""탭 0: 데이터 수집.

scripts/collect_dataset.py (빈 피킹 데이터셋 수집 스크립트, organized PCD 지원)를
GUI에서 대화형으로 실행한다. 이 탭은 이 프로젝트(BENIROBO_RTMDetTrain) 안에서만
완결된다 — 스크립트 경로, 카메라 백엔드(src/camera), config 전부 이 저장소
안에 있고, 다른 프로젝트 경로를 참조하지 않는다.

  python scripts/collect_dataset.py --config {config} --out {out} --num {num}
                                     --warmup {warmup} --start-index {start_index}

스크립트는 매 프레임마다 stdin으로 Enter(캡처)/s(스킵)/q(종료)를 기다리므로
'캡처 / 스킵 / 종료' 버튼이 곧 그 세 입력을 stdin에 써주는 역할을 한다.
카메라 워밍업 단계에서는 입력을 기다리지 않으므로 버튼은 비활성 상태로 둔다.

--config 관련 주의:
    collect_dataset.py는 cfg["camera"]["exposure_time_selector"],
    cfg["camera"]["operating_mode"] 를 그대로 읽는다. 즉 model/resolution 같은
    사람 친화적 표기가 아니라, camera.type: lucid_helios 로 시작하는 원본
    스키마(src/camera/lucid_helios.py가 그대로 받아쓰는 형식)여야 한다.
    아래 DEFAULT_HELIOS_CONFIG_TEMPLATE이 그 형태를 그대로 따른다.
"""
from __future__ import annotations

import os
from datetime import datetime

from PyQt6.QtCore import pyqtSignal, QUrl, Qt
from PyQt6.QtGui import QDesktopServices, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QMessageBox, QSpinBox, QCheckBox, QComboBox,
)

from app.core.capture_runner import DataCaptureRunner
from app.core.paths import (
    DEFAULT_DATASET_ROOT, DEFAULT_COLLECT_SCRIPT, DEFAULT_CAMERA_CONFIG_PATH,
    PROJECT_ROOT,
)
from app.widgets.log_console import LogConsole

# src/camera/lucid_helios.py가 그대로 받아쓰는 스키마 (LUCID Helios2 ToF 기준).
DEFAULT_HELIOS_CONFIG_TEMPLATE = """\
# =============================================================================
# Bin Picking Vision 설정 (collect_dataset.py / src.camera 공용) - LUCID Helios2
# =============================================================================
# 모든 길이 단위: mm

camera:
  # 카메라 타입. (lucid_helios | femto_bolt)
  type: lucid_helios

  # 시리얼 번호. null이면 첫 번째로 발견되는 카메라를 사용한다.
  serial: null

  # ---- LUCID Helios 전용 옵션 ----
  # Coord3D_ABCY16: X, Y, Z + Intensity (각 16-bit). 빈 피킹 표준 포맷.
  pixel_format: Coord3D_ABCY16

  # 노출 시간. 어두우면 길게, 모션블러가 우려되면 짧게.
  # 옵션: Exp62_5Us / Exp250Us / Exp1000Us
  exposure_time_selector: Exp250Us

  # 동작 거리 모드. 작업 영역에 맞춰 가장 짧은 것을 선택해야 정밀도가 올라간다.
  # 옵션 예: Distance1500mm / Distance3000mm / Distance4000mm / Distance5000mm / Distance6000mm
  operating_mode: Distance1500mm

  # 카메라 발견/캡처 타임아웃 (ms)
  connect_timeout_ms: 5000
  capture_timeout_ms: 2000
  warmup_frames: 3

  valid_z_range_mm: [300.0, 900.0]
"""

# src/camera/femto_bolt.py가 그대로 받아쓰는 스키마 (Orbbec Femto Bolt 기준, 검증된 드라이버).
DEFAULT_FEMTO_CONFIG_TEMPLATE = """\
# =============================================================================
# Bin Picking Vision 설정 (collect_dataset.py / src.camera 공용) - Femto Bolt
# =============================================================================
# 모든 길이 단위: mm
# Helios2용 config와 스키마가 다르니 섞어쓰지 않도록 주의.
# intensity는 IR 스트림(깊이 센서와 픽셀 정렬됨), XYZ는 PointCloudFilter로 변환.

camera:
  # 카메라 타입.
  type: femto_bolt

  # 시리얼 번호. null이면 첫 번째로 발견되는 카메라를 사용한다.
  serial: null

  # ---- Femto Bolt 전용 옵션 ----
  # depth/IR 스트림 해상도 (반드시 서로 같아야 픽셀 정렬 유지됨).
  #   640x576 (NFOV Unbinned — 약 0.5~3.86 m)
  #   512x512 (WFOV Binned   — 약 0.25~2.5 m, 근거리 유리)
  depth_width: 640
  depth_height: 576
  fps: 15

  # 카메라 캡처 타임아웃 (ms), 워밍업 프레임 수 (초기 프레임 불안정 대비)
  capture_timeout_ms: 2000
  warmup_frames: 5

  # true로 하면 depth 격자에 정렬된 RGB도 추가로 캡처한다 (실험적 옵션).
  # IR/depth 기반 기본 파이프라인과는 분리되어 있어, 이 옵션이 실패해도
  # IR/depth 캡처 자체는 영향받지 않는다. "0. 데이터 수집" 탭에서
  # "RGB PNG (Femto Bolt 전용)" 체크박스도 같이 켜야 실제로 저장된다.
  capture_rgb: false

  # ---- 컬러 카메라 조정 (capture_rgb: true 일 때만 적용) ----
  # Orbbec Viewer의 Color 패널 항목과 같은 이름/의미다. 아래 값은 Viewer 화면의 값을
  # 옮긴 시작점이며, Viewer에서 화면을 보며 맞춘 뒤 그 값을 여기에 적으면 된다.
  #   - 값을 null로 두면 그 항목은 카메라 기본값을 건드리지 않는다.
  #   - 장치가 허용하는 범위를 벗어나면 범위 안으로 조정하고 경고를 남긴다.
  #   - 실제 적용된 값은 수집 로그의 "Femto Bolt color 설정 적용: {...}" 줄에 나온다.
  #   - auto_exposure: true 이면 exposure/gain 은, auto_white_balance: true 이면
  #     white_balance 는 무시된다 (수동 값을 쓰려면 해당 auto 를 false 로).
  color:
    auto_exposure: false
    exposure: 500
    gain: 38
    brightness: 10
    auto_white_balance: true
    white_balance: null      # auto_white_balance: false 로 바꿀 때 8600 등 지정
    sharpness: 40
    saturation: 65
    contrast: 50
    hdr: false
    power_line_freq: 60      # 0=끔, 50, 60 (조명 깜빡임 줄무늬가 보이면 지역 주파수에 맞춤)
    # 필요하면 추가: gamma, hue, backlight_compensation

  # ---- depth 후처리 필터 (Orbbec Viewer의 Depth > Post Processing 과 같은 항목) ----
  # 목적: 깊이가 없는 픽셀(검은 구멍)을 메워 정렬된 RGB의 검은 영역을 줄인다.
  #   - 기본은 RGB 정렬에만 적용하고, 포인트클라우드/organized PCD 는 원본 depth 로 만든다
  #     (홀필링은 없는 깊이를 지어내므로 피킹용 3D 데이터에는 기본적으로 넣지 않음).
  #     PCD 에도 쓰려면 apply_to_pointcloud: true.
  #   - 파라미터는 null 이면 기본값. 로그의 "depth 필터 사용 가능 목록"에 이 장치가
  #     제공하는 필터 이름이 나온다. 필터 하나가 실패해도 경고만 남기고 캡처는 계속된다.
  #   - 적용 순서는 고정: decimation → threshold → noise_removal → spatial → temporal → hole_filling
  #   - decimation 은 해상도를 바꿔 IR/PCD 와 어긋나므로 권장하지 않는다 (켜지 마세요).
  depth_filters:
    enabled: true
    apply_to_pointcloud: false
    threshold:
      enabled: true
      min: 0              # mm
      max: 16000          # mm
    hole_filling:
      enabled: true
      mode: nearest       # top | nearest | furthest
    spatial:
      enabled: false
    temporal:
      enabled: false
    noise_removal:
      enabled: false
    decimation:
      enabled: false

  # ---- intensity(IR) 밝기 매핑 (Orbbec Viewer IR 패널의 Visual Range min/max 와 같은 개념) ----
  # IR 원본은 16bit 이고 8bit 이미지로 줄일 때 [min, max] 구간을 0~255 로 펴 준다.
  #   - min/max 를 둘 다 숫자로 주면 고정 범위: 프레임마다 밝기 기준이 같다 (권장).
  #     Viewer에서 화면을 보며 맞춘 값을 그대로 옮기면 된다 (예: min 0, max 1185).
  #   - 하나라도 null 이면 프레임별 percentile stretch (아래 low~high 백분위)를 쓴다.
  #   - 주의: 이 값을 바꾸면 intensity 이미지의 밝기 분포가 달라진다. 학습 데이터와
  #     추론 때 쓰는 값이 같아야 하므로, 바꾼 뒤 수집한 데이터로 학습한 모델은 같은 값으로 추론하세요.
  intensity:
    min: null
    max: null
    percentile_low: 1
    percentile_high: 99

  valid_z_range_mm: [100.0, 1500.0]
"""


class DataCollectionTab(QWidget):
    log_message = pyqtSignal(str)
    # 수집이 정상 종료되면 --out 폴더 경로를 전달한다 (학습 탭 --dataset 자동 연동용)
    dataset_captured = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.runner: DataCaptureRunner | None = None
        self._current_out_dir: str = ""
        self._intensity_selected: bool = True
        self._rgb_selected: bool = False
        self._last_idx: int | None = None
        self._preview_shown: bool = False
        self._build_ui()

        self.config_edit.setText(str(DEFAULT_CAMERA_CONFIG_PATH))
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_edit.setText(str(DEFAULT_DATASET_ROOT / current_time))

        if not DEFAULT_COLLECT_SCRIPT.is_file():
            self.log_message.emit(
                f"안내: {DEFAULT_COLLECT_SCRIPT} 가 아직 없습니다. "
                "scripts/collect_dataset.py 위치에 스크립트를 넣어주세요."
            )

    # ---------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        script_row = QHBoxLayout()
        script_row.addWidget(QLabel("수집 스크립트"))
        script_label = QLabel(str(DEFAULT_COLLECT_SCRIPT))
        script_label.setStyleSheet("color: #666;")
        script_row.addWidget(script_label, stretch=1)
        layout.addLayout(script_row)

        config_row = QHBoxLayout()
        config_row.addWidget(QLabel("--config"))
        self.config_edit = QLineEdit()
        config_row.addWidget(self.config_edit, stretch=1)
        btn_browse_config = QPushButton("찾아보기")
        btn_browse_config.clicked.connect(self._on_browse_config)
        config_row.addWidget(btn_browse_config)
        btn_new_config = QPushButton("Helios2 기본 config 생성")
        btn_new_config.setToolTip(
            "LUCID Helios2 기준 기본 config.yaml을 configs/ 폴더에 새로 만듭니다."
        )
        btn_new_config.clicked.connect(self._on_create_default_config)
        config_row.addWidget(btn_new_config)
        btn_new_femto_config = QPushButton("Femto Bolt 기본 config 생성")
        btn_new_femto_config.setToolTip(
            "Orbbec Femto Bolt 기준 기본 config.yaml을 configs/ 폴더에 새로 만듭니다."
        )
        btn_new_femto_config.clicked.connect(self._on_create_default_femto_config)
        config_row.addWidget(btn_new_femto_config)
        layout.addLayout(config_row)

        config_hint = QLabel(
            "camera.type: lucid_helios (또는 femto_bolt)로 시작하는 원본 스키마여야 합니다 "
            "(collect_dataset.py가 카메라 타입에 맞는 키를 직접 읽음). "
            "파일이 없으면 위 버튼으로 먼저 만드세요."
        )
        config_hint.setStyleSheet("color: #888; font-size: 11px;")
        config_hint.setWordWrap(True)
        layout.addWidget(config_hint)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("--out (저장 폴더)"))
        self.out_edit = QLineEdit()
        out_row.addWidget(self.out_edit, stretch=1)
        btn_browse_out = QPushButton("폴더 선택")
        btn_browse_out.clicked.connect(self._on_browse_out)
        out_row.addWidget(btn_browse_out)
        btn_open_out = QPushButton("결과 폴더 열기")
        btn_open_out.setToolTip("--out 폴더를 파일 관리자로 엽니다.")
        btn_open_out.clicked.connect(self._on_open_out_folder)
        out_row.addWidget(btn_open_out)
        layout.addLayout(out_row)

        num_row = QHBoxLayout()
        num_row.addWidget(QLabel("--num (캡처할 프레임 수)"))
        self.num_spin = QSpinBox()
        self.num_spin.setRange(1, 100000)
        self.num_spin.setValue(5)
        num_row.addWidget(self.num_spin)

        num_row.addWidget(QLabel("--warmup (워밍업 프레임 수)"))
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 100)
        self.warmup_spin.setValue(3)
        num_row.addWidget(self.warmup_spin)

        num_row.addWidget(QLabel("--start-index (시작 프레임 번호)"))
        self.start_index_spin = QSpinBox()
        self.start_index_spin.setRange(1, 999999)
        self.start_index_spin.setValue(1)
        num_row.addWidget(self.start_index_spin)
        num_row.addStretch(1)
        layout.addLayout(num_row)

        int_row = QHBoxLayout()
        self.chk_int_range = QCheckBox("Intensity 고정 밝기 범위 사용")
        self.chk_int_range.setToolTip(
            "Orbbec Viewer IR 패널의 Visual Range (min/max)와 같은 개념입니다.\n"
            "체크하면 IR 16bit 값의 [최소, 최대] 구간을 0~255로 펴서 저장합니다 (프레임마다 기준이 같음).\n"
            "체크 해제하면 config의 camera.intensity 설정(기본: 프레임별 자동)을 그대로 씁니다.\n"
            "Femto Bolt 전용이며, 학습/추론 때도 같은 값을 써야 일관됩니다."
        )
        int_row.addWidget(self.chk_int_range)
        int_row.addWidget(QLabel("최소"))
        self.int_min_spin = QSpinBox()
        self.int_min_spin.setRange(0, 65534)
        self.int_min_spin.setValue(0)
        self.int_min_spin.setEnabled(False)
        int_row.addWidget(self.int_min_spin)
        int_row.addWidget(QLabel("최대"))
        self.int_max_spin = QSpinBox()
        self.int_max_spin.setRange(1, 65535)
        self.int_max_spin.setValue(1185)
        self.int_max_spin.setEnabled(False)
        int_row.addWidget(self.int_max_spin)
        self.chk_int_range.toggled.connect(self.int_min_spin.setEnabled)
        self.chk_int_range.toggled.connect(self.int_max_spin.setEnabled)
        int_row.addStretch(1)
        layout.addLayout(int_row)

        formats_row = QHBoxLayout()
        formats_row.addWidget(QLabel("저장할 파일 종류:"))
        self.chk_intensity = QCheckBox("Intensity PNG")
        self.chk_intensity.setChecked(True)
        self.chk_pointcloud = QCheckBox("Point Cloud PLY")
        self.chk_pointcloud.setChecked(True)
        self.chk_organized = QCheckBox("Organized PCD (npy)")
        self.chk_organized.setChecked(True)
        self.chk_mask = QCheckBox("Valid Mask (npy)")
        self.chk_mask.setChecked(True)
        self.chk_metadata = QCheckBox("Metadata (json)")
        self.chk_metadata.setChecked(True)
        self.chk_rgb = QCheckBox("RGB PNG (Femto Bolt 전용)")
        self.chk_rgb.setChecked(False)
        self.chk_rgb.setToolTip(
            "config의 camera.capture_rgb: true 일 때만 실제로 저장됩니다 "
            "(꺼져 있으면 이 체크와 무관하게 저장 안 되고 경고만 뜸)."
        )
        for chk in (
            self.chk_intensity, self.chk_pointcloud, self.chk_organized,
            self.chk_mask, self.chk_metadata, self.chk_rgb,
        ):
            formats_row.addWidget(chk)
        formats_row.addStretch(1)
        layout.addLayout(formats_row)
        # RGB 저장을 켜면 미리보기도 RGB로 자동 전환 (수동으로 바꿔도 됨)
        self.chk_rgb.toggled.connect(self._on_rgb_toggled)

        split_row = QHBoxLayout()

        preview_col = QVBoxLayout()
        preview_header = QHBoxLayout()
        preview_header.addWidget(QLabel("최근 캡처 미리보기"))
        self.cmb_preview_source = QComboBox()
        self.cmb_preview_source.addItems(["Intensity", "RGB"])
        self.cmb_preview_source.setToolTip(
            "RGB는 'RGB PNG' 체크 + config의 camera.capture_rgb: true 일 때만 볼 수 있습니다 (Femto Bolt)."
        )
        self.cmb_preview_source.currentIndexChanged.connect(self._on_preview_source_changed)
        preview_header.addWidget(self.cmb_preview_source)
        preview_header.addStretch(1)
        preview_col.addLayout(preview_header)
        self.preview_label = QLabel("아직 캡처된 프레임이 없습니다.")
        self.preview_label.setMinimumHeight(320)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(
            "QLabel { background-color: #1e1e1e; color: #888; border-radius: 6px; }"
        )
        preview_col.addWidget(self.preview_label, stretch=1)
        split_row.addLayout(preview_col, stretch=1)

        log_col = QVBoxLayout()
        log_col.addWidget(QLabel("수집 스크립트 출력 로그"))
        self.capture_log = LogConsole()
        self.capture_log.setMaximumHeight(16777215)  # cards 자리 없앤 만큼 세로로 넉넉히 확장
        log_col.addWidget(self.capture_log, stretch=1)
        split_row.addLayout(log_col, stretch=1)

        layout.addLayout(split_row, stretch=1)

        hint = QLabel(
            "부품 배치를 바꾼 뒤 '캡처'를 누르면 현재 프레임을 저장합니다. "
            "워밍업 중에는 자동으로 진행되니 버튼을 누를 필요가 없습니다."
        )
        hint.setStyleSheet("color: #888; font-size: 11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("수집 시작")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_capture = QPushButton("캡처 (Enter)")
        self.btn_capture.clicked.connect(self._on_capture)
        self.btn_skip = QPushButton("스킵 (s)")
        self.btn_skip.clicked.connect(self._on_skip)
        self.btn_quit = QPushButton("종료 (q)")
        self.btn_quit.clicked.connect(self._on_quit)
        self.btn_force_stop = QPushButton("강제 중단")
        self.btn_force_stop.clicked.connect(self._on_force_stop)
        for b in (self.btn_capture, self.btn_skip, self.btn_quit, self.btn_force_stop):
            b.setEnabled(False)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_capture)
        btn_row.addWidget(self.btn_skip)
        btn_row.addWidget(self.btn_quit)
        btn_row.addWidget(self.btn_force_stop)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------ browse
    def _on_browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "config 파일 선택", str(PROJECT_ROOT / "configs"), "YAML (*.yaml *.yml)"
        )
        if path:
            self.config_edit.setText(path)

    def _on_browse_out(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "저장 폴더 선택", str(DEFAULT_DATASET_ROOT)
        )
        if folder:
            self.out_edit.setText(folder)

    def _on_open_out_folder(self) -> None:
        out_dir = self.out_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "폴더 없음", "--out 저장 폴더를 먼저 지정하세요.")
            return
        if not os.path.isdir(out_dir):
            QMessageBox.warning(
                self, "폴더 없음",
                f"아직 존재하지 않는 폴더입니다:\n{out_dir}\n"
                "수집을 한 번 실행하면 자동으로 생성됩니다.",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(out_dir))

    def _on_create_default_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "기본 config 저장 위치 (LUCID Helios2)", str(DEFAULT_CAMERA_CONFIG_PATH),
            "YAML (*.yaml *.yml)",
        )
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(DEFAULT_HELIOS_CONFIG_TEMPLATE)
        except OSError as exc:
            QMessageBox.critical(self, "생성 실패", f"config 파일을 쓰지 못했습니다:\n{exc}")
            return
        self.config_edit.setText(path)
        self.log_message.emit(f"기본 config 생성: {path} (LUCID Helios2 기준, 필요시 값 수정)")

    def _on_create_default_femto_config(self) -> None:
        default_path = str(PROJECT_ROOT / "configs" / "camera_config_femto.yaml")
        path, _ = QFileDialog.getSaveFileName(
            self, "기본 config 저장 위치 (Femto Bolt)", default_path,
            "YAML (*.yaml *.yml)",
        )
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(DEFAULT_FEMTO_CONFIG_TEMPLATE)
        except OSError as exc:
            QMessageBox.critical(self, "생성 실패", f"config 파일을 쓰지 못했습니다:\n{exc}")
            return
        self.config_edit.setText(path)
        self.log_message.emit(f"기본 config 생성: {path} (Femto Bolt 기준, 필요시 값 수정)")

    # ----------------------------------------------------------- actions
    def _on_start(self) -> None:
        if not DEFAULT_COLLECT_SCRIPT.is_file():
            QMessageBox.warning(
                self, "스크립트 없음",
                f"{DEFAULT_COLLECT_SCRIPT} 를 찾을 수 없습니다.\n"
                "scripts/collect_dataset.py 위치에 스크립트를 넣어주세요.",
            )
            return

        config_path = self.config_edit.text().strip()
        if not config_path or not os.path.isfile(config_path):
            QMessageBox.warning(
                self, "설정 확인 필요",
                "유효한 --config 파일을 지정하세요 (없다면 '기본 config 생성' 사용).",
            )
            return

        out_dir = self.out_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "설정 확인 필요", "--out 저장 폴더를 지정하세요.")
            return

        selected_formats = []
        if self.chk_intensity.isChecked():
            selected_formats.append("intensity")
        if self.chk_pointcloud.isChecked():
            selected_formats.append("pointcloud")
        if self.chk_organized.isChecked():
            selected_formats.append("organized")
        if self.chk_mask.isChecked():
            selected_formats.append("mask")
        if self.chk_metadata.isChecked():
            selected_formats.append("metadata")
        if self.chk_rgb.isChecked():
            selected_formats.append("rgb")
        if not selected_formats:
            QMessageBox.warning(
                self, "설정 확인 필요", "저장할 파일 종류를 최소 하나는 선택하세요.",
            )
            return
        self._intensity_selected = "intensity" in selected_formats
        self._rgb_selected = "rgb" in selected_formats
        self._last_idx = None
        self._preview_shown = False

        if self.chk_int_range.isChecked() and self.int_min_spin.value() >= self.int_max_spin.value():
            QMessageBox.warning(self, "설정 확인 필요", "Intensity 최소값은 최대값보다 작아야 합니다.")
            return

        command = (
            f'python "{DEFAULT_COLLECT_SCRIPT}" '
            f'--config "{config_path}" --out "{out_dir}" '
            f"--num {self.num_spin.value()} "
            f"--warmup {self.warmup_spin.value()} "
            f"--start-index {self.start_index_spin.value()} "
            f'--formats "{",".join(selected_formats)}"'
        )
        if self.chk_int_range.isChecked():
            command += f" --intensity-range {self.int_min_spin.value()},{self.int_max_spin.value()}"

        self._current_out_dir = out_dir
        self.preview_label.clear()
        self.preview_label.setText("아직 캡처된 프레임이 없습니다.")

        self.capture_log.clear()

        self.runner = DataCaptureRunner(self)
        self.runner.log_line.connect(self.capture_log.append_log)
        self.runner.progress.connect(self._on_progress)
        self.runner.finished.connect(self._on_finished)
        self.runner.error.connect(self._on_error)
        # 이 프로젝트 루트 하나로 고정 — 다른 프로젝트 경로에 의존하지 않는다.
        self.runner.start(command, working_dir=str(PROJECT_ROOT))

        self.log_message.emit(f"데이터 수집 시작 (out={out_dir})")
        self.btn_start.setEnabled(False)
        for b in (self.btn_capture, self.btn_skip, self.btn_quit, self.btn_force_stop):
            b.setEnabled(True)

    def _on_capture(self) -> None:
        if self.runner:
            self.runner.send_capture()

    def _on_skip(self) -> None:
        if self.runner:
            self.runner.send_skip()

    def _on_quit(self) -> None:
        if self.runner:
            self.runner.send_quit()

    def _on_force_stop(self) -> None:
        if self.runner:
            self.runner.stop()

    # ----------------------------------------------------------- signals
    def _on_progress(self, data: dict) -> None:
        # 프레임/valid 비율/Z 범위/누적 저장 수는 collect_dataset.py가 이미
        # "✓ saved | ... valid X% | Z ... (median ...)" 형태로 로그에 출력하므로
        # 별도 카드 없이 로그만으로 확인한다. 여기서는 미리보기 갱신만 담당.
        if "valid" in data and "idx" in data:
            self._update_preview(data["idx"])

    def _on_rgb_toggled(self, checked: bool) -> None:
        self.cmb_preview_source.setCurrentText("RGB" if checked else "Intensity")

    def _on_preview_source_changed(self, _index: int) -> None:
        # 소스를 바꾸면 마지막 캡처 프레임을 바로 다시 그린다
        if self._last_idx is not None:
            self._update_preview(self._last_idx)

    def _update_preview(self, idx: int) -> None:
        self._last_idx = idx
        use_rgb = self.cmb_preview_source.currentText() == "RGB"
        if use_rgb:
            if not self._rgb_selected:
                self.preview_label.setText(
                    "RGB PNG 저장이 꺼져 있어 미리보기를 표시할 수 없습니다."
                )
                return
            subdir = "color_rgb"
        else:
            if not self._intensity_selected:
                self.preview_label.setText(
                    "Intensity PNG 저장이 꺼져 있어 미리보기를 표시할 수 없습니다."
                )
                return
            subdir = "intensity"
        if not self._current_out_dir:
            return
        png_path = os.path.join(
            self._current_out_dir, subdir, f"frame_{idx:04d}.png"
        )
        if not os.path.isfile(png_path):
            # 스크립트는 "프레임 N" 줄을 먼저 찍고 나중에 저장하므로, 이 시점엔 파일이
            # 아직 없는 게 정상이다. 문구로 덮어쓰지 말고 직전 이미지를 그대로 둔다.
            return
        pixmap = QPixmap(png_path)
        if pixmap.isNull():
            self.preview_label.setText(f"이미지를 불러오지 못했습니다: {png_path}")
            return
        scaled = pixmap.scaledToHeight(
            self.preview_label.height(), Qt.TransformationMode.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled)
        self._preview_shown = True

    def _on_finished(self, exit_code: int) -> None:
        self.log_message.emit(f"데이터 수집 프로세스 종료 (exit code {exit_code})")
        if (
            not self._preview_shown
            and self._rgb_selected
            and self.cmb_preview_source.currentText() == "RGB"
        ):
            self.preview_label.setText(
                "RGB 파일을 찾지 못했습니다. config의 camera.capture_rgb: true 인지 확인하세요."
            )
        self.btn_start.setEnabled(True)
        for b in (self.btn_capture, self.btn_skip, self.btn_quit, self.btn_force_stop):
            b.setEnabled(False)
        if exit_code == 0:
            self.dataset_captured.emit(self.out_edit.text().strip())

    def _on_error(self, message: str) -> None:
        QMessageBox.critical(self, "데이터 수집 오류", message)