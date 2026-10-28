"""설정 탭: ICP 정합 / 검출 / 카메라 / TCP 통신 설정을 한곳에서 수정하고 저장한다.

    입력칸 = 편집 중인 값 (아직 적용 전)
    [저장] -> data/app_settings.json 에 기록 + 다른 탭이 쓰는 '적용 중인 값'으로 교체
    프로그램을 다시 켜면 마지막으로 저장한 값이 자동으로 채워진다.

저장된 값이 쓰이는 곳:
    - 5번 TCP 서버: 로봇 요청마다 그 시점에 저장돼 있는 값을 읽는다 (ICP/검출/보정/응답 형식은 서버를
      다시 켜지 않아도 다음 요청부터 반영. 호스트/포트는 서버를 다시 시작해야 한다)
    - 4번 ICP 검출 테스트: [설정값 불러오기] 버튼으로 ICP 입력칸을 이 값으로 채운다

위젯 <-> 설정값 연결은 _bind() 한 곳에서 처리한다 (읽기/쓰기/변경 감지). 파라미터 입력칸은
app/core/icp_param_specs.py 의 표로 만들어서 4번 탭과 같은 항목/범위/툴팁을 공유한다.
"""
from __future__ import annotations

import functools
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea,
    QSpinBox, QTableWidget, QTabWidget, QVBoxLayout, QWidget, QHeaderView,
)

from app.core.app_settings import (
    CAMERA_CONFIG_PATHS, SettingsData, SettingsStore, get_settings, to_project_relative,
)
from app.core.config_patcher import find_latest_best_checkpoint
from app.core.icp_param_specs import B_SECTIONS, EXTRA_SECTIONS, FGR_SPECS, PC_UPSAMPLE_METHODS
from app.core.icp_runner import ICPParams
from app.core.paths import DEFAULT_CAD_DIR, DEFAULT_CONFIG_PATH, resolve_relative_to_project
from app.core.pick_pipeline import SORT_MODES, PickCandidate
from app.core.pick_server import format_ng, format_ok
from app.core.registration import AVAILABLE_REGISTRATION_TYPES

CAD_EXTS = {".stl", ".ply", ".obj"}
TERMINATORS = [("LF  (\\n)", "LF"), ("CRLF  (\\r\\n)", "CRLF"), ("CR  (\\r)", "CR"), ("없음", "NONE")]
DEVICES = ["cuda:0", "cuda:1", "cpu"]

PAGE_ICP, PAGE_DETECT, PAGE_CAMERA, PAGE_TCP = "icp", "detect", "camera", "tcp"


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


@dataclass
class _Binding:
    page: str
    section: str
    field: str
    widget: QWidget
    get: Callable[[], Any]
    set: Callable[[Any], None]


class SettingsTab(QWidget):
    log_message = pyqtSignal(str)
    LOG_PREFIX = "설정 탭"

    def __init__(self, store: SettingsStore | None = None, parent=None):
        super().__init__(parent)
        self._store = store or get_settings()
        self._bindings: list[_Binding] = []
        self._baseline: SettingsData | None = None   # 마지막으로 저장/불러온 입력칸 상태 (변경 감지 기준)
        self._loading = False
        self._ui_ready = False    # UI를 다 만들기 전에는 변경 신호를 무시한다
        self._build_ui()
        self._ui_ready = True
        self._apply(self._store.snapshot())
        # 시작 로그는 이벤트 루프가 돈 뒤에 내보낸다 - 지금 emit하면 MainWindow가 시그널을 연결하기 전이라
        # 콘솔에 아무것도 안 보이고, 설정 파일이 깨진 경우의 경고도 사용자에게 전달되지 않는다.
        QTimer.singleShot(0, self._emit_startup_logs)

    def _emit_startup_logs(self) -> None:
        for w in self._store.load_warnings:
            self._log(f"설정 파일 경고: {w}")
        if self._store.loaded_from_file:
            self._log(f"저장된 설정 불러옴: {self._store.path}")
        else:
            self._log(f"저장된 설정이 없어 기본값으로 시작 (저장하면 {self._store.path} 에 기록)")
        fatal = [w for w in self._store.load_warnings if "읽지 못해" in w]
        if fatal:   # 파일 전체를 못 읽어 기본값으로 시작한 경우: 조용히 넘어가면 설정이 사라진 줄 모른다
            QMessageBox.warning(self, "설정 파일 문제", "\n\n".join(fatal))

    def _log(self, message: str) -> None:
        self.log_message.emit(f"[{self.LOG_PREFIX}] {message}")

    # =================================================================
    # UI 뼈대
    # =================================================================
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self.path_label = QLabel(f"설정 파일: {self._store.path}")
        self.path_label.setStyleSheet("color: #888; font-size: 10px;")
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.path_label)

        self.pages = QTabWidget()
        self.pages.addTab(self._scroll(self._build_icp_page()), "ICP 정합")
        self.pages.addTab(self._scroll(self._build_detect_page()), "검출 / CAD")
        self.pages.addTab(self._scroll(self._build_camera_page()), "카메라")
        self.pages.addTab(self._scroll(self._build_tcp_page()), "TCP 통신")
        self._page_ids = [PAGE_ICP, PAGE_DETECT, PAGE_CAMERA, PAGE_TCP]
        root.addWidget(self.pages, stretch=1)

        bar = QHBoxLayout()
        self.dirty_label = QLabel("")
        bar.addWidget(self.dirty_label)
        bar.addStretch(1)
        self.btn_page_defaults = QPushButton("이 페이지 기본값으로")
        self.btn_page_defaults.setToolTip("현재 페이지의 입력칸만 프로그램 기본값으로 채웁니다 (저장 전까지는 적용되지 않음)")
        self.btn_page_defaults.clicked.connect(lambda _=False: self._on_page_defaults())
        bar.addWidget(self.btn_page_defaults)
        self.btn_revert = QPushButton("변경 취소")
        self.btn_revert.setToolTip("입력칸을 마지막으로 저장한 값으로 되돌립니다")
        self.btn_revert.clicked.connect(lambda _=False: self._on_revert())
        bar.addWidget(self.btn_revert)
        self.btn_save = QPushButton("저장")
        self.btn_save.setStyleSheet("font-weight: 600; padding: 4px 24px;")
        self.btn_save.clicked.connect(lambda _=False: self.save())
        bar.addWidget(self.btn_save)
        root.addLayout(bar)

    @staticmethod
    def _scroll(inner: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidget(inner)
        return area

    # =================================================================
    # 위젯 <-> 설정값 연결
    # =================================================================
    def _bind(self, page: str, section: str, field: str, widget: QWidget,
              get: Callable[[], Any] | None = None, set_: Callable[[Any], None] | None = None) -> QWidget:
        """widget을 SettingsData.<section>.<field> 에 연결한다 (읽기/쓰기/변경 감지)."""
        if get is None or set_ is None:
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                get, set_ = widget.value, widget.setValue
                widget.valueChanged.connect(self._on_edited)
            elif isinstance(widget, QCheckBox):
                get, set_ = widget.isChecked, widget.setChecked
                widget.toggled.connect(self._on_edited)
            elif isinstance(widget, QLineEdit):
                get, set_ = widget.text, widget.setText
                widget.textChanged.connect(self._on_edited)
            elif isinstance(widget, QComboBox):
                get = lambda w=widget: w.currentData() if w.currentData() is not None else w.currentText()  # noqa: E731
                set_ = lambda v, w=widget: self._set_combo(w, v)  # noqa: E731
                widget.currentIndexChanged.connect(self._on_edited)
            else:
                raise TypeError(f"지원하지 않는 위젯: {type(widget).__name__}")
        self._bindings.append(_Binding(page, section, field, widget, get, set_))
        return widget

    @staticmethod
    def _set_combo(combo: QComboBox, value: Any) -> None:
        idx = combo.findData(value)
        if idx < 0:
            idx = combo.findText(str(value))
        if idx < 0 and combo.isEditable():
            combo.setEditText(str(value))
            return
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _collect(self) -> SettingsData:
        """입력칸 값으로 SettingsData를 만든다 (입력칸에 없는 필드는 저장된 값을 그대로 유지)."""
        data = self._store.snapshot()
        for b in self._bindings:
            setattr(getattr(data, b.section), b.field, b.get())
        # 경로는 프로젝트 안이면 상대경로로 저장
        data.detection.checkpoint_path = to_project_relative(data.detection.checkpoint_path.strip())
        data.detection.config_path = to_project_relative(data.detection.config_path.strip())
        data.cad.cad_path = to_project_relative(str(data.cad.cad_path).strip())
        for k in ("host", "trigger_cmd", "ping_cmd", "ok_prefix", "ng_prefix"):
            setattr(data.tcp, k, getattr(data.tcp, k).strip())
        return data

    def _apply(self, data: SettingsData, page: str | None = None) -> None:
        """SettingsData 값을 입력칸에 채운다 (page를 주면 그 페이지 입력칸만)."""
        self._loading = True
        try:
            for b in self._bindings:
                if page is None or b.page == page:
                    b.set(getattr(getattr(data, b.section), b.field))
        finally:
            self._loading = False
        if page is None:
            self._baseline = self._collect()
        self._refresh_state()

    def _on_edited(self, *_args) -> None:
        if self._loading or not self._ui_ready:
            return
        self._refresh_state()

    def _refresh_state(self) -> None:
        if not self._ui_ready:
            return
        self._update_example()
        self._on_registration_changed()
        dirty = self.has_unsaved_changes()
        if dirty:
            self.dirty_label.setText("● 저장되지 않은 변경이 있습니다")
            self.dirty_label.setStyleSheet("color: #c0392b; font-weight: 600;")
        else:
            self.dirty_label.setText("저장된 값과 같음")
            self.dirty_label.setStyleSheet("color: #888;")
        self.btn_revert.setEnabled(dirty)

    def has_unsaved_changes(self) -> bool:
        return self._baseline is not None and self._collect() != self._baseline

    # =================================================================
    # 입력 위젯 생성 도우미
    # =================================================================
    @staticmethod
    def _spin(lo, hi, step, decimals=0, width=100) -> QWidget:
        if decimals == 0 and isinstance(lo, int) and isinstance(step, int):
            w: QWidget = QSpinBox()
            w.setRange(int(lo), int(hi)); w.setSingleStep(int(step))
        else:
            w = QDoubleSpinBox()
            w.setRange(float(lo), float(hi)); w.setSingleStep(float(step)); w.setDecimals(int(decimals))
        w.setFixedWidth(width)
        return w

    def _add_spec_grid(self, grid: QGridLayout, row: int, specs: list[tuple], page: str) -> int:
        """icp_param_specs 표 -> 2열 그리드로 입력칸 생성 + ICPParams 필드에 연결. 다음 row를 반환."""
        for i, (field, label, kind, lo, hi, step, decimals, tip) in enumerate(specs):
            col = (i % 2) * 2
            if kind == "bool":
                w: QWidget = QCheckBox(label)
                grid.addWidget(w, row, col, 1, 2)
            else:
                grid.addWidget(QLabel(label), row, col)
                if kind == "int":
                    w = QSpinBox(); w.setRange(int(lo), int(hi)); w.setSingleStep(int(step))
                else:
                    w = QDoubleSpinBox(); w.setRange(float(lo), float(hi))
                    w.setSingleStep(float(step)); w.setDecimals(int(decimals))
                w.setFixedWidth(100)
                grid.addWidget(w, row, col + 1)
            if tip:
                w.setToolTip(tip)
            self._bind(page, "icp", field, w)
            if i % 2 == 1:
                row += 1
        if len(specs) % 2 == 1:
            row += 1
        return row

    @staticmethod
    def _section_header(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setStyleSheet("font-weight: 600; margin-top: 6px;")
        return lab

    @staticmethod
    def _hint(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setStyleSheet("color: #888; font-size: 10px;")
        lab.setWordWrap(True)
        return lab

    # =================================================================
    # 페이지 1: ICP 정합
    # =================================================================
    def _build_icp_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(4, 1)
        row = 0
        for title, specs in B_SECTIONS:
            grid.addWidget(self._section_header(title), row, 0, 1, 4)
            row += 1
            row = self._add_spec_grid(grid, row, specs, PAGE_ICP)
        for title, specs in EXTRA_SECTIONS:
            grid.addWidget(self._section_header(title), row, 0, 1, 4)
            row += 1
            row = self._add_spec_grid(grid, row, specs, PAGE_ICP)
            if specs and specs[0][0] == "pc_upsample_factor":
                grid.addWidget(QLabel("보간 방식"), row - 1, 2)
                combo = QComboBox(); combo.addItems(PC_UPSAMPLE_METHODS); combo.setFixedWidth(120)
                combo.setToolTip("linear/cubic: 격자 보간. probabilistic: 다중 프레임 촬영의 픽셀별 표준편차에서 "
                                 "재샘플링 (업샘플링 배수 = 점당 샘플 수, 카메라 평균화 프레임 ≥ 2 필요)")
                grid.addWidget(combo, row - 1, 3)
                self._bind(PAGE_ICP, "icp", "pc_upsample_method", combo)
        outer.addLayout(grid)

        # 정합 알고리즘 + (FGR 전용 파라미터)
        outer.addWidget(self._section_header("정합 알고리즘"))
        algo_row = QHBoxLayout()
        self.combo_registration = QComboBox()
        self.combo_registration.addItems(AVAILABLE_REGISTRATION_TYPES)
        self.combo_registration.setToolTip(
            "open3d_multistage: 초기 roll/pitch/yaw로 시작하는 다단계 ICP (기본)\n"
            "fgr_global: 초기 자세 없이 FPFH 전역 정합 후 ICP 정밀화")
        self._bind(PAGE_ICP, "icp", "registration_type", self.combo_registration)
        self.combo_registration.currentTextChanged.connect(lambda _t: self._on_registration_changed())
        algo_row.addWidget(self.combo_registration)
        algo_row.addStretch(1)
        outer.addLayout(algo_row)

        self.stage_box = QGroupBox("다단계 ICP 단계 (open3d_multistage) - 위에서 아래 순서로 실행")
        sb = QVBoxLayout(self.stage_box)
        self.stage_table = QTableWidget(0, 3)
        self.stage_table.setHorizontalHeaderLabels(["voxel (mm, 0 = 원본밀도)", "max_dist (mm)", "max_iter"])
        self.stage_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stage_table.verticalHeader().setDefaultSectionSize(30)
        self.stage_table.setMinimumHeight(150)
        sb.addWidget(self.stage_table)
        btns = QHBoxLayout()
        b_add = QPushButton("단계 추가"); b_add.clicked.connect(lambda _=False: self._add_stage_row(None))
        b_del = QPushButton("선택 단계 삭제"); b_del.clicked.connect(lambda _=False: self._del_stage_row())
        b_def = QPushButton("기본 단계로"); b_def.clicked.connect(
            lambda _=False: (self._set_stages(ICPParams().icp_stages), self._on_edited()))
        for b in (b_add, b_del, b_def):
            btns.addWidget(b)
        btns.addStretch(1)
        sb.addLayout(btns)
        sb.addWidget(self._hint("coarse -> fine 순서로 voxel을 줄이고, 마지막 단계는 0(원본밀도)으로 두는 것을 권장합니다. "
                                "max_dist는 대응점 최대 거리, max_iter는 해당 단계 최대 반복입니다."))
        outer.addWidget(self.stage_box)
        self._bind(PAGE_ICP, "icp", "icp_stages", self.stage_table, get=self._get_stages, set_=self._set_stages)

        self.fgr_box = QGroupBox("FGR 파라미터 (정합 알고리즘 = fgr_global)")
        fg = QGridLayout(self.fgr_box)
        fg.setHorizontalSpacing(12); fg.setVerticalSpacing(4)
        self._add_spec_grid(fg, 0, FGR_SPECS, PAGE_ICP)
        outer.addWidget(self.fgr_box)

        outer.addWidget(self._hint(
            "뒤집힘 보정(R[2,2] < 0 이면 뒤집힌 가시면으로 재정합)은 코드에 고정돼 있어 여기서 켜고 끌 수 없습니다. "
            "CAD 축보정/초기 자세는 CAD가 바뀔 때마다 다시 맞춰야 하는 값입니다."))
        outer.addStretch(1)
        return page

    def _on_registration_changed(self) -> None:
        algo = self.combo_registration.currentText()
        self.stage_box.setVisible(algo == "open3d_multistage")
        self.fgr_box.setVisible(algo == "fgr_global")

    # ---- ICP 단계 표 ----
    def _add_stage_row(self, stage: dict | None) -> None:
        if stage is None:   # 새 행 기본값: 직전 단계를 복사 (행을 추가하기 전에 읽어야 빈 행을 안 읽는다)
            prev = self._get_stages()
            stage = dict(prev[-1]) if prev else {"voxel": 0.003, "max_dist": 0.01, "max_iter": 50}
        row = self.stage_table.rowCount()
        self.stage_table.insertRow(row)
        voxel = QDoubleSpinBox(); voxel.setRange(0.0, 100.0); voxel.setDecimals(2); voxel.setSingleStep(0.5)
        dist = QDoubleSpinBox(); dist.setRange(0.1, 500.0); dist.setDecimals(2); dist.setSingleStep(0.5)
        it = QSpinBox(); it.setRange(1, 10000); it.setSingleStep(10)
        voxel.setValue(0.0 if stage["voxel"] is None else float(stage["voxel"]) * 1000.0)
        dist.setValue(float(stage["max_dist"]) * 1000.0)
        it.setValue(int(stage["max_iter"]))
        for col, w in enumerate((voxel, dist, it)):
            self.stage_table.setCellWidget(row, col, w)
            (w.valueChanged).connect(self._on_edited)
        self._on_edited()

    def _del_stage_row(self) -> None:
        row = self.stage_table.currentRow()
        if row < 0:
            row = self.stage_table.rowCount() - 1
        if self.stage_table.rowCount() <= 1:
            QMessageBox.information(self, "알림", "단계는 최소 1개가 필요합니다.")
            return
        self.stage_table.removeRow(row)
        self._on_edited()

    def _get_stages(self) -> list[dict]:
        out = []
        for r in range(self.stage_table.rowCount()):
            voxel_mm = self.stage_table.cellWidget(r, 0).value()
            dist_mm = self.stage_table.cellWidget(r, 1).value()
            out.append({
                "voxel": None if voxel_mm <= 0 else round(voxel_mm / 1000.0, 6),
                "max_dist": round(dist_mm / 1000.0, 6),
                "max_iter": int(self.stage_table.cellWidget(r, 2).value()),
            })
        return out

    def _set_stages(self, stages: list[dict]) -> None:
        was_loading = self._loading
        self._loading = True    # 행을 다시 만드는 동안의 변경 신호는 무시
        try:
            self.stage_table.setRowCount(0)
            for s in stages:
                self._add_stage_row(dict(s))
        finally:
            self._loading = was_loading

    # =================================================================
    # 페이지 2: 검출 / CAD
    # =================================================================
    def _build_detect_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        # --- RTMDet 검출 ---
        box = QGroupBox("RTMDet-Ins 2D 검출")
        form = QFormLayout(box)
        self.ed_ckpt = QLineEdit()
        form.addRow("체크포인트 (.pth)", self._path_row(self.ed_ckpt, self._browse_checkpoint, "찾아보기", extra=(
            "최신 best", self._find_latest_checkpoint)))
        self.ed_cfg = QLineEdit()
        form.addRow("config (.py)", self._path_row(self.ed_cfg, self._browse_config, "찾아보기"))
        conf = self._spin(0.05, 0.99, 0.05, 2)
        form.addRow("conf 임계값", conf)
        dev = QComboBox(); dev.addItems(DEVICES); dev.setEditable(True); dev.setFixedWidth(120)
        form.addRow("추론 장치", dev)
        self._bind(PAGE_DETECT, "detection", "checkpoint_path", self.ed_ckpt)
        self._bind(PAGE_DETECT, "detection", "config_path", self.ed_cfg)
        self._bind(PAGE_DETECT, "detection", "conf_threshold", conf)
        self._bind(PAGE_DETECT, "detection", "device", dev)
        form.addRow(self._hint("프로젝트 폴더 안의 파일은 상대경로로 저장됩니다 (폴더를 옮겨도 유지). "
                               "체크포인트/장치를 바꾸면 다음 요청에서 모델을 새로 로드합니다."))
        outer.addWidget(box)

        # --- ROI ---
        roi_box = QGroupBox("ROI 크롭 (영역 밖 물체는 검출하지 않음)")
        rl = QHBoxLayout(roi_box)
        chk = QCheckBox("ROI 사용")
        rl.addWidget(chk)
        self._bind(PAGE_DETECT, "detection", "roi_enabled", chk)
        roi_spins = []
        for key, label in (("roi_x1", "x1"), ("roi_y1", "y1"), ("roi_x2", "x2"), ("roi_y2", "y2")):
            rl.addWidget(QLabel(label))
            sp = self._spin(0, 20000, 10, 0, width=80)
            rl.addWidget(sp)
            self._bind(PAGE_DETECT, "detection", key, sp)
            roi_spins.append(sp)
        for sp in roi_spins:
            sp.setEnabled(False)
        chk.toggled.connect(lambda on: [s.setEnabled(on) for s in roi_spins])
        rl.addStretch(1)
        outer.addWidget(roi_box)
        outer.addWidget(self._hint("ROI 좌표는 촬영 이미지의 픽셀 좌표입니다 (4번 탭에서 이미지 위를 드래그하면 값을 확인할 수 있음). "
                                   "한 변이 16px 미만이면 무시하고 전체 이미지를 씁니다."))

        # --- 형상 보정 ---
        ref_box = QGroupBox("형상 보정 (CAD 투영 + 깊이로 마스크 정리)")
        rg = QGridLayout(ref_box)
        rg.setHorizontalSpacing(12)
        chk_ref = QCheckBox("형상 보정 사용")
        rg.addWidget(chk_ref, 0, 0, 1, 2)
        self._bind(PAGE_DETECT, "refine", "enabled", chk_ref)
        rg.addWidget(QLabel("깊이 허용(mm)"), 1, 0)
        tol = self._spin(1.0, 50.0, 0.5, 1)
        rg.addWidget(tol, 1, 1)
        self._bind(PAGE_DETECT, "refine", "depth_tol_mm", tol)
        rg.addWidget(QLabel("최소 score"), 1, 2)
        ms = self._spin(0.0, 1.0, 0.05, 2)
        rg.addWidget(ms, 1, 3)
        self._bind(PAGE_DETECT, "refine", "min_score", ms)
        c1 = QCheckBox("깊이로 φ 제한"); c2 = QCheckBox("겹침 정리")
        rg.addWidget(c1, 2, 0, 1, 2); rg.addWidget(c2, 2, 2, 1, 2)
        self._bind(PAGE_DETECT, "refine", "depth_phi", c1)
        self._bind(PAGE_DETECT, "refine", "resolve_overlap", c2)
        dc = QCheckBox("2D 기반: 표면→중심 깊이 보정")
        dc.setToolTip("4-1 탭 / 5-2 서버(ICP 없이 형상 보정 템플릿으로 픽 포인트 계산)에서, 보이는 표면 depth에\n"
                      "중심까지의 거리(원통 근사)를 더해 CAD 중심 depth로 만듭니다. 끄면 표면 depth 그대로.")
        rg.addWidget(dc, 2, 4, 1, 2)
        self._bind(PAGE_DETECT, "refine", "depth_correct", dc)
        rg.setColumnStretch(6, 1)

        # 의심 마스크 제외 (3번 오프라인 검출 테스트 탭의 '의심 마스크 숨기기'와 같은 기준)
        sus = QCheckBox("의심 마스크 제외")
        sus.setToolTip("아래 조건 중 하나라도 걸린 검출은 정합/좌표 응답에서 뺍니다. 형상 보정을 켠 경우에만 적용됩니다.\n"
                       "(3번 오프라인 검출 테스트 탭에서 같은 값으로 화면에 숨겨 보며 기준을 잡을 수 있습니다)")
        rg.addWidget(sus, 3, 0, 1, 2)
        self._bind(PAGE_DETECT, "refine", "hide_suspect", sus)
        sus_failed = QCheckBox("보정 포기 포함")
        sus_failed.setToolTip("형상 보정을 포기(원본 유지)한 마스크도 의심으로 봅니다.")
        rg.addWidget(sus_failed, 3, 2, 1, 2)
        self._bind(PAGE_DETECT, "refine", "suspect_drop_failed", sus_failed)
        sus_widgets = [sus_failed]
        for row, (field, label, hi, tip) in enumerate((
                ("suspect_min_score", "score ≥", 1.5, "템플릿 적합 점수가 이 값 미만이면 제외"),
                ("suspect_max_change", "변경률 ≤", 9.9, "변경률((추가+삭제)/원본 면적)이 이 값을 넘으면 제외. 원본이 많이 틀렸다는 뜻"),
                ("suspect_min_det_in_tpl", "원본∩템플릿 ≥", 1.0, "원본 검출 중 CAD 투영 안에 있는 비율이 이 값 미만이면 제외")), start=4):
            rg.addWidget(QLabel(label), row, 0)
            sp = self._spin(0.0, hi, 0.05, 2)
            sp.setToolTip(tip)
            rg.addWidget(sp, row, 1)
            self._bind(PAGE_DETECT, "refine", field, sp)
            sus_widgets.append(sp)
        for w in sus_widgets:
            w.setEnabled(False)
        sus.toggled.connect(lambda on: [w.setEnabled(on) for w in sus_widgets])
        outer.addWidget(ref_box)
        outer.addWidget(self._hint("보정이 실패한 인스턴스는 원본 마스크로 계속 진행합니다. 길쭉한 부품(볼트 등)에 맞는 방식이고 "
                                   "CAD가 필요합니다."))

        # --- CAD ---
        cad_box = QGroupBox("CAD 모델 (형상 보정 + 정합 공용)")
        cl = QHBoxLayout(cad_box)
        self.combo_cad = QComboBox()
        self.combo_cad.setMinimumWidth(260)
        cl.addWidget(self.combo_cad, stretch=1)
        b_ref = QPushButton("목록 새로고침"); b_ref.clicked.connect(lambda _=False: self._refresh_cad_list())
        b_browse = QPushButton("찾아보기"); b_browse.clicked.connect(lambda _=False: self._browse_cad())
        cl.addWidget(b_ref); cl.addWidget(b_browse)
        outer.addWidget(cad_box)
        self._refresh_cad_list()
        self._bind(PAGE_DETECT, "cad", "cad_path", self.combo_cad,
                   get=lambda: self.combo_cad.currentData() or "",
                   set_=self._set_cad_value)
        self.combo_cad.currentIndexChanged.connect(self._on_edited)
        outer.addStretch(1)
        return page

    def _path_row(self, edit: QLineEdit, browse: Callable, browse_text: str,
                  extra: tuple[str, Callable] | None = None) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, stretch=1)
        b = QPushButton(browse_text); b.clicked.connect(lambda _=False: browse())
        h.addWidget(b)
        if extra:
            b2 = QPushButton(extra[0]); b2.clicked.connect(lambda _=False: extra[1]())
            h.addWidget(b2)
        return w

    @_guarded
    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "체크포인트 선택", str(resolve_relative_to_project("work_dirs")),
                                              "PyTorch (*.pth)")
        if path:
            self.ed_ckpt.setText(to_project_relative(path))

    @_guarded
    def _browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "config 파일 선택", str(resolve_relative_to_project("configs")),
                                              "Python (*.py)")
        if path:
            self.ed_cfg.setText(to_project_relative(path))

    @_guarded
    def _find_latest_checkpoint(self) -> None:
        cfg = self.ed_cfg.text().strip() or (str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.is_file() else "")
        if not cfg:
            QMessageBox.information(self, "알림", "먼저 config 파일을 지정하세요.")
            return
        cfg_abs = str(resolve_relative_to_project(cfg))
        best = find_latest_best_checkpoint(cfg_abs)
        if not best:
            QMessageBox.information(self, "알림", "이 config의 work_dir에서 best_*.pth를 찾지 못했습니다.")
            return
        if not self.ed_cfg.text().strip():
            self.ed_cfg.setText(to_project_relative(cfg_abs))
        self.ed_ckpt.setText(to_project_relative(best))
        self._log(f"최신 best 체크포인트: {best}")

    # ---- CAD 콤보 ----
    def _refresh_cad_list(self) -> None:
        current = self.combo_cad.currentData()
        self._loading = True
        try:
            self.combo_cad.clear()
            self.combo_cad.addItem("(선택 안 함)", "")
            if DEFAULT_CAD_DIR.is_dir():
                for f in sorted(DEFAULT_CAD_DIR.iterdir()):
                    if f.is_file() and f.suffix.lower() in CAD_EXTS:
                        self.combo_cad.addItem(f.name, to_project_relative(str(f)))
            if current:
                self._set_cad_value(current)
        finally:
            self._loading = False
        self._on_edited()

    def _set_cad_value(self, value: str) -> None:
        value = value or ""
        idx = self.combo_cad.findData(value)
        if idx < 0:   # 목록에 없는 경로(다른 폴더/삭제됨) - 값은 유지하고 표시만 구분
            label = f"{value}  (목록 밖)" if resolve_relative_to_project(value).is_file() else f"{value}  (파일 없음)"
            self.combo_cad.addItem(label, value)
            idx = self.combo_cad.count() - 1
        self.combo_cad.setCurrentIndex(idx)

    @_guarded
    def _browse_cad(self) -> None:
        start = str(DEFAULT_CAD_DIR) if DEFAULT_CAD_DIR.is_dir() else ""
        path, _ = QFileDialog.getOpenFileName(self, "CAD 선택", start, "CAD (*.stl *.ply *.obj)")
        if path:
            self._set_cad_value(to_project_relative(path))

    # =================================================================
    # 페이지 3: 카메라
    # =================================================================
    def _build_camera_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        box = QGroupBox("5번 TCP 서버가 촬영에 쓰는 카메라")
        form = QFormLayout(box)
        cam = QComboBox()
        for name, path in CAMERA_CONFIG_PATHS.items():
            cam.addItem(name, name)
            cam.setItemData(cam.count() - 1, str(path), Qt.ItemDataRole.ToolTipRole)
        cam.setFixedWidth(180)
        form.addRow("카메라 타입", cam)
        self._bind(PAGE_CAMERA, "camera", "camera_type", cam)
        frames = self._spin(1, 30, 1, 0)
        frames.setToolTip("1 = 평균화 없음. 클수록 depth 노이즈가 줄지만 촬영 시간이 비례해서 늘어납니다.")
        form.addRow("평균화 프레임 수", frames)
        self._bind(PAGE_CAMERA, "camera", "num_frames", frames)
        ratio = self._spin(0.05, 1.0, 0.05, 2)
        ratio.setToolTip("픽셀이 유효로 인정되려면 N프레임 중 최소 이 비율 이상에서 유효해야 함. "
                         "낮출수록 빈틈은 메워지지만 노이즈 낀 픽셀도 받아들입니다.")
        form.addRow("min_valid_ratio", ratio)
        self._bind(PAGE_CAMERA, "camera", "min_valid_ratio", ratio)
        outer.addWidget(box)
        outer.addWidget(self._hint(
            "노출시간/동작거리 등 나머지 카메라 값은 configs/camera_config_helios.yaml, camera_config_femto.yaml 값을 그대로 씁니다. "
            "여기서는 평균화 프레임 수와 min_valid_ratio만 덮어씁니다. 카메라 설정을 바꾸고 저장하면 "
            "다음 요청에서 카메라를 새 설정으로 다시 엽니다."))
        outer.addStretch(1)
        return page

    # =================================================================
    # 페이지 4: TCP 통신
    # =================================================================
    def _build_tcp_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        conn = QGroupBox("접속")
        cf = QFormLayout(conn)
        host = QLineEdit(); host.setFixedWidth(180)
        host.setToolTip("0.0.0.0 = 모든 네트워크 인터페이스에서 접속 허용, 127.0.0.1 = 이 PC에서만")
        cf.addRow("호스트(바인드 주소)", host)
        self._bind(PAGE_TCP, "tcp", "host", host)
        port = self._spin(1, 65535, 1, 0)
        cf.addRow("포트 (5번 탭: ICP 기반)", port)
        self._bind(PAGE_TCP, "tcp", "port", port)
        port2d = self._spin(1, 65535, 1, 0)
        port2d.setToolTip("5-2 탭(2D 기반 TCP 서버)이 쓰는 포트. 5번 서버와 동시에 켜려면 다르게 둡니다.")
        cf.addRow("포트 (5-2 탭: 2D 기반)", port2d)
        self._bind(PAGE_TCP, "tcp", "port_2d", port2d)
        port_grip = self._spin(1, 65535, 1, 0)
        port_grip.setToolTip("13번 탭(그리퍼 동작 테스트)이 쓰는 포트. 다른 서버와 동시에 켜려면 다르게 둡니다.")
        cf.addRow("포트 (13번 탭: 그리퍼 테스트)", port_grip)
        self._bind(PAGE_TCP, "tcp", "port_gripper", port_grip)
        cf.addRow(self._hint("호스트/포트는 서버를 다시 시작해야 적용됩니다."))
        outer.addWidget(conn)

        cmd = QGroupBox("요청 명령 (대소문자 무시)")
        cmf = QFormLayout(cmd)
        trig = QLineEdit(); trig.setFixedWidth(180)
        trig.setToolTip("이 문자열을 받으면 촬영 -> 검출 -> 정합 -> 좌표 응답")
        cmf.addRow("좌표 요청", trig)
        self._bind(PAGE_TCP, "tcp", "trigger_cmd", trig)
        ping = QLineEdit(); ping.setFixedWidth(180)
        ping.setToolTip("연결 확인용. 'PONG'을 돌려주고 카메라/모델은 건드리지 않음")
        cmf.addRow("연결 확인", ping)
        self._bind(PAGE_TCP, "tcp", "ping_cmd", ping)
        outer.addWidget(cmd)

        fmt = QGroupBox("응답 형식")
        ff = QFormLayout(fmt)
        ok_p = QLineEdit(); ok_p.setFixedWidth(120)
        ng_p = QLineEdit(); ng_p.setFixedWidth(120)
        sep = QLineEdit(); sep.setFixedWidth(60); sep.setMaxLength(4)
        ff.addRow("성공 접두어", ok_p); ff.addRow("실패 접두어", ng_p); ff.addRow("필드 구분자", sep)
        self._bind(PAGE_TCP, "tcp", "ok_prefix", ok_p)
        self._bind(PAGE_TCP, "tcp", "ng_prefix", ng_p)
        self._bind(PAGE_TCP, "tcp", "field_sep", sep)
        dec = self._spin(0, 6, 1, 0)
        ff.addRow("소수 자릿수", dec)
        self._bind(PAGE_TCP, "tcp", "decimals", dec)
        term = QComboBox()
        for text, val in TERMINATORS:
            term.addItem(text, val)
        term.setFixedWidth(140)
        ff.addRow("응답 끝 문자", term)
        self._bind(PAGE_TCP, "tcp", "terminator", term)
        rpy = QCheckBox("각 후보에 roll/pitch/yaw(deg) 포함")
        ff.addRow(rpy)
        self._bind(PAGE_TCP, "tcp", "include_rpy", rpy)
        mp = self._spin(1, 20, 1, 0)
        mp.setToolTip("한 응답에 담을 최대 픽 후보 수 (정렬된 순서로 앞에서부터)")
        ff.addRow("응답당 최대 후보 수", mp)
        self._bind(PAGE_TCP, "tcp", "max_picks", mp)
        sm = QComboBox()
        for key, text in SORT_MODES.items():
            sm.addItem(text, key)
        ff.addRow("후보 정렬", sm)
        self._bind(PAGE_TCP, "tcp", "sort_mode", sm)
        outer.addWidget(fmt)

        self.example_label = QLabel("")
        self.example_label.setStyleSheet("font-family: Consolas, monospace; background: #f4f4ef; padding: 6px;")
        self.example_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.example_label.setWordWrap(True)
        outer.addWidget(QLabel("응답 예시 (현재 입력값 기준)"))
        outer.addWidget(self.example_label)
        outer.addWidget(self._hint("좌표는 카메라 좌표계 기준 mm / deg 입니다. 로봇 베이스 좌표로의 변환(핸드-아이)은 로봇 쪽에서 처리합니다. "
                                   "실패 코드: NO_DETECTION, NO_VALID_POSE, CAMERA_ERROR, CONFIG_ERROR, DETECTION_ERROR, "
                                   "INTERNAL_ERROR, BUSY(이전 요청 처리 중), UNKNOWN_CMD, TIMEOUT"))

        beh = QGroupBox("동작")
        bf = QVBoxLayout(beh)
        wu = QCheckBox("서버 시작 시 모델/CAD 미리 로드 (첫 요청 지연 방지)")
        ko = QCheckBox("카메라를 켜둔 채 대기 (요청마다 다시 열지 않음 - 빠름)")
        bf.addWidget(wu); bf.addWidget(ko)
        self._bind(PAGE_TCP, "tcp", "warmup_on_start", wu)
        self._bind(PAGE_TCP, "tcp", "keep_camera_open", ko)
        outer.addWidget(beh)
        outer.addStretch(1)
        return page

    def _update_example(self) -> None:
        if not hasattr(self, "example_label"):
            return
        tcp = self._collect().tcp
        demo = [PickCandidate(0, "bolt", 0.93, (-63.4, -29.2, 546.0), (39.6, 21.4, 52.1), 0.91, 1.2),
                PickCandidate(1, "bolt", 0.88, (69.6, 45.1, 559.1), (81.3, 31.0, -0.4), 0.76, 1.1)]
        term = tcp.terminator
        shown = {"LF": "", "CRLF": "", "CR": "", "NONE": " (끝 문자 없음)"}.get(term, "")
        self.example_label.setText(
            f"요청  {tcp.trigger_cmd}\n"
            f"성공  {format_ok(demo, tcp)}{shown}\n"
            f"실패  {format_ng('NO_DETECTION', tcp)}")

    # =================================================================
    # 버튼 동작
    # =================================================================
    def _validate(self, data: SettingsData) -> str | None:
        t = data.tcp
        if not t.trigger_cmd or any(c.isspace() for c in t.trigger_cmd):
            return "'좌표 요청' 명령은 비어 있거나 공백을 포함할 수 없습니다."
        if not t.ping_cmd or any(c.isspace() for c in t.ping_cmd):
            return "'연결 확인' 명령은 비어 있거나 공백을 포함할 수 없습니다."
        if t.trigger_cmd.upper() == t.ping_cmd.upper():
            return "'좌표 요청'과 '연결 확인' 명령이 같을 수 없습니다."
        if not t.ok_prefix or not t.ng_prefix:
            return "성공/실패 접두어는 비워둘 수 없습니다."
        if t.ok_prefix == t.ng_prefix:
            return "성공/실패 접두어가 같으면 로봇이 결과를 구분할 수 없습니다."
        if not t.field_sep:
            return "필드 구분자는 비워둘 수 없습니다."
        if not t.host:
            return "호스트(바인드 주소)를 입력하세요."
        d = data.detection
        if d.roi_enabled and (d.roi_x2 - d.roi_x1 < 16 or d.roi_y2 - d.roi_y1 < 16):
            return "ROI를 쓰려면 x2-x1, y2-y1 이 각각 16 이상이어야 합니다 (또는 ROI 사용을 끄세요)."
        if not data.icp.icp_stages:
            return "ICP 단계가 1개 이상 필요합니다."
        return None

    @_guarded
    def save(self) -> bool:
        """입력칸 값을 검증하고 파일에 저장한다. 저장했으면 True."""
        data = self._collect()
        problem = self._validate(data)
        if problem:
            QMessageBox.warning(self, "저장할 수 없음", problem)
            return False
        warns = []
        if data.icp.registration_type == "open3d_multistage" and data.icp.icp_stages[-1]["voxel"] is not None:
            warns.append("마지막 ICP 단계가 원본밀도(voxel 0)가 아닙니다.")
        if data.cad.cad_path and not resolve_relative_to_project(data.cad.cad_path).is_file():
            warns.append(f"CAD 파일을 찾을 수 없습니다: {data.cad.cad_path}")
        if data.detection.checkpoint_path and not resolve_relative_to_project(data.detection.checkpoint_path).is_file():
            warns.append(f"체크포인트 파일을 찾을 수 없습니다: {data.detection.checkpoint_path}")
        try:
            self._store.save(data)
        except OSError as exc:
            QMessageBox.critical(self, "저장 실패", f"{self._store.path}\n\n{exc}")
            self._log(f"저장 실패: {exc}")
            return False
        self._baseline = self._collect()
        self._refresh_state()
        self._log(f"설정 저장 완료: {self._store.path}")
        for w in warns:
            self._log(f"경고: {w}")
        if warns:
            QMessageBox.information(self, "저장했습니다 (확인 필요)", "\n".join(warns))
        return True

    @_guarded
    def _on_revert(self) -> None:
        self._apply(self._store.snapshot())
        self._log("입력칸을 저장된 값으로 되돌림")

    @_guarded
    def _on_page_defaults(self) -> None:
        page = self._page_ids[self.pages.currentIndex()]
        name = self.pages.tabText(self.pages.currentIndex())
        if QMessageBox.question(self, "기본값으로", f"'{name}' 페이지의 입력칸을 프로그램 기본값으로 채울까요?\n"
                                "(저장 버튼을 누르기 전까지는 적용되지 않습니다)") != QMessageBox.StandardButton.Yes:
            return
        defaults = SettingsData()
        if page == PAGE_DETECT:   # 모델/CAD 경로는 기본값이 빈 값이라 지우지 않는다
            defaults.detection.checkpoint_path = self.ed_ckpt.text()
            defaults.detection.config_path = self.ed_cfg.text()
            defaults.cad.cad_path = self.combo_cad.currentData() or ""
        self._apply(defaults, page=page)
        self._refresh_state()
        self._log(f"'{name}' 페이지 기본값으로 채움 (저장 전)")
