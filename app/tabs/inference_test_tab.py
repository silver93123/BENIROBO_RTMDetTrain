"""탭 3: 오프라인 2D 검출 테스트 (ICP/3D pose 없음).

실제 파이프라인(3_Detect_and_PickPoint.py)과 동일하게 체크포인트 + config
두 가지를 모두 지정해야 RTMDetInferencer를 만들 수 있다.

이미지는 한 장씩 파일 다이얼로그로 불러오는 대신, 폴더를 지정하면 그 안의
이미지들을 목록으로 펼쳐두고 목록에서 골라 미리보기 -> 추론 실행하는 방식.
[카메라로 촬영]으로 바로 찍은 프레임도 같은 목록에 추가돼 똑같이 추론/형상 보정할 수 있다
(촬영본은 점군을 메모리에 같이 보관하므로 세션 폴더 없이도 형상 보정이 된다).

체크포인트 / config / CAD 경로와 카메라 값은 이 탭에서 입력하지 않고 '12. 설정' 탭에
저장된 값을 쓴다 (app/core/detect_test_common.py).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QFileDialog, QSlider, QMessageBox, QLineEdit,
    QCheckBox, QDoubleSpinBox, QSpinBox,
)

from app.core.app_settings import get_settings
from app.core.camera_intrinsics import estimate_intrinsics_from_organized_pcd
from app.core.detect_test_common import (
    busy_cursor, capture_frame, load_settings_paths, new_capture_name, save_intensity_png, summary_text,
)
from app.core.detector import Detector, Detection
from app.widgets.image_viewer import ImageViewer
from src.detection.mask_refine import RefineParams, RefineResult, get_bank, load_session_arrays, refine_masks

DEFAULT_SCORE_THRESHOLD = 0.3  # 실제 파이프라인 스크립트의 SCORE_THRESHOLD와 동일
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_BOX_OPACITY = 50  # % - 박스가 물체를 가리지 않게 기본은 반투명


class InferenceTestTab(QWidget):
    log_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checkpoint_path: str | None = None
        self._config_path: str | None = None
        self._image_path: str | None = None
        self._image_paths: list[str] = []  # 현재 폴더의 이미지 전체 경로 (image_list와 index 일치)
        self._last_detections: list[Detection] = []  # 모델 원본 출력 (export는 항상 이것)
        # 형상 보정 결과 (_last_detections와 index 일치). 보정 전이면 빈 리스트.
        self._refined_detections: list[Detection] = []
        self._refine_results: list[RefineResult] = []
        # 카메라 촬영본: 임시 PNG 경로 -> (표시 이름, 점군 (H,W,3) mm, 유효마스크). 이번 실행 중 메모리에만 보관.
        self._captures: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
        self._build_ui()
        self._refresh_settings_label()
        get_settings().changed.connect(self._refresh_settings_label)   # 설정 탭에서 저장하면 안내 문구 갱신

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        # ------------------------------------------------------- 좌측: 폴더 + 이미지 목록
        left = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(220)

        left.addWidget(QLabel("이미지 폴더"))
        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("폴더를 선택하세요")
        self.folder_edit.setReadOnly(True)
        folder_row.addWidget(self.folder_edit, stretch=1)
        btn_browse_folder = QPushButton("폴더 선택")
        btn_browse_folder.clicked.connect(self._on_browse_image_folder)
        folder_row.addWidget(btn_browse_folder)
        left.addLayout(folder_row)

        self.btn_capture = QPushButton("카메라로 촬영")
        self.btn_capture.setToolTip(
            "'12. 설정' 탭 > '카메라'의 값(카메라 타입, 평균화 프레임 수)으로 한 장 촬영해 아래 목록에 추가합니다.\n"
            "점군도 함께 보관하므로 형상 보정도 쓸 수 있습니다 (이번 실행 중 메모리에만 보관).\n"
            "촬영하는 동안 화면이 잠깐 멈춥니다 (평균화 프레임이 많을수록 길어짐)."
        )
        self.btn_capture.clicked.connect(lambda _=False: self._on_capture())
        left.addWidget(self.btn_capture)

        self.image_count_label = QLabel("이미지 목록 (0장)")
        self.image_count_label.setStyleSheet("color: #666; font-size: 11px; margin-top: 6px;")
        left.addWidget(self.image_count_label)

        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._on_image_row_changed)
        left.addWidget(self.image_list, stretch=1)

        root.addWidget(left_widget)

        # ------------------------------------------------------- 중앙: 설정 + 미리보기
        center = QVBoxLayout()

        # 체크포인트 / config / CAD는 '12. 설정' 탭 저장값을 쓴다 - 지금 무엇이 쓰이는지만 보여준다.
        self.settings_label = QLabel("")
        self.settings_label.setStyleSheet("color: #888; font-size: 11px;")
        center.addWidget(self.settings_label)

        run_row = QHBoxLayout()
        self.btn_run = QPushButton("추론 실행")
        self.btn_run.clicked.connect(self._on_run_inference)
        run_row.addWidget(self.btn_run)
        run_row.addStretch(1)
        center.addLayout(run_row)

        # --- ROI (검출 영역 제한) --------------------------------------------
        roi_row = QHBoxLayout()
        self.chk_roi = QCheckBox("ROI 사용")
        self.chk_roi.setToolTip(
            "켜면 이미지 위에서 마우스로 드래그해 사각형을 그립니다 (숫자로 미세 조정 가능).\n"
            "추론은 이 영역만 잘라서 수행하고, 밖의 물체는 검출되지 않습니다.\n"
            "ROI는 이미지를 바꿔도 유지됩니다 (같은 카메라/빈 고정 가정)."
        )
        self.chk_roi.toggled.connect(self._on_roi_toggled)
        roi_row.addWidget(self.chk_roi)
        self.roi_spins: dict[str, QSpinBox] = {}
        for key, label in (("x1", "x1"), ("y1", "y1"), ("x2", "x2"), ("y2", "y2")):
            roi_row.addWidget(QLabel(label))
            sp = QSpinBox()
            sp.setRange(0, 99999)
            sp.setEnabled(False)
            sp.valueChanged.connect(lambda _=None: self._on_roi_spin_changed())
            self.roi_spins[key] = sp
            roi_row.addWidget(sp)
        self.btn_roi_reset = QPushButton("전체로")
        self.btn_roi_reset.setToolTip("ROI를 이미지 전체로 되돌린다")
        self.btn_roi_reset.setEnabled(False)
        self.btn_roi_reset.clicked.connect(self._on_roi_reset)
        roi_row.addWidget(self.btn_roi_reset)
        roi_row.addStretch(1)
        center.addLayout(roi_row)

        # --- 형상 보정 (CAD 투영 템플릿) -------------------------------------
        self.chk_refine = QCheckBox("형상 보정 (CAD 투영 템플릿 + 깊이)")
        self.chk_refine.setToolTip(
            "추론 직후 마스크를 볼트 투영 실루엣에 맞춰 다듬는다.\n"
            "세션 폴더 구조(<세션>/intensity/*.png + pointcloud_organized + valid_mask)의 이미지이거나,\n"
            "[카메라로 촬영]한 프레임이어야 한다. CAD는 '12. 설정' 탭의 CAD 모델을 쓴다."
        )
        center.addWidget(self.chk_refine)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("깊이 허용(mm)"))
        self.spin_depth_tol = QDoubleSpinBox()
        self.spin_depth_tol.setRange(1.0, 50.0)
        self.spin_depth_tol.setSingleStep(1.0)
        self.spin_depth_tol.setValue(RefineParams().depth_tol_mm)
        self.spin_depth_tol.setToolTip("물체 평면에서 이 거리 이내의 점을 물체로 인정. 센서 노이즈가 크면 키운다.")
        param_row.addWidget(self.spin_depth_tol)
        param_row.addWidget(QLabel("최소 score"))
        self.spin_min_score = QDoubleSpinBox()
        self.spin_min_score.setRange(0.0, 1.0)
        self.spin_min_score.setSingleStep(0.05)
        self.spin_min_score.setValue(RefineParams().min_score)
        self.spin_min_score.setToolTip("템플릿 적합 점수가 이 값 미만이면 보정을 포기하고 원본 마스크를 유지.")
        param_row.addWidget(self.spin_min_score)
        self.btn_refine = QPushButton("보정 적용")
        self.btn_refine.setToolTip("재추론 없이 현재 검출 결과에 보정만 다시 실행 (파라미터 튜닝용)")
        self.btn_refine.clicked.connect(self._on_apply_refine)
        param_row.addWidget(self.btn_refine)
        self.chk_show_refined = QCheckBox("보정 마스크 표시")
        self.chk_show_refined.setChecked(True)
        self.chk_show_refined.toggled.connect(lambda _=None: self._update_view())
        param_row.addWidget(self.chk_show_refined)
        self.chk_depth_phi = QCheckBox("깊이로 φ 제한")
        self.chk_depth_phi.setChecked(RefineParams().depth_phi)
        self.chk_depth_phi.setToolTip(
            "축이 카메라 쪽으로 기울어진 방향(어느 끝이 가까운지)을 깊이 기울기로 정한다.\n"
            "보정이 자주 포기되면 꺼서 비교해 보세요 (끄면 실루엣만으로 추정)."
        )
        param_row.addWidget(self.chk_depth_phi)
        self.chk_overlap = QCheckBox("겹침 정리")
        self.chk_overlap.setChecked(RefineParams().resolve_overlap)
        self.chk_overlap.setToolTip(
            "보정된 마스크끼리 겹친 픽셀을 한쪽에만 남긴다 (깊이가 맞는 쪽, 모호하면 마스크 안쪽 깊은 쪽).\n"
            "같은 물체의 중복 검출(IoU 0.8 이상)은 건드리지 않는다."
        )
        param_row.addWidget(self.chk_overlap)
        param_row.addStretch(1)
        center.addLayout(param_row)

        # 의심 마스크 거르기: 보정 결과의 품질 지표로 이상한 마스크를 화면/목록에서 숨긴다 (원본 데이터는 그대로)
        filt_row = QHBoxLayout()
        self.chk_suspect = QCheckBox("의심 마스크 숨기기")
        self.chk_suspect.setToolTip("아래 조건 중 하나라도 걸리면 숨깁니다 (목록에는 '숨김 N건'으로 표시, 보정 정보는 JSON에 남음).")
        self.chk_drop_failed = QCheckBox("보정 포기 포함")
        self.chk_drop_failed.setChecked(True)
        self.chk_drop_failed.setToolTip("형상 보정을 포기(원본 유지)한 마스크도 숨깁니다.")
        self.spin_f_score = QDoubleSpinBox()
        self.spin_f_score.setRange(0.0, 1.5)
        self.spin_f_score.setSingleStep(0.05)
        self.spin_f_score.setValue(0.8)
        self.spin_f_score.setToolTip("템플릿 적합 점수가 이 값 미만이면 숨김.")
        self.spin_f_change = QDoubleSpinBox()
        self.spin_f_change.setRange(0.0, 9.9)
        self.spin_f_change.setSingleStep(0.05)
        self.spin_f_change.setValue(0.5)
        self.spin_f_change.setToolTip("변경률((추가+삭제)/원본 면적)이 이 값을 넘으면 숨김. 원본이 많이 틀렸다는 뜻.")
        self.spin_f_det = QDoubleSpinBox()
        self.spin_f_det.setRange(0.0, 1.0)
        self.spin_f_det.setSingleStep(0.05)
        self.spin_f_det.setValue(0.6)
        self.spin_f_det.setToolTip("원본 검출 중 CAD 투영 안에 있는 비율이 이 값 미만이면 숨김.")
        for w in (self.chk_suspect, self.chk_drop_failed):
            filt_row.addWidget(w)
        for lab, sp in (("score ≥", self.spin_f_score), ("변경률 ≤", self.spin_f_change), ("원본∩템플릿 ≥", self.spin_f_det)):
            filt_row.addWidget(QLabel(lab))
            filt_row.addWidget(sp)
        filt_row.addStretch(1)
        for w in (self.chk_suspect, self.chk_drop_failed):
            w.toggled.connect(lambda _=None: self._update_view())
        for sp in (self.spin_f_score, self.spin_f_change, self.spin_f_det):
            sp.valueChanged.connect(lambda _=None: self._update_view())
        center.addLayout(filt_row)

        # 보정이 몇 건 적용/포기됐는지와 포기 사유를 항상 보여준다 (포기하면 화면이 원본과 같아 보이므로)
        self.refine_status = QLabel("")
        self.refine_status.setWordWrap(True)
        self.refine_status.setStyleSheet("color: #666; font-size: 11px;")
        center.addWidget(self.refine_status)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("박스 투명도"))
        self.box_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.box_opacity_slider.setRange(0, 100)  # 0 = 박스/라벨 숨김, 100 = 불투명
        self.box_opacity_slider.setValue(DEFAULT_BOX_OPACITY)
        self.box_opacity_slider.setToolTip("바운딩 박스 테두리와 라벨의 불투명도 (마스크는 영향 없음)")
        self.box_opacity_label = QLabel(f"{DEFAULT_BOX_OPACITY}%")
        self.box_opacity_slider.valueChanged.connect(self._on_box_opacity_changed)
        opacity_row.addWidget(self.box_opacity_slider, stretch=1)
        opacity_row.addWidget(self.box_opacity_label)
        center.addLayout(opacity_row)

        self.image_viewer = ImageViewer()
        self.image_viewer.set_box_opacity(DEFAULT_BOX_OPACITY / 100.0)
        self.image_viewer.roi_changed.connect(self._on_roi_drawn)
        center.addWidget(self.image_viewer, stretch=1)
        root.addLayout(center, stretch=2)

        # ------------------------------------------------------- 우측: 검출 결과
        right = QVBoxLayout()
        right.addWidget(QLabel("검출 결과"))
        self.result_list = QListWidget()
        right.addWidget(self.result_list, stretch=1)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("conf threshold"))
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(int(DEFAULT_SCORE_THRESHOLD * 100))
        self.thresh_label = QLabel(f"{DEFAULT_SCORE_THRESHOLD:.2f}")
        self.thresh_slider.valueChanged.connect(self._on_threshold_changed)
        thresh_row.addWidget(self.thresh_slider, stretch=1)
        thresh_row.addWidget(self.thresh_label)
        right.addLayout(thresh_row)

        self.btn_export = QPushButton("검출 결과 내보내기 (json)")
        self.btn_export.clicked.connect(self._on_export)
        right.addWidget(self.btn_export)

        root.addLayout(right, stretch=1)

    def _refresh_settings_label(self) -> None:
        """'12. 설정' 탭의 체크포인트 / config / CAD가 지금 무엇인지 보여준다 (문제가 있으면 빨갛게)."""
        text, tooltip, ok = summary_text(load_settings_paths(), need_cad=False)
        self.settings_label.setText(text)
        self.settings_label.setToolTip(tooltip)
        self.settings_label.setStyleSheet(
            "color: #888; font-size: 11px;" if ok else "color: #c0392b; font-size: 11px;")

    # ----------------------------------------------------------- actions
    def _on_capture(self) -> None:
        """카메라로 한 장 촬영해 목록에 추가하고 바로 선택한다 (설정 탭 카메라 값 사용)."""
        self.btn_capture.setEnabled(False)
        try:
            with busy_cursor():
                frame, cam = capture_frame()
            name = new_capture_name()
            png_path = save_intensity_png(frame.intensity, name)
        except Exception as exc:  # noqa: BLE001
            self.log_message.emit(f"카메라 촬영 실패: {exc}")
            QMessageBox.critical(self, "촬영 오류", str(exc))
            return
        finally:
            self.btn_capture.setEnabled(True)

        self._captures[png_path] = (
            name,
            np.asarray(frame.points_organized, dtype=np.float32),
            np.asarray(frame.valid_mask, dtype=bool),
        )
        self._image_paths.append(png_path)
        self.image_list.addItem(QListWidgetItem(f"[촬영] {name}"))
        self.image_count_label.setText(f"이미지 목록 ({len(self._image_paths)}장)")
        valid_ratio = 100.0 * float(frame.valid_mask.sum()) / frame.valid_mask.size
        self.log_message.emit(
            f"카메라 촬영 완료: {name} ({cam.camera_type}, 평균화 {cam.num_frames}프레임, 유효 픽셀 {valid_ratio:.1f}%)")
        self.image_list.setCurrentRow(self.image_list.count() - 1)   # -> _on_image_row_changed 로 미리보기

    def _on_browse_image_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택")
        if not folder:
            return
        self.folder_edit.setText(folder)
        self._load_image_folder(folder)

    def _load_image_folder(self, folder: str) -> None:
        folder_path = Path(folder)
        files = sorted(
            f for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )

        self._image_paths = [str(f) for f in files]
        self.image_list.clear()
        for f in files:
            self.image_list.addItem(QListWidgetItem(f.name))
        for png_path, (name, _pcd, _valid) in self._captures.items():   # 촬영본은 폴더를 바꿔도 목록에 유지
            self._image_paths.append(png_path)
            self.image_list.addItem(QListWidgetItem(f"[촬영] {name}"))
        self.image_count_label.setText(f"이미지 목록 ({len(self._image_paths)}장)")

        self._image_path = None
        self._last_detections = []
        self._clear_refined()
        self.result_list.clear()
        self.image_viewer.load_image("")  # 미리보기 초기화

        if not files:
            QMessageBox.information(self, "알림", "선택한 폴더에 이미지가 없습니다.")
            self.log_message.emit(f"이미지 폴더 스캔: {folder} (0장)")
            return

        self.log_message.emit(f"이미지 폴더 스캔: {folder} ({len(files)}장)")
        self.image_list.setCurrentRow(0)  # 첫 이미지를 자동으로 미리보기

    def _on_image_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._image_paths):
            return
        self._image_path = self._image_paths[row]
        self.image_viewer.load_image(self._image_path)
        self._sync_roi_to_image()
        self._last_detections = []
        self._clear_refined()
        self.result_list.clear()
        self.log_message.emit(f"이미지 선택: {self._image_path}")

    def _on_run_inference(self) -> None:
        if not self._image_path:
            QMessageBox.warning(self, "알림", "먼저 이미지 폴더를 선택하거나 카메라로 촬영한 뒤 목록에서 고르세요.")
            return
        paths = load_settings_paths()   # 체크포인트 / config는 '12. 설정' 탭 저장값
        problem = paths.problem()
        if problem:
            QMessageBox.warning(self, "설정 필요", problem)
            return

        threshold = self.thresh_slider.value() / 100.0

        detector = Detector(
            checkpoint_path=str(paths.checkpoint),
            config_path=str(paths.config),
            score_threshold=threshold,
        )
        try:
            detections = detector.predict(self._image_path, conf_threshold=threshold, roi=self._current_roi())
        except ImportError as exc:
            QMessageBox.critical(
                self, "추론 엔진 없음",
                f"{exc}\n\n"
                "이 오류는 실제 모델을 로드하지 못했다는 뜻입니다 - 더미 결과가 아닙니다."
            )
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "추론 오류", str(exc))
            return

        self._last_detections = detections
        self._clear_refined()
        self.log_message.emit(f"검출 완료: {len(detections)}건 (threshold={threshold:.2f})")
        if self.chk_refine.isChecked():
            self._apply_refine()  # 실패해도 내부에서 팝업 + 원본 표시
        else:
            self._update_view()

    def _on_threshold_changed(self, value: int) -> None:
        self.thresh_label.setText(f"{value / 100:.2f}")
        self._update_view()

    def _on_box_opacity_changed(self, value: int) -> None:
        self.box_opacity_label.setText(f"{value}%")
        self.image_viewer.set_box_opacity(value / 100.0)

    # ------------------------------------------------------- ROI
    def _current_roi(self) -> tuple[int, int, int, int] | None:
        """ROI 사용이 켜져 있으면 (x1, y1, x2, y2), 아니면 None."""
        if not self.chk_roi.isChecked():
            return None
        r = tuple(self.roi_spins[k].value() for k in ("x1", "y1", "x2", "y2"))
        return r if (r[2] - r[0] >= 16 and r[3] - r[1] >= 16) else None

    def _set_roi_spins(self, roi: tuple[int, int, int, int]) -> None:
        for k, v in zip(("x1", "y1", "x2", "y2"), roi):
            sp = self.roi_spins[k]
            sp.blockSignals(True)
            sp.setValue(int(v))
            sp.blockSignals(False)

    def _sync_roi_to_image(self) -> None:
        """이미지가 바뀌면 스핀박스 범위를 이미지 크기에 맞추고, ROI를 (크기가 같으면 유지) 다시 표시한다."""
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
            self.log_message.emit("ROI 사용: 이미지 위에서 드래그해 영역을 지정하세요")
        else:
            self.image_viewer.set_roi(None)
            self.log_message.emit("ROI 해제: 전체 이미지로 검출합니다")

    def _on_roi_drawn(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self._set_roi_spins((x1, y1, x2, y2))
        self.log_message.emit(f"ROI 설정: ({x1}, {y1}) ~ ({x2}, {y2}) [{x2 - x1}x{y2 - y1}px]")

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

    # ------------------------------------------------------- 형상 보정
    def _clear_refined(self) -> None:
        self._refined_detections = []
        self._refine_results = []
        if hasattr(self, "refine_status"):
            self.refine_status.setText("")

    def _load_session_arrays(self, image_path: str):
        cap = self._captures.get(image_path)
        if cap is not None:
            return cap[1], cap[2]   # 카메라 촬영본: 점군/유효마스크를 메모리에 가지고 있다
        return load_session_arrays(image_path)  # 세션 구조가 아니면 ValueError

    def _on_apply_refine(self) -> None:
        if not self._last_detections:
            QMessageBox.warning(self, "알림", "먼저 추론을 실행하세요.")
            return
        self._apply_refine()

    def _apply_refine(self) -> None:
        """_last_detections(원본)에 보정을 적용해 _refined_detections / _refine_results를 채운다."""
        try:
            paths = load_settings_paths()   # CAD는 '12. 설정' 탭 저장값
            problem = paths.problem(need_detector=False, need_cad=True)
            if problem:
                raise ValueError(problem)
            cad = str(paths.cad)
            if not self._image_path:
                raise ValueError("이미지가 선택되지 않았습니다.")

            pcd, valid = self._load_session_arrays(self._image_path)
            intr = estimate_intrinsics_from_organized_pcd(pcd, valid)
            bank = get_bank(cad)
            params = dataclasses.replace(
                RefineParams(),
                depth_tol_mm=float(self.spin_depth_tol.value()),
                min_score=float(self.spin_min_score.value()),
                depth_phi=self.chk_depth_phi.isChecked(),
                resolve_overlap=self.chk_overlap.isChecked(),
            )

            refined: list[Detection] = []
            results: list[RefineResult] = [RefineResult(mask=None, ok=False, reason="no_mask")
                                           for _ in self._last_detections]
            with_mask = [i for i, d in enumerate(self._last_detections) if d.mask is not None]
            batch = refine_masks([self._last_detections[i].mask for i in with_mask], pcd, valid, intr, bank, params)
            for i, r in zip(with_mask, batch):
                results[i] = r
            for det, res in zip(self._last_detections, results):
                if res.ok:
                    ys, xs = np.nonzero(res.mask)
                    bbox = (float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1)
                    refined.append(dataclasses.replace(det, mask=res.mask, bbox=bbox))
                else:
                    refined.append(det)  # 포기 -> 원본 그대로
            self._refined_detections = refined
            self._refine_results = results
            n_ok = sum(r.ok for r in results)
            self.log_message.emit(
                f"형상 보정: {n_ok}/{len(results)}건 적용 (CAD aspect {bank.aspect:.1f}, "
                f"depth_tol={params.depth_tol_mm:.1f}mm, min_score={params.min_score:.2f}, "
                f"depth_phi={params.depth_phi})"
            )
            for i, r in enumerate(results):
                self.log_message.emit(f"  obj{i}: {r.summary()}")
            reasons: dict[str, int] = {}
            for r in results:
                if not r.ok:
                    reasons[r.reason] = reasons.get(r.reason, 0) + 1
            status = f"보정 {n_ok}/{len(results)}건 적용"
            if reasons:
                status += " | 원본 유지: " + ", ".join(f"{k}×{v}" for k, v in reasons.items())
            self.refine_status.setText(status)
            if bank.aspect < params.elong_min:
                self.log_message.emit("경고: CAD가 길쭉하지 않아(aspect 낮음) 이 보정 방식에 적합하지 않을 수 있습니다.")
        except Exception as exc:  # noqa: BLE001
            self._clear_refined()
            self.refine_status.setText(f"보정 실패: {exc}")
            self.log_message.emit(f"형상 보정 실패: {exc}")
            QMessageBox.warning(self, "형상 보정 실패", f"{exc}\n\n원본 마스크를 표시합니다.")
        self._update_view()

    # ------------------------------------------------------- 표시
    def _visible(self):
        """threshold를 통과한 (표시용 검출, 보정정보) 쌍. 보정 표시가 꺼져 있거나 보정 전이면 원본."""
        thr = self.thresh_slider.value() / 100.0
        use_refined = bool(self._refined_detections) and self.chk_show_refined.isChecked()
        src = self._refined_detections if use_refined else self._last_detections
        out_d, out_r = [], []
        self._n_hidden_suspect = 0
        filt = self.chk_suspect.isChecked() and bool(self._refine_results)
        for i, det in enumerate(src):
            if det.confidence < thr:
                continue
            r = self._refine_results[i] if i < len(self._refine_results) else None
            if filt and r is not None and r.is_suspect(
                    self.spin_f_score.value(), self.spin_f_change.value(), self.spin_f_det.value(),
                    self.chk_drop_failed.isChecked()):
                self._n_hidden_suspect += 1
                continue
            out_d.append(det)
            out_r.append(r)
        return out_d, out_r

    def _update_view(self) -> None:
        if not self._last_detections:
            return
        dets, infos = self._visible()
        self.image_viewer.set_detections(dets)
        self._refresh_result_list(dets, infos)

    def _refresh_result_list(self, detections=None, infos=None) -> None:
        if detections is None:
            detections, infos = self._visible()
        self.result_list.clear()
        for i, det in enumerate(detections):
            text = f"object_{i}  |  class: {det.label}  |  conf {det.confidence:.2f}"
            info = infos[i] if infos and i < len(infos) else None
            if info is not None:
                text += f"\n    {info.summary()}"
            self.result_list.addItem(QListWidgetItem(text))
        n_h = getattr(self, "_n_hidden_suspect", 0)
        if n_h:
            self.result_list.addItem(QListWidgetItem(f"(의심 마스크 {n_h}건 숨김)"))

    def _on_export(self) -> None:
        if not self._last_detections:
            QMessageBox.warning(self, "알림", "먼저 추론을 실행하세요.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "결과 저장", "detections.json", "JSON (*.json)"
        )
        if not path:
            return
        import json

        data = []
        for i, d in enumerate(self._last_detections):
            row = {"label": d.label, "confidence": d.confidence, "bbox": d.bbox}  # 항상 모델 원본
            if i < len(self._refine_results):
                r = self._refine_results[i]
                row["refine"] = {
                    "ok": bool(r.ok), "reason": r.reason, "score": float(r.score),
                    "phi_deg": float(r.phi_deg), "theta_deg": float(r.theta_deg),
                    "area_before": int(r.n_before), "area_after": int(r.n_after),
                    "det_in_tpl": float(r.det_in_tpl), "coverage": float(r.coverage),
                    "change_ratio": float(r.change_ratio),
                }
            data.append(row)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.log_message.emit(f"검출 결과 저장: {path}")