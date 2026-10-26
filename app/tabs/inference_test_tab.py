"""탭 3: 오프라인 2D 검출 테스트 (ICP/3D pose 없음).

실제 파이프라인(3_Detect_and_PickPoint.py)과 동일하게 체크포인트 + config
두 가지를 모두 지정해야 RTMDetInferencer를 만들 수 있다.

이미지는 한 장씩 파일 다이얼로그로 불러오는 대신, 폴더를 지정하면 그 안의
이미지들을 목록으로 펼쳐두고 목록에서 골라 미리보기 -> 추론 실행하는 방식.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QFileDialog, QSlider, QMessageBox, QLineEdit,
    QCheckBox, QDoubleSpinBox,
)

from app.core.camera_intrinsics import estimate_intrinsics_from_organized_pcd
from app.core.detector import Detector, Detection
from app.core.config_patcher import find_latest_best_checkpoint
from app.core.paths import DEFAULT_CONFIG_PATH, DEFAULT_CAD_DIR
from app.widgets.image_viewer import ImageViewer
from src.detection.mask_refine import RefineParams, RefineResult, get_bank, load_session_arrays, refine_mask

DEFAULT_SCORE_THRESHOLD = 0.3  # 실제 파이프라인 스크립트의 SCORE_THRESHOLD와 동일
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
CAD_EXTS = {".stl", ".ply", ".obj"}
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
        self._build_ui()
        self._prefill_latest_checkpoint()
        self._prefill_cad()

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

        self.image_count_label = QLabel("이미지 목록 (0장)")
        self.image_count_label.setStyleSheet("color: #666; font-size: 11px; margin-top: 6px;")
        left.addWidget(self.image_count_label)

        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._on_image_row_changed)
        left.addWidget(self.image_list, stretch=1)

        root.addWidget(left_widget)

        # ------------------------------------------------------- 중앙: 설정 + 미리보기
        center = QVBoxLayout()

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("체크포인트"))
        self.checkpoint_edit = QLineEdit()
        path_row.addWidget(self.checkpoint_edit, stretch=1)
        btn_browse_ckpt = QPushButton("선택")
        btn_browse_ckpt.clicked.connect(self._on_browse_checkpoint)
        path_row.addWidget(btn_browse_ckpt)
        center.addLayout(path_row)

        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("config"))
        self.config_edit = QLineEdit()
        self.config_edit.setPlaceholderText("보통 체크포인트와 같은 work_dir 안의 .py 파일")
        cfg_row.addWidget(self.config_edit, stretch=1)
        btn_browse_cfg = QPushButton("선택")
        btn_browse_cfg.clicked.connect(self._on_browse_config)
        cfg_row.addWidget(btn_browse_cfg)
        center.addLayout(cfg_row)

        run_row = QHBoxLayout()
        self.btn_run = QPushButton("추론 실행")
        self.btn_run.clicked.connect(self._on_run_inference)
        run_row.addWidget(self.btn_run)
        run_row.addStretch(1)
        center.addLayout(run_row)

        # --- 형상 보정 (CAD 투영 템플릿) -------------------------------------
        self.chk_refine = QCheckBox("형상 보정 (CAD 투영 템플릿 + 깊이)")
        self.chk_refine.setToolTip(
            "추론 직후 마스크를 볼트 투영 실루엣에 맞춰 다듬는다.\n"
            "세션 폴더 구조(<세션>/intensity/*.png + pointcloud_organized + valid_mask)가 필요하다."
        )
        center.addWidget(self.chk_refine)

        cad_row = QHBoxLayout()
        cad_row.addWidget(QLabel("CAD"))
        self.cad_edit = QLineEdit()
        self.cad_edit.setPlaceholderText("볼트 CAD 파일 (.stl/.ply/.obj) - 단위 m 또는 mm 자동 판별")
        cad_row.addWidget(self.cad_edit, stretch=1)
        btn_browse_cad = QPushButton("선택")
        btn_browse_cad.clicked.connect(self._on_browse_cad)
        cad_row.addWidget(btn_browse_cad)
        center.addLayout(cad_row)

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
        param_row.addStretch(1)
        center.addLayout(param_row)

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

    def _prefill_latest_checkpoint(self) -> None:
        """탭이 열릴 때, 기본 config의 work_dir에서 가장 최근 best 체크포인트를 자동으로 채운다."""
        if not DEFAULT_CONFIG_PATH.is_file():
            return
        cfg_path = str(DEFAULT_CONFIG_PATH)
        self.config_edit.setText(cfg_path)

        best = find_latest_best_checkpoint(cfg_path)
        if best:
            self.checkpoint_edit.setText(best)
            self.log_message.emit(f"최신 best 체크포인트 자동 설정: {best}")
        else:
            self.log_message.emit(
                "work_dir에 best 체크포인트가 아직 없어 자동 설정을 건너뜁니다. "
                "직접 선택해주세요."
            )

    def _prefill_cad(self) -> None:
        """data/cad 안에 'bolt'가 들어간 CAD가 정확히 하나면 자동으로 채운다."""
        try:
            if not DEFAULT_CAD_DIR.is_dir():
                return
            cands = [f for f in DEFAULT_CAD_DIR.iterdir()
                     if f.suffix.lower() in CAD_EXTS and "bolt" in f.name.lower()]
            if len(cands) == 1:
                self.cad_edit.setText(str(cands[0]))
        except OSError:
            pass

    # ----------------------------------------------------------- actions
    def _on_browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "체크포인트 선택", "", "PyTorch (*.pth)")
        if not path:
            return
        self.checkpoint_edit.setText(path)
        self.log_message.emit(f"체크포인트 설정: {path}")

        # 체크포인트와 같은 폴더에 config .py가 하나뿐이면 자동으로 제안
        if not self.config_edit.text():
            candidates = list(Path(path).parent.glob("*.py"))
            if len(candidates) == 1:
                self.config_edit.setText(str(candidates[0]))
                self.log_message.emit(f"같은 폴더에서 config 자동 제안: {candidates[0]}")

    def _on_browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "config 파일 선택", "", "Python (*.py)")
        if path:
            self.config_edit.setText(path)

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
        self.image_count_label.setText(f"이미지 목록 ({len(files)}장)")

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
        self._last_detections = []
        self._clear_refined()
        self.result_list.clear()
        self.log_message.emit(f"이미지 선택: {self._image_path}")

    def _on_run_inference(self) -> None:
        if not self._image_path:
            QMessageBox.warning(self, "알림", "먼저 이미지 폴더를 선택하고 목록에서 이미지를 고르세요.")
            return
        checkpoint = self.checkpoint_edit.text().strip()
        config = self.config_edit.text().strip()
        if not checkpoint or not config:
            QMessageBox.warning(self, "알림", "체크포인트와 config를 모두 지정하세요.")
            return

        threshold = self.thresh_slider.value() / 100.0

        detector = Detector(
            checkpoint_path=checkpoint,
            config_path=config,
            score_threshold=threshold,
        )
        try:
            detections = detector.predict(self._image_path, conf_threshold=threshold)
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

    # ------------------------------------------------------- 형상 보정
    def _clear_refined(self) -> None:
        self._refined_detections = []
        self._refine_results = []

    def _on_browse_cad(self) -> None:
        start = str(DEFAULT_CAD_DIR) if DEFAULT_CAD_DIR.is_dir() else ""
        path, _ = QFileDialog.getOpenFileName(self, "볼트 CAD 선택", start, "CAD (*.stl *.ply *.obj)")
        if path:
            self.cad_edit.setText(path)
            self.log_message.emit(f"CAD 설정: {path}")

    def _load_session_arrays(self, image_path: str):
        return load_session_arrays(image_path)  # 세션 구조가 아니면 ValueError

    def _on_apply_refine(self) -> None:
        if not self._last_detections:
            QMessageBox.warning(self, "알림", "먼저 추론을 실행하세요.")
            return
        self._apply_refine()

    def _apply_refine(self) -> None:
        """_last_detections(원본)에 보정을 적용해 _refined_detections / _refine_results를 채운다."""
        try:
            cad = self.cad_edit.text().strip()
            if not cad or not Path(cad).is_file():
                raise ValueError("볼트 CAD 파일을 지정하세요 (.stl/.ply/.obj).")
            if not self._image_path:
                raise ValueError("이미지가 선택되지 않았습니다.")

            pcd, valid = self._load_session_arrays(self._image_path)
            intr = estimate_intrinsics_from_organized_pcd(pcd, valid)
            bank = get_bank(cad)
            params = dataclasses.replace(
                RefineParams(),
                depth_tol_mm=float(self.spin_depth_tol.value()),
                min_score=float(self.spin_min_score.value()),
            )

            refined: list[Detection] = []
            results: list[RefineResult] = []
            for det in self._last_detections:
                if det.mask is None:
                    refined.append(det)
                    results.append(RefineResult(mask=None, ok=False, reason="no_mask"))
                    continue
                res = refine_mask(det.mask, pcd, valid, intr, bank, params)
                results.append(res)
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
                f"depth_tol={params.depth_tol_mm:.1f}mm, min_score={params.min_score:.2f})"
            )
            if bank.aspect < params.elong_min:
                self.log_message.emit("경고: CAD가 길쭉하지 않아(aspect 낮음) 이 보정 방식에 적합하지 않을 수 있습니다.")
        except Exception as exc:  # noqa: BLE001
            self._clear_refined()
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
        for i, det in enumerate(src):
            if det.confidence < thr:
                continue
            out_d.append(det)
            out_r.append(self._refine_results[i] if i < len(self._refine_results) else None)
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
                }
            data.append(row)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.log_message.emit(f"검출 결과 저장: {path}")