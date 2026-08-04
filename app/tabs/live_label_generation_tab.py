"""탭 6: 회전 라벨 생성 (실시간 촬영 기반).

기존 generate_rotation_labels.py(배치/subprocess 방식)는 결과를 텍스트 로그로만
보여줘서 "검출이 왜 안 되는지" 바로 확인하기 어려웠다. 이 탭은 그 문제를
해결하려고 아예 다른 접근을 쓴다: "5. ICP 정합테스트(TCP)"(LiveCaptureICPTab)를
그대로 상속해서 [촬영] -> [2D 검출 실행] -> [ICP 정합 실행]까지 100% 동일한
화면(오버레이 이미지 뷰어 + 인스턴스별 결과 카드)을 그대로 쓰고, 마지막에
"라벨로 저장" 버튼 하나만 추가했다.

    [촬영]              -- LiveCaptureICPTab 그대로 (카메라 타입/평균화)
      -> [2D 검출 실행]  -- ICPWorkbenchTab 그대로 (오버레이 이미지에 bbox/mask 표시)
      -> [ICP 정합 실행] -- ICPWorkbenchTab 그대로 (우측에 성공/실패 카드, fitness 표시)
      -> [라벨로 저장]   -- 이 파일에서 추가. 이미 화면에서 확인한 결과 중
                            fitness가 기준 이상인 인스턴스만 학습 라벨로 저장.

이렇게 하면 검출이 안 되는 원인(모델 문제/ICP 문제/config 문제)을 화면에서
바로 보면서 디버깅한 뒤, 잘 나온 것만 라벨로 채택하는 흐름이 된다 - subprocess
로그를 눈으로 훑을 필요가 없다.

주의: 상속 관계상 카메라 촬영/평균화 UI, 체크포인트·config 선택, CAD 선택,
ICP/FGR 파라미터, 3D 뷰어까지 전부 LiveCaptureICPTab -> ICPWorkbenchTab에서
그대로 물려받는다. 이 파일이 새로 정의하는 건 라벨 채택 기준 입력칸과
"라벨로 저장" 버튼, 그리고 그 저장 로직뿐이다.
"""
from __future__ import annotations

import json
from datetime import datetime

import cv2
import numpy as np
from PyQt6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QWidget

from app.core.paths import PROJECT_ROOT
from app.tabs.live_capture_icp_tab import LiveCaptureICPTab

DEFAULT_MASK_OUT_DIR = PROJECT_ROOT / "data" / "rotation_labels_masks"
DEFAULT_LABELS_OUT = PROJECT_ROOT / "data" / "rotation_labels.json"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "data" / "rotation_labels_preview"
DEFAULT_LABEL_FITNESS_MIN = 0.85  # ICP 파라미터 박스의 fitness threshold(보통 0.6~0.7)보다
                                   # 엄격하게 - "정합 성공"과 "학습 라벨로 쓸 만큼 확실함"은 다른 기준.
DEFAULT_PREVIEW_MAX_WIDTH = 480


class LiveLabelGenerationTab(LiveCaptureICPTab):
    LOG_PREFIX = "라벨 생성(실시간) 탭"

    def __init__(self, parent=None):
        self._saved_labels: list[dict] = []
        self._n_saved_session = 0
        super().__init__(parent)
        self._load_existing_labels()

    # ----------------------------------------------------- UI 확장
    def _build_acquisition_panel(self) -> QWidget:
        """LiveCaptureICPTab의 촬영 패널(카메라 타입/평균화/촬영/이력)을 그대로
        받아서, 그 밑에 라벨 저장 관련 컨트롤만 이어붙인다."""
        panel = super()._build_acquisition_panel()
        layout = panel.layout()

        layout.addWidget(QLabel("라벨 채택 기준"))
        fitness_row = QHBoxLayout()
        fitness_row.addWidget(QLabel("fitness ≥"))
        self.spin_label_fitness_min = QDoubleSpinBox()
        self.spin_label_fitness_min.setRange(0.0, 1.0)
        self.spin_label_fitness_min.setSingleStep(0.01)
        self.spin_label_fitness_min.setValue(DEFAULT_LABEL_FITNESS_MIN)
        self.spin_label_fitness_min.setToolTip(
            "위 'ICP 파라미터' 박스의 fitness threshold(정합 성공/실패 판정)와는\n"
            "별개입니다. 여기 지정한 값 이상인 인스턴스만 학습 라벨로 저장됩니다."
        )
        fitness_row.addWidget(self.spin_label_fitness_min)
        layout.addLayout(fitness_row)

        self.btn_save_labels = QPushButton("라벨로 저장 (현재 프레임)")
        self.btn_save_labels.setToolTip("먼저 2D 검출 실행 -> ICP 정합 실행을 마쳐야 활성화됩니다.")
        self.btn_save_labels.clicked.connect(self._on_save_labels)
        self.btn_save_labels.setEnabled(False)
        layout.addWidget(self.btn_save_labels)

        self.label_count_label = QLabel(f"누적 저장: {self._n_saved_session}건")
        layout.addWidget(self.label_count_label)

        return panel

    # ----------------------------------------------------- 프레임/ICP 훅
    def _on_new_frame_acquired(self, frame_label: str) -> None:
        super()._on_new_frame_acquired(frame_label)
        self.btn_save_labels.setEnabled(False)  # 새 프레임이면 검출/ICP 다시 해야 함

    def _on_run_icp(self) -> None:
        super()._on_run_icp()
        self.btn_save_labels.setEnabled(
            bool(self._last_icp_results) and any(r.ok for r in self._last_icp_results)
        )

    # ----------------------------------------------------- 라벨 저장
    def _on_save_labels(self) -> None:
        if not self._last_icp_results:
            QMessageBox.warning(self, "알림", "먼저 2D 검출과 ICP 정합을 실행하세요.")
            return

        threshold = self.spin_label_fitness_min.value()
        DEFAULT_MASK_OUT_DIR.mkdir(parents=True, exist_ok=True)
        DEFAULT_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

        frame_bgr = None
        if self._current_image_path:
            gray = cv2.imread(self._current_image_path, cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                frame_bgr = np.stack([gray, gray, gray], axis=-1)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_tag = self._current_frame or "frame"
        n_saved_this_frame = 0
        n_skipped = 0

        for i, (det, result) in enumerate(zip(self._last_detections, self._last_icp_results)):
            if not result.ok or result.fitness is None or result.fitness < threshold:
                n_skipped += 1
                continue

            mask_filename = f"{stamp}_{frame_tag}_obj{i}.npy"
            mask_path = DEFAULT_MASK_OUT_DIR / mask_filename
            np.save(mask_path, det.mask.astype(bool))

            self._saved_labels.append({
                "image": self._current_image_path,
                "mask": str(mask_path),
                "bbox": [float(v) for v in det.bbox],
                "rotation_matrix": result.T[:3, :3].tolist(),
                "fitness": float(result.fitness),
            })
            n_saved_this_frame += 1
            self._n_saved_session += 1

            if frame_bgr is not None:
                preview_path = DEFAULT_PREVIEW_DIR / mask_filename.replace(".npy", ".jpg")
                self._save_preview_image(frame_bgr, det.mask, det.bbox, result.fitness, preview_path)

        if n_saved_this_frame == 0:
            QMessageBox.information(
                self, "저장할 라벨 없음",
                f"fitness ≥ {threshold:.2f}인 성공 인스턴스가 없습니다.\n"
                "우측 결과 카드에서 각 인스턴스의 fitness 값을 확인하세요.",
            )
            return

        DEFAULT_LABELS_OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(DEFAULT_LABELS_OUT, "w", encoding="utf-8") as f:
            json.dump(self._saved_labels, f, ensure_ascii=False, indent=2)

        self.label_count_label.setText(f"누적 저장: {self._n_saved_session}건")
        self.log_message.emit(
            f"[{self.LOG_PREFIX}] 라벨 {n_saved_this_frame}건 저장 "
            f"(건너뜀 {n_skipped}건, 누적 {self._n_saved_session}건) -> {DEFAULT_LABELS_OUT}"
        )

    def _load_existing_labels(self) -> None:
        """탭을 다시 켰을 때 기존에 저장해둔 라벨 파일이 있으면 이어서 누적한다
        (매번 새로 켤 때마다 labels.json이 통째로 덮어써지지 않도록)."""
        if not DEFAULT_LABELS_OUT.is_file():
            return
        try:
            with open(DEFAULT_LABELS_OUT, "r", encoding="utf-8") as f:
                self._saved_labels = json.load(f)
            self._n_saved_session = len(self._saved_labels)
            self.label_count_label.setText(f"누적 저장: {self._n_saved_session}건 (기존 파일에 이어서 씀)")
        except (json.JSONDecodeError, OSError) as exc:
            self.log_message.emit(f"[{self.LOG_PREFIX}] 기존 라벨 파일 로드 실패 (무시하고 새로 시작): {exc}")

    # ----------------------------------------------------- 미리보기 이미지
    @staticmethod
    def _save_preview_image(image_bgr, mask, bbox, fitness, out_path, max_width=DEFAULT_PREVIEW_MAX_WIDTH):
        """generate_rotation_labels.py의 _save_preview()와 동일한 로직.

        마스크 오버레이(초록) + bbox(빨강) + fitness 텍스트를 그려서 검토용
        이미지로 저장한다. 학습 라벨(mask .npy)과는 별개 파일이라 지워도 무방.
        """
        overlay = image_bgr.copy()
        color = np.array([60, 220, 60], dtype=np.uint8)
        overlay[mask] = (
            overlay[mask].astype(np.float32) * 0.5 + color.astype(np.float32) * 0.5
        ).astype(np.uint8)

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            overlay, f"fitness={fitness:.3f}", (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
        )

        h, w = overlay.shape[:2]
        if w > max_width:
            scale = max_width / w
            overlay = cv2.resize(overlay, (max_width, int(h * scale)))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), overlay)