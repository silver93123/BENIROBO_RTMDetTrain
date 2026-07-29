"""RTMDet-Ins + 2단계 회전 회귀 헤드 파이프라인.

이 파이프라인만의 고유 상태(rotation checkpoint 경로 등)는 FrameContext에
넣지 않고 이 탭 자신의 UI 상태로 관리한다 - FrameContext는 "여러 파이프라인이
공통으로 필요로 하는 것"만 담는다는 원칙 때문.

register()는 오버라이드하지 않는다 - detect()가 Detection.initial_pose를
채워서 반환하면, ICPPipelineTab의 기본 구현이 알아서 T_init_override로
써준다. 이 탭이 하는 일은 정확히 detect() 하나뿐이다.
"""
from __future__ import annotations

from typing import List

from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from app.core.detector import Detection, Detector
from app.core.pipeline_context import FrameContext
from app.tabs.icp_pipelines.base import ICPPipelineTab

BACKEND_NAME = "rtmdet_ins_rothead"


class RotHeadPipelineTab(ICPPipelineTab):
    pipeline_name = "RotHead"

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        hint = QLabel(
            "회전 헤드 검출. 성공한 인스턴스는 ICP 초기 pose를 자동으로 받습니다\n"
            "(마스크 crop이 실패하는 등 회전 헤드가 pose를 못 낸 인스턴스만\n"
            "'ICP 파라미터' 박스의 기본 fallback을 씁니다)."
        )
        hint.setStyleSheet("color: #888; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(QLabel("회전 헤드 checkpoint"))
        self.rot_checkpoint_edit = QLineEdit()
        self.rot_checkpoint_edit.setPlaceholderText(
            "미지정 시 미학습(ImageNet 초기값) 모델 - 의미있는 회전을 못 냄"
        )
        ckpt_row.addWidget(self.rot_checkpoint_edit, stretch=1)
        btn_browse = QPushButton("선택")
        btn_browse.clicked.connect(self._on_browse_rotation_checkpoint)
        ckpt_row.addWidget(btn_browse)
        layout.addLayout(ckpt_row)

        layout.addStretch(1)

    def _on_browse_rotation_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "회전 헤드 checkpoint 선택", "", "PyTorch (*.pth)")
        if path:
            self.rot_checkpoint_edit.setText(path)

    def detect(self, ctx: FrameContext) -> List[Detection]:
        detector = Detector(
            checkpoint_path=ctx.checkpoint_path,
            config_path=ctx.config_path,
            score_threshold=ctx.score_threshold,
            backend=BACKEND_NAME,
            # Detector.**backend_kwargs를 통해 RTMDetInferencerRotHead의
            # 생성자로 그대로 전달됨 (app/core/detector.py 참고).
            rotation_checkpoint=self.rot_checkpoint_edit.text().strip() or None,
        )
        detections = detector.predict(
            ctx.image_path, conf_threshold=ctx.score_threshold,
            pcd_organized_mm=ctx.pcd_organized_mm, valid_mask=ctx.valid_mask,
        )
        return [d for d in detections if d.mask is not None]