"""RTMDet-Ins + FoundationPose 파이프라인.

RotHeadPipelineTab과 동일한 이유로 CAD 경로를 이 탭 자신의 UI 상태로
관리한다 - 단, RotHead와 달리 "왜 FrameContext.cad_pcd를 못 쓰는지"는
RotHead와 다른 이유다:

    RotHead: CAD가 애초에 필요 없어서 상관없음.
    FoundationPose: CAD가 필요하지만, ICPWorkbenchTab._build_context()는
        detect() 시점에 cad_loaded=False로 호출한다(CAD는 register()
        직전에만 로드하는 게 기존 워크벤치의 설계). 그런데
        FoundationPose는 render-and-compare 자체가 detect() 단계에서
        일어나야 하므로 ctx.cad_pcd를 기다릴 수 없다. 그래서 이 탭은
        ICP 정합용 CAD 콤보박스(워크벤치 상단)와는 별개로, part별 CAD
        메쉬 경로를 자기 자신의 표에서 직접 관리한다.

register()는 오버라이드하지 않는다 - detect()가 이미 Detection.initial_pose를
채워서 반환하면, ICPPipelineTab의 기본 구현이 T_init_override로 그대로
써준다(base.py 참고). 이 탭이 하는 일은 정확히 detect()와 CAD 경로 UI뿐이다.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from app.core.detector import Detection, Detector
from app.core.pipeline_context import FrameContext
from app.tabs.icp_pipelines.base import ICPPipelineTab

BACKEND_NAME = "rtmdet_ins_foundationpose"

# 프로젝트에 이미 등록된 4개 파트 - configs/rtmdet-ins_*.py 이름과 맞춤.
KNOWN_PARTS = ["bolt_m10_80", "bracket", "kkokkalcon", "kkokkalcon_homrunball"]


class FoundationPosePipelineTab(ICPPipelineTab):
    pipeline_name = "FoundationPose"

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        hint = QLabel(
            "FoundationPose 검출. RGBD render-and-compare로 (R,t) 전체를 추정하며,\n"
            "CAD가 등록된 파트만 initial_pose를 받습니다 (미등록 파트는 워크벤치의\n"
            "기본 fallback 정합을 씁니다). ToF intensity 이미지는 3채널로 자동 복제됩니다."
        )
        hint.setStyleSheet("color: #888; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # part별 CAD 메쉬 경로 테이블 - RotHead의 checkpoint 필드 하나짜리와
        # 달리 4개 파트 각각 CAD가 다르므로 테이블로 관리한다.
        self.cad_table = QTableWidget(len(KNOWN_PARTS), 2)
        self.cad_table.setHorizontalHeaderLabels(["파트", "CAD 메쉬 경로"])
        for row, part in enumerate(KNOWN_PARTS):
            part_item = QTableWidgetItem(part)
            part_item.setFlags(part_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # 읽기전용
            self.cad_table.setItem(row, 0, part_item)
            self.cad_table.setItem(row, 1, QTableWidgetItem(""))
        layout.addWidget(self.cad_table)

        browse_row = QHBoxLayout()
        browse_row.addWidget(QLabel("선택한 행에 CAD 지정:"))
        btn_browse = QPushButton("파일 선택")
        btn_browse.clicked.connect(self._on_browse_cad_for_selected_row)
        browse_row.addWidget(btn_browse)
        browse_row.addStretch(1)
        layout.addLayout(browse_row)

        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Refinement iteration"))
        self.iter_edit = QLineEdit("5")
        self.iter_edit.setFixedWidth(50)
        iter_row.addWidget(self.iter_edit)
        iter_row.addStretch(1)
        layout.addLayout(iter_row)

        layout.addStretch(1)
        self._detector: Detector | None = None

    def _on_browse_cad_for_selected_row(self) -> None:
        row = self.cad_table.currentRow()
        if row < 0:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "CAD 파일 선택", "", "CAD (*.stl *.ply *.obj)"
        )
        if path:
            self.cad_table.setItem(row, 1, QTableWidgetItem(path))

    def _collect_cad_paths(self) -> dict[str, str]:
        paths = {}
        for row in range(self.cad_table.rowCount()):
            part = self.cad_table.item(row, 0).text()
            path_item = self.cad_table.item(row, 1)
            path = path_item.text().strip() if path_item else ""
            if path:
                paths[part] = path
        return paths

    def detect(self, ctx: FrameContext) -> List[Detection]:
        cad_paths = self._collect_cad_paths()
        try:
            est_refine_iter = int(self.iter_edit.text().strip() or "5")
        except ValueError:
            est_refine_iter = 5

        # Detector는 checkpoint/config가 바뀌지 않는 한 재사용 가능하지만,
        # CAD 경로/iteration은 UI에서 바뀔 수 있으므로 이 탭에서 직접
        # 백엔드 인스턴스를 새로 만들어 register_cad를 호출한다.
        detector = Detector(
            checkpoint_path=ctx.checkpoint_path,
            config_path=ctx.config_path,
            score_threshold=ctx.score_threshold,
            backend=BACKEND_NAME,
            est_refine_iter=est_refine_iter,
        )
        detector.load_model()
        for part_name, path in cad_paths.items():
            detector._inferencer.register_cad(part_name, path)

        detections = detector.predict(
            ctx.image_path, conf_threshold=ctx.score_threshold,
            pcd_organized_mm=ctx.pcd_organized_mm, valid_mask=ctx.valid_mask,
        )
        return [d for d in detections if d.mask is not None]