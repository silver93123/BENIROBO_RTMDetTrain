"""탭 7: RotHead 정합테스트 (RotHead 전용, 독립 탭).

"5. ICP 정합테스트(TCP)"/"6. 회전 라벨 생성"은 상단에 RTMDet/RotHead 서브탭이
같이 있어서 매번 전환해야 했다. 이 탭은 RotHead만 전용으로 테스트하고 싶을 때
쓰는 것 - 서브탭 전환 없이 항상 RotHead 파이프라인만 쓴다.

LiveCaptureICPTab을 그대로 상속해서 화면/동작은 100% 동일하게 재사용하고,
pipeline_tabs에서 "RTMDet" 서브탭만 제거한다 (RotHeadPipelineTab 자체를 새로
만들지 않음 - app/tabs/icp_pipelines/rothead_pipeline_tab.py 그대로 재사용).

이 탭만의 추가 기능: 우측 "ICP 결과" 카드 목록의 세로 폭을 줄이고, 그 아래
"RotHead 초기자세" 패널을 새로 넣었다. 기존 결과 카드는 ICP 정합 *이후*
fitness/pick point만 보여주고, RotHead가 ICP 돌리기 *전에* 예측한 raw
회전/이동값은 어디에도 안 보였다 - 이 탭은 그 초기 추정치를 그대로 노출해서,
"RotHead가 초기에 얼마나 정확했는지"와 "ICP가 최종적으로 얼마나 보정했는지"를
나란히 비교할 수 있게 한다.

icp_workbench_base.py는 손대지 않는다 - 이미 만들어진 위젯 트리를 후처리로
조정하는 방식(result_scroll의 부모 레이아웃을 가져와 옆에 새 위젯을 끼워 넣음).

주의: 이 탭은 "RotHead(RotationHeadNet, 크롭 전체 -> 6D 직접 회귀)" 전용이다.
PVNet(키포인트 투표 + PnP)과는 다른 알고리즘 - PVNet은 아직 학습/추론 파이프라인이
없어서(핵심 알고리즘만 src/detection/pvnet/에 있음) 이 탭으로 테스트할 수 없다.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QFrame, QLabel, QScrollArea, QVBoxLayout, QWidget

from app.core import icp_runner
from app.tabs.live_capture_icp_tab import LiveCaptureICPTab

ROTHEAD_TAB_NAME = "RotHead"


class RotHeadOnlyICPTab(LiveCaptureICPTab):
    LOG_PREFIX = "RotHead 정합테스트 탭"
    RESULT_PANEL_MAX_HEIGHT = 260  # 'ICP 결과' 카드 목록 세로 폭 제한 - 나머지
                                    # 공간은 아래 RotHead 초기자세 패널이 차지

    def __init__(self, parent=None):
        super().__init__(parent)
        self._restrict_to_rothead()
        self._build_initial_pose_panel()

    def _restrict_to_rothead(self) -> None:
        """pipeline_tabs(QTabWidget)에서 RotHead 서브탭만 남기고 나머지를 제거.

        탭이 하나만 남으므로 탭바 자체도 숨겨서, 사용자가 "여기는 RotHead
        고정"이라는 걸 UI로도 바로 알 수 있게 한다.
        """
        remove_indices = [
            i for i in range(self.pipeline_tabs.count())
            if self.pipeline_tabs.tabText(i) != ROTHEAD_TAB_NAME
        ]
        for i in reversed(remove_indices):  # 뒤에서부터 제거해야 인덱스가 안 밀림
            widget = self.pipeline_tabs.widget(i)
            self.pipeline_tabs.removeTab(i)
            widget.deleteLater()

        if self.pipeline_tabs.count() != 1:
            raise RuntimeError(
                f"RotHead 서브탭을 정확히 1개 남기지 못했습니다 (남은 개수: "
                f"{self.pipeline_tabs.count()}) - AVAILABLE_ICP_PIPELINES의 "
                f"탭 이름이 '{ROTHEAD_TAB_NAME}'과 다른지 확인하세요."
            )
        self.pipeline_tabs.tabBar().setVisible(False)

    # ------------------------------------------------------- 초기자세 패널
    def _build_initial_pose_panel(self) -> None:
        """우측 'ICP 결과' 세로 폭을 줄이고, 그 아래 RotHead 초기자세 패널 추가.

        result_scroll이 이미 속해있는 레이아웃(우측 패널의 QVBoxLayout)을
        `parentWidget().layout()`으로 그대로 가져와서, 바로 다음 위치에
        제목 + 새 스크롤 영역을 끼워 넣는다.
        """
        self.result_scroll.setMaximumHeight(self.RESULT_PANEL_MAX_HEIGHT)

        right_layout = self.result_scroll.parentWidget().layout()
        result_idx = right_layout.indexOf(self.result_scroll)
        right_layout.setStretch(result_idx, 0)  # 높이 제한을 실제로 지키도록 stretch 제거

        title = QLabel("RotHead 초기자세 (ICP 정합 전)")
        title.setStyleSheet("font-weight: 600; margin-top: 8px;")
        right_layout.insertWidget(result_idx + 1, title)

        self.initial_pose_scroll = QScrollArea()
        self.initial_pose_scroll.setWidgetResizable(True)
        self.initial_pose_container = QWidget()
        self.initial_pose_layout = QVBoxLayout(self.initial_pose_container)
        self.initial_pose_layout.addStretch(1)
        self.initial_pose_scroll.setWidget(self.initial_pose_container)
        # stretch=1 - 위에서 줄인 만큼 남는 공간을 이 패널이 전부 차지하게 함
        right_layout.insertWidget(result_idx + 2, self.initial_pose_scroll, 1)

    def _on_run_detection(self) -> None:
        super()._on_run_detection()
        self._render_initial_pose_panel()

    def _clear_initial_pose_panel(self) -> None:
        # 마지막 addStretch(1) 항목은 남겨두고 그 앞의 카드들만 지운다.
        while self.initial_pose_layout.count() > 1:
            item = self.initial_pose_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _render_initial_pose_panel(self) -> None:
        """RotHead가 이번 검출에서 낸 raw initial_pose를 인스턴스별 카드로 표시.

        기존 'ICP 결과' 카드(_render_result_panel)와 같은 스타일을 맞춰서
        두 패널을 나란히 봤을 때 이질감이 없게 했다.
        """
        self._clear_initial_pose_panel()
        for i, det in enumerate(self._last_detections):
            card = QFrame()
            card.setFrameShape(QFrame.Shape.StyledPanel)
            card.setStyleSheet(
                "QFrame { border: 1px solid #ddd; border-radius: 6px; padding: 4px; margin-bottom: 4px; }"
            )
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 6, 8, 6)

            title = QLabel(f"obj{i}")
            title.setStyleSheet("font-weight: 600;")
            card_layout.addWidget(title)

            pose = getattr(det, "initial_pose", None)
            if pose is None:
                info = QLabel("pose 없음 - ICP는 fallback 초기값 사용")
                info.setStyleSheet("color: #c0392b;")
                info.setWordWrap(True)
                card_layout.addWidget(info)
            else:
                # 회전행렬을 그대로 보여주면 직관적으로 읽기 어려우므로,
                # icp_runner.transform_to_pose()(픽포인트 등 다른 화면과 동일한
                # 함수)로 roll/pitch/yaw(deg) + 위치(mm)로 바꿔서 표시한다.
                pose_info = icp_runner.transform_to_pose(pose)
                x, y, z = pose_info["xyz_mm"]
                euler = pose_info["euler_deg"]

                pos_label = QLabel(f"위치(mm): x={x:.1f}  y={y:.1f}  z={z:.1f}")
                pos_label.setStyleSheet("color: #2a8a2a;")
                pos_label.setWordWrap(True)
                card_layout.addWidget(pos_label)

                rot_label = QLabel(
                    f"회전(deg): roll={euler['roll_deg']:.1f}  "
                    f"pitch={euler['pitch_deg']:.1f}  yaw={euler['yaw_deg']:.1f}"
                )
                rot_label.setStyleSheet("color: #2a6f97;")
                rot_label.setWordWrap(True)
                card_layout.addWidget(rot_label)

            self.initial_pose_layout.insertWidget(self.initial_pose_layout.count() - 1, card)