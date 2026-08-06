"""탭 9: FoundationPose 정합테스트 (FoundationPose 전용, 독립 탭).

RotHeadOnlyICPTab(app/tabs/rothead_only_icp_tab.py)과 동일한 이유·동일한
확장 방식이다: "5. ICP 정합테스트(TCP)"는 상단에 RTMDet/RotHead/FoundationPose
서브탭이 같이 있어서 매번 전환해야 했다. 이 탭은 FoundationPose만 전용으로
테스트할 때 쓴다 - 서브탭 전환 없이 항상 FoundationPose 파이프라인만 쓴다.

LiveCaptureICPTab을 그대로 상속해서 화면/동작은 100% 동일하게 재사용하고,
pipeline_tabs에서 "FoundationPose"를 제외한 서브탭을 제거한다
(FoundationPosePipelineTab 자체를 새로 만들지 않음 - app/tabs/icp_pipelines/
foundationpose_pipeline_tab.py 그대로 재사용).

RotHeadOnlyICPTab과의 차이점 - 요청받은 대로 레이아웃을 조정했다:
    - "ICP 파라미터 (모든 파이프라인 공유)"/"FGR 파라미터" 박스(params_scroll)를
      숨긴다. ICPWorkbenchTab._build_icp_params_box() 등은 여전히 내부적으로
      생성되고 self.thresh_slider 등 다른 위젯과는 무관하므로, 숨겨도
      ICP 정합 자체는 기본 파라미터 값으로 정상 동작한다 - 사용자가 조절할
      수 없게 될 뿐이다. 파라미터를 바꿔야 하면 "5. ICP 정합테스트(TCP)"
      탭에서 FoundationPose 서브탭을 선택해 쓰면 된다.
    - 그렇게 비워진 세로 공간과, root 레이아웃에서 중앙 영역의 stretch
      비중을 넓혀서 카메라 이미지+오버레이(image_viewer)가 더 크게 보이게
      했다.

icp_workbench_base.py는 손대지 않는다 - 이미 만들어진 위젯 트리를 후처리로
조정하는 방식만 쓴다 (RotHeadOnlyICPTab과 동일 원칙).
"""
from __future__ import annotations

from app.tabs.live_capture_icp_tab import LiveCaptureICPTab

FOUNDATIONPOSE_TAB_NAME = "FoundationPose"

# root(QHBoxLayout)에서 [좌측 340px 고정] [중앙(center)] [우측 280px 고정]
# 순서로 addWidget/addLayout된다 (icp_workbench_base.py._build_ui 참고).
# 중앙 레이아웃은 원래 root.addLayout(center, stretch=2)로 들어가 있는데,
# 좌/우가 고정폭이라 이 stretch 값 자체는 큰 의미가 없다(고정폭 위젯들과
# 경쟁하는 게 아니라 남는 공간을 전부 가져감) - 그래도 명시적으로 더 키워
# 의도를 분명히 한다.
CENTER_LAYOUT_INDEX = 1  # root: [0]=left_widget, [1]=center(QVBoxLayout), [2]=right_widget
CENTER_STRETCH = 4


class FoundationPoseOnlyICPTab(LiveCaptureICPTab):
    LOG_PREFIX = "FoundationPose 정합테스트 탭"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._restrict_to_foundationpose()
        self._hide_icp_params_panel()
        self._enlarge_image_viewer()

    # --------------------------------------------------- 파이프라인 제한
    def _restrict_to_foundationpose(self) -> None:
        """pipeline_tabs(QTabWidget)에서 FoundationPose 서브탭만 남기고
        나머지를 제거. 탭이 하나만 남으므로 탭바도 숨긴다."""
        remove_indices = [
            i for i in range(self.pipeline_tabs.count())
            if self.pipeline_tabs.tabText(i) != FOUNDATIONPOSE_TAB_NAME
        ]
        for i in reversed(remove_indices):  # 뒤에서부터 제거해야 인덱스가 안 밀림
            widget = self.pipeline_tabs.widget(i)
            self.pipeline_tabs.removeTab(i)
            widget.deleteLater()

        if self.pipeline_tabs.count() != 1:
            raise RuntimeError(
                f"FoundationPose 서브탭을 정확히 1개 남기지 못했습니다 (남은 개수: "
                f"{self.pipeline_tabs.count()}) - AVAILABLE_ICP_PIPELINES의 "
                f"탭 이름이 '{FOUNDATIONPOSE_TAB_NAME}'과 다른지 확인하세요."
            )
        self.pipeline_tabs.tabBar().setVisible(False)

    # --------------------------------------------------- ICP 파라미터 패널 숨김
    def _hide_icp_params_panel(self) -> None:
        """"ICP 파라미터 (모든 파이프라인 공유)"/"FGR 파라미터" 박스를 숨긴다.

        setVisible(False)만으로는 QVBoxLayout이 레이아웃을 즉시 재계산하지
        않아 빈 공간이 남을 수 있으므로, setMaximumHeight(0)까지 같이 줘서
        확실히 공간을 접는다. 파라미터 위젯 자체(self.thresh_slider 등,
        params_scroll 밖에 있는 것들)는 그대로 남아있어 검출/정합 로직에는
        영향 없다 - 오직 "ICP 파라미터"/"FGR 파라미터" 박스만 접힌다.
        """
        self.params_scroll.setVisible(False)
        self.params_scroll.setMaximumHeight(0)
        self.params_scroll.updateGeometry()

    # --------------------------------------------------- 이미지 뷰어 확대
    def _enlarge_image_viewer(self) -> None:
        """중앙 레이아웃의 stretch 비중을 넓혀 image_viewer가 남는 공간을
        최대한 차지하게 한다. root는 icp_workbench_base.py에서
        `QHBoxLayout(self)`로 self의 레이아웃 자체로 설정돼 있으므로
        self.layout()으로 그대로 가져올 수 있다."""
        root = self.layout()
        root.setStretch(CENTER_LAYOUT_INDEX, CENTER_STRETCH)

        # image_viewer 자신도 최소 크기를 키워 처음부터 넉넉하게 보이게 함.
        self.image_viewer.setMinimumHeight(520)
