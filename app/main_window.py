"""메인 윈도우: 좌측 트리 네비게이션 + 우측 스택 콘텐츠 + 하단 공용 로그.

기존 QListWidget(플랫 5항목)을 QTreeWidget으로 확장했다. 완전히 다른
워크플로우(모델 학습: RTMDet vs RotHead - 스크립트도, 로그 포맷도, 지표도
다름)만 하위 트리 항목으로 분리하고, 한 파이프라인 안에서 교체 가능한
파라미터(ICP 정합 테스트의 detector/registration 알고리즘)는 여전히 탭
콘텐츠 내부의 콤보박스/내부 탭으로 남겨둔다 - app/tabs/icp_test_tab.py
참고. 이 구분 기준은 대화로 정리된 것: "완전히 다른 파이프라인"만 트리로
쪼갠다.

새 학습 파이프라인이 추가되면 app/tabs/training_pipelines/
AVAILABLE_TRAINING_PIPELINES에 한 줄만 추가하면 이 파일은 손댈 필요 없이
트리 하위 항목이 자동으로 늘어난다.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTreeWidget,
    QTreeWidgetItem, QStackedWidget, QLabel,
)
from PyQt6.QtCore import Qt

from app.tabs.data_collection_tab import DataCollectionTab
from app.tabs.data_session_tab import DataSessionTab
from app.tabs.inference_test_tab import InferenceTestTab
from app.tabs.icp_test_tab import ICPTestTab
from app.tabs.training_pipelines import AVAILABLE_TRAINING_PIPELINES
from app.widgets.log_console import LogConsole

# 트리 아이템에 스택 페이지 인덱스를 저장할 때 쓰는 데이터 role.
PAGE_INDEX_ROLE = Qt.ItemDataRole.UserRole


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Vision Bin-Picking Toolkit")
        self.resize(1200, 780)
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)

        top_bar = self._build_top_bar()
        outer.addWidget(top_bar)

        body = QHBoxLayout()
        outer.addLayout(body, stretch=1)

        self.nav_tree = QTreeWidget()
        self.nav_tree.setFixedWidth(200)
        self.nav_tree.setHeaderHidden(True)
        body.addWidget(self.nav_tree)

        self.stack = QStackedWidget()
        body.addWidget(self.stack, stretch=1)

        # ---- 페이지 구성 (탭 인스턴스 생성 + 스택에 추가 + 트리 항목 연결) ----
        self.collection_tab = DataCollectionTab()
        self._add_leaf("0. 데이터 수집", self.collection_tab)

        self.data_tab = DataSessionTab()
        self._add_leaf("1. 데이터 세션", self.data_tab)

        # "2. 모델 학습" - 완전히 다른 워크플로우라 하위 트리 항목으로 분리.
        # AVAILABLE_TRAINING_PIPELINES를 그대로 순회하므로 새 파이프라인이
        # 추가돼도 이 파일은 안 바뀐다.
        training_root = QTreeWidgetItem(["2. 모델 학습"])
        self.nav_tree.addTopLevelItem(training_root)
        self.training_tabs: dict[str, QWidget] = {}
        for name, cls in AVAILABLE_TRAINING_PIPELINES:
            tab_instance = cls()
            self.training_tabs[name] = tab_instance
            self._add_leaf(name, tab_instance, parent_item=training_root)
        training_root.setExpanded(True)
        # 학습 탭 중 --dataset 연동이 되는 건 RTMDet 학습뿐 (RotHead는
        # --labels-json 파일 경로를 쓰므로 세션 폴더명과 개념이 다름).
        self.rtmdet_training_tab = self.training_tabs["RTMDet 학습"]

        self.inference_tab = InferenceTestTab()
        self._add_leaf("3. 오프라인 검출 테스트", self.inference_tab)

        self.icp_tab = ICPTestTab()
        self._add_leaf("4. ICP 정합 테스트", self.icp_tab)

        self.nav_tree.currentItemChanged.connect(self._on_nav_changed)

        self.log_console = LogConsole()
        outer.addWidget(self.log_console)

        # 로그 시그널 연결
        self.collection_tab.log_message.connect(self.log_console.append_log)
        self.data_tab.log_message.connect(self.log_console.append_log)
        for tab in self.training_tabs.values():
            tab.log_message.connect(self.log_console.append_log)
        self.inference_tab.log_message.connect(self.log_console.append_log)
        self.icp_tab.log_message.connect(self.log_console.append_log)

        # 데이터 수집 탭에서 수집이 끝나면 -> RTMDet 학습 탭 / ICP 탭에 바로 연동
        self.collection_tab.dataset_captured.connect(self.rtmdet_training_tab.set_session_path)
        self.collection_tab.dataset_captured.connect(self.icp_tab.set_session_path)

        # 데이터 세션 탭에서 세션 선택 -> RTMDet 학습 탭 / ICP 탭에 참고용으로 전달
        self.data_tab.session_selected.connect(self.rtmdet_training_tab.set_session_path)
        self.data_tab.session_selected.connect(self.icp_tab.set_session_path)

        # 첫 화면: 트리의 첫 leaf 항목 선택
        first_leaf = self.nav_tree.topLevelItem(0)
        self.nav_tree.setCurrentItem(first_leaf)

    def _add_leaf(self, label: str, page: QWidget, parent_item: QTreeWidgetItem | None = None) -> None:
        """스택에 페이지를 추가하고, 그 인덱스를 담은 트리 leaf 항목을 만든다."""
        page_index = self.stack.addWidget(page)
        item = QTreeWidgetItem([label])
        item.setData(0, PAGE_INDEX_ROLE, page_index)
        if parent_item is not None:
            parent_item.addChild(item)
        else:
            self.nav_tree.addTopLevelItem(item)

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(40)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 16, 0)

        title = QLabel("3D Vision Bin-Picking Toolkit")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)
        layout.addStretch(1)
        return bar

    def _on_nav_changed(self, current: QTreeWidgetItem | None, previous: QTreeWidgetItem | None) -> None:
        if current is None:
            return
        page_index = current.data(0, PAGE_INDEX_ROLE)
        if page_index is None:
            # "2. 모델 학습" 같은 카테고리(부모) 항목 - 페이지가 없으므로
            # 그냥 펼치기/접기만 하고 스택은 안 바꾼다.
            current.setExpanded(not current.isExpanded())
            return
        self.stack.setCurrentIndex(page_index)