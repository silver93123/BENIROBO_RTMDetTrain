"""메인 윈도우: 좌측 사이드바 네비게이션 + 우측 탭 콘텐츠 + 하단 공용 로그."""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QListWidget,
    QListWidgetItem, QStackedWidget, QLabel,
)
from PyQt6.QtCore import Qt

from app.tabs.data_collection_tab import DataCollectionTab
from app.tabs.data_session_tab import DataSessionTab
from app.tabs.training_tab import TrainingTab
from app.tabs.inference_test_tab import InferenceTestTab
from app.widgets.log_console import LogConsole


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

        self.nav_list = QListWidget()
        self.nav_list.setFixedWidth(180)
        self.nav_list.addItem(QListWidgetItem("0. 데이터 수집"))
        self.nav_list.addItem(QListWidgetItem("1. 데이터 세션"))
        self.nav_list.addItem(QListWidgetItem("2. 모델 학습"))
        self.nav_list.addItem(QListWidgetItem("3. 오프라인 검출 테스트"))
        self.nav_list.currentRowChanged.connect(self._on_nav_changed)
        body.addWidget(self.nav_list)

        self.stack = QStackedWidget()
        self.collection_tab = DataCollectionTab()
        self.data_tab = DataSessionTab()
        self.training_tab = TrainingTab()
        self.inference_tab = InferenceTestTab()
        self.stack.addWidget(self.collection_tab)
        self.stack.addWidget(self.data_tab)
        self.stack.addWidget(self.training_tab)
        self.stack.addWidget(self.inference_tab)
        body.addWidget(self.stack, stretch=1)

        self.log_console = LogConsole()
        outer.addWidget(self.log_console)

        # 로그 시그널 연결
        self.collection_tab.log_message.connect(self.log_console.append_log)
        self.data_tab.log_message.connect(self.log_console.append_log)
        self.training_tab.log_message.connect(self.log_console.append_log)
        self.inference_tab.log_message.connect(self.log_console.append_log)

        # 데이터 수집 탭에서 수집이 끝나면 -> 학습 탭에 --dataset 값으로 바로 연동
        self.collection_tab.dataset_captured.connect(self.training_tab.set_session_path)

        # 데이터 세션 탭에서 세션 선택 -> 학습 탭에 참고용으로 전달
        self.data_tab.session_selected.connect(self.training_tab.set_session_path)

        self.nav_list.setCurrentRow(0)

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

    def _on_nav_changed(self, row: int) -> None:
        self.stack.setCurrentIndex(row)