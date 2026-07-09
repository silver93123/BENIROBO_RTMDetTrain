"""탭 1: 데이터 세션 관리.

카메라 라이브 캡처 대신, 이미 존재하는 세션 폴더들(루트 하위 폴더)을 스캔해서
intensity / pointcloud_organized / annotations 상태를 확인하는 용도.
라벨링 자체는 CVAT 등 외부 툴에서 하고, 이 탭은 상태 확인만 한다.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox, QFrame,
)

from app.core.session_manager import scan_sessions, SessionInfo, SubfolderStatus
from app.core.paths import DEFAULT_DATASET_ROOT


class DataSessionTab(QWidget):
    log_message = pyqtSignal(str)
    session_selected = pyqtSignal(str)  # 선택된 세션 폴더 경로 (학습 탭에 전달됨)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sessions: list[SessionInfo] = []
        self._current_session_path: str | None = None
        self._build_ui()
        # 프로젝트 폴더 내 data/dataset을 기본값으로 채워둔다 (없으면 만들어서라도 안내)
        self.root_edit.setText(str(DEFAULT_DATASET_ROOT))
        if DEFAULT_DATASET_ROOT.is_dir():
            self._on_refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("데이터셋 루트 폴더"))
        self.root_edit = QLineEdit()
        top_row.addWidget(self.root_edit, stretch=1)
        btn_browse = QPushButton("폴더 선택")
        btn_browse.clicked.connect(self._on_browse_root)
        top_row.addWidget(btn_browse)
        btn_refresh = QPushButton("새로고침")
        btn_refresh.clicked.connect(self._on_refresh)
        top_row.addWidget(btn_refresh)
        layout.addLayout(top_row)

        body = QHBoxLayout()
        layout.addLayout(body, stretch=1)

        self.session_list = QListWidget()
        self.session_list.setFixedWidth(260)
        self.session_list.currentRowChanged.connect(self._on_session_row_changed)
        body.addWidget(self.session_list)

        detail_panel = QVBoxLayout()
        self.detail_title = QLabel("세션을 선택하세요")
        self.detail_title.setStyleSheet("font-weight: 600; font-size: 13px;")
        detail_panel.addWidget(self.detail_title)

        self.intensity_status = self._status_row(detail_panel, "intensity/ (RGB)")
        self.pointcloud_status = self._status_row(detail_panel, "pointcloud_organized/")
        self.annotations_status = self._status_row(detail_panel, "annotations/ (CVAT 결과)")

        self.btn_open_folder = QPushButton("세션 폴더 열기 (탐색기)")
        self.btn_open_folder.clicked.connect(self._on_open_folder)
        self.btn_open_folder.setEnabled(False)
        detail_panel.addWidget(self.btn_open_folder)
        detail_panel.addStretch(1)

        body.addLayout(detail_panel, stretch=1)

    def _status_row(self, parent_layout: QVBoxLayout, label: str) -> QLabel:
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { border: 0.5px solid #ccc; border-radius: 6px; padding: 6px; }"
        )
        row = QHBoxLayout(frame)
        row.addWidget(QLabel(label))
        row.addStretch(1)
        status_label = QLabel("-")
        row.addWidget(status_label)
        parent_layout.addWidget(frame)
        return status_label

    # ------------------------------------------------------------ actions
    def _on_browse_root(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "데이터셋 루트 폴더 선택")
        if not folder:
            return
        self.root_edit.setText(folder)
        self._on_refresh()

    def _on_refresh(self) -> None:
        root = self.root_edit.text().strip()
        if not root:
            QMessageBox.warning(self, "알림", "먼저 데이터셋 루트 폴더를 선택하세요.")
            return
        try:
            self._sessions = scan_sessions(root)
        except FileNotFoundError as exc:
            QMessageBox.warning(self, "알림", str(exc))
            return

        self.session_list.clear()
        for s in self._sessions:
            marker = "✓" if s.training_ready else "·"
            item = QListWidgetItem(f"{marker} {s.name}   [{s.status_text}]")
            self.session_list.addItem(item)
        self.log_message.emit(f"세션 {len(self._sessions)}개를 스캔했습니다: {root}")

    def _on_session_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._sessions):
            return
        session = self._sessions[row]
        self.detail_title.setText(session.name)
        self._set_status(self.intensity_status, session.intensity)
        self._set_status(self.pointcloud_status, session.pointcloud)
        self._set_annotation_status(session)
        self.btn_open_folder.setEnabled(True)
        self._current_session_path = session.path
        self.session_selected.emit(session.path)

    @staticmethod
    def _set_status(label: QLabel, status: SubfolderStatus) -> None:
        if not status.exists:
            label.setText("폴더 없음")
            label.setStyleSheet("color: #999;")
        elif status.count == 0:
            label.setText("0개")
            label.setStyleSheet("color: #b8860b;")
        else:
            label.setText(f"{status.count}개")
            label.setStyleSheet("color: #1D9E75; font-weight: 600;")

    def _set_annotation_status(self, session: SessionInfo) -> None:
        label = self.annotations_status
        if not session.annotation_file_exists:
            label.setText("instances_Train.json 없음")
            label.setStyleSheet("color: #b8860b;")
        elif session.annotation_image_count is not None:
            label.setText(f"있음 ({session.annotation_image_count}장)")
            label.setStyleSheet("color: #1D9E75; font-weight: 600;")
        else:
            label.setText("있음 (파싱 실패)")
            label.setStyleSheet("color: #b8860b;")

    def _on_open_folder(self) -> None:
        if self._current_session_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._current_session_path))
