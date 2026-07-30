"""탭 4: ICP 정합 테스트 (세션 기반).

data/dataset/<세션>/에 저장된 프레임을 로드해서 검출+ICP를 테스트한다.
공통 로직(파이프라인 실행, ICP/FGR 파라미터, 결과 패널, 3D 뷰어)은
ICPWorkbenchTab(app/tabs/icp_workbench_base.py)에 있고, 이 파일은 "프레임을
어떻게 얻는지"(세션 폴더 선택 + 프레임 목록)만 구현한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox, QLineEdit,
)

from app.core.paths import DEFAULT_DATASET_ROOT
from app.tabs.icp_workbench_base import ICPWorkbenchTab


class SessionICPTab(ICPWorkbenchTab):
    LOG_PREFIX = "ICP 탭"

    def __init__(self, parent=None):
        self._frame_names: list[str] = []
        super().__init__(parent)

    def _build_acquisition_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("세션 폴더"))
        session_row = QHBoxLayout()
        self.session_edit = QLineEdit()
        self.session_edit.setReadOnly(True)
        self.session_edit.setPlaceholderText("세션 폴더를 선택하세요")
        session_row.addWidget(self.session_edit, stretch=1)
        btn_browse_session = QPushButton("선택")
        btn_browse_session.clicked.connect(self._on_browse_session)
        session_row.addWidget(btn_browse_session)
        layout.addLayout(session_row)

        self.frame_count_label = QLabel("프레임 목록 (0장)")
        self.frame_count_label.setStyleSheet("color: #666; font-size: 11px; margin-top: 8px;")
        layout.addWidget(self.frame_count_label)

        self.frame_list = QListWidget()
        self.frame_list.currentRowChanged.connect(self._on_frame_row_changed)
        layout.addWidget(self.frame_list, stretch=1)

        return panel

    # -------------------------------------------------------- 세션 / 프레임
    def set_session_path(self, session_path: str) -> None:
        """탭0/탭1에서 세션이 정해지면(시그널) 여기도 같이 갱신."""
        self.session_edit.setText(session_path)
        self._load_session(session_path)

    def _on_browse_session(self) -> None:
        start_dir = str(DEFAULT_DATASET_ROOT) if DEFAULT_DATASET_ROOT.is_dir() else ""
        folder = QFileDialog.getExistingDirectory(self, "세션 폴더 선택", start_dir)
        if not folder:
            return
        self.session_edit.setText(folder)
        self._load_session(folder)

    def _load_session(self, session_path: str) -> None:
        base = Path(session_path)
        intensity_dir = base / "intensity"
        organized_dir = base / "pointcloud_organized"
        mask_dir = base / "valid_mask"

        if not intensity_dir.is_dir():
            QMessageBox.warning(self, "알림", "이 폴더에는 intensity/ 폴더가 없습니다.")
            return
        if not organized_dir.is_dir() or not mask_dir.is_dir():
            QMessageBox.warning(
                self, "알림",
                "이 세션에는 pointcloud_organized/ 또는 valid_mask/ 폴더가 없습니다.\n"
                "ICP 정합에는 두 폴더가 모두 필요합니다 (collect_dataset.py로 수집한 세션인지 확인하세요)."
            )
            return

        self._session_path = str(base)
        stems = sorted(f.stem for f in intensity_dir.glob("*.png"))
        usable = [s for s in stems if (organized_dir / f"{s}.npy").is_file()
                  and (mask_dir / f"{s}.npy").is_file()]

        self._frame_names = usable
        self.frame_list.clear()
        for s in usable:
            self.frame_list.addItem(QListWidgetItem(s))
        self.frame_count_label.setText(f"프레임 목록 ({len(usable)}장)")

        self._reset_frame_state()
        if not usable:
            QMessageBox.information(self, "알림", "3D 데이터(pointcloud_organized+valid_mask)가 있는 프레임이 없습니다.")
            self.log_message.emit(f"[{self.LOG_PREFIX}] 세션 로드: {base} (사용 가능 프레임 0장)")
            return

        self.log_message.emit(f"[{self.LOG_PREFIX}] 세션 로드: {base} (사용 가능 프레임 {len(usable)}장)")
        self.frame_list.setCurrentRow(0)

    def _on_frame_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._frame_names) or not self._session_path:
            return
        name = self._frame_names[row]
        base = Path(self._session_path)

        self._pcd_organized = np.load(base / "pointcloud_organized" / f"{name}.npy")
        self._valid_mask = np.load(base / "valid_mask" / f"{name}.npy")
        self._current_image_path = str(base / "intensity" / f"{name}.png")

        self._on_new_frame_acquired(name)