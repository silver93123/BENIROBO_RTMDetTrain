"""3번(오프라인 검출 테스트) / 4번(ICP 검출 테스트) / 4-1번 탭 공용 도우미.

    경로    체크포인트 / config / CAD 는 탭마다 따로 입력하지 않고 '12. 설정' 탭에서 [저장]한 값을 쓴다
            (5번 TCP 서버와 같은 기준: get_settings().snapshot()). 설정 탭에서 바꾸고 저장하면
            다음 실행부터 바로 반영된다.
    촬영    설정 탭 '카메라' 페이지의 값(카메라 타입, 평균화 프레임 수, min_valid_ratio)으로 한 프레임을 얻는다.
            나머지 카메라 값(노출시간 등)은 configs/camera_config_*.yaml 값 그대로.
"""
from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication

from app.core.app_settings import get_settings
from app.core.paths import resolve_relative_to_project

SETTINGS_HINT = "'12. 설정' 탭 > '검출 / CAD' 페이지에서 지정하고 [저장]하세요."


# =============================================================================
# 설정 탭의 체크포인트 / config / CAD 경로
# =============================================================================
@dataclass(frozen=True)
class SettingsPaths:
    """설정 탭에 저장된 경로를 절대경로로 푼 것. 미지정이면 None (파일 존재 여부는 problem()에서 확인)."""
    checkpoint: Path | None
    config: Path | None
    cad: Path | None

    def problem(self, need_detector: bool = True, need_cad: bool = False) -> str | None:
        """필요한 파일이 미지정/없음이면 사용자에게 보여줄 안내문을, 문제가 없으면 None을 돌려준다."""
        items: list[tuple[str, Path | None]] = []
        if need_detector:
            items += [("체크포인트", self.checkpoint), ("config", self.config)]
        if need_cad:
            items.append(("CAD 모델", self.cad))
        lines = []
        for label, path in items:
            if path is None:
                lines.append(f"- {label}: 지정되지 않았습니다.")
            elif not path.is_file():
                lines.append(f"- {label}: 파일을 찾을 수 없습니다 ({path})")
        if not lines:
            return None
        return "\n".join(lines) + "\n\n" + SETTINGS_HINT


def _resolve(raw: str) -> Path | None:
    raw = (raw or "").strip()
    return resolve_relative_to_project(raw) if raw else None


def load_settings_paths() -> SettingsPaths:
    """지금 적용 중인(저장된) 설정값에서 체크포인트 / config / CAD 경로를 읽는다."""
    snap = get_settings().snapshot()
    return SettingsPaths(
        checkpoint=_resolve(snap.detection.checkpoint_path),
        config=_resolve(snap.detection.config_path),
        cad=_resolve(snap.cad.cad_path),
    )


def summary_text(paths: SettingsPaths, need_cad: bool) -> tuple[str, str, bool]:
    """탭에 보여줄 요약. (표시 문장, 툴팁(전체 경로), 문제 없음 여부)를 돌려준다."""
    def state(p: Path | None) -> str:
        if p is None:
            return "(미지정)"
        return p.name if p.is_file() else f"{p.name}  (파일 없음)"

    text = ("검출 설정은 '12. 설정' 탭 저장값을 사용합니다\n"
            f"체크포인트: {state(paths.checkpoint)}\n"
            f"config: {state(paths.config)}\n"
            f"CAD: {state(paths.cad)}")
    tooltip = "\n".join(f"{k}: {v if v is not None else '(미지정)'}" for k, v in (
        ("체크포인트", paths.checkpoint), ("config", paths.config), ("CAD", paths.cad)))
    return text, tooltip, paths.problem(need_detector=True, need_cad=need_cad) is None


# =============================================================================
# 카메라 촬영 (설정 탭의 카메라 값 사용)
# =============================================================================
@contextmanager
def busy_cursor():
    """촬영처럼 UI를 잠깐 막는 동기 작업 동안 대기 커서를 보여준다."""
    QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
    QApplication.processEvents()   # 커서가 바뀐 상태로 작업이 시작되게
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()


def capture_frame():
    """설정 탭 카메라 값으로 카메라를 열어 한 프레임 촬영하고 바로 닫는다.

    Returns: (FrameData, CameraSettings)
    촬영은 동기 실행이다 - 평균화 프레임 수가 클수록 그만큼 UI가 멈춘다 (5-1 탭과 같은 제약).
    """
    from app.core.pick_server import build_camera_cfg   # 5번 TCP 서버와 같은 설정 -> 카메라 cfg 변환
    from src.camera import create_camera                # 카메라 SDK는 실제로 촬영할 때만 import

    cam_settings = get_settings().snapshot().camera
    cfg = build_camera_cfg(cam_settings)
    with create_camera(cfg) as cam:
        return cam.capture(), cam_settings


def new_capture_name() -> str:
    """촬영본 이름 (밀리초까지 넣어 연속 촬영해도 겹치지 않게)."""
    return datetime.now().strftime("capture_%H%M%S_%f")[:-3]


def save_intensity_png(intensity: np.ndarray, name: str) -> str:
    """검출기는 이미지 경로를 받으므로 촬영한 intensity를 임시 PNG로 저장하고 경로를 돌려준다."""
    path = Path(tempfile.gettempdir()) / f"detect_test_{name}.png"
    if not cv2.imwrite(str(path), intensity):
        raise OSError(f"촬영 이미지를 임시 파일로 저장하지 못했습니다: {path}")
    return str(path)
