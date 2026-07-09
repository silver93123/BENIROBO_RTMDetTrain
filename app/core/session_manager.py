"""세션 폴더(루트 하위 폴더) 스캔 및 라벨링 상태 확인.

실제 학습 스크립트(Train_rtmdet_model.py)의 validate_dataset_dir()과
동일한 기준으로 검사한다:
  - <session>/intensity/                          존재해야 함
  - <session>/annotations/instances_Train.json    정확히 이 파일명으로 존재해야 함
    (여러 개의 라벨 파일이 아니라, CVAT 등에서 내보낸 COCO 포맷 파일 하나)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
REQUIRED_ANNOTATION_FILENAME = "instances_Train.json"  # 대소문자까지 정확히 일치해야 함


@dataclass
class SubfolderStatus:
    exists: bool
    count: int


@dataclass
class SessionInfo:
    name: str
    path: str
    intensity: SubfolderStatus
    pointcloud: SubfolderStatus
    annotation_file_exists: bool
    annotation_image_count: int | None  # COCO json의 images 개수 (파싱 성공 시)

    @property
    def training_ready(self) -> bool:
        """실제 학습 스크립트의 validate_dataset_dir() 기준 통과 여부."""
        return self.intensity.exists and self.annotation_file_exists

    @property
    def status_text(self) -> str:
        if not self.intensity.exists:
            return "intensity 폴더 없음"
        if not self.annotation_file_exists:
            return f"{REQUIRED_ANNOTATION_FILENAME} 없음"
        if self.annotation_image_count is not None:
            return f"학습 가능 (라벨 {self.annotation_image_count}장)"
        return "학습 가능"


def _count_files(folder: Path, exts: set[str] | None = None) -> SubfolderStatus:
    if not folder.is_dir():
        return SubfolderStatus(exists=False, count=0)
    if exts:
        n = sum(1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() in exts)
    else:
        n = sum(1 for f in folder.iterdir() if f.is_file())
    return SubfolderStatus(exists=True, count=n)


def _read_coco_image_count(ann_path: Path) -> int | None:
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        images = data.get("images")
        return len(images) if isinstance(images, list) else None
    except (OSError, ValueError):
        return None


def inspect_session(session_path: str) -> SessionInfo:
    base = Path(session_path)
    intensity = _count_files(base / "intensity", IMAGE_EXTS)
    pointcloud = _count_files(base / "pointcloud_organized")

    ann_path = base / "annotations" / REQUIRED_ANNOTATION_FILENAME
    ann_exists = ann_path.is_file()
    ann_image_count = _read_coco_image_count(ann_path) if ann_exists else None

    return SessionInfo(
        name=base.name,
        path=str(base),
        intensity=intensity,
        pointcloud=pointcloud,
        annotation_file_exists=ann_exists,
        annotation_image_count=ann_image_count,
    )


def scan_sessions(root_dir: str) -> list[SessionInfo]:
    """root_dir 바로 아래 폴더들을 세션으로 간주해서 상태를 스캔한다."""
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"루트 폴더를 찾을 수 없습니다: {root_dir}")
    return [inspect_session(str(entry)) for entry in sorted(root.iterdir()) if entry.is_dir()]
