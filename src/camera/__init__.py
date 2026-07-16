"""카메라 팩토리. cfg["camera"]["type"]에 따라 알맞은 백엔드를 생성한다.

collect_dataset.py를 포함한 이 프로젝트 전체가 여기서만 카메라를 임포트한다.
다른 프로젝트에 의존하지 않고, 이 저장소 안(src/camera)에서 전부 해결한다.

사용 예:
    from src.camera import create_camera
    with create_camera(cfg["camera"]) as cam:
        frame = cam.capture()
"""
from __future__ import annotations

from .base import CameraBase, FrameData

_BACKENDS = {
    "lucid_helios": "src.camera.lucid_helios:LucidHeliosCamera",
    # 추후 다른 카메라 추가 시 여기에 등록.
    # "femto_bolt": "src.camera.femto_bolt:FemtoBoltCamera",
}


def create_camera(cfg: dict) -> CameraBase:
    """cfg (config.yaml의 camera: 섹션)를 받아 알맞은 카메라 인스턴스를 만든다."""
    cam_type = cfg.get("type")
    if cam_type not in _BACKENDS:
        supported = ", ".join(sorted(_BACKENDS)) or "(없음)"
        raise ValueError(
            f"지원하지 않는 카메라 type='{cam_type}'. 지원 목록: {supported}"
        )

    if cam_type == "lucid_helios":
        from .lucid_helios import LucidHeliosCamera

        return LucidHeliosCamera(cfg)

    raise AssertionError("unreachable")  # _BACKENDS와 분기가 어긋난 경우 방지용


__all__ = ["create_camera", "CameraBase", "FrameData"]