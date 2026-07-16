"""Camera abstraction layer.

사용 예시:
    from src.camera import create_camera
    
    cfg = {"type": "lucid_helios", "serial": None, ...}
    with create_camera(cfg) as cam:
        frame = cam.capture()
        print(frame.points.shape, frame.intensity.shape)
"""

from .base import CameraBase, FrameData

__all__ = ["CameraBase", "FrameData", "create_camera"]


def create_camera(config: dict) -> CameraBase:
    """설정 dict로부터 카메라 인스턴스 생성 (factory).

    Args:
        config: config.yaml의 'camera' 섹션 dict.

    Returns:
        CameraBase 하위 클래스 인스턴스.

    Raises:
        ValueError: 지원하지 않는 카메라 타입.
    """
    cam_type = config.get("type", "").lower()

    if cam_type == "lucid_helios":
        from .lucid_helios import LucidHeliosCamera
        return LucidHeliosCamera(
            serial=config.get("serial"),
            pixel_format=config.get("pixel_format", "Coord3D_ABCY16"),
            exposure_time_selector=config.get("exposure_time_selector", "Exp1000Us"),
            operating_mode=config.get("operating_mode", "Distance3000mm"),
            connect_timeout_ms=config.get("connect_timeout_ms", 5000),
            capture_timeout_ms=config.get("capture_timeout_ms", 2000),
            valid_z_range_mm=tuple(config.get("valid_z_range_mm", (100.0, 1500.0))),
        )

    if cam_type == "femto_bolt":
        from .femto_bolt import FemtoBoltCamera
        return FemtoBoltCamera(
            serial=config.get("serial"),
            depth_width=config.get("depth_width", 640),
            depth_height=config.get("depth_height", 576),
            fps=config.get("fps", 15),
            capture_timeout_ms=config.get("capture_timeout_ms", 2000),
            valid_z_range_mm=tuple(config.get("valid_z_range_mm", (100.0, 1500.0))),
            warmup_frames=config.get("warmup_frames", 5),
            capture_rgb=config.get("capture_rgb", False),
        )

    raise ValueError(
        f"지원하지 않는 카메라 타입: '{cam_type}'. "
        f"지원: ['lucid_helios', 'femto_bolt']"
    )