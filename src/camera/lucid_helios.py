"""LUCID Helios2 (ToF) 카메라 백엔드. arena_api(LUCID Arena SDK) 기반.

확인 필요: 이 파일은 LUCID의 공식 문서/앱노트에 나온 패턴
(PixelFormat=Coord3D_ABCY16, Scan3dCoordinateScale/Offset 노드로 mm 변환)을
기준으로 작성했지만, 실제 하드웨어에서 한 번도 테스트해보지 않았다.
특히 아래 노드 이름들은 설치된 Arena SDK 버전 / 펌웨어에 따라 다를 수 있으므로,
ArenaView GUI의 노드 목록(Feature List)과 대조해서 확인해야 한다.

    - ExposureTimeSelector   (Exp62_5Us / Exp250Us / Exp1000Us 같은 선택지)
    - Scan3dOperatingMode 또는 OperatingMode  (Distance1500mm 등)
    - Scan3dCoordinateScale / Scan3dCoordinateOffset (+ Scan3dCoordinateSelector)

노드 이름이 다르면 KeyError/AttributeError가 나므로, 처음 연결할 때
ArenaView에서 실제 노드 이름을 한 번 확인하고 아래 상수만 고쳐주면 된다.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .base import CameraBase, FrameData

# 실제 노드 이름과 다르면 이 두 줄만 고치면 된다.
NODE_OPERATING_MODE = "Scan3dOperatingMode"
NODE_EXPOSURE_SELECTOR = "ExposureTimeSelector"


class LucidHeliosCamera(CameraBase):
    """config_mapper.py가 만드는 backend dict를 그대로 받는다.

    기대하는 cfg 키 (configs/camera_config.yaml과 동일):
        type: "lucid_helios"
        serial: str | None
        pixel_format: "Coord3D_ABCY16"
        exposure_time_selector: "Exp62_5Us" | "Exp250Us" | "Exp1000Us"
        operating_mode: "Distance1500mm" 등
        connect_timeout_ms: int
        capture_timeout_ms: int
        valid_z_range_mm: [min, max]
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._device = None
        self._system = None
        self._scale_mm: float = 1.0
        self._offset_mm = (0.0, 0.0, 0.0)  # (x, y, z)

    # ------------------------------------------------------------- lifecycle
    def open(self) -> None:
        # 지연 임포트: arena_api가 설치돼 있지 않은 환경(예: 학습 서버)에서도
        # 이 프로젝트의 다른 기능(학습/추론)은 문제없이 동작하도록 한다.
        from arena_api.system import system  # noqa: PLC0415

        self._system = system

        serial = self.cfg.get("serial")
        timeout_ms = self.cfg.get("connect_timeout_ms", 5000)
        device_infos = self._wait_for_device(serial, timeout_ms)

        self._device = system.create_device(device_infos=device_infos)[0]
        nodemap = self._device.nodemap

        nodemap["PixelFormat"].value = self.cfg.get("pixel_format", "Coord3D_ABCY16")
        nodemap[NODE_EXPOSURE_SELECTOR].value = self.cfg.get(
            "exposure_time_selector", "Exp250Us"
        )
        nodemap[NODE_OPERATING_MODE].value = self.cfg.get(
            "operating_mode", "Distance1500mm"
        )

        # mm 변환에 필요한 스케일/오프셋을 미리 읽어둔다.
        self._scale_mm = float(nodemap["Scan3dCoordinateScale"].value)
        offsets = []
        for axis in ("CoordinateA", "CoordinateB", "CoordinateC"):
            nodemap["Scan3dCoordinateSelector"].value = axis
            offsets.append(float(nodemap["Scan3dCoordinateOffset"].value))
        self._offset_mm = tuple(offsets)

        self._device.start_stream()

        for _ in range(self.cfg.get("warmup_frames", 0)):
            self._device.get_buffer(timeout=self.cfg.get("capture_timeout_ms", 2000))

    def capture(self) -> FrameData:
        if self._device is None:
            raise RuntimeError("카메라가 열려있지 않습니다. open()을 먼저 호출하세요.")

        timeout_ms = self.cfg.get("capture_timeout_ms", 2000)

        buffer = self._device.get_buffer(timeout=timeout_ms)
        try:
            height, width = buffer.height, buffer.width
            # Coord3D_ABCY16: 픽셀당 (A, B, C, Y) = (X, Y, Z, Intensity), 각 16-bit unsigned
            # (총 8바이트/픽셀). ctypes로 uint16 포인터로 직접 캐스팅해서
            # 바이트/워드 카운트 실수를 피한다.
            import ctypes
            ptr16 = ctypes.cast(buffer.pdata, ctypes.POINTER(ctypes.c_uint16))
            raw16 = np.ctypeslib.as_array(
                ptr16, shape=(height, width, 4)
            ).astype(np.float32)

            xyz = raw16[..., :3] * self._scale_mm
            xyz[..., 0] += self._offset_mm[0]
            xyz[..., 1] += self._offset_mm[1]
            xyz[..., 2] += self._offset_mm[2]

            # intensity16 = raw16[..., 3]
            # intensity8 = np.clip(intensity16 / 257.0, 0, 255).astype(np.uint8)

            intensity16 = raw16[..., 3]
            # ToF Intensity 채널은 실제 신호가 16bit 전체 범위를 안 쓰는 경우가 많아서
            # (특히 짧은 노출/약한 조명), 고정 /257 나눗셈은 화면이 거의 까맣게 나온다.
            # 프레임의 1~99 percentile로 대비를 맞춰서 8bit로 변환한다.
            lo, hi = np.percentile(intensity16, [1.0, 99.0])
            if hi <= lo:
                intensity8 = np.zeros_like(intensity16, dtype=np.uint8)
            else:
                stretched = (intensity16 - lo) / (hi - lo) * 255.0
                intensity8 = np.clip(stretched, 0, 255).astype(np.uint8)

            zmin, zmax = self.cfg.get("valid_z_range_mm", [-1e9, 1e9])
            valid_mask = (
                (xyz[..., 2] >= zmin)
                & (xyz[..., 2] <= zmax)
                & np.isfinite(xyz[..., 2])
                & (xyz[..., 2] != 0.0)
            )

            points_organized = xyz.copy()
            points_organized[~valid_mask] = np.nan
            points = xyz[valid_mask]

            return FrameData(
                intensity=intensity8,
                points=points,
                points_organized=points_organized,
                valid_mask=valid_mask,
                height=height,
                width=width,
            )
        finally:
            self._device.requeue_buffer(buffer)

    def close(self) -> None:
        if self._device is None:
            return
        try:
            self._device.stop_stream()
        except Exception:
            pass
        try:
            if self._system is not None:
                self._system.destroy_device(self._device)
        except Exception:
            pass
        self._device = None

    # ----------------------------------------------------------------- 내부
    def _wait_for_device(self, serial: Optional[str], timeout_ms: int):
        from arena_api.system import system  # noqa: PLC0415

        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            infos = system.device_infos
            if serial:
                infos = [d for d in infos if d.get("serial") == serial]
            if infos:
                return [infos[0]]
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "카메라를 찾을 수 없습니다"
                    + (f" (serial={serial})" if serial else "")
                    + f". {timeout_ms}ms 동안 대기했습니다."
                )
            time.sleep(0.2)