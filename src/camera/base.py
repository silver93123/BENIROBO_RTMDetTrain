"""카메라 백엔드 공통 인터페이스.

collect_dataset.py와 (나중에 추가될) 실시간 추론 파이프라인이 카메라
종류(LUCID Helios2, 추후 다른 ToF/스테레오 카메라)에 상관없이 동일한
방식으로 쓸 수 있도록 하는 얇은 추상 계층이다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FrameData:
    """한 번의 capture() 결과.

    Attributes:
        intensity: (H, W) uint8 mono 이미지. RTMDet 입력용.
        points: (N, 3) float32, mm 단위, valid_mask==True인 점만 모아놓은 배열.
        points_organized: (H, W, 3) float32, mm 단위, invalid 픽셀은 NaN.
        valid_mask: (H, W) bool.
        height, width: 센서 해상도.
    """
    intensity: np.ndarray
    points: np.ndarray
    points_organized: np.ndarray
    valid_mask: np.ndarray
    height: int
    width: int


class CameraBase(ABC):
    """모든 카메라 백엔드가 구현해야 하는 최소 인터페이스."""

    @abstractmethod
    def open(self) -> None:
        """카메라 연결 및 스트리밍 시작."""
        raise NotImplementedError

    @abstractmethod
    def capture(self) -> FrameData:
        """한 프레임을 캡처해서 FrameData로 반환."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """스트리밍 종료 및 연결 해제. 이미 닫혀 있어도 안전해야 한다."""
        raise NotImplementedError

    # ------------------------------------------------------- context manager
    def __enter__(self) -> "CameraBase":
        self.open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()