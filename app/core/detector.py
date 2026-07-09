"""오프라인 2D 객체 검출 추론 (RTMDet-Ins).

실제 파이프라인 스크립트(3_Detect_and_PickPoint.py)의 [A] Detection 단계만
그대로 가져왔다. PCD 분리 / ICP 정합 / 픽포인트 계산은 포함하지 않는다
(2D 검출까지만 -> ICP는 나중 단계에서 필요할 때 추가).

의존성: <프로젝트 루트>/src/detection.py 의 RTMDetInferencer 클래스가 필요하다.
이 파일은 실제 추론 엔진 구현이 담긴 파일로, 업로드되지 않아서 이 앱 zip에는
포함되어 있지 않다. 프로젝트 루트에 src/detection.py를 두면 그대로 동작하고,
없으면 import 시점에 명확한 에러를 낸다 (더 이상 더미/랜덤 결과를 만들지 않는다).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (픽셀 좌표)
    mask: np.ndarray | None = None  # (H, W) bool - 있으면 마스크 오버레이에 쓸 수 있음


class Detector:
    def __init__(
        self,
        checkpoint_path: str | None = None,
        config_path: str | None = None,
        device: str = "cuda:0",
        score_threshold: float = 0.3,
    ):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device
        self.score_threshold = score_threshold
        self._inferencer = None

    def load_model(self) -> None:
        """실제 파이프라인 스크립트와 동일하게 RTMDetInferencer를 생성한다."""
        if not self.checkpoint_path:
            raise ValueError("체크포인트 경로가 지정되지 않았습니다.")
        if not self.config_path:
            raise ValueError(
                "config 경로가 지정되지 않았습니다. "
                "실제 파이프라인에서는 체크포인트와 같은 work_dir 안의 "
                "rtmdet-ins_bracket.py(학습 시 저장된 사본)를 사용합니다."
            )

        try:
            from src.detection import RTMDetInferencer  # <프로젝트 루트>/src/detection.py
        except ImportError as exc:
            raise ImportError(
                "src/detection.py의 RTMDetInferencer를 import할 수 없습니다. "
                "이 클래스가 실제 추론 엔진 구현체인데, 이 앱에는 포함되어 있지 않습니다. "
                "프로젝트 루트에 src/detection.py 파일이 있는지, "
                "그리고 그 안에 RTMDetInferencer 클래스가 있는지 확인하세요."
            ) from exc

        self._inferencer = RTMDetInferencer(
            config=self.config_path,
            checkpoint=self.checkpoint_path,
            device=self.device,
            score_threshold=self.score_threshold,
        )

    def predict(self, image_path: str, conf_threshold: float | None = None) -> list[Detection]:
        """이미지 한 장에 대해 검출을 수행한다 (원본 스크립트의 process_frame_detection과 동일 흐름)."""
        if self._inferencer is None:
            self.load_model()

        threshold = self.score_threshold if conf_threshold is None else conf_threshold

        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
        bgr = np.stack([gray, gray, gray], axis=-1)

        results = self._inferencer.infer(bgr)

        detections: list[Detection] = []
        for r in results:
            score = float(r.score)
            if score < threshold:
                continue
            x1, y1, x2, y2 = [float(v) for v in r.bbox]
            detections.append(
                Detection(
                    label=str(r.class_name),
                    confidence=score,
                    bbox=(x1, y1, x2, y2),
                    mask=getattr(r, "mask", None),
                )
            )
        return detections
