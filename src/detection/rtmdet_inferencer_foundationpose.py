"""RTMDet-Ins + FoundationPose(render-and-compare 기반 6D pose).

rtmdet_inferencer_rothead.py와 동일한 확장 방식을 따른다:
    1) super().infer()로 기존 RTMDet-Ins 마스크/bbox 검출을 그대로 수행
    2) 각 인스턴스에 대해 FoundationPose.register()를 호출해 (R,t) 전체를
       한 번에 얻는다 (RotHead처럼 회전/이동을 따로 계산할 필요 없음 -
       FoundationPose는 depth 기반 render-and-compare로 이미 결합된
       pose를 반환한다)
    3) 결과를 DetectionResult.initial_pose(4x4, m 단위)에 채운다

RotHead와의 핵심 차이: RotHead는 RGB crop만 있으면 되지만, FoundationPose는
CAD 메쉬(part별)를 register() 시점에 이미 들고 있어야 한다. 이 클래스는
CADRegistry를 생성자에서 주입받는 형태로 그 의존성을 명시한다 - 호출부
(FoundationPosePipelineTab)가 CAD 경로를 이 시점 이전에 등록해줘야 한다.

intensity(ToF) 카메라 대응: rgb 인자로 3채널 pseudo-RGB(채널 복제)를
넘겨도 동작하도록 설계했다 - 실제 변환은 호출부(app/core/detector.py의
기존 관례와 동일하게 gray -> np.stack([gray]*3, axis=-1))에서 수행하고,
이 클래스는 이미 3채널인 rgb를 받는다고만 가정한다.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from app.core.camera_intrinsics import estimate_intrinsics_from_organized_pcd
from src.detection.base import DetectionResult
from src.detection.foundationpose_estimator import CADRegistry, FoundationPoseEstimator
from src.detection.rtmdet_inferencer import RTMDetInferencer

DEFAULT_EST_REFINE_ITER = 5


class RTMDetInferencerFoundationPose(RTMDetInferencer):
    provides_pose_init = True

    def __init__(
        self,
        config: str,
        checkpoint: str,
        device: str = "cuda:0",
        score_threshold: float = 0.3,
        cad_registry: Optional[CADRegistry] = None,
        est_refine_iter: int = DEFAULT_EST_REFINE_ITER,
    ) -> None:
        """
        Args:
            cad_registry: part_name -> CAD 메쉬 경로가 이미 등록된 레지스트리.
                None이면 빈 레지스트리로 시작 - register_cad()로 나중에
                채울 수 있다 (FoundationPosePipelineTab이 UI에서 경로를
                받아 이 메서드를 호출하는 흐름을 상정).
        """
        super().__init__(config, checkpoint, device, score_threshold)
        self.cad_registry = cad_registry or CADRegistry()
        self._pose_estimator: Optional[FoundationPoseEstimator] = None
        self._est_refine_iter = est_refine_iter

    def register_cad(self, part_name: str, mesh_path: str) -> None:
        """호출부(GUI)가 CAD 파일 선택 시 이걸 불러 등록한다."""
        self.cad_registry.register_path(part_name, mesh_path)

    def _ensure_pose_estimator(self) -> FoundationPoseEstimator:
        # FoundationPose(nvdiffrast 등) import는 실제로 필요해질 때까지
        # 미룬다 - detector.py의 기존 lazy-import 관례와 동일.
        if self._pose_estimator is None:
            self._pose_estimator = FoundationPoseEstimator(
                self.cad_registry, est_refine_iter=self._est_refine_iter
            )
        return self._pose_estimator

    def infer(
        self,
        image: np.ndarray,
        pcd_organized_mm: Optional[np.ndarray] = None,
        valid_mask: Optional[np.ndarray] = None,
    ) -> List[DetectionResult]:
        """image + pcd_organized_mm(H,W,3 mm) + valid_mask(H,W bool)
        -> DetectionResult 리스트.

        image는 (H,W) mono 또는 (H,W,3) 어느 쪽이든 받는다 - mono면
        내부에서 pseudo-RGB로 채널 복제한다 (ToF intensity 카메라 대응,
        app/core/detector.py의 기존 gray->bgr 변환 관례와 동일).
        """
        base_results = super().infer(image)  # 기존 RTMDet-Ins 검출, 완전히 그대로

        if pcd_organized_mm is None or valid_mask is None or not base_results:
            return base_results

        rgb = image if image.ndim == 3 else np.stack([image] * 3, axis=-1)
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)

        depth_m = np.where(valid_mask, pcd_organized_mm[..., 2] / 1000.0, 0.0).astype(np.float32)
        fx, fy, cx, cy = estimate_intrinsics_from_organized_pcd(pcd_organized_mm, valid_mask)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        estimator = self._ensure_pose_estimator()

        enriched: List[DetectionResult] = []
        for det in base_results:
            initial_pose = None
            if self.cad_registry.is_registered(det.class_name):
                initial_pose = estimator.register(
                    part_name=det.class_name, rgb=rgb, depth_m=depth_m,
                    mask=det.mask, K=K,
                )
                # 실패(가림 심함, register 예외 등)면 None으로 남겨
                # icp_runner의 기존 fallback(build_icp_init)이 대신 쓰이게 함.

            enriched.append(
                DetectionResult(
                    mask=det.mask,
                    bbox=det.bbox,
                    score=det.score,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    initial_pose=initial_pose,
                )
            )
        return enriched