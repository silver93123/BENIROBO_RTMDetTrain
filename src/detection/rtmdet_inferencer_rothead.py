"""RTMDet-Ins + 2단계(decoupled) 회전 회귀 헤드.

RTMDetInferencer(기존 검출, 안 건드림)를 상속해서 infer()만 확장한다:
    1) super().infer()로 기존 마스크/bbox 검출을 그대로 수행
    2) 각 인스턴스를 마스크로 크롭 -> CropRotationRegressor(작은 CNN)로
       6D rotation 회귀
    3) translation은 icp_runner.extract_instance_points_mm과 동일한 방식으로
       pointcloud_organized에서 centroid를 뽑아 m 단위로 계산
    4) 회전 + 이동을 합쳐 initial_pose(4x4, m 단위)로 채움

RTMDet-Ins의 dense head/loss(SimOTA 동적 라벨 할당, 동적 마스크 커널)는
전혀 손대지 않는다 - 회전 회귀는 완전히 분리된 2단계 모델
(rotation_head_model.py)이 담당한다. 자세한 설계 배경은 그 파일의
모듈 docstring 참고.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from app.core.icp_runner import extract_instance_points_mm
from src.detection.base import DetectionResult
from src.detection.rotation_head_model import CropRotationRegressor
from src.detection.rotation_utils import compose_pose, rot6d_to_matrix
from src.detection.rtmdet_inferencer import RTMDetInferencer

# icp_runner.py의 마스크 침식 관례와 동일하게 맞춤 (ICPParams 기본값과 일치).
DEFAULT_MASK_ERODE_PX = 3
MIN_POINTS_FOR_TRANSLATION = 10


class RTMDetInferencerRotHead(RTMDetInferencer):
    provides_pose_init = True

    def __init__(
        self,
        config: str,
        checkpoint: str,
        device: str = "cuda:0",
        score_threshold: float = 0.3,
        rotation_checkpoint: Optional[str] = None,
        rotation_backbone: str = "resnet18",
        rotation_crop_size: int = 128,
    ) -> None:
        """
        Args:
            rotation_checkpoint: 회전 회귀 헤드의 학습된 가중치 경로.
                None이면 ImageNet 사전학습 초기값만 쓰는 미학습 모델이라
                의미있는 회전을 못 낸다 - 실제 사용 전에는 반드시 지정할 것
                (scripts/train_rotation_head.py로 학습).
        """
        super().__init__(config, checkpoint, device, score_threshold)
        self._rot_regressor = CropRotationRegressor(
            checkpoint_path=rotation_checkpoint,
            device=device,
            backbone=rotation_backbone,
            crop_size=rotation_crop_size,
        )

    def infer(
        self,
        image: np.ndarray,
        pcd_organized_mm: Optional[np.ndarray] = None,
        valid_mask: Optional[np.ndarray] = None,
    ) -> List[DetectionResult]:
        """image + pcd_organized_mm(H,W,3 mm) + valid_mask(H,W bool)
        -> DetectionResult 리스트.

        pcd_organized_mm/valid_mask를 안 주면 baseline과 동일하게
        initial_pose=None만 채워서 반환한다 (호출부가 아직 3D 데이터를
        준비 못 했을 때도 안전하게 동작하도록).
        """
        base_results = super().infer(image)  # 기존 RTMDet-Ins 검출, 완전히 그대로

        if pcd_organized_mm is None or valid_mask is None or not base_results:
            return base_results

        image_bgr = image if image.ndim == 3 else np.stack([image] * 3, axis=-1)
        if image_bgr.dtype != np.uint8:
            image_bgr = image_bgr.astype(np.uint8)

        masks = [d.mask for d in base_results]
        bboxes = [d.bbox for d in base_results]
        rot6d_list = self._rot_regressor.predict(image_bgr, masks, bboxes)

        enriched: List[DetectionResult] = []
        for det, rot6d in zip(base_results, rot6d_list):
            initial_pose = None
            if rot6d is not None:
                rot_matrix = rot6d_to_matrix(rot6d)
                translation_m = self._estimate_translation_m(
                    det.mask, pcd_organized_mm, valid_mask
                )
                if translation_m is not None:
                    initial_pose = compose_pose(rot_matrix, translation_m)
                # depth 크롭 실패(occlusion 등)면 initial_pose=None으로 남겨
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

    @staticmethod
    def _estimate_translation_m(
        mask: np.ndarray, pcd_organized_mm: np.ndarray, valid_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """icp_runner.extract_instance_points_mm과 동일한 정의로 포인트를
        뽑아 centroid를 m 단위로 반환. 포인트가 너무 적으면 None."""
        pts_mm = extract_instance_points_mm(
            mask, pcd_organized_mm, valid_mask, erode_px=DEFAULT_MASK_ERODE_PX
        )
        if len(pts_mm) < MIN_POINTS_FOR_TRANSLATION:
            return None
        centroid_mm = pts_mm.mean(axis=0)
        return centroid_mm / 1000.0  # m 단위로 변환, icp_runner의 T_init과 일치