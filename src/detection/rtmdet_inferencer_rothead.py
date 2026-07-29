"""RTMDet-Ins + 회전 헤드 추론 래퍼.

RTMDetInferencer를 상속해서 infer()만 확장한다: 기존 마스크/bbox 검출은
그대로 두고, 각 인스턴스에 대해 커스텀 rotation head 출력(6D 표현)을
읽어 initial_pose(4x4, m 단위)를 추가로 채운다.

번역(translation) 추정 방식(2026-07 수정):
    처음에는 depth+camera_intrinsics로 직접 역투영하는 방식으로 짰었으나,
    세션 데이터가 이미 픽셀별 3D 좌표(pointcloud_organized, H,W,3 mm)를
    들고 있어서 역투영이 불필요한 중복 계산이었다. icp_runner.py의
    extract_instance_points_mm()과 완전히 동일한 방식(마스크 x
    pointcloud_organized x valid_mask, 동일한 erode_px 관례)으로 포인트를
    뽑아 centroid를 취하는 것으로 통일했다. 이렇게 하면 두 스테이지
    (translation 추정, ICP 본 정합)가 "어떤 포인트를 물체로 보는지"에
    대해 정확히 같은 정의를 공유하게 된다.

회전(rotation)은 RGB 특징 기반, 이동(translation)은 depth 기반으로
분리해서 계산한다 - 순수 RGB만으로는 물체의 실제 스케일/거리가
근본적으로 모호하기 때문에, 이동은 항상 depth에서 가져온다.

*** 실제 회전 헤드 모델 구조 추가(멀티태스크 학습)는 mmdet config/커스텀
    head 정의 쪽에서 이뤄져야 함. 여기서는 pred_instances.rot6d 필드이
    있다고 가정하고 이를 읽어 DetectionResult.initial_pose로 조립하는
    어댑터 계층만 구현한다. ***
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from app.core.icp_runner import extract_instance_points_mm
from src.detection.base import DetectionResult
from src.detection.rtmdet_inferencer import RTMDetInferencer

# icp_runner.py의 마스크 침식 관례와 동일하게 맞춤 (ICPParams 기본값과 일치).
DEFAULT_MASK_ERODE_PX = 3
MIN_POINTS_FOR_TRANSLATION = 10


class RTMDetInferencerRotHead(RTMDetInferencer):
    provides_pose_init = True

    def infer(
        self,
        image: np.ndarray,
        pcd_organized_mm: Optional[np.ndarray] = None,
        valid_mask: Optional[np.ndarray] = None,
    ) -> List[DetectionResult]:
        """image + pcd_organized_mm(H,W,3 mm) + valid_mask(H,W bool)
        -> DetectionResult 리스트.

        pcd_organized_mm/valid_mask를 안 주면 회전 헤드 백엔드라도
        baseline과 동일하게 initial_pose=None만 채워서 반환한다
        (호출부가 아직 3D 데이터를 준비 못 했을 때도 안전하게 동작하도록).
        """
        base_results = super().infer(image)  # 기존 마스크/bbox 검출 재사용

        if pcd_organized_mm is None or valid_mask is None:
            return base_results

        from mmdet.apis import inference_detector

        image_bgr = image if image.ndim == 3 else np.stack([image] * 3, axis=-1)
        if image_bgr.dtype != np.uint8:
            image_bgr = image_bgr.astype(np.uint8)

        result = inference_detector(self._model, image_bgr)
        rot6d_batch = getattr(result.pred_instances, "rot6d", None)
        if rot6d_batch is None:
            # 회전 헤드가 없는 체크포인트로 로드된 경우 -> baseline과 동일 동작
            return base_results

        rot6d_np = np.asarray(rot6d_batch)
        # base_results는 score_threshold로 이미 필터링/정렬되어 있어 원본
        # pred_instances 인덱스와 어긋난다. score로 재매칭한다.
        scores_all = result.pred_instances.scores.cpu().numpy()

        enriched: List[DetectionResult] = []
        for det in base_results:
            match_idx = self._match_index(det, scores_all)
            initial_pose = None
            if match_idx is not None:
                from src.detection.rotation_utils import compose_pose, rot6d_to_matrix

                rot_matrix = rot6d_to_matrix(rot6d_np[match_idx])
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
    def _match_index(det: DetectionResult, scores_all: np.ndarray) -> Optional[int]:
        """score 값으로 원본 pred_instances 인덱스를 역매칭.

        간단하지만 score가 float32라 근사 비교 필요. 더 견고하게 하려면
        rotation head 학습 시 pred_instances에 instance_id를 같이 실어
        두는 걸 권장 (score 충돌 가능성 완전 배제).
        """
        candidates = np.where(np.isclose(scores_all, det.score, atol=1e-6))[0]
        return int(candidates[0]) if len(candidates) else None

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