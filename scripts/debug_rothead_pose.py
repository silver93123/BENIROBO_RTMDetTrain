"""RotHead 파이프라인이 실제로 initial_pose를 계산하는지 진단하는 스크립트.

"RotHead 탭으로 돌려도 RTMDet 탭과 똑같은 결과가 나온다"의 원인을 좁히기
위한 것. GUI를 거치지 않고 RTMDetInferencerRotHead.infer()를 직접 호출해서,
인스턴스별로:
    1) 회전 헤드가 rot6d를 냈는지 (크롭 실패 여부)
    2) translation(centroid)을 냈는지 (포인트 부족 여부)
    3) 최종 initial_pose가 None인지 아닌지
를 그대로 출력한다. 여기서 None이 나오면 -> ICP가 RTMDet과 똑같은 fallback을
타는 게 확인되는 것이고(버그 아니라 통과 조건 문제), 전부 정상 값이 나오는데
ICP 결과가 그대로라면 -> "ICP가 초기값에 안 민감한 쉬운 장면이라 최종
fitness가 어차피 비슷하게 수렴하는 것" 쪽에 무게가 실린다.

사용법: 아래 설정값만 고쳐서 실행.
    python scripts/debug_rothead_pose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# =============================================================================
# 설정값 - "6. 회전 라벨 생성" 탭에 입력한 값과 동일하게 맞출 것
# =============================================================================
RTMDET_CHECKPOINT = "/home/silver/binpicking_vision/BENIROBO_RTMDetTrain/work_dirs/rtmdet-ins_bolt_m10_80_v1/best_coco_bbox_mAP_epoch_40.pth"
RTMDET_CONFIG = "/home/silver/binpicking_vision/BENIROBO_RTMDetTrain/work_dirs/rtmdet-ins_bolt_m10_80_v1/rtmdet-ins_bolt_m10_80_used.py"
ROTATION_CHECKPOINT = "/home/silver/binpicking_vision/BENIROBO_RTMDetTrain/work_dirs_rohead/rotation_head_epoch30.pth"

# rotation_labels.json에 이미 저장된 항목 중 하나를 그대로 재사용 (image + mask
# 경로가 다 있으니 별도로 새로 촬영할 필요 없음). 몇 번째 항목을 쓸지 인덱스로 지정.
LABELS_JSON = ROOT / "data" / "rotation_labels.json"
LABEL_INDEX = 0

# scripts/train_rotation_head.py에서 이 라벨을 학습시킬 때 쓴 대칭군과 반드시
# 동일하게 맞출 것 - 안 맞으면 아래 각도 오차가 실제보다 훨씬 크게(최대 180도
# 가까이) 나와서 "학습이 하나도 안 됐다"고 오판하게 된다 (cylindrical_y로
# 학습했는데 여기서 "none"으로 비교하면 위상 차이를 전부 오차로 잡아버림).
SYMMETRY_GROUP = "cylindrical_y"

DEVICE = "cuda:0"
SCORE_THRESHOLD = 0.3
# =============================================================================


def main() -> None:
    import json
    import cv2

    from app.core.icp_runner import extract_instance_points_mm
    from src.detection.rtmdet_inferencer_rothead import RTMDetInferencerRotHead

    with open(LABELS_JSON, "r", encoding="utf-8") as f:
        items = json.load(f)
    item = items[LABEL_INDEX]
    print(f"테스트 이미지: {item['image']}")
    print(f"저장된 라벨의 bbox: {item['bbox']}")

    gray = cv2.imread(item["image"], cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {item['image']}")
    image_bgr = np.stack([gray, gray, gray], axis=-1)

    # rotation_labels.json은 pcd_organized/valid_mask를 따로 저장 안 하므로,
    # 여기서는 "회전 헤드가 rot6d를 내는지"만 순수하게 확인한다 (마스크는
    # 저장된 mask.npy를 그대로 씀 - RTMDet을 다시 돌릴 필요 없음).
    mask = np.load(item["mask"])
    bbox = np.array(item["bbox"])

    print("\n=== 1) RotationHead 체크포인트 로드 + rot6d 예측 ===")
    from src.detection.rotation_head_model import CropRotationRegressor
    regressor = CropRotationRegressor(checkpoint_path=ROTATION_CHECKPOINT, device=DEVICE)
    rot6d_list = regressor.predict(image_bgr, [mask], [bbox])
    rot6d = rot6d_list[0]

    if rot6d is None:
        print("  -> rot6d = None (크롭 실패! crop_and_preprocess가 이 마스크/bbox로 크롭을 못 만듦)")
        print("     원인 후보: bbox가 이미지 밖으로 나감, 마스크가 비어있음 등")
    else:
        print(f"  -> rot6d 정상 출력: {rot6d.round(4)}")
        from src.detection.rotation_utils import rot6d_to_matrix
        R = rot6d_to_matrix(rot6d)
        print(f"  -> 회전행렬:\n{R.round(4)}")

    print("\n=== 2) 저장된 라벨의 rotation_matrix(학습 시 ICP가 계산한 정답)과 비교 ===")
    gt_R = np.array(item["rotation_matrix"])
    print(f"  라벨 GT 회전행렬:\n{gt_R.round(4)}")
    if rot6d is not None:
        from scripts.train_rotation_head import SYMMETRY_GROUPS  # noqa: E402

        symmetry_group = SYMMETRY_GROUPS[SYMMETRY_GROUP]

        # 대칭을 무시한 raw 각도 오차 - 참고용으로만 같이 찍는다. cylindrical_*
        # 대칭에서는 위상만 다른 "실질적으로 맞는 답"도 이 값은 크게 나올 수
        # 있으므로, 아래 "대칭 고려" 쪽이 실제 품질 판단 기준이다.
        raw_err = np.degrees(np.arccos(np.clip((np.trace(R.T @ gt_R) - 1) / 2, -1, 1)))
        print(f"  대칭 무시한 raw 각도 오차: {raw_err:.1f}도 (참고용, 판단 기준 아님)")

        best_err = None
        for S in symmetry_group:
            gt_variant = gt_R @ S  # symmetry_aware_geodesic_loss와 동일한 곱셈 순서
            R_diff = R.T @ gt_variant
            cos_theta = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
            angle = np.degrees(np.arccos(cos_theta))
            if best_err is None or angle < best_err:
                best_err = angle
        print(f"  대칭({SYMMETRY_GROUP}) 고려한 실제 각도 오차: {best_err:.1f}도  <- 이게 진짜 판단 기준")
        print(
            "  (완전 랜덤 예측이면 평균 90도 근처가 나옴 - 이보다 훨씬 작으면 "
            "학습이 실제로 뭔가 배운 것)"
        )

    print("\n=== 3) translation(centroid) 계산은 pcd_organized/valid_mask가 있어야 확인 가능 ===")
    print("  (rotation_labels.json엔 저장 안 되는 데이터라 이 스크립트에서는 스킵)")
    print("  GUI에서 새로 촬영 -> RotHead 탭 -> '2D 검출 실행' 직후, 결과 카드의")
    print("  fitness 위에 pose_init 관련 로그가 있는지 학습 스크립트 출력 로그를 확인하세요.")


if __name__ == "__main__":
    main()