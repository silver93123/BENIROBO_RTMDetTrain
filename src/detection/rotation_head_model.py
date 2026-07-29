"""2단계(decoupled) 회전 회귀 헤드.

RTMDet-Ins의 dense head(SimOTA 동적 라벨 할당, sep-BN 레벨별 conv tower,
동적 마스크 커널)는 전혀 건드리지 않는다. 대신:

    1) RTMDet-Ins가 이미 낸 마스크/bbox로 이미지를 크롭 (배경은 마스크로 제거)
    2) 크롭 패치를 이 작은 CNN(RotationHeadNet)에 통과시켜 6D rotation을 회귀

이렇게 분리하면:
    - RTMDet-Ins 학습 파이프라인(2_Train_rtmdet_model.py, config)은 손댈
      필요가 없다.
    - 회전 예측이 잘못돼도 탐지 자체 성능에는 전혀 영향을 주지 않아 원인
      분리가 쉽다.
    - 데이터셋이 이미 CAD당 1개 config(rtmdet-ins_bracket.py 등)로
      나뉘어 있으므로, 이 회전 헤드도 "부품 하나당 모델 하나"로 학습하는 게
      자연스럽다 - 즉 대칭군(symmetry group)은 학습 스크립트 실행 단위에서
      상수 하나로 고정해도 충분하다 (배치 내 인스턴스마다 다른 대칭군을
      고려할 필요가 없음).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

# --- 하드코딩 상수 (CLI/config로 override 가능) ---
DEFAULT_CROP_SIZE = 128
DEFAULT_BACKBONE = "resnet18"
DEFAULT_PRETRAINED = True  # ImageNet 사전학습 가중치로 초기화 (수렴 빠름)


# =============================================================================
# 모델
# =============================================================================
class RotationHeadNet(nn.Module):
    """마스크로 배경 제거된 RGB 크롭 -> 6D rotation.

    입력: (B, 3, H, W) float32, [0, 1] 정규화.
    출력: (B, 6) raw 6D 표현. Gram-Schmidt 정규화는 여기서 하지 않고
        rotation_utils.rot6d_to_matrix()(추론) 또는 rot6d_to_matrix_torch()
        (학습, 이 파일 안)에서 한다 - loss 계산 시점에 정규화해야
        gradient가 올바르게 흐른다.
    """

    def __init__(self, backbone: str = DEFAULT_BACKBONE, pretrained: bool = DEFAULT_PRETRAINED):
        super().__init__()
        if backbone != "resnet18":
            raise ValueError(f"지원하지 않는 backbone: {backbone} (지금은 resnet18만 지원)")

        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(net.children())[:-1])  # avgpool까지, fc 제거
        feat_dim = net.fc.in_features

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = torch.flatten(feat, 1)
        return self.head(feat)


# =============================================================================
# 6D rotation <-> 회전행렬, torch 미분가능 버전 (학습용)
# rotation_utils.py의 numpy 버전과 수식은 동일하나, 배치 텐서 + autograd 호환.
# =============================================================================
def rot6d_to_matrix_torch(rot6d: torch.Tensor) -> torch.Tensor:
    """(B, 6) -> (B, 3, 3). Zhou et al. 2019 Gram-Schmidt 방식."""
    a1 = rot6d[:, 0:3]
    a2 = rot6d[:, 3:6]

    b1 = torch.nn.functional.normalize(a1, dim=1)
    a2_proj = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2_proj, dim=1)
    b3 = torch.cross(b1, b2, dim=1)

    return torch.stack([b1, b2, b3], dim=-1)  # (B,3,3), 열벡터로 쌓음


# =============================================================================
# 대칭 인식 geodesic loss
# =============================================================================
def symmetry_aware_geodesic_loss(
    pred_rot6d: torch.Tensor,
    gt_rotation: torch.Tensor,
    symmetry_group: Sequence[np.ndarray],
) -> torch.Tensor:
    """예측 6D와 GT 회전행렬 사이의 geodesic angle loss (라디안), 대칭군 고려.

    Args:
        pred_rot6d: (B, 6).
        gt_rotation: (B, 3, 3). source(CAD)->scene GT 회전.
        symmetry_group: 이 배치(=보통 부품 하나) 전체에 공통 적용되는 대칭
            변환 리스트, 각 원소 (3,3) np.ndarray. 예: Z축 180도 대칭 부품이면
            [I, diag(-1,-1,1)]. 최소 [I]는 항상 포함(대칭 없음).

    Returns:
        스칼라 loss (라디안 단위 평균 각도 오차, 대칭 등가 중 최솟값).
    """
    pred_R = rot6d_to_matrix_torch(pred_rot6d)  # (B,3,3)

    per_symmetry_theta = []
    for S in symmetry_group:
        S_t = torch.as_tensor(S, dtype=gt_rotation.dtype, device=gt_rotation.device)
        gt_variant = torch.matmul(gt_rotation, S_t)  # (B,3,3)
        R_diff = torch.matmul(pred_R.transpose(1, 2), gt_variant)
        trace = R_diff.diagonal(dim1=1, dim2=2).sum(-1)
        cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        per_symmetry_theta.append(torch.acos(cos_theta))

    stacked = torch.stack(per_symmetry_theta, dim=0)  # (K, B)
    min_theta, _ = stacked.min(dim=0)  # (B,) - 대칭 등가 중 가장 가까운 것
    return min_theta.mean()


# =============================================================================
# 전처리: 검출 마스크/bbox -> 모델 입력 텐서
# =============================================================================
def crop_and_preprocess(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: np.ndarray,
    out_size: int = DEFAULT_CROP_SIZE,
    mask_background: bool = True,
) -> Optional[torch.Tensor]:
    """bbox로 크롭 -> (옵션) 마스크 밖 배경 제거 -> out_size 정사각형 리사이즈
    -> CHW float32 [0,1] 텐서.

    bbox가 이미지 범위를 벗어나 크롭 영역이 비어버리면 None을 반환한다
    (호출부는 이 인스턴스의 initial_pose를 그냥 None으로 남기면 됨).
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image_bgr[y1:y2, x1:x2].copy()
    if mask_background:
        m = mask[y1:y2, x1:x2]
        crop[~m] = 0

    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    crop = crop.astype(np.float32) / 255.0
    return torch.from_numpy(crop).permute(2, 0, 1).contiguous()  # HWC -> CHW


# =============================================================================
# 추론 wrapper - RTMDetInferencerRotHead가 이 클래스를 통해 rot6d를 얻는다.
# =============================================================================
class CropRotationRegressor:
    """RotationHeadNet 로드 + 인스턴스별 배치 추론.

    checkpoint_path가 None이면 ImageNet 사전학습 가중치만으로 초기화된
    미학습 모델을 쓴다 - 당연히 의미있는 회전을 못 내므로, 이 경우 호출부는
    반환값을 신뢰하지 말고 fallback pose_init을 쓰는 게 맞다 (지금은 별도
    플래그 없이 그냥 학습 안 된 값이 나가니, 실제 배포 전에는 반드시
    checkpoint_path를 지정할 것).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        backbone: str = DEFAULT_BACKBONE,
        crop_size: int = DEFAULT_CROP_SIZE,
    ):
        self.device = device
        self.crop_size = crop_size
        self.model = RotationHeadNet(backbone=backbone, pretrained=(checkpoint_path is None))
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location=device)
            state_dict = state["model_state_dict"] if "model_state_dict" in state else state
            self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        image_bgr: np.ndarray,
        masks: Sequence[np.ndarray],
        bboxes: Sequence[np.ndarray],
    ) -> List[Optional[np.ndarray]]:
        """이미지 한 장 + 인스턴스별 마스크/bbox -> 인스턴스별 rot6d((6,) np)
        또는 크롭 실패 시 None. 입력 순서와 1:1로 대응하는 리스트를 반환한다."""
        tensors: List[torch.Tensor] = []
        valid_idx: List[int] = []
        for i, (mask, bbox) in enumerate(zip(masks, bboxes)):
            t = crop_and_preprocess(image_bgr, mask, bbox, out_size=self.crop_size)
            if t is not None:
                tensors.append(t)
                valid_idx.append(i)

        results: List[Optional[np.ndarray]] = [None] * len(masks)
        if not tensors:
            return results

        batch = torch.stack(tensors).to(self.device)
        rot6d_pred = self.model(batch).cpu().numpy()  # (N, 6)
        for j, idx in enumerate(valid_idx):
            results[idx] = rot6d_pred[j]
        return results