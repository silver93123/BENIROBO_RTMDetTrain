"""GaussianUpsampler 학습용 손실 함수.

브루트포스 cdist 기반 Chamfer distance - 인스턴스당 포인트 수(수백~수천)
수준에서는 충분히 빠르다. 포인트 수가 수만 개를 넘어가면 KD-tree 기반
구현으로 바꿀 것 (pytorch3d.ops.knn_points 등).
"""
from __future__ import annotations

import torch


def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """대칭 Chamfer distance. a:(B,N,3), b:(B,M,3) -> (B,) 배치별 스칼라."""
    dist = torch.cdist(a, b)  # (B,N,M)
    min_a = dist.min(dim=2).values  # (B,N) - a의 각 점에서 b까지 최근접 거리
    min_b = dist.min(dim=1).values  # (B,M) - b의 각 점에서 a까지 최근접 거리
    return min_a.mean(dim=1) + min_b.mean(dim=1)


def gaussian_regularization_loss(
    gt_points: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, R: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """PU-Gaussian 논문 식(7): 각 GT 점의 최근접 예측 가우시안 평균을 찾아
    그 가우시안 기준 마할라노비스 거리를 최소화한다.

    gt_points:(B,M,3), mean/scale:(B,N,3), R:(B,N,3,3) -> (B,) 배치별 스칼라.
    이 항이 있어야 가우시안들이 GT 표면을 실제로 덮도록 유도된다 - Chamfer만
    쓰면 가우시안이 다른 이유로도(예: 원점 근처로 뭉침) 낮은 loss를 낼 수 있음.
    """
    dist = torch.cdist(gt_points, mean)        # (B,M,N)
    nearest_idx = dist.argmin(dim=2)            # (B,M)

    B, M, _ = gt_points.shape
    batch_idx = torch.arange(B, device=gt_points.device).view(B, 1).expand(-1, M)
    mu = mean[batch_idx, nearest_idx]            # (B,M,3)
    s = scale[batch_idx, nearest_idx]             # (B,M,3)
    Rn = R[batch_idx, nearest_idx]                 # (B,M,3,3)

    diff = (gt_points - mu).unsqueeze(-1)          # (B,M,3,1)
    local = torch.matmul(Rn.transpose(-1, -2), diff).squeeze(-1)  # 가우시안 로컬 축으로 투영
    maha = (local ** 2 / (s ** 2 + eps)).sum(-1)     # (B,M)
    return maha.mean(dim=1)


def stage1_loss(
    gt_points: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, R: torch.Tensor,
    sampled_points: torch.Tensor, w_chamfer: float = 1.0, w_gaussian: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """1단계(Gaussian 예측+샘플링) 전체 손실. (스칼라 loss, 로그용 dict) 반환."""
    cd = chamfer_distance(sampled_points, gt_points)
    reg = gaussian_regularization_loss(gt_points, mean, scale, R)
    total = w_chamfer * cd + w_gaussian * reg
    return total.mean(), {
        "chamfer": cd.mean().item(),
        "gaussian_reg": reg.mean().item(),
    }