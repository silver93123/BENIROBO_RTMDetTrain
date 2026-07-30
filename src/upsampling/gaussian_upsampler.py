"""지역 비등방(anisotropic) 3D 가우시안 기반 포인트클라우드 업샘플러.

PU-Gaussian(Khater et al., 2025, arXiv:2509.20207)의 1단계(Gaussian
prediction + sampling)를 참고해서 구현했다. 입력 포인트 하나하나마다
"이 지점 주변 지역 표면이 어느 방향으로 얼마나 퍼져있는지"를 나타내는
비등방 가우시안 N(mu_i, Sigma_i)을 예측하고, 거기서 r개씩 재샘플링해
밀도를 높인다.

원 논문과의 의도적인 차이 두 가지:
    1. 백본을 Point Transformer 대신 훨씬 가벼운 kNN 기반 로컬
       특징추출기(PointNet류)로 바꿨다 - 원 논문은 장면 전체(2048pt)를
       다루지만, 우리는 이미 마스크로 크롭된 부품 하나(수백~수천 점)만
       다루므로 무거운 어텐션 백본이 굳이 필요 없다고 판단했다. 성능이
       부족하면 이 백본만 Point Transformer로 교체하면 된다(아래
       LocalFeatureExtractor를 감싸는 인터페이스로 분리해둠).
    2. 스케일 파라미터화를 논문의 softmax(3축 합=1, 상대 비율만 표현) 대신
       softplus(양수 절대값)로 바꿨다. 논문은 벤치마크 관례상 포인트클라우드를
       단위구/큐브로 정규화해서 쓰지만, 우리는 mm 단위 절대 좌표를 그대로
       다뤄야 하므로 스케일도 절대 표준편차(mm)로 나와야 downstream(ICP)에
       의미가 있다.

*** 이 파일은 1단계(Gaussian 예측+샘플링)까지만 구현한다. 논문의 2단계
    refinement 네트워크(edge 보존용 후처리)는 포함하지 않았다 - 우선
    1단계만으로 충분한지 확인한 뒤 필요하면 추가하는 게 순서라고 판단했다. ***
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_FEATURE_DIM = 64
DEFAULT_K_NEIGHBORS = 16
DEFAULT_CLIP_STD = 2.0  # 논문과 동일: 평균에서 2 표준편차 넘는 샘플은 버림(재추출)


# =============================================================================
# 기하 유틸
# =============================================================================
def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) 쿼터니언(w,x,y,z) -> (..., 3, 3) 회전행렬."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2),
    ], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


def knn_indices(points: torch.Tensor, k: int) -> torch.Tensor:
    """(B,N,3) -> (B,N,k) 최근접 이웃 인덱스 (자기 자신 포함, 브루트포스 - 인스턴스당
    포인트 수가 수천 개 수준이라 충분히 빠름. 수만 개 넘어가면 KD-tree로 바꿀 것)."""
    dist = torch.cdist(points, points)
    k = min(k, points.shape[1])
    return dist.topk(k, dim=-1, largest=False).indices


def gather_neighbors(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """points:(B,N,C), idx:(B,N,k) -> (B,N,k,C)."""
    B = points.shape[0]
    batch_idx = torch.arange(B, device=points.device).view(B, 1, 1).expand_as(idx)
    return points[batch_idx, idx]


# =============================================================================
# 백본 (경량 kNN 로컬 특징추출기 - Point Transformer 자리에 넣은 것)
# =============================================================================
class LocalFeatureExtractor(nn.Module):
    """입력 포인트마다 k-이웃의 상대좌표를 shared MLP + max-pool로 인코딩한다.

    상대좌표(neighbor - center)를 쓰기 때문에 전체 좌표계 이동에는 불변이다
    (회전 불변은 아님 - 필요하면 PCA로 로컬 프레임을 맞추는 전처리를 추가할 것).
    """

    def __init__(self, feature_dim: int = DEFAULT_FEATURE_DIM, k: int = DEFAULT_K_NEIGHBORS):
        super().__init__()
        self.k = k
        self.mlp_local = nn.Sequential(
            nn.Conv2d(3, 32, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
        )
        self.mlp_global = nn.Sequential(
            nn.Conv1d(64, feature_dim, 1), nn.BatchNorm1d(feature_dim), nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """points: (B,N,3) -> (B,N,feature_dim)."""
        idx = knn_indices(points, self.k)
        neighbors = gather_neighbors(points, idx)          # (B,N,k,3)
        rel = neighbors - points.unsqueeze(2)               # (B,N,k,3) 지역 상대좌표
        rel = rel.permute(0, 3, 1, 2)                        # (B,3,N,k) - Conv2d 입력형
        feat = self.mlp_local(rel).max(dim=-1).values         # (B,64,N) - 이웃 축 max-pool
        feat = self.mlp_global(feat)                           # (B,feature_dim,N)
        return feat.permute(0, 2, 1)                            # (B,N,feature_dim)


# =============================================================================
# 가우시안 파라미터 회귀 헤드
# =============================================================================
class GaussianRegressor(nn.Module):
    def __init__(self, feature_dim: int = DEFAULT_FEATURE_DIM):
        super().__init__()

        def head(out_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Conv1d(feature_dim, feature_dim, 1), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(feature_dim, out_dim, 1),
            )

        self.head_scale = head(3)     # 비등방 스케일(표준편차, mm) 원값 - softplus는 밖에서 적용
        self.head_rotation = head(4)  # 쿼터니언
        self.head_offset = head(3)    # 평균 위치 오프셋 (mm)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = feat.permute(0, 2, 1)  # (B,C,N)
        raw_scale = self.head_scale(f).permute(0, 2, 1)
        quat = self.head_rotation(f).permute(0, 2, 1)
        offset = self.head_offset(f).permute(0, 2, 1)
        return raw_scale, quat, offset


# =============================================================================
# 전체 모델
# =============================================================================
class GaussianUpsampler(nn.Module):
    """입력 포인트 (B,N,3) mm -> 포인트별 (mean, scale, R) 가우시안 파라미터.

    scale_init_bias: softplus(raw_scale + bias) 형태로 스케일을 만든다. bias를
    음수로 주면 학습 초반에 스케일이 작게(=원본 위치 주변에 촘촘히) 시작해서
    학습이 안정적으로 출발한다 - 처음부터 크게 흩어지면 Chamfer loss가 불안정.
    """

    def __init__(self, feature_dim: int = DEFAULT_FEATURE_DIM, k_neighbors: int = DEFAULT_K_NEIGHBORS,
                 scale_init_bias: float = -3.0):
        super().__init__()
        self.backbone = LocalFeatureExtractor(feature_dim, k_neighbors)
        self.regressor = GaussianRegressor(feature_dim)
        self.scale_init_bias = scale_init_bias

    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """points: (B,N,3) mm -> mean(B,N,3), scale(B,N,3) mm(양수), R(B,N,3,3)."""
        feat = self.backbone(points)
        raw_scale, quat, offset = self.regressor(feat)
        scale = F.softplus(raw_scale + self.scale_init_bias) + 1e-4
        mean = points + offset
        R = quaternion_to_matrix(quat)
        return mean, scale, R


def sample_gaussians(
    mean: torch.Tensor, scale: torch.Tensor, R: torch.Tensor,
    r: int, clip_std: float = DEFAULT_CLIP_STD,
) -> torch.Tensor:
    """포인트별 가우시안에서 r개씩 재샘플링 (reparameterization trick).

    mean:(B,N,3), scale:(B,N,3), R:(B,N,3,3) -> (B, N*r, 3).
    논문과 동일하게 |eps| > clip_std인 샘플은 clamp한다(근사 - 엄밀한
    reject-resample은 배치 크기가 안 맞아 구현이 번거로워 clamp로 대체).
    """
    B, N, _ = mean.shape
    eps = torch.randn(B, N, r, 3, device=mean.device, dtype=mean.dtype)
    eps = eps.clamp(-clip_std, clip_std)
    scaled = eps * scale.unsqueeze(2)                           # (B,N,r,3)
    rotated = torch.einsum('bnij,bnrj->bnri', R, scaled)         # 지역 축 -> 월드 축
    sampled = mean.unsqueeze(2) + rotated                         # (B,N,r,3)
    return sampled.reshape(B, N * r, 3)