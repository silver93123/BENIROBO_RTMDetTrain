"""GaussianUpsampler(src/upsampling/) 학습 스크립트.

핵심 아이디어: PU-Gaussian 원 논문은 PUGAN/PU1K 같은 일반 벤치마크 형상으로
학습하지만, 우리는 부품별 정확한 CAD가 있으므로 그걸 GT로 쓴다 - 도메인 갭을
줄이는 핵심 포인트.

학습쌍 만드는 법 (build_training_pair_from_icp_result):
    이미 성공한 ICP 결과(fitness 높은 것)의 T(4x4, CAD->scene)를 거꾸로
    적용해서, 실측 스캔 포인트를 CAD의 로컬 좌표계로 되돌린다. 그러면:
        - sparse_input = T^-1 적용한 실측 스캔 (CAD 로컬 프레임, mm)
        - dense_gt     = CAD 표면에서 균일 샘플링한 점 (같은 프레임, mm)
    이 둘이 "같은 좌표계 안의 sparse/dense 쌍"이 되어 그대로 학습에 쓸 수 있다.

실행 (프로젝트 루트에서, 학습쌍이 이미 준비돼 있다는 전제):
    python scripts/train_pc_upsampler.py --pairs-json data/pc_upsample_pairs.json

pairs-json 포맷, 리스트의 각 항목:
    {"sparse": "data/pc_pairs/sparse_0001.npy", "gt": "data/pc_pairs/gt_0001.npy"}
    두 파일 다 (N,3)/(M,3) float32, mm 단위, 같은 로컬 좌표계.

*** 학습쌍을 세션 전체에서 자동으로 뽑아 이 JSON을 만들어주는 스크립트
    (generate_rotation_labels.py와 같은 역할)는 아직 없다 - 이번엔
    build_training_pair_from_icp_result() 유틸리티까지만 만들었고,
    세션 일괄 처리는 다음 단계로 남겨뒀다. ***
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.upsampling import GaussianUpsampler, sample_gaussians, stage1_loss  # noqa: E402

# --- 하드코딩 상수 (CLI로 override 가능) ---
DEFAULT_PAIRS_JSON = str(ROOT / "data" / "pc_upsample_pairs.json")
DEFAULT_OUTPUT_DIR = str(ROOT / "checkpoints" / "pc_upsampler")
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 4  # 인스턴스마다 포인트 수가 달라 배치 내 padding이 필요 - 작게 시작
DEFAULT_LR = 1e-3
DEFAULT_SAMPLES_PER_POINT = 8  # r (업샘플 배수)
DEFAULT_FEATURE_DIM = 64
DEFAULT_K_NEIGHBORS = 16
DEFAULT_CHECKPOINT_INTERVAL = 10
DEFAULT_MAX_INPUT_POINTS = 512  # 배치 padding 단순화를 위해 입력을 이 개수로 서브샘플/패딩


# =============================================================================
# 학습쌍 생성 유틸 - icp_test_tab / generate_rotation_labels.py에서 이미
# 확보한 ICPResult를 바로 재활용할 수 있게 만든 함수.
# =============================================================================
def build_training_pair_from_icp_result(
    pts_scene_mm: np.ndarray, T_cad_to_scene: np.ndarray, cad_pcd_m,
) -> tuple[np.ndarray, np.ndarray]:
    """ICP 결과 하나로부터 (sparse_input_mm, dense_gt_mm) 학습쌍을 만든다.

    Args:
        pts_scene_mm: (N,3) 실측 스캔 포인트, scene(카메라) 좌표계, mm.
        T_cad_to_scene: (4,4) ICPResult.T와 동일 - source(CAD)->scene 변환, m 단위 관례.
        cad_pcd_m: icp_runner.load_cad_as_pcd()가 반환한 것과 동일한
            o3d.geometry.PointCloud (CAD 로컬 좌표계, m 단위).

    Returns:
        sparse_input_mm: (N,3) 실측 스캔을 CAD 로컬 좌표계로 되돌린 것, mm.
        dense_gt_mm: (M,3) CAD 표면 샘플, 같은 좌표계, mm.
    """
    T_inv = np.linalg.inv(T_cad_to_scene)
    pts_scene_m = pts_scene_mm.astype(np.float64) / 1000.0
    pts_h = np.concatenate([pts_scene_m, np.ones((len(pts_scene_m), 1))], axis=1)
    pts_cad_frame_m = (T_inv @ pts_h.T).T[:, :3]
    sparse_input_mm = (pts_cad_frame_m * 1000.0).astype(np.float32)

    dense_gt_mm = (np.asarray(cad_pcd_m.points, dtype=np.float64) * 1000.0).astype(np.float32)
    return sparse_input_mm, dense_gt_mm


# =============================================================================
# Dataset
# =============================================================================
class PairedPointCloudDataset(Dataset):
    def __init__(self, pairs_json: str, max_input_points: int = DEFAULT_MAX_INPUT_POINTS):
        with open(pairs_json, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        if not self.items:
            raise ValueError(f"학습쌍이 비어있습니다: {pairs_json}")
        self.max_input_points = max_input_points

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        sparse = np.load(item["sparse"]).astype(np.float32)
        gt = np.load(item["gt"]).astype(np.float32)

        n = len(sparse)
        if n >= self.max_input_points:
            keep = np.random.choice(n, self.max_input_points, replace=False)
            sparse = sparse[keep]
        else:
            pad_idx = np.random.choice(n, self.max_input_points - n, replace=True)
            sparse = np.concatenate([sparse, sparse[pad_idx]], axis=0)

        return torch.from_numpy(sparse), torch.from_numpy(gt)


def collate_variable_gt(batch):
    """sparse는 max_input_points로 고정 길이라 그대로 stack, gt는 인스턴스마다
    길이가 달라 리스트로 유지 (Chamfer는 배치 내 길이가 달라도 개별 계산 가능)."""
    sparses, gts = zip(*batch)
    return torch.stack(sparses), list(gts)


# =============================================================================
# 학습 루프
# =============================================================================
def train(args: argparse.Namespace) -> None:
    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"[경고] CUDA를 쓸 수 없어 device를 '{args.device}' -> '{device}'로 변경합니다.")

    dataset = PairedPointCloudDataset(args.pairs_json, max_input_points=args.max_input_points)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_variable_gt)
    print(f"학습쌍: {len(dataset)}개")

    model = GaussianUpsampler(feature_dim=args.feature_dim, k_neighbors=args.k_neighbors).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_cd = 0.0
        for sparse, gts in loader:
            sparse = sparse.to(device)  # (B, max_input_points, 3)
            mean, scale, R = model(sparse)
            sampled = sample_gaussians(mean, scale, R, r=args.samples_per_point)

            # gt는 인스턴스마다 길이가 달라 배치 내에서 개별적으로 loss 계산 후 평균
            losses = []
            cds = []
            for i, gt in enumerate(gts):
                gt_b = gt.unsqueeze(0).to(device)
                loss_i, logs_i = stage1_loss(
                    gt_b, mean[i:i + 1], scale[i:i + 1], R[i:i + 1], sampled[i:i + 1],
                )
                losses.append(loss_i)
                cds.append(logs_i["chamfer"])
            loss = torch.stack(losses).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * sparse.shape[0]
            epoch_cd += float(np.mean(cds)) * sparse.shape[0]

        avg_loss = epoch_loss / len(dataset)
        avg_cd = epoch_cd / len(dataset)
        print(f"[epoch {epoch + 1}/{args.epochs}] loss={avg_loss:.4f} chamfer={avg_cd:.4f}")

        is_last = epoch == args.epochs - 1
        if (epoch + 1) % args.checkpoint_interval == 0 or is_last:
            ckpt_path = output_dir / f"pc_upsampler_epoch{epoch + 1}.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "feature_dim": args.feature_dim,
                "k_neighbors": args.k_neighbors,
            }, ckpt_path)
            print(f"  체크포인트 저장: {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-json", default=DEFAULT_PAIRS_JSON)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--samples-per-point", type=int, default=DEFAULT_SAMPLES_PER_POINT)
    parser.add_argument("--feature-dim", type=int, default=DEFAULT_FEATURE_DIM)
    parser.add_argument("--k-neighbors", type=int, default=DEFAULT_K_NEIGHBORS)
    parser.add_argument("--max-input-points", type=int, default=DEFAULT_MAX_INPUT_POINTS)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()