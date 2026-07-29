"""회전 회귀 헤드(RotationHeadNet) 학습 스크립트.

실행 (프로젝트 루트에서):
    python scripts/train_rotation_head.py --labels-json data/rotation_labels_bracket.json

RTMDet-Ins(2_Train_rtmdet_model.py)와는 완전히 독립된 별도 학습 파이프라인이다.
이 부품(CAD)에 대한 rtmdet-ins config가 이미 따로 있는 것과 동일한 이유로,
회전 헤드도 "부품 하나당 모델 하나"로 학습한다 - 그래서 대칭군(SYMMETRY_GROUP)은
배치마다 다른 게 아니라 이 스크립트 실행 전체에 적용되는 상수 하나로 고정한다.

라벨 JSON 포맷 (labels_json), 리스트의 각 항목:
    {
        "image": "data/dataset/<세션>/intensity/frame_0001.png",
        "mask": "data/rotation_labels/frame_0001_obj0_mask.npy",   # (H,W) bool
        "bbox": [x1, y1, x2, y2],
        "rotation_matrix": [[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]]
            # source(CAD) -> scene 회전, icp_runner의 T[:3,:3]과 동일한 정의.
    }

라벨 출처 권장: icp_runner.ICPResult.stage_logs에서 fitness가 높은(신뢰도
높은) 프레임만 필터링해서 T[:3,:3]을 pseudo-label로 재사용하는 방식
(본문 대화에서 논의한 방식 - 별도 스크립트로 이 JSON을 만들어서 넣으면 됨).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.rotation_head_model import (  # noqa: E402
    RotationHeadNet,
    crop_and_preprocess,
    symmetry_aware_geodesic_loss,
)

# --- 하드코딩 상수 (CLI로 override 가능) ---
DEFAULT_LABELS_JSON = str(ROOT / "data" / "rotation_labels.json")
DEFAULT_OUTPUT_DIR = str(ROOT / "checkpoints" / "rotation_head")
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-4
DEFAULT_CROP_SIZE = 128
DEFAULT_DEVICE = "cuda:0"
DEFAULT_CHECKPOINT_INTERVAL = 10

# 이 실행에서 학습할 부품의 대칭군. CAD 형상을 보고 판단해서 골라 넣을 것.
# "none": 대칭 없음 (일반적인 비대칭 부품)
# "z180": Z축 180도 회전 대칭 (예: 양끝이 똑같이 생긴 막대형 부품)
# 필요하면 여기에 새 그룹을 추가.
SYMMETRY_GROUPS: dict[str, list[np.ndarray]] = {
    "none": [np.eye(3, dtype=np.float32)],
    "z180": [np.eye(3, dtype=np.float32), np.diag([-1.0, -1.0, 1.0]).astype(np.float32)],
}
DEFAULT_SYMMETRY_GROUP = "none"


class RotationDataset(Dataset):
    """라벨 JSON 하나를 통째로 메모리에 올려 쓰는 단순한 Dataset.
    데이터가 수만 장 단위로 늘어나면 lazy-loading으로 바꿀 것."""

    def __init__(self, labels_json: str, crop_size: int = DEFAULT_CROP_SIZE):
        with open(labels_json, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        if not self.items:
            raise ValueError(f"라벨이 비어있습니다: {labels_json}")
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        image = cv2.imread(item["image"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {item['image']}")
        mask = np.load(item["mask"]).astype(bool)
        bbox = np.array(item["bbox"], dtype=np.float32)

        crop = crop_and_preprocess(image, mask, bbox, out_size=self.crop_size)
        if crop is None:
            # bbox가 이미지 범위를 벗어난 불량 라벨 - 검은 화면으로 대체하지 않고
            # 학습 데이터 정제 단계에서 걸러내는 게 맞음. 여기서는 명확히 에러.
            raise ValueError(f"크롭 실패 (bbox 범위 이상): {item['image']}, bbox={bbox}")

        gt_R = np.array(item["rotation_matrix"], dtype=np.float32)
        return crop, torch.from_numpy(gt_R)


def train(args: argparse.Namespace) -> None:
    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"[경고] CUDA를 쓸 수 없어 device를 '{args.device}' -> '{device}'로 변경합니다.")

    symmetry_group = SYMMETRY_GROUPS.get(args.symmetry_group)
    if symmetry_group is None:
        raise ValueError(
            f"알 수 없는 symmetry_group: '{args.symmetry_group}'. "
            f"지원: {list(SYMMETRY_GROUPS)}"
        )

    dataset = RotationDataset(args.labels_json, crop_size=args.crop_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"학습 데이터: {len(dataset)}개 인스턴스, symmetry_group='{args.symmetry_group}'")

    model = RotationHeadNet(backbone=args.backbone, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for crops, gt_R in loader:
            crops = crops.to(device)
            gt_R = gt_R.to(device)

            pred_rot6d = model(crops)
            loss = symmetry_aware_geodesic_loss(pred_rot6d, gt_R, symmetry_group)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * crops.shape[0]

        avg_loss_rad = epoch_loss / len(dataset)
        print(
            f"[epoch {epoch + 1}/{args.epochs}] geodesic loss = {avg_loss_rad:.4f} rad "
            f"({np.degrees(avg_loss_rad):.1f} deg)"
        )

        is_last = epoch == args.epochs - 1
        if (epoch + 1) % args.checkpoint_interval == 0 or is_last:
            ckpt_path = output_dir / f"rotation_head_epoch{epoch + 1}.pth"
            torch.save({"model_state_dict": model.state_dict(),
                        "backbone": args.backbone,
                        "crop_size": args.crop_size}, ckpt_path)
            print(f"  체크포인트 저장: {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-json", default=DEFAULT_LABELS_JSON)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument(
        "--symmetry-group", default=DEFAULT_SYMMETRY_GROUP,
        choices=list(SYMMETRY_GROUPS),
        help="이 부품의 대칭군 (rotation_head_model.py의 SYMMETRY_GROUPS 참고)",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()