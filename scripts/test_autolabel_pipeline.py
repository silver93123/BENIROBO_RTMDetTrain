"""오토라벨링 파이프라인 end-to-end 테스트 (GPU/mmdet 불필요, 가짜 검출기 사용).

실행: python scripts/test_autolabel_pipeline.py   (레포 루트에서)
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.autolabel.config import DEFAULT  # noqa: E402
from src.autolabel.labelme_io import read_labelme, write_labelme  # noqa: E402
from src.autolabel.merge import export_coco, merge_round  # noqa: E402
from src.autolabel.predict import DetectorPredictor, predict_folder  # noqa: E402
from src.autolabel.queue_builder import build_queue, labelme_command  # noqa: E402

H, W = 240, 320


@dataclass
class FakeResult:
    class_id: int
    class_name: str
    score: float
    mask: np.ndarray


class FakeDetector:
    class_names = ["box", "bolt"]

    def infer(self, image):
        # 배경 밝기(모서리 픽셀)로 시나리오를 정하고, 마스크는 밝은 사각형 영역을 그대로 돌려준다.
        # 마스크가 이미지 내용을 따라가야 좌우반전 TTA 일치도가 현실처럼 계산된다.
        level = int(image[0, 0, 0])
        mask = image[..., 0] > 128
        if not mask.any():
            return []
        if level == 10:  # 확신 높은 쉬운 이미지
            return [FakeResult(0, "box", 0.97, mask)]
        return [FakeResult(1, "bolt", 0.35, mask)]  # 확신 낮은 이미지


def make_images(d: Path):
    d.mkdir(parents=True)
    for name, level in (("easy", 10), ("hard", 20), ("empty", 30)):
        img = np.full((H, W), level, np.uint8)
        if name == "easy":
            img[40:120, 40:140] = 200
        elif name == "hard":
            img[60:100, 150:230] = 200
        assert cv2.imwrite(str(d / f"{name}.png"), img)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        imgs, cache, rnd, ds = tmp / "imgs", tmp / "cache", tmp / "round", tmp / "ds"
        make_images(imgs)

        pred = DetectorPredictor(FakeDetector(), tag="fake")
        predict_folder(imgs, cache, pred, DEFAULT, progress=lambda m: None)
        assert len(list(cache.glob("*.pred.json"))) == 3

        summary = build_queue(imgs, cache, rnd, progress=lambda m: None)
        print("queue:", summary)
        manifest = json.loads((rnd / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["classes"] == ["box", "bolt"]
        by_stem = {v["original_stem"]: (k, v) for k, v in manifest["items"].items()}
        assert by_stem["easy"][1]["bucket"] == "auto", by_stem["easy"][1]
        assert by_stem["empty"][1]["bucket"] != "auto"  # 0개 검출은 최우선 검수
        assert by_stem["empty"][1]["priority"] == 1.0

        # 덮어쓰기 가드
        try:
            build_queue(imgs, cache, rnd, progress=lambda m: None)
            raise AssertionError("overwrite guard failed")
        except FileExistsError:
            pass

        # labelme 명령
        cmd = labelme_command(rnd, "urgent")
        assert cmd[0] == "labelme" and "--flags" in cmd and "--nodata" not in cmd

        # 사람 작업 시뮬레이션: hard 는 폴리곤 수정, empty 는 직접 그림 (reviewed ON), 그 외는 손대지 않음
        hk, hv = by_stem["hard"]
        hp = rnd / hv["bucket"] / f"{hk}.json"
        hp_payload = read_labelme(hp)
        assert hp_payload["flags"] == {"reviewed": False}
        for s in hp_payload["shapes"]:
            assert isinstance(s["flags"]["hole"], bool) and s["group_id"] is None
        hp_payload["shapes"][0]["points"] = [[150, 60], [230, 60], [230, 110], [150, 110]]  # 일부 확장
        hp_payload["shapes"][0]["description"] = ""  # 사람이 새로 그린 것처럼
        write_labelme(hp, hp_payload)

        ek, ev = by_stem["empty"]
        ep = rnd / ev["bucket"] / f"{ek}.json"
        e_payload = read_labelme(ep)
        e_payload["flags"]["reviewed"] = True  # "객체 없음" 확인
        write_labelme(ep, e_payload)

        # 모델 초안을 그대로 둔 미검수 케이스를 만들기 위해 hard 하나를 더 복제
        stats = merge_round(rnd, ds, DEFAULT, progress=lambda m: None)
        print("merge:", {k: stats[k] for k in ("images", "instances", "review_states", "unreviewed_excluded")})
        assert stats["review_states"] == {"auto": 1, "edited": 2}, stats["review_states"]
        assert stats["unreviewed_excluded"] == 0

        # 미검수 제외 확인: hard 를 다시 초안으로 되돌림
        hp.write_bytes(json.dumps(read_labelme(hp), ensure_ascii=False).encode())  # 해시가 달라지므로 edited 로 보임
        # 진짜 초안 복원: 대기열을 overwrite 로 재생성 후 아무것도 안 건드림
        build_queue(imgs, cache, rnd, overwrite=True, progress=lambda m: None)
        stats2 = merge_round(rnd, ds, DEFAULT, progress=lambda m: None)
        print("merge2:", stats2["review_states"], "excluded:", stats2["unreviewed_excluded"])
        assert stats2["review_states"].get("unreviewed") == 2
        assert stats2["images"] == 1  # auto 한 장만
        assert not (ds / "hard.json").exists() and not (ds / "empty.json").exists()  # 옛 병합 파일 정리됨

        coco = export_coco(ds, progress=lambda m: None)
        assert len(coco["images"]) == 1 and len(coco["annotations"]) == 1
        assert [c["name"] for c in coco["categories"]] == ["box", "bolt"]
        a = coco["annotations"][0]
        assert a["category_id"] == 1 and a["area"] > 7000, a

    print("ALL OK")


if __name__ == "__main__":
    main()
