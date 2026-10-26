"""mask_refine 자체 테스트 (합성 데이터, GPU/CAD/카메라 불필요).

볼트 모양 메쉬(원통 몸통 + 육각 머리)를 numpy로 만들고, 간단한 z-buffer로 두 볼트와 바닥이
있는 장면의 depth/ID 맵을 렌더링한 뒤, 일부러 망가뜨린 마스크(이웃 번짐, 중간 끊김, 바닥 번짐)를
보정해서 정답 마스크와의 IoU가 좋아지는지 확인한다.

실행 (레포 루트에서):
    python scripts/test_mask_refine.py            # 결정적 테스트 (ALL OK 확인)
    python scripts/test_mask_refine.py --stress   # 무작위 장면 통계 (참고용, 노이즈 0.4/2/4mm)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.mask_refine import RefineParams, SilhouetteBank, refine_mask  # noqa: E402


# ----------------------------------------------------------------------------- 메쉬
def _ring(x: float, r: float, n: int, phase: float = 0.0) -> np.ndarray:
    a = phase + np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([np.full(n, x), r * np.cos(a), r * np.sin(a)], axis=1)


def _band(v: list, i0: int, i1: int, n: int) -> list:
    return [t for k in range(n) for t in ([i0 + k, i0 + (k + 1) % n, i1 + k],
                                          [i0 + (k + 1) % n, i1 + (k + 1) % n, i1 + k])]


def _fan(v: list, ring0: int, n: int, x: float) -> list:
    v.append(np.array([x, 0.0, 0.0]))
    c = len(v) - 1
    return [[ring0 + k, ring0 + (k + 1) % n, c] for k in range(n)]


def make_bolt_mesh(length=80.0, shaft_r=5.0, head_h=6.4, head_r=9.2) -> tuple[np.ndarray, np.ndarray]:
    """축 = x. 머리: x in [-head_h, 0], 몸통: x in [0, length]."""
    v, f = [], []
    n_s, n_h = 24, 6
    # 몸통
    base = len(v); v.extend(_ring(0.0, shaft_r, n_s)); v.extend(_ring(length, shaft_r, n_s))
    f += _band(v, base, base + n_s, n_s)
    f += _fan(v, base + n_s, n_s, length)
    # 머리
    hb = len(v); v.extend(_ring(-head_h, head_r, n_h)); v.extend(_ring(0.0, head_r, n_h))
    f += _band(v, hb, hb + n_h, n_h)
    f += _fan(v, hb, n_h, -head_h)
    f += _fan(v, hb + n_h, n_h, 0.0)
    return np.array(v), np.array(f)


def rot_x_to(a: np.ndarray) -> np.ndarray:
    """x축을 단위벡터 a로 보내는 회전행렬 (roll은 임의)."""
    x = np.array([1.0, 0, 0]); a = a / np.linalg.norm(a)
    v = np.cross(x, a); c = float(x @ a)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([-1.0, -1, 1])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def axis_dir(phi_deg: float, theta_deg: float) -> np.ndarray:
    p, t = math.radians(phi_deg), math.radians(theta_deg)
    return np.array([math.cos(p) * math.cos(t), math.cos(p) * math.sin(t), math.sin(p)])


# ----------------------------------------------------------------------------- 렌더러
def render(meshes: list[tuple[np.ndarray, np.ndarray]], f: float, cx: float, cy: float,
           w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """메쉬(카메라 좌표 mm) 리스트 -> (depth[mm], id). 배경 depth=inf, id=-1."""
    depth = np.full((h, w), np.inf)
    ident = np.full((h, w), -1, dtype=np.int32)
    for k, (verts, faces) in enumerate(meshes):
        z = verts[:, 2]
        uv = np.stack([f * verts[:, 0] / z + cx, f * verts[:, 1] / z + cy], axis=1)
        for tri in faces:
            p, zz = uv[tri], z[tri]
            x0, x1 = max(int(math.floor(p[:, 0].min())), 0), min(int(math.ceil(p[:, 0].max())), w - 1)
            y0, y1 = max(int(math.floor(p[:, 1].min())), 0), min(int(math.ceil(p[:, 1].max())), h - 1)
            if x1 < x0 or y1 < y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
            den = (p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0]) + (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1])
            if abs(den) < 1e-12:
                continue
            l0 = ((p[1, 1] - p[2, 1]) * (gx - p[2, 0]) + (p[2, 0] - p[1, 0]) * (gy - p[2, 1])) / den
            l1 = ((p[2, 1] - p[0, 1]) * (gx - p[2, 0]) + (p[0, 0] - p[2, 0]) * (gy - p[2, 1])) / den
            l2 = 1 - l0 - l1
            inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
            zi = 1.0 / (l0 / zz[0] + l1 / zz[1] + l2 / zz[2])
            sub_d, sub_i = depth[y0:y1 + 1, x0:x1 + 1], ident[y0:y1 + 1, x0:x1 + 1]
            upd = inside & (zi < sub_d)
            sub_d[upd] = zi[upd]
            sub_i[upd] = k
    return depth, ident


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / u) if u else 0.0


def pcd_from_depth(depth: np.ndarray, f: float, cx: float, cy: float, rng, noise=0.4, hole=0.015):
    h, w = depth.shape
    z = depth.copy()
    fin = np.isfinite(z)
    z[fin] += rng.normal(0, noise, size=int(fin.sum()))
    valid = fin & (rng.random(z.shape) > hole)
    gy, gx = np.mgrid[0:h, 0:w]
    pcd = np.stack([(gx - cx) * z / f, (gy - cy) * z / f, z], axis=-1)
    pcd[~valid] = np.nan
    return pcd, valid



def stress(n: int = 40, noise: float = 0.4, hole: float = 0.015, seed: int = 0) -> None:
    """무작위 자세/이웃/가림/노이즈로 n개 장면을 만들어 보정 전후 IoU 통계를 낸다 (참고용, 합격 기준 없음)."""
    verts, faces = make_bolt_mesh()
    bank = SilhouetteBank(verts, faces, phi_step_deg=5.0)
    rng = np.random.default_rng(seed)
    w, h, f = 640, 480, 600.0
    cx, cy = w / 2, h / 2
    floor = np.array([[-400, -300, 650], [400, -300, 650], [400, 300, 650], [-400, 300, 650]], float)
    floor_mesh = (floor, np.array([[0, 1, 2], [0, 2, 3]]))
    rows = []
    for i in range(n):
        phi, th = rng.uniform(-75, 75), rng.uniform(0, 360)
        ta = np.array([rng.uniform(-60, 60), rng.uniform(-40, 40), rng.uniform(480, 600)])
        poses = [(phi, th, ta)]
        # 뒤 이웃(필수), 앞 가림 물체(50%)
        poses.append((rng.uniform(-75, 75), rng.uniform(0, 360),
                      ta + np.array([rng.uniform(-25, 25), rng.uniform(-25, 25), rng.uniform(25, 60)])))
        if rng.random() < 0.5:
            poses.append((rng.uniform(-30, 30), rng.uniform(0, 360),
                          ta + np.array([rng.uniform(-10, 10), rng.uniform(-10, 10), -rng.uniform(25, 45)])))
        meshes = [(verts @ rot_x_to(axis_dir(a, b)).T + t, faces) for a, b, t in poses] + [floor_mesh]
        depth, ident = render(meshes, f, cx, cy, w, h)
        gt = ident == 0
        if gt.sum() < 150:
            continue
        pcd, valid = pcd_from_depth(depth, f, cx, cy, rng, noise=noise, hole=hole)
        # 망가뜨리기: 이웃/바닥 번짐 + 무작위 띠 제거
        k = int(rng.integers(15, 36)) | 1
        near = cv2.dilate(gt.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
        det = gt | (near & (ident != 0) & (ident != 2))
        ys, xs = np.nonzero(gt)
        ang = math.radians(th)
        proj = (xs - xs.mean()) * math.cos(ang) + (ys - ys.mean()) * math.sin(ang)
        span = max(proj.max() - proj.min(), 1.0)
        if rng.random() < 0.7:
            cut = np.abs(proj - rng.uniform(-0.3, 0.3) * span) < rng.uniform(0.04, 0.10) * span
            det[ys[cut], xs[cut]] = False
        if det.sum() < 80:
            continue
        res = refine_mask(det, pcd, valid, (f, f, cx, cy), bank, RefineParams())
        rows.append((phi, iou(det, gt), iou(res.mask, gt), res.ok, res.reason))
    if not rows:
        print("[stress] 유효 장면 없음")
        return
    b = np.array([r[1] for r in rows]); a = np.array([r[2] for r in rows]); okm = np.array([r[3] for r in rows])
    print(f"[stress noise={noise}mm hole={hole:.1%}] 장면 {len(rows)}개 | 보정 적용 {okm.sum()}개, 포기 {(~okm).sum()}개")
    print(f"   IoU 평균 {b.mean():.3f} -> {a.mean():.3f} | 중앙값 {np.median(b):.3f} -> {np.median(a):.3f}")
    print(f"   개선(+0.05↑) {(a - b > 0.05).sum()}개, 악화(-0.05↓) {(a - b < -0.05).sum()}개")
    bad = [(round(r[0]), round(r[1], 2), round(r[2], 2), r[4]) for r in rows if r[2] - r[1] < -0.05]
    if bad:
        print("   악화 사례 (phi, before, after, reason):", bad[:8])
    reasons = {}
    for r in rows:
        if not r[3]:
            reasons[r[4]] = reasons.get(r[4], 0) + 1
    if reasons:
        print("   포기 사유:", reasons)

# ----------------------------------------------------------------------------- 테스트
def main() -> int:
    ok_all = True
    verts, faces = make_bolt_mesh()
    bank = SilhouetteBank(verts, faces, phi_step_deg=5.0)
    print(f"[bank] length={bank.length_mm:.1f}mm width={bank.width_mm:.1f}mm aspect={bank.aspect:.2f} "
          f"phi={bank.n_phi}개 base={bank._size}px")

    # --- T1. theta 부호/방향: phi=0 템플릿의 PCA 주축 각도가 theta와 같아야 한다 ---
    pi0 = int(np.argmin(np.abs(bank.phi_deg)))
    for th in (0.0, 30.0, 100.0):
        tpl = bank.render(pi0, th, 2.0, 400)
        ys, xs = np.nonzero(tpl > 0.5)
        evals, evecs = np.linalg.eigh(np.cov(np.stack([xs - xs.mean(), ys - ys.mean()])))
        ang = math.degrees(math.atan2(evecs[1, 1], evecs[0, 1])) % 180
        good = abs(((ang - th % 180) + 90) % 180 - 90) < 2.0
        ok_all &= good
        print(f"[T1] theta={th:5.1f} -> 템플릿 주축 {ang:6.1f}°  {'OK' if good else 'FAIL'}")

    # --- 장면: A(대상), B(A보다 40mm 뒤, 옆), C(A보다 ~35mm 앞에서 A를 가로질러 가림), 바닥 ---
    w, h, f = 640, 480, 600.0
    cx, cy = w / 2, h / 2
    rng = np.random.default_rng(1)
    poses = [
        (25.0, 35.0, np.array([-30.0, 0.0, 520.0])),      # A
        (-10.0, 110.0, np.array([-8.0, 20.0, 562.0])),    # B (뒤)
        (5.0, 105.0, np.array([-28.0, 0.0, 485.0])),      # C (앞, A를 가림)
    ]
    meshes = []
    for phi, th, t in poses:
        rot = rot_x_to(axis_dir(phi, th))
        meshes.append((verts @ rot.T + t, faces))
    floor = np.array([[-400, -300, 650], [400, -300, 650], [400, 300, 650], [-400, 300, 650]], float)
    meshes.append((floor, np.array([[0, 1, 2], [0, 2, 3]])))
    depth, ident = render(meshes, f, cx, cy, w, h)
    gt_a, gt_b, gt_c, floor_id = ident == 0, ident == 1, ident == 2, 3
    pcd, valid = pcd_from_depth(depth, f, cx, cy, rng)
    n_cc = cv2.connectedComponents(gt_a.astype(np.uint8), connectivity=8)[0] - 1
    # 볼트 A 안에 반사로 인한 무효 구멍
    ays, axs = np.nonzero(gt_a)
    mid = np.argmin(np.abs(axs - np.median(axs)))
    valid[ays[mid] - 3:ays[mid] + 4, axs[mid] - 3:axs[mid] + 4] = False
    print(f"[scene] A {gt_a.sum()}px (C에 가려 {n_cc}조각), B {gt_b.sum()}px, C {gt_c.sum()}px")

    intr = (f, f, cx, cy)
    params = RefineParams()

    def run(name: str, det: np.ndarray, gt: np.ndarray, min_gain: float | None, forbid: np.ndarray | None = None):
        nonlocal ok_all
        res = refine_mask(det, pcd, valid, intr, bank, params)
        before, after = iou(det, gt), iou(res.mask, gt)
        extra = ""
        good = True
        if forbid is not None:  # 가림 물체(C) 픽셀을 새로 채워 넣으면 안 된다
            leaked = int(np.count_nonzero(res.mask & forbid & ~det))
            extra = f" | 가린 물체로 샌 픽셀 {leaked}"
            good &= leaked <= 0.1 * np.count_nonzero(forbid)
        if min_gain is None:       # 이미 정답: 나빠지면 안 됨
            good &= after >= before - 0.05
        else:
            good &= res.ok and after >= before + min_gain
        ok_all &= good
        print(f"[{name}] IoU {before:.3f} -> {after:.3f} | {res.summary()}{extra}   " + ("OK" if good else "FAIL"))
        return res

    # T2. 이미 정확한 (C에 가려 둘로 갈라진) 마스크는 망가뜨리지 않고, 가려진 곳을 채우지도 않는다
    run("T2 정답(가림) A  ", gt_a.copy(), gt_a, None, forbid=gt_c)

    # T3. 이웃(B) 번짐 + 중간 끊김(A)
    near_a = cv2.dilate(gt_a.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))).astype(bool)
    bleed = gt_b & near_a
    ang = math.radians(35)
    proj = (axs - axs.mean()) * math.cos(ang) + (ays - ays.mean()) * math.sin(ang)
    span = proj.max() - proj.min()
    cut = np.zeros_like(gt_a)
    sel = np.abs(proj - 0.30 * span) < 0.07 * span
    cut[ays[sel], axs[sel]] = True
    det_a_bad = (gt_a | bleed) & ~cut
    print(f"       (번짐 {int(bleed.sum())}px, 제거한 띠 {int((gt_a & cut).sum())}px)")
    ok_all &= bleed.sum() > 30
    run("T3 A: 이웃번짐+끊김", det_a_bad, gt_a, 0.05, forbid=gt_c)

    # T4. 바닥으로 번짐 (B): 마스크의 대부분이 바닥이어도 물체 평면을 골라야 한다
    near_b = cv2.dilate(gt_b.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))).astype(bool)
    det_b = gt_b | ((ident == floor_id) & near_b)
    print(f"       (바닥 번짐 {int((det_b & ~gt_b).sum())}px vs 물체 {int(gt_b.sum())}px)")
    run("T4 B: 바닥번짐      ", det_b, gt_b, 0.05)

    # T5. 쓰레기 마스크(모양/깊이 모두 이상)는 보정을 포기해야 한다 (원본 유지)
    junk = np.zeros_like(gt_a)
    junk[300:330, 100:400] = True
    res = refine_mask(junk, pcd, valid, intr, bank, params)
    good = (not res.ok) and np.array_equal(res.mask, junk)
    print(f"[T5 쓰레기 마스크] {res.summary()}   {'OK' if good else 'FAIL'}")
    ok_all &= good

    print("\n결과:", "ALL OK" if ok_all else "일부 FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    if "--stress" in sys.argv:
        for nz in (0.4, 2.0, 4.0):
            stress(n=40, noise=nz, hole=0.03 if nz > 1 else 0.015)
        sys.exit(0)
    sys.exit(main())
