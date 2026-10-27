"""2D 검출 마스크 형상 보정 (CAD 투영 템플릿 기반, pose/ICP 불필요).

문제: RTMDet-Ins 마스크가 (1) 이웃 물체로 번지거나 (2) 중간이 끊겨서 나오면
ICP에 넘어가는 점군에 배경/이웃 점이 섞이거나 일부가 빠진다.

아이디어: 볼트처럼 축대칭에 가까운 부품은 "축이 이미지 평면에 기울어진 각도 phi"
하나와 이미지 평면 회전 theta, 위치, 크기(깊이 Z로 결정)만으로 실루엣이 거의
정해진다. 그래서 CAD를 phi별로 미리 투영해 둔 작은 템플릿 뱅크를 만들고,

    1) 마스크 안의 depth로 평면(Z = a*u + b*v + c)을 RANSAC으로 적합해
       "이 물체와 같은 깊이"인 코어 픽셀만 남기고 (이웃 번짐 제거),
    2) 코어에 가장 잘 맞는 템플릿(phi, theta, 위치)을 찾은 뒤,
    3) 최종 마스크 = 템플릿 영역 ∩ (깊이가 평면과 일치하는 픽셀 ∪ 깊이 무효 픽셀)
       로 다시 만든다 (끊긴 부분은 이어 붙고, 가려진 부분은 depth로 걸러진다).

호출 순서가 ICP 이전이므로 pose가 필요 없다. 필요한 입력은 검출 마스크, 세션의
pointcloud_organized(H,W,3 mm), valid_mask, 근사 intrinsic(fx, fy)뿐이다.

한계 (코드 아래 docstring에도 반복):
    - 축대칭 가정: CAD가 가늘고 긴 부품(aspect >= 2 권장)일 때만 의미가 있다.
    - 약원근 투영(weak perspective) 근사: 물체 크기 << 거리일 때 정확하다.
    - 서로 닿은 두 물체가 깊이도 비슷하면 분리하지 못한다.
    - 템플릿 일치 점수가 낮으면 보정하지 않고 원본 마스크를 그대로 돌려준다.

의존성: numpy, opencv-python (CAD 파일 로드 시에만 open3d 또는 trimesh).
"""
from __future__ import annotations

import math
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

__all__ = [
    "RefineParams", "RefineResult", "SilhouetteBank",
    "load_mesh_mm", "get_bank", "refine_mask",
]


# =============================================================================
# 파라미터 / 결과
# =============================================================================
@dataclass
class RefineParams:
    # --- 깊이 일관성 (mm). 기준 하나(depth_tol_mm)에서 나머지를 파생시킨다 ---
    # depth_tol_mm: RANSAC 평면 인라이어 거리 = "이 물체와 같은 깊이"로 볼 허용 폭.
    #   볼트 지름/머리 두께(약 10~18mm) 때문에 표면이 평면에서 수 mm 벗어나므로 너무 작게 잡지 말 것.
    depth_tol_mm: float = 8.0
    fill_ratio: float = 1.5   # 최종 마스크에 포함시킬 허용 = depth_tol_mm * fill_ratio
    bad_ratio: float = 2.5    # 평면보다 이만큼 "더 먼" 픽셀을 템플릿이 덮으면 감점(그 자리엔 물체가 있어야 하므로 모순).
                              # 평면보다 "더 가까운" 픽셀(=앞에서 가리는 물체)은 감점하지 않는다.

    join_mm: float = 20.0       # 같은 깊이의 조각 사이 틈이 이 거리(mm) 이하면 한 물체로 묶음
                                #   (다른 볼트에 가려져 둘로 갈라진 경우를 이어 붙이기 위함)
    ransac_iters: int = 200
    max_ransac_points: int = 4000
    min_valid_points: int = 30  # 마스크 안 유효 depth 점이 이보다 적으면 보정 포기
    min_core_px: int = 30
    plane_min_frac: float = 0.3  # 여러 평면 후보 중, 최다 후보 대비 이 비율 이상인 것만 고려
    plane_max_planes: int = 3    # 순차 RANSAC으로 찾을 최대 평면 수 (물체/이웃/바닥 등)
    keep_tol_px: int = 4         # 원 검출 픽셀을 템플릿 경계에서 이만큼 바깥까지 유지

    # --- 템플릿 피팅 ---
    alpha: float = 0.5   # 템플릿이 코어 밖을 덮는 면적에 대한 페널티 (작을수록 끊긴 조각을 관대하게 봄)
    lam: float = 1.0     # "깊이 불일치" 픽셀을 덮는 것에 대한 페널티
    beta: float = 0.1    # 원래 검출 마스크와 겹치는 것에 대한 가산(위치 모호성 해소용)
    elong_min: float = 1.6      # 코어 PCA 장단축비가 이보다 작으면(=끝면에서 본 듯) theta를 전체 탐색
    theta_search_deg: float = 8.0
    theta_step_deg: float = 4.0
    theta_full_step_deg: float = 15.0
    coarse_scale: float = 0.5   # 1단계 탐색 해상도 배율
    coarse_phi_step_deg: float = 15.0
    top_k: int = 3
    # 깊이 기울기로 phi(축이 카메라 쪽으로 기울어진 각)를 제한한다. 실루엣만으로는 phi와 -phi가
    # 구분되지 않아(정사영은 z 반전에 불변) 어느 끝이 카메라에 가까운지 알 수 없지만, 깊이 평면은 안다.
    # 마스크 안에 평면 후보가 여러 개(물체/이웃/바닥)면, 각 후보로 템플릿을 맞춰 보고 점수가 가장 좋은 쪽을 물체로 쓴다.
    # (끄면 '카메라에 가장 가까운 평면 = 물체'로 고정 -> 앞쪽 이웃이 번졌을 때 이웃을 물체로 오인)
    plane_select_by_fit: bool = True
    # 여러 검출을 한꺼번에 보정할 때(refine_masks), 두 마스크가 겹친 픽셀은 한쪽에만 남긴다.
    resolve_overlap: bool = True
    overlap_dup_iou: float = 0.8      # 두 보정 마스크의 IoU가 이 이상이면 같은 물체의 중복 검출로 보고 건드리지 않음
    depth_phi: bool = True
    phi_depth_tol_deg: float = 12.0   # 깊이로 구한 phi에서 이 범위 안의 phi 템플릿만 시도
    phi_depth_min_pts: int = 60       # 기울기를 구할 최소 inlier 점 수 (미만이면 제한 없이 전체 탐색)

    fill_invalid_px: int = 4    # 깊이 무효 픽셀은 원 검출 마스크에서 이 거리 안일 때만 채움
    min_score: float = 0.5      # 이보다 낮으면 보정 포기(원본 유지)

    @property
    def fill_tol_mm(self) -> float:
        return self.depth_tol_mm * self.fill_ratio

    @property
    def bad_tol_mm(self) -> float:
        return self.depth_tol_mm * self.bad_ratio


# 실패 사유 코드 -> 사용자에게 보여줄 설명 (무엇을 만져 보면 되는지 포함)
REASON_HELP = {
    "mask_too_small": "마스크가 너무 작음",
    "shape_mismatch": "마스크와 점군의 크기가 다름 (이미지/점군 해상도 확인)",
    "no_depth": "마스크 안에 유효한 깊이 점이 부족",
    "plane_fit_failed": "물체 평면 적합 실패",
    "bad_depth": "깊이 값 이상",
    "bad_scale": "스케일 이상 (내부 파라미터/점군 단위 확인)",
    "small_core": "깊이가 일치하는 영역이 너무 작음 -> '깊이 허용(mm)'을 올려 보세요",
    "scale_inconsistent": "마스크 안 깊이가 크게 퍼짐 (여러 물체/배경 혼입)",
    "no_candidate": "맞는 템플릿 후보가 없음",
    "low_score": "CAD 실루엣이 잘 안 맞음 -> '최소 score'를 낮추거나 CAD/깊이 허용 확인",
    "empty_result": "보정 결과가 비어 원본 유지",
    "no_mask": "검출에 마스크가 없음",
    "no_session_data": "세션 점군 파일 없음",
}


@dataclass
class RefineResult:
    mask: np.ndarray          # 보정 결과. ok=False면 입력 원본 마스크 그대로.
    ok: bool
    reason: str = ""          # ok=False일 때 사유 코드
    score: float = 0.0
    phi_deg: float = 0.0      # 축의 기울기 (+/-90 = 끝면, 0 = 옆면)
    theta_deg: float = 0.0    # 이미지 평면 안 회전 (u축에서 v축 방향, deg)
    n_before: int = 0
    n_after: int = 0
    n_added: int = 0
    n_removed: int = 0
    plane: tuple | None = None   # 물체 평면 (u0, v0, a, b, c): z = a*(u-u0) + b*(v-v0) + c  (겹침 정리에 사용)
    n_overlap_removed: int = 0   # 이웃 마스크와 겹쳐서 이웃에게 넘긴 픽셀 수
    # --- 품질 지표 (의심 마스크를 거르는 데 사용; ok=True일 때만 의미 있음) ---
    det_in_tpl: float = 0.0      # 원본 검출 중 CAD 투영 템플릿 안에 있는 비율 (낮으면 원본이 엉뚱한 모양/번짐)
    coverage: float = 0.0        # 템플릿 중 최종 마스크가 채운 비율 (낮으면 끊김/가림이 큼)
    change_ratio: float = 0.0    # (추가+삭제 픽셀) / 원본 면적 (크면 보정이 많이 고쳤다 = 원본 신뢰도 낮음)
    center_uv: tuple | None = None  # 최종 템플릿의 모델 중심 픽셀 (u, v), 이미지 좌표. 2D 기반 픽 포인트 계산용

    def is_suspect(self, min_score: float = 0.0, max_change: float = 9.9, min_det_in_tpl: float = 0.0,
                   drop_failed: bool = True) -> str:
        """의심 마스크면 사유 문자열, 아니면 ''. 보정 포기(ok=False)는 drop_failed일 때 의심으로 본다."""
        if not self.ok:
            return f"보정포기({self.reason})" if drop_failed else ""
        if self.score < min_score:
            return f"score {self.score:.2f}<{min_score:.2f}"
        if self.change_ratio > max_change:
            return f"변경률 {self.change_ratio:.2f}>{max_change:.2f}"
        if self.det_in_tpl < min_det_in_tpl:
            return f"원본∩템플릿 {self.det_in_tpl:.2f}<{min_det_in_tpl:.2f}"
        return ""

    def summary(self) -> str:
        if not self.ok:
            tail = f" score={self.score:.2f}" if self.score else ""
            return f"보정 안 함({self.reason}{tail}): {REASON_HELP.get(self.reason, '')}"
        return (f"보정 score={self.score:.2f} phi={self.phi_deg:+.0f}° theta={self.theta_deg:.0f}° "
                f"면적 {self.n_before}→{self.n_after} (+{self.n_added}/-{self.n_removed}) "
                f"[원본∩템플릿 {self.det_in_tpl:.2f}, 채움 {self.coverage:.2f}, 변경률 {self.change_ratio:.2f}]"
                + (f" 겹침정리 -{self.n_overlap_removed}" if self.n_overlap_removed else ""))


# =============================================================================
# 세션 데이터 로드 (탭 3, 오토라벨링 공용)
# =============================================================================
def load_session_arrays(image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """<세션>/intensity/<stem>.png -> (pcd_mm (H,W,3) float32, valid (H,W) bool).

    세션 구조(collect_dataset.py)가 아니거나 파일이 없으면 ValueError.
    """
    img = Path(image_path)
    if img.parent.name != "intensity":
        raise ValueError(
            "이미지가 세션 폴더 구조가 아닙니다.\n"
            "형상 보정은 <세션>/intensity/ 폴더의 이미지가 필요하며,\n"
            "같은 세션의 pointcloud_organized/, valid_mask/ 가 있어야 합니다."
        )
    session = img.parent.parent
    pcd_path = session / "pointcloud_organized" / f"{img.stem}.npy"
    valid_path = session / "valid_mask" / f"{img.stem}.npy"
    for pth in (pcd_path, valid_path):
        if not pth.is_file():
            raise ValueError(f"파일이 없습니다: {pth}")
    pcd = np.load(pcd_path)
    valid = np.load(valid_path).astype(bool)
    if pcd.ndim != 3 or pcd.shape[2] != 3 or valid.shape != pcd.shape[:2]:
        raise ValueError(f"점군/유효마스크 형상이 이상합니다: pcd {pcd.shape}, valid {valid.shape}")
    return pcd.astype(np.float32), valid


# =============================================================================
# CAD 로드
# =============================================================================
def load_mesh_mm(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """CAD(.stl/.ply/.obj) -> (vertices[mm], faces). 단위는 icp_runner.load_cad_as_pcd와
    같은 기준으로 판단한다 (bbox 최대 길이가 10을 넘으면 이미 mm, 아니면 m로 보고 x1000)."""
    path = str(path)
    try:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(path)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.triangles, dtype=np.int64)
    except ImportError:
        import trimesh  # open3d가 없을 때만 대체
        m = trimesh.load(path, force="mesh")
        verts = np.asarray(m.vertices, dtype=np.float64)
        faces = np.asarray(m.faces, dtype=np.int64)

    if len(verts) == 0 or len(faces) == 0:
        raise ValueError(f"CAD에서 정점/면을 읽지 못했습니다: {path}")
    ext = verts.max(axis=0) - verts.min(axis=0)
    if ext.max() <= 10.0:
        verts = verts * 1000.0
    return verts, faces


# =============================================================================
# 템플릿 뱅크
# =============================================================================
class SilhouetteBank:
    """CAD 실루엣 템플릿 뱅크.

    모델 좌표계: PCA 최장축 = x축, bbox 중심 = 원점.
    축 방향 a = (cos(phi)cos(theta), cos(phi)sin(theta), sin(phi)) (카메라 좌표, z=광축).
    roll(축 둘레 회전)은 무시한다 (축대칭 근사). 그러면 실루엣은 (phi, theta)로만 정해지고,
    theta는 이미지 평면 회전이라 phi별 기본 템플릿(theta=0)을 2D 회전하면 얻는다.
    phi는 [-90, +90] 전체를 둔다 (+/-는 머리/꼬리 중 어느 쪽이 카메라 쪽인지를 구분).
    """

    def __init__(self, vertices_mm: np.ndarray, faces: np.ndarray,
                 phi_step_deg: float = 5.0, res_mm: float = 0.5, roll_samples: int = 1):
        verts = np.asarray(vertices_mm, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        if verts.ndim != 2 or verts.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("vertices는 (N,3), faces는 (M,3)이어야 합니다.")

        center = verts.mean(axis=0)
        eigval, eigvec = np.linalg.eigh(np.cov((verts - center).T))
        order = np.argsort(eigval)[::-1]
        e1, e2 = eigvec[:, order[0]], eigvec[:, order[1]]
        e3 = np.cross(e1, e2)
        vm = (verts - center) @ np.stack([e1, e2, e3], axis=1)
        lo, hi = vm.min(axis=0), vm.max(axis=0)
        # 모델 좌표 -> CAD 파일 좌표: v_cad = model_to_cad_R @ x_model + model_origin_cad_mm
        # (2D 템플릿 매칭 결과를 CAD 기준 자세로 되돌릴 때 사용)
        self.model_to_cad_R = np.stack([e1, e2, e3], axis=1)
        self.model_origin_cad_mm = center + self.model_to_cad_R @ ((lo + hi) / 2.0)
        self.cad_center_mm = center  # 정점 평균
        vm = vm - (lo + hi) / 2.0
        ext = hi - lo

        self.length_mm = float(ext[0])
        self.width_mm = float(max(ext[1], ext[2]))
        self.aspect = self.length_mm / max(self.width_mm, 1e-6)
        self.radius_mm = float(np.linalg.norm(vm, axis=1).max())  # 어떤 방향/회전에서도 이 반경 안에 들어옴
        self.res_mm = float(res_mm)

        n_phi = int(round(180.0 / phi_step_deg)) + 1
        self.phi_deg = np.linspace(-90.0, 90.0, n_phi)
        size = int(math.ceil(2.0 * self.radius_mm / res_mm)) + 8
        self._size = size
        self.bases = np.zeros((n_phi, size, size), dtype=np.uint8)

        # roll(축 둘레 회전)은 영상에서 알 수 없다. roll_samples>1이면 여러 roll 실루엣의 다수결(>50%)을
        # 템플릿으로 쓰는 옵션이지만, 합성 테스트에서 오히려 phi 추정이 나빠져(IoU 0.99->0.95)
        # 기본은 1(CAD 원래 roll 하나)이다. 육각 머리 roll 불일치로 인한 오차는 IoU 0.88~0.99 수준.
        n_roll = max(1, int(roll_samples))
        for i, phi in enumerate(self.phi_deg):
            beta = -math.radians(phi)  # Ry(-phi)가 x축을 (cos phi, 0, sin phi)로 보낸다
            c, s = math.cos(beta), math.sin(beta)
            rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            acc = np.zeros((size, size), dtype=np.float32)
            for k in range(n_roll):
                a = math.pi * k / n_roll  # 0~180deg 균등 (육각 머리는 60deg 주기라 충분)
                ca, sa = math.cos(a), math.sin(a)
                rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
                xy = (vm @ rx.T @ rot.T)[:, :2] / res_mm + size / 2.0
                tri = np.round(xy[faces] * 16.0).astype(np.int32)  # (M,3,2), shift=4 -> 1/16 px 정밀도
                img = np.zeros((size, size), dtype=np.uint8)
                cv2.fillPoly(img, list(tri), 1, lineType=cv2.LINE_8, shift=4)
                acc += img
            self.bases[i] = ((acc / n_roll) > 0.5).astype(np.uint8) * 255

    @classmethod
    def from_cad_file(cls, path: str | Path, **kw) -> "SilhouetteBank":
        verts, faces = load_mesh_mm(path)
        return cls(verts, faces, **kw)

    @property
    def n_phi(self) -> int:
        return len(self.phi_deg)

    @property
    def phi_step_deg(self) -> float:
        return float(self.phi_deg[1] - self.phi_deg[0])

    def render(self, phi_idx: int, theta_deg: float, px_per_mm: float, patch: int) -> np.ndarray:
        """(patch, patch) float32 {0,1}. 모델 중심이 패치 중심(patch/2)에 오도록 그린다."""
        base = self.bases[phi_idx]
        s = px_per_mm * self.res_mm
        if s < 0.7:  # 많이 줄일 때 얇은 부분이 사라지지 않도록 미리 흐림
            base = cv2.GaussianBlur(base, (0, 0), sigmaX=0.5 / s)
        half = self._size / 2.0
        # 화면은 y가 아래이므로 theta(u축→v축)는 OpenCV 각도의 부호를 반대로 준다
        mat = cv2.getRotationMatrix2D((half, half), -theta_deg, s)
        mat[:, 2] += (patch / 2.0 - half, patch / 2.0 - half)
        out = cv2.warpAffine(base, mat, (patch, patch), flags=cv2.INTER_LINEAR)
        return (out > 127).astype(np.float32)


_BANK_CACHE: dict[tuple, SilhouetteBank] = {}


def get_bank(cad_path: str | Path, phi_step_deg: float = 5.0) -> SilhouetteBank:
    """같은 CAD면 한 번만 만든다 (파일 수정시각이 바뀌면 다시 만든다)."""
    p = Path(cad_path)
    key = (str(p.resolve()), p.stat().st_mtime_ns, float(phi_step_deg))
    bank = _BANK_CACHE.get(key)
    if bank is None:
        bank = SilhouetteBank.from_cad_file(p, phi_step_deg=phi_step_deg)
        _BANK_CACHE[key] = bank
    return bank


# =============================================================================
# 내부 유틸
# =============================================================================
def _fit_plane_ransac(u: np.ndarray, v: np.ndarray, z: np.ndarray, thresh: float,
                      iters: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    """z = a*u + b*v + c. 반환 ((a,b,c), inlier bool 배열). 적합 불가면 None."""
    n = len(z)
    if n < 3:
        return None
    a_mat = np.stack([u, v, np.ones(n)], axis=1)
    best_cnt, best_inl = -1, None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            coef = np.linalg.solve(a_mat[idx], z[idx])
        except np.linalg.LinAlgError:
            continue
        inl = np.abs(a_mat @ coef - z) < thresh
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt, best_inl = cnt, inl
    if best_inl is None or best_cnt < 3:
        return None
    coef, *_ = np.linalg.lstsq(a_mat[best_inl], z[best_inl], rcond=None)
    return coef, np.abs(a_mat @ coef - z) < thresh


def _find_planes(u, v, z, p: RefineParams, rng) -> list[np.ndarray]:
    """마스크 안 depth에서 평면 후보를 순차 RANSAC으로 찾는다 (물체/이웃/바닥 등).

    충분히 큰 후보(최다 대비 plane_min_frac 이상)만 남기고, 평균 깊이가 작은(=카메라에 가까운) 순으로
    정렬해 돌려준다. 맨 앞(가장 가까운 것)이 기본 '물체' 후보다 (바닥 번짐 대응).
    plane_select_by_fit이 켜져 있으면 refine_mask가 후보마다 템플릿을 맞춰 보고 최종 선택한다.
    """
    cands = []
    ru, rv, rz = u, v, z
    for _ in range(p.plane_max_planes):
        if len(rz) < p.min_valid_points:
            break
        fit = _fit_plane_ransac(ru, rv, rz, p.depth_tol_mm, p.ransac_iters, rng)
        if fit is None:
            break
        coef, inl = fit
        cnt = int(inl.sum())
        if cnt < p.min_valid_points:
            break
        cands.append((coef, cnt, float(rz[inl].mean())))
        ru, rv, rz = ru[~inl], rv[~inl], rz[~inl]
    if not cands:
        return []
    top_cnt = max(c[1] for c in cands)
    ok = sorted((c for c in cands if c[1] >= p.plane_min_frac * top_cnt), key=lambda c: c[2])
    return [c[0] for c in ok]


def _score_map(f_map, bad_map, det_map, exp_map, tpl, f_sum, p: RefineParams):
    """템플릿을 모든 위치에 놓았을 때의 점수 맵.

    a: 템플릿이 덮은 코어(증거) 면적
    b: 덮은 "모순" 면적 (그 자리엔 물체가 있어야 하는데 더 먼 것이 보임)
    c: 원 검출 마스크와 겹친 면적 (위치 모호성 해소용 약한 가산)
    e: 덮은 "설명되는" 면적 = 물체 평면과 깊이가 맞는 픽셀(검출이 놓쳤어도) + 앞에서 가리는 물체 + 깊이 무효인 검출 픽셀.
       템플릿 면적 중 설명되지 않는 부분만 벌점으로 센다 (끊긴 띠나 가려진 곳은 벌점 없음).
    """
    tarea = float(tpl.sum())
    if tarea < 1.0:
        return None
    a = cv2.matchTemplate(f_map, tpl, cv2.TM_CCORR)
    b = cv2.matchTemplate(bad_map, tpl, cv2.TM_CCORR)
    c = cv2.matchTemplate(det_map, tpl, cv2.TM_CCORR)
    e = cv2.matchTemplate(exp_map, tpl, cv2.TM_CCORR)
    unexplained = np.maximum(tarea - e, 0.0)
    return a / (f_sum + p.alpha * unexplained + 1e-6) - p.lam * b / tarea + p.beta * c / tarea


def _components_with_seed(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """mask의 연결성분 중 seed(코어)를 하나라도 포함한 것만 남긴다.
    다른 물체에 가려 둘 이상으로 갈라진 정상 조각은 유지하고, 코어와 무관하게 템플릿이 만든 섬은 버린다."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    keep = np.unique(lab[mask & seed])
    keep = keep[keep > 0]
    return np.isin(lab, keep)


def _fail(det: np.ndarray, reason: str, n_before: int, score: float = 0.0,
          phi: float = 0.0, theta: float = 0.0) -> RefineResult:
    return RefineResult(mask=det, ok=False, reason=reason, score=score, phi_deg=phi,
                        theta_deg=theta, n_before=n_before, n_after=n_before)


# =============================================================================
# 메인
# =============================================================================
def _refine_impl(det_mask: np.ndarray, pcd_mm: np.ndarray, valid_mask: np.ndarray,
                 intrinsics: Sequence[float], bank: SilhouetteBank,
                 params: RefineParams | None = None, plane_rank: int = 0,
                 info: dict | None = None) -> RefineResult:
    """refine_mask의 본체. plane_rank번째(가까운 순) 평면 후보를 물체로 가정해 보정한다.
    info가 주어지면 info["n_planes"]에 평면 후보 수를 채운다.

    Args:
        det_mask: (H,W) bool 검출 마스크.
        pcd_mm: (H,W,3) float, organized 점군(mm). 광축 방향 Z는 [..., 2].
        valid_mask: (H,W) bool, pcd 유효 픽셀.
        intrinsics: (fx, fy, cx, cy) 또는 (fx, fy). 약원근 스케일 계산에는 fx, fy만 쓴다.
        bank: SilhouetteBank (get_bank로 CAD당 한 번 생성).
        params: RefineParams.

    Returns:
        RefineResult. ok=False이면 .mask는 입력 det_mask 그대로다.
    """
    p = params or RefineParams()
    det = np.asarray(det_mask, dtype=bool)
    h_img, w_img = det.shape
    n_before = int(det.sum())
    if n_before < p.min_core_px:
        return _fail(det, "mask_too_small", n_before)
    if pcd_mm.shape[:2] != det.shape or valid_mask.shape != det.shape:
        return _fail(det, "shape_mismatch", n_before)

    fx, fy = float(intrinsics[0]), float(intrinsics[1])
    z_all = pcd_mm[..., 2]
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(z_all) & (z_all > 0)

    # ---- 1. 마스크 안 depth로 평면 적합 -----------------------------------
    ys, xs = np.nonzero(det & valid)
    if len(ys) < p.min_valid_points:
        return _fail(det, "no_depth", n_before)
    u0, v0 = float(xs.mean()), float(ys.mean())
    rng = np.random.default_rng(0)
    if len(ys) > p.max_ransac_points:
        sel = rng.choice(len(ys), p.max_ransac_points, replace=False)
        ys_s, xs_s = ys[sel], xs[sel]
    else:
        ys_s, xs_s = ys, xs
    planes = _find_planes(xs_s - u0, ys_s - v0, z_all[ys_s, xs_s].astype(np.float64), p, rng)
    if info is not None:
        info["n_planes"] = len(planes)
    if plane_rank >= len(planes):
        return _fail(det, "plane_fit_failed", n_before)
    pa, pb, pc = planes[plane_rank]

    # 물체 평면 inlier 점으로 카메라 좌표계의 기울기 dz/dX, dz/dY 를 구한다 (phi 부호/크기의 근거).
    # 픽셀 좌표가 아니라 (X, Y, Z) mm 로 적합해야 시야각에 따른 왜곡 없이 축 방향 기울기가 나온다.
    depth_grad = None
    if p.depth_phi:
        inl = np.abs(z_all[ys, xs] - (pa * (xs - u0) + pb * (ys - v0) + pc)) < p.depth_tol_mm
        if int(inl.sum()) >= p.phi_depth_min_pts:
            xx = pcd_mm[ys[inl], xs[inl], 0].astype(np.float64)
            yy = pcd_mm[ys[inl], xs[inl], 1].astype(np.float64)
            zz = z_all[ys[inl], xs[inl]].astype(np.float64)
            if np.all(np.isfinite(xx)) and np.all(np.isfinite(yy)) and np.ptp(xx) + np.ptp(yy) > 1e-6:
                g, *_ = np.linalg.lstsq(np.stack([xx, yy, np.ones_like(xx)], axis=1), zz, rcond=None)
                if np.all(np.isfinite(g[:2])):
                    depth_grad = (float(g[0]), float(g[1]))

    def _phi_candidates(th: float, phi_idx_all: list[int]) -> list[int]:
        """이 theta(축의 이미지 평면 방향)에서 깊이가 말하는 phi 근처의 phi 인덱스만."""
        if depth_grad is None:
            return phi_idx_all
        t = math.radians(th)
        phi_d = math.degrees(math.atan(depth_grad[0] * math.cos(t) + depth_grad[1] * math.sin(t)))
        near = [pi for pi in phi_idx_all if abs(float(bank.phi_deg[pi]) - phi_d) <= p.phi_depth_tol_deg]
        if near:
            return near
        return [min(phi_idx_all, key=lambda pi: abs(float(bank.phi_deg[pi]) - phi_d))]

    # ---- 2. 스케일/패치/캔버스 --------------------------------------------
    # 코어 중심에서의 평면 깊이로 px/mm를 정한다 (코어는 아래에서 구하므로 우선 마스크 중심 사용)
    z_c = float(pc)
    if z_c <= 1.0:
        return _fail(det, "bad_depth", n_before)
    px_per_mm = 0.5 * (fx + fy) / z_c
    patch = int(math.ceil(2.0 * bank.radius_mm * px_per_mm)) + 4
    if patch < 8 or patch > 2000:
        return _fail(det, "bad_scale", n_before)

    ys_d, xs_d = np.nonzero(det)
    bx1, bx2, by1, by2 = int(xs_d.min()), int(xs_d.max()), int(ys_d.min()), int(ys_d.max())
    cx0, cy0 = bx1 - patch, by1 - patch                      # 캔버스 원점(이미지 좌표, 음수 가능)
    cw, ch = (bx2 - bx1 + 1) + 2 * patch, (by2 - by1 + 1) + 2 * patch
    ix0, iy0 = max(cx0, 0), max(cy0, 0)
    ix1, iy1 = min(cx0 + cw, w_img), min(cy0 + ch, h_img)
    reg = (slice(iy0, iy1), slice(ix0, ix1))                 # 이미지 쪽 영역
    can = (slice(iy0 - cy0, iy1 - cy0), slice(ix0 - cx0, ix1 - cx0))  # 캔버스 쪽 같은 영역

    det_r = det[reg]
    valid_r = valid[reg]
    gy, gx = np.mgrid[iy0:iy1, ix0:ix1]
    plane = pa * (gx - u0) + pb * (gy - v0) + pc
    sdz = np.where(valid_r, z_all[reg] - plane, 0.0)       # 부호 있는 깊이차 (+: 평면보다 멀다)
    absdz = np.where(valid_r, np.abs(sdz), np.inf)

    # ---- 3. 코어: 같은 깊이 + 가까운 조각만 ------------------------------
    inl = det_r & (absdz < p.depth_tol_mm)
    if int(inl.sum()) < p.min_core_px:
        return _fail(det, "small_core", n_before)
    join_px = int(np.clip(math.ceil(p.join_mm * px_per_mm / 2.0), 2, 60))  # 양쪽에서 부풀리므로 /2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * join_px + 1, 2 * join_px + 1))
    dil = cv2.dilate(inl.astype(np.uint8), k)
    n_lab, lab = cv2.connectedComponents(dil, connectivity=8)
    cnt = np.bincount(lab[inl], minlength=n_lab)[1:]
    best_lab = 1 + int(np.argmax(cnt))
    comp = lab == best_lab
    core = inl & comp
    if int(core.sum()) < p.min_core_px:
        return _fail(det, "small_core", n_before)

    f_r = core | (det_r & ~valid_r & comp)        # 깊이 무효 픽셀(반사/어두움)은 코어 근처일 때만 증거로 인정
    bad_r = valid_r & (sdz > p.bad_tol_mm)             # 더 먼 것이 보임 -> 템플릿 자리로는 모순
    occ_r = valid_r & (sdz < -p.fill_tol_mm)           # 더 가까운 것 = 앞에서 가리는 물체 -> 중립
    exp_r = (valid_r & (absdz < p.fill_tol_mm)) | occ_r | (det_r & ~valid_r)  # 템플릿이 덮어도 이상하지 않은 곳

    f_can = np.zeros((ch, cw), np.float32)
    bad_can = np.zeros((ch, cw), np.float32)
    exp_can = np.zeros((ch, cw), np.float32)
    det_can = np.zeros((ch, cw), np.float32)
    f_can[can], bad_can[can], det_can[can], exp_can[can] = f_r, bad_r, det_r, exp_r

    # ---- 4. 코어 PCA -> theta 초기값, 코어 중심 깊이로 스케일 재계산 -------
    cys, cxs = np.nonzero(core)
    cxs = cxs + ix0
    cys = cys + iy0
    cu, cv_ = float(cxs.mean()), float(cys.mean())
    z_core = pa * (cu - u0) + pb * (cv_ - v0) + pc
    if z_core <= 1.0:
        return _fail(det, "bad_depth", n_before)
    px_per_mm = 0.5 * (fx + fy) / z_core
    patch2 = int(math.ceil(2.0 * bank.radius_mm * px_per_mm)) + 4
    if abs(patch2 - patch) > 2:
        # 코어 중심 깊이가 마스크 중심과 많이 달라 패치 크기가 어긋남 -> 안전하게 포기
        return _fail(det, "scale_inconsistent", n_before)

    cov = np.cov(np.stack([cxs - cu, cys - cv_]))
    if cov.ndim == 0 or not np.all(np.isfinite(cov)):
        return _fail(det, "small_core", n_before)
    evals, evecs = np.linalg.eigh(cov)
    elong = math.sqrt(max(evals[1], 1e-9) / max(evals[0], 1e-9))
    vx, vy = evecs[:, 1]
    theta0 = math.degrees(math.atan2(vy, vx)) % 180.0

    if elong >= p.elong_min:
        offs = np.arange(-p.theta_search_deg, p.theta_search_deg + 1e-6, p.theta_step_deg)
        thetas = [float((t + d) % 360.0) for t in (theta0, theta0 + 180.0) for d in offs]
    else:
        thetas = [float(t) for t in np.arange(0.0, 360.0, p.theta_full_step_deg)]

    # ---- 5. 1단계: 저해상도 전체 탐색 -------------------------------------
    ws = p.coarse_scale if patch * p.coarse_scale >= 16 else 1.0
    pc_patch = max(8, int(round(patch * ws)))

    def _resize(m):
        if ws == 1.0:
            return m
        return cv2.resize(m, (max(1, int(round(cw * ws))), max(1, int(round(ch * ws)))),
                          interpolation=cv2.INTER_AREA)

    f_c, bad_c, det_c, exp_c = _resize(f_can), _resize(bad_can), _resize(det_can), _resize(exp_can)
    f_sum_c = float(f_c.sum())
    step_c = max(1, int(round(p.coarse_phi_step_deg / bank.phi_step_deg)))
    phi_idx_c = sorted(set(list(range(0, bank.n_phi, step_c)) + [bank.n_phi - 1]))

    cands = []  # (score, phi_idx, theta, x, y) -- 좌표는 coarse 캔버스 기준 좌상단
    if f_c.shape[0] >= pc_patch and f_c.shape[1] >= pc_patch:
        for th in thetas:
            for pi in _phi_candidates(th, phi_idx_c):
                tpl = bank.render(pi, th, px_per_mm * ws, pc_patch)
                sm = _score_map(f_c, bad_c, det_c, exp_c, tpl, f_sum_c, p)
                if sm is None:
                    continue
                _, mx, _, loc = cv2.minMaxLoc(sm)
                cands.append((float(mx), pi, th, loc[0], loc[1]))
    if not cands:
        return _fail(det, "no_candidate", n_before)
    cands.sort(key=lambda t: -t[0])
    top = cands[: max(1, p.top_k)]

    # ---- 6. 2단계: 풀해상도에서 phi/theta/위치 미세 조정 -------------------
    f_sum = float(f_can.sum())
    win = max(3, int(math.ceil(1.5 / ws)) + 2)
    best = None  # (score, phi_idx, theta, x, y)
    for _, pi, th, xc, yc in top:
        tl_x = int(round((xc + pc_patch / 2.0) / ws - patch / 2.0))
        tl_y = int(round((yc + pc_patch / 2.0) / ws - patch / 2.0))
        for dpi in (-1, 0, 1):
            pi2 = pi + dpi
            if not 0 <= pi2 < bank.n_phi:
                continue
            for dth in (-3.0, 0.0, 3.0):
                th2 = (th + dth) % 360.0
                size_w = patch + 2 * win
                x0 = int(np.clip(tl_x - win, 0, max(cw - size_w, 0)))
                y0 = int(np.clip(tl_y - win, 0, max(ch - size_w, 0)))
                fc = f_can[y0:y0 + size_w, x0:x0 + size_w]
                if fc.shape[0] < patch or fc.shape[1] < patch:
                    continue
                tpl = bank.render(pi2, th2, px_per_mm, patch)
                sm = _score_map(fc, bad_can[y0:y0 + size_w, x0:x0 + size_w],
                                det_can[y0:y0 + size_w, x0:x0 + size_w],
                                exp_can[y0:y0 + size_w, x0:x0 + size_w], tpl, f_sum, p)
                if sm is None:
                    continue
                _, mx, _, loc = cv2.minMaxLoc(sm)
                if best is None or mx > best[0]:
                    best = (float(mx), pi2, th2, x0 + loc[0], y0 + loc[1])
    if best is None:
        return _fail(det, "no_candidate", n_before)
    score, pi, th, bx, by = best
    phi = float(bank.phi_deg[pi])
    if score < p.min_score:
        return _fail(det, "low_score", n_before, score, phi, th)

    # ---- 7. 최종 마스크: 템플릿 ∩ 깊이 일치 -------------------------------
    tpl = bank.render(pi, th, px_per_mm, patch) > 0.5
    t_can = np.zeros((ch, cw), bool)
    t_can[by:by + patch, bx:bx + patch] = tpl
    t_r = t_can[can]
    consistent = valid_r & (absdz < p.fill_tol_mm)
    ki = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.fill_invalid_px + 1, 2 * p.fill_invalid_px + 1))
    near_det = cv2.dilate(det_r.astype(np.uint8), ki).astype(bool)
    # 템플릿은 "모양 제약 + 틈 메우기" 용도. 깊이가 맞는 원 검출 픽셀은 템플릿 경계 근처(keep_tol_px)까지
    # 그대로 유지해서, 이미 정확한 경계를 템플릿 오차(1~2px)로 깎아먹지 않는다.
    kt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.keep_tol_px + 1, 2 * p.keep_tol_px + 1))
    near_t = cv2.dilate(t_r.astype(np.uint8), kt).astype(bool)
    keep_det = det_r & consistent & near_t
    add_t = t_r & (consistent | (~valid_r & near_det))
    final_r = keep_det | add_t
    final_r = cv2.morphologyEx(final_r.astype(np.uint8), cv2.MORPH_CLOSE,
                               np.ones((3, 3), np.uint8)).astype(bool)
    final_r = _components_with_seed(final_r, core)

    out = np.zeros_like(det)
    out[reg] = final_r
    n_after = int(out.sum())
    if n_after < p.min_core_px:
        return _fail(det, "empty_result", n_before, score, phi, th)

    n_added, n_removed = int((out & ~det).sum()), int((det & ~out).sum())
    t_area = max(int(t_r.sum()), 1)
    return RefineResult(
        mask=out, ok=True, score=score, phi_deg=phi, theta_deg=float(th),
        n_before=n_before, n_after=n_after, n_added=n_added, n_removed=n_removed,
        det_in_tpl=float((det_r & t_r).sum()) / max(int(det_r.sum()), 1),
        coverage=float((final_r & t_r).sum()) / t_area,
        change_ratio=(n_added + n_removed) / max(n_before, 1),
        plane=(float(u0), float(v0), float(pa), float(pb), float(pc)),
        center_uv=(float(cx0 + bx + patch / 2.0), float(cy0 + by + patch / 2.0)),
    )


def refine_mask(det_mask: np.ndarray, pcd_mm: np.ndarray, valid_mask: np.ndarray,
                intrinsics: Sequence[float], bank: SilhouetteBank,
                params: RefineParams | None = None) -> RefineResult:
    """검출 마스크 한 개를 보정한다.

    Args:
        det_mask: (H,W) bool 검출 마스크.
        pcd_mm: (H,W,3) float, organized 점군(mm). 광축 방향 Z는 [..., 2].
        valid_mask: (H,W) bool, pcd 유효 픽셀.
        intrinsics: (fx, fy, cx, cy) 또는 (fx, fy). 약원근 스케일 계산에는 fx, fy만 쓴다.
        bank: SilhouetteBank (get_bank로 CAD당 한 번 생성).
        params: RefineParams.

    Returns:
        RefineResult. ok=False이면 .mask는 입력 det_mask 그대로다.

    평면 후보가 여러 개면(물체+이웃/바닥) 후보마다 보정해 보고, '템플릿 점수 x 검출 마스크 설명 비율'이
    가장 큰 것을 고른다. 앞쪽 이웃이 번진 경우에도 대상 볼트를 물체로 고를 수 있게 한다.
    """
    p = params or RefineParams()
    info: dict = {}
    best = _refine_impl(det_mask, pcd_mm, valid_mask, intrinsics, bank, p, 0, info)
    n_planes = int(info.get("n_planes", 1))
    if not p.plane_select_by_fit or n_planes <= 1:
        return best
    det = np.asarray(det_mask, dtype=bool)
    det_n = max(int(det.sum()), 1)

    def _eff(r: RefineResult) -> float:
        return -1.0 if not r.ok else r.score * float((r.mask & det).sum()) / det_n

    best_eff = _eff(best)
    for k in range(1, n_planes):
        r = _refine_impl(det_mask, pcd_mm, valid_mask, intrinsics, bank, p, k, None)
        e = _eff(r)
        if e > best_eff:
            best, best_eff = r, e
    return best


def refine_masks(det_masks: Sequence[np.ndarray], pcd_mm: np.ndarray, valid_mask: np.ndarray,
                 intrinsics: Sequence[float], bank: SilhouetteBank,
                 params: RefineParams | None = None) -> list[RefineResult]:
    """한 프레임의 검출 마스크 여러 개를 보정하고, 마스크끼리 겹친 픽셀을 정리한다.

    각 마스크를 refine_mask로 보정한 뒤, 보정에 성공한 두 마스크가 겹친 픽셀은 한쪽에만 남긴다.
      - 한쪽 평면에만 깊이가 맞으면(|dz| < depth_tol) 그쪽이 가져간다.
      - 둘 다 맞거나 둘 다 안 맞으면 자기 마스크 안쪽 깊이(경계에서 먼 쪽)가 큰 쪽이 가져간다.
      - 보정 후 두 마스크의 IoU가 overlap_dup_iou 이상이면(같은 물체 중복 검출) 건드리지 않는다.
    보정을 포기한 마스크(ok=False)는 원본 그대로 두고, 이웃에게 픽셀을 빼앗지도 않는다.
    """
    p = params or RefineParams()
    results = [refine_mask(m, pcd_mm, valid_mask, intrinsics, bank, p) for m in det_masks]
    if not p.resolve_overlap or len(results) < 2:
        return results

    idx = [i for i, r in enumerate(results) if r.ok and r.plane is not None]
    if len(idx) < 2:
        return results
    base = {i: results[i].mask for i in idx}
    cur = {i: results[i].mask.copy() for i in idx}
    dist = {}
    z_all = pcd_mm[..., 2]
    zvalid = np.asarray(valid_mask, dtype=bool) & np.isfinite(z_all) & (z_all > 0)
    h, w = z_all.shape
    gy, gx = np.mgrid[0:h, 0:w]

    def _consistent(i: int) -> np.ndarray:
        u0, v0, a, b, c = results[i].plane
        dz = np.abs(z_all - (a * (gx - u0) + b * (gy - v0) + c))
        return (~zvalid) | (dz < p.depth_tol_mm)   # 깊이가 없는 픽셀은 판단 보류(일치로 취급)

    def _dist(i: int) -> np.ndarray:
        if i not in dist:
            dist[i] = cv2.distanceTransform(base[i].astype(np.uint8), cv2.DIST_L2, 3)
        return dist[i]

    changed: set[int] = set()
    cons_cache: dict[int, np.ndarray] = {}
    for ai in range(len(idx)):
        for bi in range(ai + 1, len(idx)):
            i, j = idx[ai], idx[bi]
            ov = cur[i] & cur[j]
            if not ov.any():
                continue
            union = int((base[i] | base[j]).sum())
            if union and int((base[i] & base[j]).sum()) / union >= p.overlap_dup_iou:
                continue  # 같은 물체의 중복 검출 (이 정리의 대상이 아님)
            ci = cons_cache.setdefault(i, _consistent(i))
            cj = cons_cache.setdefault(j, _consistent(j))
            i_wins = (ci & ~cj) | ((ci == cj) & (_dist(i) >= _dist(j)))
            cur[i][ov & ~i_wins] = False
            cur[j][ov & i_wins] = False
            changed.update((i, j))

    for i in changed:
        m = cur[i]
        # 조각 정리: 겹침 정리로 생긴 작은 파편 제거 (큰 조각은 가림으로 갈라진 정상 조각일 수 있어 유지)
        n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if n_lab > 1:
            min_keep = max(15, int(0.03 * int(m.sum())))
            keep = np.zeros(n_lab, bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_keep
            m = keep[lab]
        if int(m.sum()) < p.min_core_px:
            continue  # 거의 다 빼앗겼으면 정리 취소 (원래 보정 결과 유지)
        det_i = np.asarray(det_masks[i], dtype=bool)
        r = results[i]
        results[i] = dataclasses.replace(
            r, mask=m, n_after=int(m.sum()),
            n_added=int((m & ~det_i).sum()), n_removed=int((det_i & ~m).sum()),
            n_overlap_removed=int((base[i] & ~m).sum()),
        )
    return results