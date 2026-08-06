"""FoundationPose(NVlabs) 코어 래퍼.

RTMDet-Ins/RotHead와 마찬가지로 "무거운 외부 의존성은 실제 추론이 필요할
때만 import"하는 관례를 따른다 (app/core/detector.py의 ImportError 처리
패턴 참고). vendor/FoundationPose가 없으면 이 모듈을 import하는 시점이
아니라, 실제로 인스턴스를 만드는 시점에 명확한 에러를 낸다.

part(class_name)별로 CAD가 다르므로 FoundationPose 인스턴스도 part별로
분리 관리한다 - CADRegistry가 이 매핑을 담당한다.

좌표계/단위 관례: 이 프로젝트 전체가 m 단위를 쓰므로(icp_runner.py,
rotation_head_model.py 등과 동일), CAD는 항상 m 단위로 정규화해서
FoundationPose에 넘긴다. icp_runner.load_cad_as_pcd()와 동일한
"bbox extent > 10 이면 mm로 보고 1/1000 스케일" 관례를 그대로 재사용한다.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

# vendor/FoundationPose를 파이썬 경로에 추가 (서브모듈로 관리 예정).
_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "FoundationPose"


@dataclass
class PartMeshEntry:
    part_name: str
    mesh_path: Path
    vertices: np.ndarray = None       # (N,3) m 단위
    vertex_normals: np.ndarray = None  # (N,3)
    triangles: np.ndarray = None      # (M,3) int


class CADRegistry:
    """class_name(=part_name) -> 메쉬 정보 매핑, lazy loading.

    icp_runner.load_cad_as_pcd()와 동일한 mm->m 스케일 규칙을 쓰지만,
    거기서는 최종적으로 포인트클라우드(ICP용)로 샘플링하는 반면 여기서는
    삼각형 메쉬(vertices/normals/triangles)를 그대로 보존한다 -
    FoundationPose의 render-and-compare가 메쉬 표면 렌더링을 요구하기
    때문이다.
    """

    def __init__(self) -> None:
        self._entries: dict[str, PartMeshEntry] = {}

    def register_path(self, part_name: str, mesh_path: str | Path) -> None:
        self._entries[part_name] = PartMeshEntry(part_name=part_name, mesh_path=Path(mesh_path))

    def is_registered(self, part_name: str) -> bool:
        return part_name in self._entries

    def get(self, part_name: str) -> PartMeshEntry:
        if part_name not in self._entries:
            raise KeyError(
                f"'{part_name}'에 대한 CAD 경로가 등록되지 않았습니다. "
                f"등록된 파트: {list(self._entries.keys())}"
            )
        entry = self._entries[part_name]
        if entry.vertices is None:
            self._load(entry)
        return entry

    @staticmethod
    def _load(entry: PartMeshEntry) -> None:
        mesh = o3d.io.read_triangle_mesh(str(entry.mesh_path))
        if len(mesh.vertices) == 0:
            raise ValueError(f"CAD를 읽을 수 없거나 정점이 없습니다: {entry.mesh_path}")

        # icp_runner.load_cad_as_pcd()와 동일한 mm 판별/스케일 규칙.
        ext = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
        if ext.max() > 10.0:
            mesh.scale(1.0 / 1000.0, center=np.zeros(3))

        mesh.compute_vertex_normals()
        entry.vertices = np.asarray(mesh.vertices, dtype=np.float64)
        entry.vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        entry.triangles = np.asarray(mesh.triangles, dtype=np.int64)


class FoundationPoseEstimator:
    """part별 FoundationPose(register/track) 인스턴스를 관리하는 래퍼.

    실제 FoundationPose/ScorePredictor/PoseRefinePredictor import는
    __init__에서 시도한다 - vendor 서브모듈이 없거나 nvdiffrast 등
    의존성이 안 깔려 있으면 여기서 바로 ImportError를 내서, 호출부
    (RTMDetInferencerFoundationPose)가 사용자에게 명확한 메시지를
    보여줄 수 있게 한다.
    """

    def __init__(self, cad_registry: CADRegistry, est_refine_iter: int = 5):
        self.cad_registry = cad_registry
        self.est_refine_iter = est_refine_iter
        self._estimators: dict[str, object] = {}
        self._scorer = None
        self._refiner = None
        self._FoundationPose = None
        self._import_foundationpose()

    def _import_foundationpose(self) -> None:
        if not _VENDOR_ROOT.is_dir():
            raise ImportError(
                f"FoundationPose 서브모듈을 찾을 수 없습니다: {_VENDOR_ROOT}\n"
                "다음으로 서브모듈을 추가하세요:\n"
                "  git submodule add https://github.com/NVlabs/FoundationPose vendor/FoundationPose\n"
                "그리고 해당 repo의 install.md대로 의존성(nvdiffrast, warp-lang 등)을 설치하세요."
            )
        if str(_VENDOR_ROOT) not in sys.path:
            sys.path.insert(0, str(_VENDOR_ROOT))
        try:
            from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor  # noqa
        except ImportError as exc:
            raise ImportError(
                "FoundationPose 의존성 import 실패. vendor/FoundationPose/install.md의 "
                "환경(nvdiffrast, warp-lang, kaolin 등)이 갖춰졌는지 확인하세요.\n"
                f"원본 오류: {exc}"
            ) from exc

        self._FoundationPose = FoundationPose
        self._scorer = ScorePredictor()
        self._refiner = PoseRefinePredictor()

    def _get_estimator(self, part_name: str):
        if part_name not in self._estimators:
            entry = self.cad_registry.get(part_name)
            self._estimators[part_name] = self._FoundationPose(
                model_pts=entry.vertices,
                model_normals=entry.vertex_normals,
                mesh_vertices=entry.vertices,
                mesh_faces=entry.triangles,
                scorer=self._scorer,
                refiner=self._refiner,
            )
        return self._estimators[part_name]

    def register(
        self,
        part_name: str,
        rgb: np.ndarray,       # (H,W,3) uint8, pseudo-RGB(채널 복제) 가능
        depth_m: np.ndarray,   # (H,W) float32, m 단위, 무효 픽셀=0
        mask: np.ndarray,      # (H,W) bool
        K: np.ndarray,         # (3,3)
    ) -> Optional[np.ndarray]:
        """part 하나에 대해 6D pose(4x4, m 단위)를 추정. 실패 시 None."""
        estimator = self._get_estimator(part_name)
        try:
            pose = estimator.register(
                K=K, rgb=rgb, depth=depth_m, ob_mask=mask,
                iteration=self.est_refine_iter,
            )
        except Exception:  # noqa: BLE001 - FoundationPose 내부 예외 형태가 다양함
            return None
        return np.asarray(pose, dtype=np.float64)