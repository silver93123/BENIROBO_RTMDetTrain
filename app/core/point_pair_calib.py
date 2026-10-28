"""카메라 좌표 -> 로봇 좌표 변환 추정 (점 짝 방식).

같은 물리적 점을 카메라 점군에서 읽은 좌표(cam)와 로봇 티치 펜던트에서 읽은 좌표(robot)로 짝지어 모으고,
    robot = R @ cam + t
인 강체 변환(회전 R + 이동 t)을 최소제곱으로 푼다 (Kabsch/Umeyama의 SVD 풀이).

주의 - 이 변환은 점 짝을 모은 '그 촬영 자세'에서만 유효하다. 로봇 손에 달린 카메라(eye-in-hand)는 촬영 자세가
바뀌면 변환도 바뀌므로, 실제 피킹도 보정할 때와 같은 촬영 자세에서 찍어야 한다. 그래서 자세를 메모로 같이 저장한다.

GUI/소켓과 무관한 순수 numpy 코드 (단위 테스트 가능).
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.core.paths import PROJECT_ROOT

CALIB_PATH = PROJECT_ROOT / "data" / "gripper_calib.json"
DEFAULT_RMS_LIMIT_MM = 2.0     # 이보다 잔차가 크면 응답에 변환을 적용하지 않는다


class CalibError(ValueError):
    """변환을 풀 수 없는 입력 (점 부족, 일직선 배치 등)."""


@dataclass
class CalibPair:
    cam: tuple[float, float, float]                    # 카메라 좌표 (mm)
    robot: tuple[float, float, float] | None = None    # 로봇 좌표 (mm) - 아직 입력 안 했으면 None
    use: bool = True                                   # 풀이에 사용 (False면 검증용으로 잔차만 본다)


@dataclass
class RigidFit:
    R: np.ndarray                         # (3, 3)
    t: np.ndarray                         # (3,)
    rms_mm: float                         # 사용한 점들의 잔차 RMS
    max_mm: float
    loo_rms_mm: float | None              # leave-one-out 오차 RMS (사용 점 4개 이상일 때)
    n_used: int
    errors_mm: list[float | None]         # 입력 순서대로 점별 오차 (robot 미입력이면 None). 검증용(use=False) 점도 계산한다
    extent_mm: float                      # 사용한 점들이 퍼진 크기(최대 거리)
    scale_ratio: float                    # 로봇/카메라 거리 비 (1에 가까워야 한다)
    warnings: list[str] = field(default_factory=list)

    def apply(self, cam_xyz) -> tuple[float, float, float]:
        v = self.R @ np.asarray(cam_xyz, dtype=float) + self.t
        return float(v[0]), float(v[1]), float(v[2])

    def euler_zyx_deg(self) -> tuple[float, float, float]:
        """회전 R을 Rz*Ry*Rx 순서의 (rx, ry, rz) 도 단위로 표시용 변환 (로봇 규약과 다를 수 있다 - 참고용)."""
        R = self.R
        sy = float(np.hypot(R[0, 0], R[1, 0]))
        if sy > 1e-9:
            rx = np.arctan2(R[2, 1], R[2, 2]); ry = np.arctan2(-R[2, 0], sy); rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1]); ry = np.arctan2(-R[2, 0], sy); rz = 0.0
        return tuple(float(np.degrees(a)) for a in (rx, ry, rz))   # type: ignore[return-value]


def fit_rigid(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dst ~= R @ src + t 를 만족하는 최소제곱 강체 변환 (Kabsch). 반사(det<0)는 회전으로 보정한다."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d if d != 0 else 1.0])
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


def solve(pairs: list[CalibPair]) -> RigidFit:
    """pairs 중 robot이 입력되고 use=True인 점으로 변환을 푼다. 풀 수 없으면 CalibError."""
    used = [i for i, p in enumerate(pairs) if p.use and p.robot is not None]
    if len(used) < 3:
        raise CalibError(f"사용할 점 짝이 {len(used)}개입니다. 최소 3개(권장 10개 이상)가 필요합니다.")
    src = np.array([pairs[i].cam for i in used], dtype=np.float64)
    dst = np.array([pairs[i].robot for i in used], dtype=np.float64)
    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        raise CalibError("좌표에 숫자가 아닌 값이 있습니다.")

    sv = np.linalg.svd(src - src.mean(axis=0), compute_uv=False)
    sv_d = np.linalg.svd(dst - dst.mean(axis=0), compute_uv=False)
    if sv[0] < 1e-6 or sv[1] / sv[0] < 0.02 or sv_d[1] / max(sv_d[0], 1e-9) < 0.02:
        raise CalibError("점들이 거의 일직선 위에 있습니다. 작업 영역에 넓게 흩어 점을 잡으세요 (회전을 정할 수 없음).")

    R, t = fit_rigid(src, dst)
    pred = src @ R.T + t
    err = np.linalg.norm(pred - dst, axis=1)

    warnings: list[str] = []
    n = len(used)
    if n < 4:
        warnings.append("점이 3개뿐이라 잔차가 0에 가깝게 나와도 검증이 되지 않습니다. 10개 이상 권장.")
    elif n < 8:
        warnings.append(f"점이 {n}개입니다. 10~20개로 늘리면 더 믿을 수 있습니다.")
    extent = float(max(np.linalg.norm(a - b) for a in src for b in src)) if n > 1 else 0.0
    if extent < 150.0:
        warnings.append(f"점들이 {extent:.0f} mm 안에 모여 있습니다. 작업 영역 전체(수백 mm)에 퍼뜨리면 회전 오차가 줄어듭니다.")
    if sv[2] / sv[0] < 0.05:
        warnings.append("점들이 거의 한 평면에 있습니다. 높이(Z)를 2~3단으로 달리해 모으면 기울기 추정이 안정됩니다.")
    d_cam = np.linalg.norm(src[:, None] - src[None], axis=2)
    d_rob = np.linalg.norm(dst[:, None] - dst[None], axis=2)
    mask = d_cam > 1e-6
    ratio = float(np.median(d_rob[mask] / d_cam[mask])) if mask.any() else 1.0
    if abs(ratio - 1.0) > 0.02:
        warnings.append(f"점 사이 거리가 카메라와 로봇에서 {ratio:.3f}배 차이납니다. 단위(mm)나 점 대응이 맞는지 확인하세요.")

    loo = None
    if n >= 4:
        errs = []
        for k in range(n):
            keep = [j for j in range(n) if j != k]
            if np.linalg.svd(src[keep] - src[keep].mean(axis=0), compute_uv=False)[1] < 1e-6:
                continue
            Rk, tk = fit_rigid(src[keep], dst[keep])
            errs.append(float(np.linalg.norm(Rk @ src[k] + tk - dst[k])))
        if errs:
            loo = float(np.sqrt(np.mean(np.square(errs))))

    per_point: list[float | None] = []
    for p in pairs:
        if p.robot is None:
            per_point.append(None)
        else:
            v = R @ np.asarray(p.cam, dtype=float) + t
            per_point.append(float(np.linalg.norm(v - np.asarray(p.robot, dtype=float))))
    return RigidFit(R=R, t=t, rms_mm=float(np.sqrt(np.mean(err ** 2))), max_mm=float(err.max()), loo_rms_mm=loo,
                    n_used=n, errors_mm=per_point, extent_mm=extent, scale_ratio=ratio, warnings=warnings)


# =============================================================================
# 저장 / 불러오기
# =============================================================================
def save_calibration(path: Path, pairs: list[CalibPair], fit: RigidFit | None, pose_memo: str,
                     rms_limit_mm: float = DEFAULT_RMS_LIMIT_MM) -> None:
    data = {
        "version": 1,
        "pose_memo": pose_memo,
        "rms_limit_mm": rms_limit_mm,
        "pairs": [{"cam": list(p.cam), "robot": list(p.robot) if p.robot is not None else None, "use": p.use} for p in pairs],
        "transform": None if fit is None else {
            "R": fit.R.tolist(), "t": fit.t.tolist(), "rms_mm": fit.rms_mm, "max_mm": fit.max_mm,
            "loo_rms_mm": fit.loo_rms_mm, "n_used": fit.n_used},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_calibration(path: Path) -> tuple[list[CalibPair], dict | None, str, float]:
    """(점 짝, 저장된 transform dict 또는 None, 촬영 자세 메모, RMS 한계). 파일이 없으면 빈 값."""
    if not Path(path).is_file():
        return [], None, "", DEFAULT_RMS_LIMIT_MM
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = [CalibPair(cam=tuple(float(v) for v in p["cam"]),
                       robot=None if p.get("robot") is None else tuple(float(v) for v in p["robot"]),
                       use=bool(p.get("use", True))) for p in raw.get("pairs", [])]
    return pairs, raw.get("transform"), str(raw.get("pose_memo", "")), float(raw.get("rms_limit_mm", DEFAULT_RMS_LIMIT_MM))


# =============================================================================
# 응답 쪽에서 쓰는 스레드 안전 보관소
# =============================================================================
class OutputConfig:
    """PICK 응답 좌표계 설정. GUI 스레드가 쓰고 소켓 스레드가 읽는다."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame = "camera"                 # "camera" | "robot"
        self._R: np.ndarray | None = None
        self._t: np.ndarray | None = None
        self._rms: float | None = None
        self._limit = DEFAULT_RMS_LIMIT_MM

    def set(self, frame: str, R=None, t=None, rms: float | None = None, limit: float | None = None) -> None:
        with self._lock:
            self._frame = frame
            self._R = None if R is None else np.asarray(R, dtype=float)
            self._t = None if t is None else np.asarray(t, dtype=float)
            self._rms = rms
            if limit is not None:
                self._limit = float(limit)

    def snapshot(self) -> tuple[str, np.ndarray | None, np.ndarray | None, float | None, float]:
        with self._lock:
            return self._frame, self._R, self._t, self._rms, self._limit


# =============================================================================
# 저장된 변환을 쓰는 출력 설정 (4-1 표시 / 5-2 TCP 응답용)
# =============================================================================
class SavedTransformOutput(OutputConfig):
    """13번 탭이 '저장'한 변환(data/gripper_calib.json)을 읽어 쓰는 OutputConfig.

    체크박스(enabled)가 켜져 있으면 frame="robot", 꺼져 있으면 "camera". 변환은 파일 수정시각이 바뀔 때마다
    다시 읽으므로 13번 탭에서 다시 저장하면 서버를 재시작하지 않아도 다음 요청부터 반영된다.
    저장 파일에 transform이 없으면(점 짝만 저장됨) 점 짝으로 다시 푼다.
    """

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self._path = Path(path) if path is not None else CALIB_PATH
        self._enabled = False
        self._mtime: float | None = None
        self._memo = ""
        self._n_used = 0
        self._err = ""

    def set_enabled(self, on: bool) -> None:
        with self._lock:
            self._enabled = bool(on)

    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def _reload_locked(self) -> None:
        try:
            mtime = self._path.stat().st_mtime if self._path.is_file() else None
        except OSError:
            mtime = None
        if mtime == self._mtime and (mtime is not None or self._err):
            return
        self._mtime = mtime
        self._R = self._t = self._rms = None
        self._memo, self._n_used, self._err = "", 0, ""
        if mtime is None:
            self._err = f"저장된 변환 파일이 없습니다 ({self._path.name}) - 13번 탭에서 변환을 계산하고 [저장]하세요"
            return
        try:
            pairs, tf, memo, limit = load_calibration(self._path)
            if tf is None:
                fit = solve(pairs)
                R, t, rms, n = fit.R, fit.t, fit.rms_mm, fit.n_used
            else:
                R, t, rms, n = np.asarray(tf["R"], float), np.asarray(tf["t"], float), float(tf["rms_mm"]), int(tf.get("n_used", 0))
            if R.shape != (3, 3) or t.shape != (3,):
                raise ValueError("변환 행렬 모양이 올바르지 않습니다")
            self._R, self._t, self._rms, self._limit, self._memo, self._n_used = R, t, rms, limit, memo, n
        except Exception as exc:  # noqa: BLE001 - 깨진 파일이어도 앱은 계속 돈다
            self._err = f"저장된 변환을 읽지 못했습니다: {exc}"

    def snapshot(self):
        with self._lock:
            self._reload_locked()
            return ("robot" if self._enabled else "camera"), self._R, self._t, self._rms, self._limit

    def check(self) -> tuple[str, str]:
        """(코드, 설명). 코드: "" 정상 | "NO_CALIB" | "CALIB_BAD". 로봇 좌표 변환을 지금 쓸 수 있는지."""
        with self._lock:
            self._reload_locked()
            if self._R is None or self._t is None:
                return "NO_CALIB", self._err or "저장된 좌표 변환이 없습니다"
            if self._rms is not None and self._rms > self._limit:
                return "CALIB_BAD", f"변환 잔차 RMS {self._rms:.2f} mm 가 한계 {self._limit:.2f} mm 를 넘습니다"
            memo = f" · 촬영 자세: {self._memo}" if self._memo else ""
            return "", f"점 {self._n_used}개, 잔차 RMS {self._rms:.2f} mm{memo}"

    def convert(self, xyz) -> tuple[float, float, float] | None:
        """카메라 좌표 -> 로봇 좌표. 변환을 쓸 수 없으면 None."""
        if self.check()[0]:
            return None
        with self._lock:
            v = self._R @ np.asarray(xyz, dtype=float) + self._t
        return float(v[0]), float(v[1]), float(v[2])
