"""TCP 오프셋 보정: 로봇 Rz 회전 중심(플랜지)과 그리퍼 중심이 달라서 생기는 XY 어긋남을 소프트웨어로 보정한다.

로봇에 툴 좌표계(TCP)를 등록하는 대신, 좌표를 줄 때 이 어긋남을 미리 반영한다.

모델:  펜던트에 보이는 위치(플랜지) q = c - Rz(rz) @ d0
       c  : 그리퍼 중심의 실제 위치 (로봇 베이스 XY)
       d0 : Rz=0일 때 '플랜지 -> 그리퍼 중심' 오프셋 (베이스 XY, mm)  <- 이 모듈의 (dx, dy)
       rz : 그때 로봇의 Rz (deg)

그러므로 그리퍼 중심을 점 P에 놓고 Rz=theta로 가려면 플랜지(로봇에 보낼 좌표)는  P - Rz(theta) @ d0  이다.
카메라->로봇 변환의 점 짝을 Rz=rz_cal 상태로 모았다면 변환 결과에는 -Rz(rz_cal)@d0 이 이미 섞여 있으므로,

    보정 후 XY = 변환 XY + (Rz(rz_cal) - Rz(theta)) @ d0

(rz_cal = 보정점을 모을 때의 Rz, 보통 0). Z는 그대로 둔다 (그리퍼 길이는 변환에 이미 포함, Ry 기울기 영향은 무시).
툴 좌표계를 로봇에 이미 등록했다면 이중 보정이 되므로 이 기능을 끈다.

GUI/소켓과 무관한 순수 numpy 코드 + 저장 파일(data/tcp_offset.json) 읽기/쓰기.
"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.core.paths import PROJECT_ROOT

TCP_OFFSET_PATH = PROJECT_ROOT / "data" / "tcp_offset.json"


@dataclass
class TcpOffset:
    enabled: bool = False
    dx: float = 0.0          # Rz=0일 때 플랜지 -> 그리퍼 중심, 베이스 X (mm)
    dy: float = 0.0          # 베이스 Y (mm)
    rz_cal_deg: float = 0.0  # 카메라->로봇 변환의 점 짝을 모을 때 로봇 Rz (deg)

    @property
    def active(self) -> bool:
        return bool(self.enabled and (abs(self.dx) > 1e-9 or abs(self.dy) > 1e-9))


def _rz2(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s], [s, c]])


def shift_xy(off: TcpOffset, rz_deg: float) -> tuple[float, float]:
    """로봇 좌표에 더할 XY 보정량 (Rz(rz_cal) - Rz(rz)) @ d0. 보정이 꺼져 있으면 (0, 0)."""
    if not off.active:
        return 0.0, 0.0
    d0 = np.array([off.dx, off.dy], dtype=float)
    v = (_rz2(off.rz_cal_deg) - _rz2(rz_deg)) @ d0
    return float(v[0]), float(v[1])


def compensate(xyz, rz_deg: float, off: TcpOffset) -> tuple[float, float, float]:
    """로봇 좌표 xyz(mm)를 Rz=rz_deg로 갈 때의 플랜지 좌표로 보정한다 (Z 불변)."""
    sx, sy = shift_xy(off, rz_deg)
    return float(xyz[0]) + sx, float(xyz[1]) + sy, float(xyz[2])


def fit_offset(points, flipped: bool = True) -> dict:
    """측정 (rz_deg, x, y) 목록 -> d0. 그리퍼 중심을 바닥의 한 점에 맞춘 채 Rz만 바꿔 가며 잰 값.

    돌려주는 dict: d0(base) = (dx, dy)  <- 이 모듈에 넣는 값, tool = 툴 좌표계에 등록할 때의 값(Y 부호는 Rx≈180 가정),
    fixed_point, radius, rms, angle_span.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < 3:
        raise ValueError("측정 점이 3개 이상 필요합니다")
    th = np.radians(pts[:, 0])
    if float(np.ptp(pts[:, 0])) < 30.0:
        raise ValueError("Rz 범위가 30도 미만입니다. 90도 이상 벌려서 재세요")
    A, b = [], []
    for t, (x, y) in zip(th, pts[:, 1:]):
        c, s = math.cos(t), math.sin(t)
        A.append([1, 0, -c, s]); b.append(x)       # [I  -Rz(t)] [c; d0] = q
        A.append([0, 1, -s, -c]); b.append(y)
    A, b = np.asarray(A), np.asarray(b)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, d0x, d0y = (float(v) for v in sol)
    rms = float(np.sqrt(np.mean((A @ sol - b) ** 2)))
    return {"d0": (d0x, d0y), "tool": (d0x, -d0y if flipped else d0y), "fixed_point": (cx, cy),
            "radius": math.hypot(d0x, d0y), "rms": rms, "angle_span": float(np.ptp(pts[:, 0])), "n": len(pts)}


def parse_points(text: str) -> list[tuple[float, float, float]]:
    """'rz,x,y; rz,x,y; ...' (줄바꿈도 구분자) -> [(rz, x, y), ...]. 형식이 틀리면 ValueError."""
    out = []
    for chunk in text.replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals = [v for v in chunk.replace(",", " ").split() if v]
        if len(vals) != 3:
            raise ValueError(f"'{chunk}': rz, x, y 세 값이 필요합니다")
        out.append(tuple(float(v) for v in vals))
    return out


# --------------------------------------------------------------------------- 저장 / 불러오기
def load_tcp_offset(path: Path | None = None) -> TcpOffset:
    path = Path(path) if path is not None else TCP_OFFSET_PATH
    if not path.is_file():
        return TcpOffset()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return TcpOffset(enabled=bool(raw.get("enabled", False)), dx=float(raw.get("dx", 0.0)),
                         dy=float(raw.get("dy", 0.0)), rz_cal_deg=float(raw.get("rz_cal_deg", 0.0)))
    except (OSError, ValueError, TypeError):
        return TcpOffset()   # 깨진 파일이어도 앱은 계속 돈다 (보정 없음)


def save_tcp_offset(off: TcpOffset, path: Path | None = None) -> Path:
    path = Path(path) if path is not None else TCP_OFFSET_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"version": 1, **asdict(off)}, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


class TcpOffsetStore:
    """저장 파일을 수정시각이 바뀔 때마다 다시 읽는 스레드 안전 보관소 (13번 탭에서 저장하면 다음 요청부터 반영)."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path is not None else TCP_OFFSET_PATH
        self._lock = threading.Lock()
        self._mtime: float | None = None
        self._cached = TcpOffset()
        self._loaded = False

    def get(self) -> TcpOffset:
        with self._lock:
            try:
                mtime = self._path.stat().st_mtime if self._path.is_file() else None
            except OSError:
                mtime = None
            if not self._loaded or mtime != self._mtime:
                self._cached = load_tcp_offset(self._path)
                self._mtime, self._loaded = mtime, True
            return self._cached
