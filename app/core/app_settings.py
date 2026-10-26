"""앱 전체 설정 저장소 (data/app_settings.json).

설정 탭에서 '저장'을 누르면 이 파일에 기록되고, 프로그램을 다시 켜면 마지막으로
저장된 값이 그대로 적용된다. 규칙은 하나다:

    - 설정 탭의 입력칸 = 아직 저장 전인 편집 중인 값
    - '저장' 버튼을 눌러야 파일에 쓰이고, 그 순간부터 다른 탭(5번 TCP 서버 등)이 그 값을 쓴다

다른 탭/스레드는 항상 `get_settings().snapshot()`으로 복사본을 받아서 쓴다
(TCP 서버 워커 스레드가 요청 처리 도중에 GUI 쪽 저장과 겹쳐도 값이 반쯤 바뀌지 않도록).

파일이 없거나 깨져 있어도 앱은 기본값으로 정상 기동한다 - 깨진 파일은
app_settings.json.corrupt로 옮겨 두고 경고를 남긴다. 저장은 임시파일에 쓴 뒤
os.replace로 교체하는 원자적 쓰기라서 저장 도중 꺼져도 기존 파일이 안 깨진다.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from app.core.icp_runner import ICPParams
from app.core.paths import PROJECT_ROOT

SETTINGS_PATH = PROJECT_ROOT / "data" / "app_settings.json"
SCHEMA_VERSION = 1

# 카메라 타입 -> 카메라 설정 yaml (노출시간/동작거리 등은 이 파일 값을 그대로 쓰고,
# 평균화 프레임 수와 min_valid_ratio만 CameraSettings로 덮어쓴다).
CAMERA_CONFIG_PATHS = {
    "lucid_helios": PROJECT_ROOT / "configs" / "camera_config_helios.yaml",
    "femto_bolt": PROJECT_ROOT / "configs" / "camera_config_femto.yaml",
}


# =============================================================================
# 섹션 정의
# =============================================================================
@dataclass
class DetectionSettings:
    """2D 검출 (RTMDet-Ins) + ROI 크롭."""
    checkpoint_path: str = ""
    config_path: str = ""
    conf_threshold: float = 0.30
    device: str = "cuda:0"
    roi_enabled: bool = False
    roi_x1: int = 0
    roi_y1: int = 0
    roi_x2: int = 0
    roi_y2: int = 0

    def roi(self) -> tuple[int, int, int, int] | None:
        """ROI가 켜져 있고 크기가 유효하면 (x1, y1, x2, y2), 아니면 None(전체 이미지)."""
        if not self.roi_enabled:
            return None
        if self.roi_x2 - self.roi_x1 < 16 or self.roi_y2 - self.roi_y1 < 16:
            return None
        return (self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2)


@dataclass
class RefineSettings:
    """형상 보정: CAD 실루엣 + 깊이 마스크 보정 (src/detection/mask_refine.RefineParams의 일부)."""
    enabled: bool = True
    depth_tol_mm: float = 8.0
    min_score: float = 0.5
    depth_phi: bool = True
    resolve_overlap: bool = True
    # 의심 마스크 제외: 보정 결과의 품질 지표가 아래 조건에 걸린 검출은 정합/좌표 응답에서 뺀다
    # (3번 오프라인 검출 테스트 탭의 '의심 마스크 숨기기'와 같은 기준 - 거기는 화면에서만 숨기고, 여기는 실제로 뺀다).
    # 형상 보정을 켠 경우에만 의미가 있다 (꺼져 있으면 품질 지표가 없어 적용하지 않음).
    hide_suspect: bool = False
    suspect_drop_failed: bool = True      # 보정을 포기한(원본 유지) 마스크도 의심으로 본다
    suspect_min_score: float = 0.80       # 템플릿 적합 점수가 이 값 미만이면 의심
    suspect_max_change: float = 0.50      # 변경률((추가+삭제)/원본 면적)이 이 값 초과면 의심
    suspect_min_det_in_tpl: float = 0.60  # 원본 검출 중 CAD 투영 안에 있는 비율이 이 값 미만이면 의심


@dataclass
class CadSettings:
    cad_path: str = ""   # 프로젝트 안이면 상대경로로 저장 (프로젝트 폴더를 옮겨도 유지)


@dataclass
class CameraSettings:
    camera_type: str = "lucid_helios"   # configs/camera_config_<...>.yaml 과 연결
    num_frames: int = 8                  # 다중 프레임 평균화 (1 = 끔)
    min_valid_ratio: float = 0.6


@dataclass
class TcpSettings:
    """5번 탭 TCP 서버: 통신 설정과 응답 형식."""
    host: str = "0.0.0.0"
    port: int = 5000
    # 요청 명령 (대소문자 무시)
    trigger_cmd: str = "PICK"
    ping_cmd: str = "PING"
    # 응답 형식: <ok_prefix><sep><개수><sep>x<sep>y<sep>z[<sep>roll<sep>pitch<sep>yaw]...<terminator>
    ok_prefix: str = "OK"
    ng_prefix: str = "NG"
    field_sep: str = ","
    decimals: int = 2
    terminator: str = "LF"          # LF | CRLF | CR | NONE
    include_rpy: bool = True
    max_picks: int = 1              # 한 응답에 담을 최대 픽 후보 수
    sort_mode: str = "z_asc"        # z_asc | fitness_desc | conf_desc
    # 동작
    warmup_on_start: bool = True    # 서버 시작 시 모델/CAD를 미리 로드 (첫 요청 지연 방지)
    keep_camera_open: bool = True   # 요청마다 카메라를 새로 열지 않음 (빠름)

    def terminator_bytes(self) -> bytes:
        return {"LF": b"\n", "CRLF": b"\r\n", "CR": b"\r", "NONE": b""}.get(self.terminator, b"\n")


@dataclass
class SettingsData:
    icp: ICPParams = field(default_factory=ICPParams)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    refine: RefineSettings = field(default_factory=RefineSettings)
    cad: CadSettings = field(default_factory=CadSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    tcp: TcpSettings = field(default_factory=TcpSettings)

    SECTIONS = ("icp", "detection", "refine", "cad", "camera", "tcp")

    # ---- 직렬화 ----
    def to_dict(self) -> dict:
        out: dict[str, Any] = {"_meta": {"version": SCHEMA_VERSION,
                                          "saved_at": datetime.now().isoformat(timespec="seconds")}}
        for name in self.SECTIONS:
            out[name] = dataclasses.asdict(getattr(self, name))
        return out

    @classmethod
    def from_dict(cls, raw: dict, warnings: list[str] | None = None) -> "SettingsData":
        """알 수 없는 키/잘못된 타입은 무시하고 기본값을 쓴다 (경고만 남김) - 부분 손상에 강하다."""
        warnings = warnings if warnings is not None else []
        data = cls()
        for name in cls.SECTIONS:
            section_raw = raw.get(name)
            if section_raw is None:
                continue
            if not isinstance(section_raw, dict):
                warnings.append(f"[{name}] 섹션 형식이 올바르지 않아 기본값 사용")
                continue
            _fill_dataclass(getattr(data, name), section_raw, name, warnings)
        _validate_icp_stages(data.icp, warnings)
        return data


def _coerce(value: Any, default: Any, key: str, warnings: list[str]) -> Any:
    """JSON 값을 기본값의 타입에 맞춰 변환한다. 못 하면 default를 돌려주고 경고."""
    try:
        if isinstance(default, bool):
            if isinstance(value, bool):
                return value
            raise ValueError("bool이 아님")
        if isinstance(default, int):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("숫자가 아님")
            return int(value)
        if isinstance(default, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("숫자가 아님")
            return float(value)
        if isinstance(default, str):
            if not isinstance(value, str):
                raise ValueError("문자열이 아님")
            return value
        if isinstance(default, list):
            if not isinstance(value, list):
                raise ValueError("리스트가 아님")
            return value
    except ValueError as exc:
        warnings.append(f"{key}: {exc} -> 기본값 사용")
        return default
    return value


def _fill_dataclass(target: Any, raw: dict, section: str, warnings: list[str]) -> None:
    known = {f.name for f in dataclasses.fields(target)}
    for key, value in raw.items():
        if key not in known:
            warnings.append(f"[{section}] 알 수 없는 키 무시: {key}")
            continue
        default = getattr(target, key)
        setattr(target, key, _coerce(value, default, f"{section}.{key}", warnings))


def _validate_icp_stages(icp: ICPParams, warnings: list[str]) -> None:
    """icp_stages: [{"voxel": float|None, "max_dist": float>0, "max_iter": int>=1}, ...] (1개 이상)."""
    stages = icp.icp_stages
    ok = isinstance(stages, list) and len(stages) >= 1
    cleaned: list[dict] = []
    if ok:
        for s in stages:
            try:
                voxel = s.get("voxel")
                voxel = None if voxel is None else float(voxel)
                max_dist = float(s["max_dist"])
                max_iter = int(s["max_iter"])
                if (voxel is not None and voxel <= 0) or max_dist <= 0 or max_iter < 1:
                    raise ValueError("범위 오류")
                cleaned.append({"voxel": voxel, "max_dist": max_dist, "max_iter": max_iter})
            except (AttributeError, KeyError, TypeError, ValueError):
                ok = False
                break
    if ok:
        icp.icp_stages = cleaned
    else:
        warnings.append("icp.icp_stages 형식 오류 -> 기본 단계 사용")
        icp.icp_stages = ICPParams().icp_stages


# =============================================================================
# 저장소
# =============================================================================
class SettingsStore(QObject):
    """파일 <-> 메모리 동기화 + 변경 알림. 스레드 안전."""

    changed = pyqtSignal()   # 저장/기본값 복원 등으로 '적용 중인 값'이 바뀔 때

    def __init__(self, path: Path | None = None, parent: QObject | None = None):
        super().__init__(parent)
        self._path = Path(path) if path is not None else SETTINGS_PATH
        self._lock = threading.RLock()
        self._data = SettingsData()
        self.load_warnings: list[str] = []
        self.loaded_from_file = False
        self.load()

    @property
    def path(self) -> Path:
        return self._path

    # ---- 읽기 ----
    def snapshot(self) -> SettingsData:
        with self._lock:
            return copy.deepcopy(self._data)

    # ---- 파일 읽기 ----
    def load(self) -> list[str]:
        """파일에서 읽어 적용한다. 파일이 없으면 기본값. 경고 목록을 반환한다."""
        warnings: list[str] = []
        data = SettingsData()
        loaded = False
        if self._path.is_file():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError("최상위가 객체가 아님")
                data = SettingsData.from_dict(raw, warnings)
                loaded = True
            except (OSError, ValueError) as exc:   # JSONDecodeError는 ValueError 하위
                backup = self._path.with_suffix(self._path.suffix + ".corrupt")
                try:
                    os.replace(self._path, backup)
                    warnings.append(f"설정 파일을 읽지 못해 기본값으로 시작합니다 ({exc}). 원본은 {backup.name}로 보관했습니다.")
                except OSError:
                    warnings.append(f"설정 파일을 읽지 못해 기본값으로 시작합니다 ({exc}).")
        with self._lock:
            self._data = data
            self.load_warnings = warnings
            self.loaded_from_file = loaded
        self.changed.emit()
        return warnings

    # ---- 쓰기 ----
    def save(self, data: SettingsData) -> None:
        """data를 파일에 원자적으로 쓰고 적용 중인 값으로 바꾼다. 실패하면 예외 (메모리도 안 바뀜)."""
        payload = json.dumps(data.to_dict(), indent=2, ensure_ascii=False)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".app_settings_", suffix=".tmp", dir=str(self._path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, self._path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        with self._lock:
            self._data = copy.deepcopy(data)
            self.loaded_from_file = True
        self.changed.emit()


_store: SettingsStore | None = None
_store_lock = threading.Lock()


def get_settings() -> SettingsStore:
    """프로세스 전체에서 공유하는 설정 저장소 (처음 호출될 때 파일을 읽는다)."""
    global _store
    with _store_lock:
        if _store is None:
            _store = SettingsStore()
        return _store


def to_project_relative(path_str: str) -> str:
    """프로젝트 폴더 안의 경로면 상대경로로, 밖이면 그대로 돌려준다 (저장 시 이식성 확보용)."""
    if not path_str:
        return ""
    p = Path(path_str)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.resolve())) if p.is_absolute() else path_str
    except ValueError:
        return path_str