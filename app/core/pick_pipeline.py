"""헤드리스 픽 파이프라인: 카메라 프레임 1장 -> 2D 검출 -> 3D 정합 -> 정렬된 픽 후보.

탭 4(ICP 검출 테스트)가 GUI 안에서 단계별로 하던 일을, 화면 없이 한 번에 돌리는 버전이다.
5번 탭의 TCP 서버(app/core/pick_server.py)가 로봇 요청을 받을 때마다 run()을 호출한다.
계산 함수는 전부 기존 코드(icp_runner, mask_refine, Detector)를 그대로 쓰고, 이 파일은
"설정 -> 단계 연결 -> 결과 정리"만 맡는다.

    mode="2d" 이면 3D 정합(ICP) 대신 형상 보정 템플릿에서 픽 포인트/회전을 바로 구한다
    (app/core/pose_from_template.py, 5-2 탭). 형상 보정이 켜져 있어야 하고, 보정이 성공한 검출만 후보가 된다.

    ROI (픽점 기준)         settings.detection.roi_*  (전체 이미지로 검출 -> 픽점 산출 -> 픽점이 ROI 안인 것만 후보)
    2D 검출 (RTMDet-Ins)    settings.detection.*
    형상 보정 (선택)        settings.refine.*      (실패하면 원본 마스크로 계속 진행)
    의심 마스크 제외 (선택) settings.refine.suspect_*
    3D 정합 (ICP)           settings.icp (= ICPParams)
    정렬                    settings.tcp.sort_mode

좌표 규약: 픽 좌표는 카메라 좌표계, 단위 mm / deg (icp_runner.transform_to_pose와 동일).
로봇 베이스 좌표계로의 변환(핸드-아이 캘리브레이션)은 이 파이프라인 밖의 일이다.

스레드: Detector/CAD 캐시를 들고 있으므로 한 스레드에서만 호출한다 (PickWorker가 보장).
"""
from __future__ import annotations

import dataclasses
import os
import tempfile
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from app.core import icp_runner, pose_from_template, roi_filter
from app.core.app_settings import SettingsData
from app.core.camera_intrinsics import estimate_intrinsics_from_organized_pcd
from app.core.detector import Detection, Detector
from app.core.icp_runner import ICPParams, ICPResult
from app.core.paths import resolve_relative_to_project
from src.detection.mask_refine import RefineParams, RefineResult, get_bank, refine_masks

SORT_MODES = {
    "z_asc": "z 오름차순 (카메라에 가까운 = 위에 있는 것 먼저)",
    "fitness_desc": "fitness 내림차순 (정합이 가장 확실한 것 먼저)",
    "conf_desc": "검출 confidence 내림차순",
}


class PipelineError(Exception):
    """code는 TCP 응답의 NG 코드로 그대로 쓰인다 (공백 없는 대문자 스네이크)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class PickCandidate:
    instance_id: int
    label: str
    confidence: float
    xyz_mm: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]
    fitness: float
    rmse_mm: float
    was_flipped: bool = False
    axis_cam: tuple[float, float, float] | None = None   # 부품 긴 축 방향 (카메라 좌표, 2D 기반만). 로봇 좌표 변환 시 Rz/Ry 계산용


@dataclass
class PickRunResult:
    ok: bool = False
    error_code: str = ""            # NO_DETECTION | NO_VALID_POSE | CONFIG_ERROR | CAMERA_ERROR | INTERNAL_ERROR ...
    error_message: str = ""
    candidates: list[PickCandidate] = field(default_factory=list)   # 정렬 완료, 성공한 것만
    started_at: str = ""
    image_path: str | None = None   # 이번 프레임의 intensity 이미지 (표시용)
    detections: list[Detection] = field(default_factory=list)       # ICP에 쓰인 검출 (보정 켜면 보정본)
    refine_results: list[RefineResult] = field(default_factory=list)
    # 의심 마스크로 제외된 검출: (제외 전 번호, 사유, confidence)
    suspect_excluded: list[tuple[int, str, float]] = field(default_factory=list)
    # 픽점이 ROI 밖이라 제외된 검출: (번호, 사유). ROI는 검출 후 픽점 기준으로 적용한다 (app/core/roi_filter.py)
    roi_excluded: list[tuple[int, str]] = field(default_factory=list)
    icp_results: list[ICPResult] = field(default_factory=list)
    # mode="2d" 전용: 인스턴스별 2D 기반 pose (detections와 index 일치), 화면 표시용 카메라 모델/CAD 크기
    mode: str = "icp"
    pose2d_results: list = field(default_factory=list)
    intrinsics: tuple | None = None
    cad_extent_mm: float = 60.0
    timings_ms: dict[str, float] = field(default_factory=dict)
    coord_frame: str = "camera"     # candidates의 xyz_mm 좌표계: "camera" | "robot" (좌표 변환 적용 여부)

    def summary(self) -> str:
        n_det = len(self.detections)
        n_ok = sum(p.ok for p in self.pose2d_results) if self.mode == "2d" else sum(r.ok for r in self.icp_results)
        total = self.timings_ms.get("total")
        t = f", {total:.0f}ms" if total is not None else ""
        if self.suspect_excluded:
            t = f", 의심 마스크 {len(self.suspect_excluded)}건 제외" + t
        if self.roi_excluded:
            t = f", ROI 밖 {len(self.roi_excluded)}건 제외" + t
        if self.ok:
            what = "2D 계산 성공" if self.mode == "2d" else "정합 성공"
            return f"성공: 검출 {n_det} / {what} {n_ok} / 후보 {len(self.candidates)}{t}"
        what = "2D 계산 성공" if self.mode == "2d" else "정합 성공"
        return f"실패 [{self.error_code}] {self.error_message} (검출 {n_det}, {what} {n_ok}){t}"


def sort_candidates(cands: list[PickCandidate], mode: str) -> list[PickCandidate]:
    if mode == "fitness_desc":
        return sorted(cands, key=lambda c: -c.fitness)
    if mode == "conf_desc":
        return sorted(cands, key=lambda c: -c.confidence)
    return sorted(cands, key=lambda c: c.xyz_mm[2])   # z_asc (기본)


class PickPipeline:
    def __init__(self, log: Callable[[str], None] | None = None, mode: str = "icp"):
        if mode not in ("icp", "2d"):
            raise ValueError(f"mode는 'icp' 또는 '2d': {mode!r}")
        self.mode = mode
        self._log = log or (lambda _m: None)
        self._detector: Detector | None = None
        self._detector_key: tuple | None = None
        # CAD 캐시
        self._cad_pcd = None
        self._cad_visible_normal = None
        self._cad_visible_flipped = None
        self._cad_load_key: tuple | None = None
        self._cad_vis_key: tuple | None = None
        # 표시용 프레임 파일 (최근 몇 장만 남기고 지운다)
        self._frame_dir = Path(tempfile.mkdtemp(prefix="benirobo_pick_"))
        self._frame_files: deque[Path] = deque()
        self._frame_seq = 0

    # ------------------------------------------------------------------ 준비
    def warmup(self, snap: SettingsData) -> None:
        """모델과 CAD를 미리 올린다 (첫 요청 지연 방지). 설정이 잘못됐으면 PipelineError."""
        detector = self._ensure_detector(snap)
        if detector._inferencer is None:
            detector.load_model()   # predict() 첫 호출 때 일어나는 모델 로드를 미리 해둔다
        if self.mode == "2d":
            self._ensure_cad_pcd(snap)
            get_bank(self._cad_path(snap))              # 템플릿 뱅크(실루엣)도 미리 만든다
            return
        self._ensure_cad(snap)
        icp_runner._build_default_estimator(snap.icp)   # 알고리즘 설정 오류를 미리 발견

    def _ensure_detector(self, snap: SettingsData) -> Detector:
        det = snap.detection
        ckpt = resolve_relative_to_project(det.checkpoint_path) if det.checkpoint_path else None
        cfg = resolve_relative_to_project(det.config_path) if det.config_path else None
        if ckpt is None or not ckpt.is_file():
            raise PipelineError("CONFIG_ERROR", f"체크포인트 파일이 없습니다: {det.checkpoint_path or '(미지정)'}")
        if cfg is None or not cfg.is_file():
            raise PipelineError("CONFIG_ERROR", f"config 파일이 없습니다: {det.config_path or '(미지정)'}")
        key = (str(ckpt), str(cfg), det.device)
        if self._detector is None or self._detector_key != key:
            self._detector = Detector(checkpoint_path=str(ckpt), config_path=str(cfg), device=det.device,
                                      score_threshold=det.conf_threshold, backend="rtmdet_ins")
            self._detector_key = key
            self._log(f"검출 모델 준비: {ckpt.name} ({det.device})")
        return self._detector

    def _cad_path(self, snap: SettingsData) -> Path:
        if not snap.cad.cad_path:
            raise PipelineError("CONFIG_ERROR", "CAD 모델이 지정되지 않았습니다 (설정 탭 > 검출/CAD).")
        p = resolve_relative_to_project(snap.cad.cad_path)
        if not p.is_file():
            raise PipelineError("CONFIG_ERROR", f"CAD 파일이 없습니다: {snap.cad.cad_path}")
        return p

    def _ensure_cad_pcd(self, snap: SettingsData) -> None:
        """CAD 로드(+축 보정)만. 값이 바뀔 때만 다시 계산한다."""
        p = self._cad_path(snap)
        params = snap.icp
        load_key = (str(p), p.stat().st_mtime_ns, params.cad_axis_correction_deg)
        if self._cad_pcd is None or self._cad_load_key != load_key:
            self._log(f"CAD 로드: {p.name} (축보정 R{params.cad_axis_roll_deg:.0f} "
                      f"P{params.cad_axis_pitch_deg:.0f} Y{params.cad_axis_yaw_deg:.0f})")
            self._cad_pcd = icp_runner.load_cad_as_pcd(str(p), params)
            self._cad_load_key = load_key
            self._cad_vis_key = None

    def _ensure_cad(self, snap: SettingsData) -> None:
        """CAD 준비: CAD 로드(+축 보정)와 가시면 계산. 값이 바뀔 때만 다시 계산한다."""
        self._ensure_cad_pcd(snap)
        params = snap.icp
        vis_key = (params.init_rotation_deg, params.cad_hpr_ref_distance_m, params.cad_hpr_radius_factor)
        if self._cad_vis_key != vis_key:
            normal, flipped = icp_runner.build_visible_cad_pair(self._cad_pcd, params)
            total, vis = len(self._cad_pcd.points), len(normal.points)
            if total == 0 or vis / total < 0.05:
                self._log(f"경고: CAD 가시면이 비정상적으로 적음({vis}/{total}점) - "
                          f"'카메라~부품 거리' 설정 확인. CAD 전체로 폴백합니다.")
                normal = flipped = self._cad_pcd
            self._cad_visible_normal, self._cad_visible_flipped = normal, flipped
            self._cad_vis_key = vis_key
            self._log(f"CAD 가시면 준비: {total}점 -> {vis}점")

    # ------------------------------------------------------------------ 실행
    def run(self, frame, snap: SettingsData) -> PickRunResult:
        """frame: src.camera.FrameData. 예외를 던지지 않고 항상 PickRunResult를 돌려준다."""
        res = PickRunResult(started_at=datetime.now().isoformat(timespec="milliseconds"))
        t0 = time.perf_counter()
        try:
            self._run(frame, snap, res)
        except PipelineError as exc:
            res.ok, res.error_code, res.error_message = False, exc.code, str(exc)
        except Exception as exc:  # noqa: BLE001 - 서버가 죽지 않게 모두 결과로 바꾼다
            self._log(f"파이프라인 예외: {exc}\n{traceback.format_exc()}")
            res.ok, res.error_code, res.error_message = False, "INTERNAL_ERROR", f"{type(exc).__name__}: {exc}"
        res.timings_ms["total"] = (time.perf_counter() - t0) * 1000.0
        return res

    def _run(self, frame, snap: SettingsData, res: PickRunResult) -> None:
        pcd, valid = frame.points_organized, frame.valid_mask.astype(bool)
        res.image_path = self._save_frame_image(frame.intensity)
        res.mode = self.mode

        # ---- 준비 (설정 오류는 여기서 CONFIG_ERROR로 끝낸다) ----
        detector = self._ensure_detector(snap)
        params: ICPParams = snap.icp
        if self.mode == "2d":
            if not snap.refine.enabled:
                raise PipelineError("CONFIG_ERROR", "2D 기반 모드는 형상 보정이 켜져 있어야 합니다 "
                                    "(설정 탭 > 검출/CAD > 형상 보정 사용).")
            self._ensure_cad_pcd(snap)
        else:
            self._ensure_cad(snap)
            try:
                icp_runner._build_default_estimator(params)
            except Exception as exc:  # noqa: BLE001
                raise PipelineError("CONFIG_ERROR", f"정합 알고리즘 설정 오류: {exc}") from exc

        # ---- 2D 검출 (전체 이미지. ROI는 픽점 산출 뒤에 픽점 기준으로 적용한다) ----
        t = time.perf_counter()
        det_cfg = snap.detection
        try:
            raw = detector.predict(res.image_path, conf_threshold=det_cfg.conf_threshold, roi=None)
        except ImportError as exc:
            self._detector = None
            raise PipelineError("CONFIG_ERROR", f"추론 엔진 없음: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            self._detector = None   # 모델 로드 실패일 수 있으니 다음엔 새로 만든다
            raise PipelineError("DETECTION_ERROR", f"{type(exc).__name__}: {exc}") from exc
        raw = [d for d in raw if d.mask is not None]
        res.timings_ms["detect"] = (time.perf_counter() - t) * 1000.0
        if not raw:
            res.detections = []
            raise PipelineError("NO_DETECTION", f"검출 0건 (conf≥{det_cfg.conf_threshold:.2f})")

        # ---- 형상 보정 + 의심 마스크 제외 ----
        t = time.perf_counter()
        res.detections, res.refine_results = self._refine(raw, pcd, valid, snap)
        res.detections, res.refine_results, res.suspect_excluded = self._drop_suspect(
            res.detections, res.refine_results, snap.refine)
        res.timings_ms["refine"] = (time.perf_counter() - t) * 1000.0
        if res.suspect_excluded:
            self._log(f"의심 마스크 {len(res.suspect_excluded)}건 제외: "
                      + ", ".join(f"obj{i}({why})" for i, why, _c in res.suspect_excluded))
        if not res.detections:
            raise PipelineError("NO_DETECTION", f"검출 {len(raw)}건이 모두 의심 마스크로 제외됨 "
                                f"({'; '.join(why for _i, why, _c in res.suspect_excluded[:3])})")

        if self.mode == "2d":
            self._run_2d(pcd, valid, snap, res)
            return

        # ---- 3D 정합 ----
        t = time.perf_counter()
        pcd_std = getattr(frame, "points_std", None)
        for i, det in enumerate(res.detections):
            try:
                if params.pc_upsample_method == "probabilistic" and params.pc_upsample_factor > 1:
                    pts_mm = icp_runner.extract_instance_points_probabilistic(
                        det.mask, pcd, pcd_std, valid,
                        erode_px=params.mask_erode_px, samples_per_point=params.pc_upsample_factor)
                else:
                    pts_mm = icp_runner.extract_instance_points_mm(
                        det.mask, pcd, valid, erode_px=params.mask_erode_px,
                        upsample_factor=params.pc_upsample_factor, upsample_method=params.pc_upsample_method)
                r = icp_runner.run_icp_for_instance(
                    i, pts_mm, self._cad_pcd, self._cad_visible_normal, self._cad_visible_flipped,
                    params=params, T_init_override=getattr(det, "initial_pose", None))
            except Exception as exc:  # noqa: BLE001 - 한 인스턴스 실패가 나머지를 막지 않게
                self._log(f"obj{i} 정합 예외: {exc}\n{traceback.format_exc()}")
                r = ICPResult(instance_id=i, ok=False, error=f"예외: {exc}")
            res.icp_results.append(r)
        res.timings_ms["icp"] = (time.perf_counter() - t) * 1000.0

        # ---- ROI: 픽점이 ROI 밖인 정합 결과는 실패 처리 ----
        roi = snap.detection.roi()
        if roi is not None:
            intr = estimate_intrinsics_from_organized_pcd(pcd, valid)
            for r in res.icp_results:
                if r.ok and r.pick_point_mm is not None:
                    uv = roi_filter.pick_pixel(r.pick_point_mm, pcd, valid, intr)
                    why = roi_filter.outside_reason(uv, roi)
                    if why:
                        r.ok, r.error = False, why
                        res.roi_excluded.append((r.instance_id, why))
            self._log_roi_excluded(res)

        # ---- 픽 후보 정리 ----
        cands: list[PickCandidate] = []
        for r in res.icp_results:
            if not r.ok:
                continue
            det = res.detections[r.instance_id]
            e = r.pose["euler_deg"]
            cands.append(PickCandidate(
                instance_id=r.instance_id, label=det.label, confidence=float(det.confidence),
                xyz_mm=tuple(float(v) for v in r.pick_point_mm),
                rpy_deg=(float(e["roll_deg"]), float(e["pitch_deg"]), float(e["yaw_deg"])),
                fitness=float(r.fitness), rmse_mm=float(r.rmse_m) * 1000.0, was_flipped=r.was_flipped))
        res.candidates = sort_candidates(cands, snap.tcp.sort_mode)
        if not res.candidates:
            fits = [r.fitness for r in res.icp_results if r.fitness is not None]
            best = f", 최고 fitness {max(fits):.3f}" if fits else ""
            first_err = next((r.error for r in res.icp_results if r.error), "")
            raise PipelineError("NO_VALID_POSE",
                                f"정합 성공 0/{len(res.icp_results)}{best} ({first_err})")
        res.ok = True

    def _log_roi_excluded(self, res: PickRunResult) -> None:
        if res.roi_excluded:
            self._log(f"ROI 밖 픽점 {len(res.roi_excluded)}건 제외: "
                      + ", ".join(f"obj{i}" for i, _w in res.roi_excluded))

    def _run_2d(self, pcd, valid, snap: SettingsData, res: PickRunResult) -> None:
        """ICP 없이: 형상 보정이 고른 템플릿에서 픽 포인트/회전을 바로 구한다."""
        t = time.perf_counter()
        if len(res.refine_results) != len(res.detections):
            raise PipelineError("REFINE_ERROR", "형상 보정을 실행하지 못했습니다 (로그 확인)")
        bank = get_bank(self._cad_path(snap))
        intr = estimate_intrinsics_from_organized_pcd(pcd, valid)
        res.intrinsics = tuple(float(v) for v in intr)
        res.cad_extent_mm = float(self._cad_pcd.get_axis_aligned_bounding_box().get_max_extent()) * 1000.0
        center_mm = np.asarray(self._cad_pcd.get_center(), dtype=np.float64) * 1000.0   # ICP와 같은 정의의 CAD 중심
        roi = snap.detection.roi()
        cands: list[PickCandidate] = []
        for i, rr in enumerate(res.refine_results):
            kw = dict(cad_axis_deg=snap.icp.cad_axis_correction_deg, cad_center_mm=center_mm,
                      depth_correct=bool(snap.refine.depth_correct))
            if snap.refine.pos_from_pcd:   # 위치를 점군에서 직접 읽는다 (13번 탭 클릭 좌표와 같은 방식)
                p = pose_from_template.pose_from_refine_pcd(rr, bank, intr, pcd, valid, **kw)
            else:
                p = pose_from_template.pose_from_refine(rr, bank, intr, **kw)
            res.pose2d_results.append(p)
            if p.ok and roi is not None:   # 픽점이 ROI 밖이면 실패 처리
                uv = roi_filter.pick_pixel(p.pick_point_mm, pcd, valid, intr, uv=p.uv or rr.center_uv)
                why = roi_filter.outside_reason(uv, roi)
                if why:
                    p.ok, p.error = False, why
                    res.roi_excluded.append((i, why))
            if not p.ok:
                continue
            det = res.detections[i]
            e = p.pose["euler_deg"]
            cands.append(PickCandidate(
                instance_id=i, label=det.label, confidence=float(det.confidence),
                xyz_mm=tuple(float(v) for v in p.pick_point_mm),
                rpy_deg=(float(e["roll_deg"]), float(e["pitch_deg"]), float(e["yaw_deg"])),
                fitness=float(rr.score), rmse_mm=0.0,   # 2D 기반에서 fitness 자리는 형상 보정 score
                axis_cam=None if p.axis_cam is None else tuple(float(v) for v in p.axis_cam)))
        res.timings_ms["pose2d"] = (time.perf_counter() - t) * 1000.0
        self._log_roi_excluded(res)
        res.candidates = sort_candidates(cands, snap.tcp.sort_mode)
        if not res.candidates:
            first_err = next((p.error for p in res.pose2d_results if p.error), "")
            raise PipelineError("NO_VALID_POSE", f"2D 기반 계산 성공 0/{len(res.pose2d_results)} ({first_err})")
        res.ok = True

    def _refine(self, detections: list[Detection], pcd, valid, snap: SettingsData
                ) -> tuple[list[Detection], list[RefineResult]]:
        """형상 보정. 꺼져 있거나 실패하면 (원본 검출, [])를 돌려준다 -> ICP는 항상 진행."""
        cfg = snap.refine
        if not cfg.enabled:
            return detections, []
        try:
            bank = get_bank(self._cad_path(snap))
            intr = estimate_intrinsics_from_organized_pcd(pcd, valid)
            params = dataclasses.replace(
                RefineParams(), depth_tol_mm=float(cfg.depth_tol_mm), min_score=float(cfg.min_score),
                depth_phi=bool(cfg.depth_phi), resolve_overlap=bool(cfg.resolve_overlap))
            results = refine_masks([d.mask for d in detections], pcd, valid, intr, bank, params)
            out: list[Detection] = []
            for det, rr in zip(detections, results):
                if rr.ok:
                    ys, xs = np.nonzero(rr.mask)
                    bbox = (float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1)
                    out.append(dataclasses.replace(det, mask=rr.mask, bbox=bbox))
                else:
                    out.append(det)
            return out, list(results)
        except Exception as exc:  # noqa: BLE001
            self._log(f"형상 보정 실패 -> 원본 마스크 사용: {exc}")
            return detections, []

    @staticmethod
    def _drop_suspect(detections: list[Detection], refine_results: list[RefineResult], cfg
                      ) -> tuple[list[Detection], list[RefineResult], list[tuple[int, str, float]]]:
        """의심 마스크 제외. 형상 보정 결과가 없으면(꺼짐/실패) 판단 근거가 없으므로 아무것도 빼지 않는다."""
        if not cfg.hide_suspect or not refine_results or len(refine_results) != len(detections):
            return detections, refine_results, []
        keep_d: list[Detection] = []
        keep_r: list[RefineResult] = []
        excluded: list[tuple[int, str, float]] = []
        for i, (det, rr) in enumerate(zip(detections, refine_results)):
            why = rr.is_suspect(cfg.suspect_min_score, cfg.suspect_max_change,
                                cfg.suspect_min_det_in_tpl, cfg.suspect_drop_failed,
                                getattr(cfg, "suspect_max_occ_edge", 1.0))
            if why:
                excluded.append((i, why, float(det.confidence)))
            else:
                keep_d.append(det)
                keep_r.append(rr)
        return keep_d, keep_r, excluded

    # ------------------------------------------------------------------ 표시용 이미지
    def _save_frame_image(self, intensity: np.ndarray) -> str:
        self._frame_seq += 1
        path = self._frame_dir / f"frame_{self._frame_seq:06d}.png"
        img = intensity
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if not cv2.imwrite(str(path), img):
            raise PipelineError("INTERNAL_ERROR", f"프레임 이미지를 쓰지 못했습니다: {path}")
        self._frame_files.append(path)
        while len(self._frame_files) > 4:   # 표시용 최근 몇 장만 보관
            old = self._frame_files.popleft()
            try:
                os.unlink(old)
            except OSError:
                pass
        return str(path)
