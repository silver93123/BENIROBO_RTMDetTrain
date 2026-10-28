"""탭 4-1: ICP 검출 테스트 (2D 기반).

4번 탭(ICP 검출 테스트)과 같은 화면에, ICP 없이 "형상 보정이 고른 CAD 템플릿"만으로 구한 픽 포인트/회전을
추가로 보여주고 ICP 결과와 비교한다.

    위치  템플릿 중심 픽셀 + 물체 평면 depth -> 카메라 좌표 (선택: 표면 -> CAD 중심 깊이 보정)
    회전  템플릿의 phi / theta 로 만든 회전. roll(축 둘레 회전)은 영상으로 알 수 없어 CAD 원래 상태로 고정.

계산 로직은 app/core/pose_from_template.py, 화면/ICP 흐름은 4번 탭(ICPDetectionTestTab)을 그대로 상속한다.
형상 보정이 켜져 있고 그 인스턴스의 보정이 성공해야 2D 기반 값이 나온다 (보정 포기 -> 계산 불가).
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QCheckBox, QFrame, QLabel, QMessageBox, QPushButton, QVBoxLayout

from app.core.app_settings import get_settings
from app.core.point_pair_calib import SavedTransformOutput
from app.widgets.elided_label import ElidedLabel

from app.core import pose_from_template
from app.core.camera_intrinsics import estimate_intrinsics_from_organized_pcd, project_points
from app.core.detect_test_common import load_settings_paths
from app.core.icp_runner import ICPResult
from app.core.pose_from_template import Pose2D
from app.tabs.icp_detection_test_tab import ICPDetectionTestTab, _guarded
from src.detection.mask_refine import get_bank

DOT_2D = QColor(0, 200, 220)  # 2D 기반 픽 포인트 원 색 (ICP는 노란색)


class ICPDetection2DTab(ICPDetectionTestTab):
    LOG_PREFIX = "ICP 2D 기반 탭"

    _pick2d: list = []          # 인스턴스별 Pose2D (검출 index와 일치)
    _pick2d_pcd: list = []      # 같은 검출을 점군 직접 읽기 방식으로 계산한 Pose2D (기존 방식과의 비교용)
    _bank = None
    _intr = None
    _axis_deg = (0.0, 0.0, 0.0)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pick2d = []
        self._pick2d_pcd = []
        self.btn_run_all.setText("검출 + 2D 계산 + ICP 비교")
        self.btn_run_all.setToolTip("2D 검출 + 형상 보정 -> 2D 기반 픽 포인트 -> ICP 정합 -> 결과 비교")
        self.btn_run_2d = QPushButton("검출 + 2D 기반 계산 (ICP 없이)")
        self.btn_run_2d.clicked.connect(lambda _=False: self._on_run_2d_only())
        self.btn_calc_2d = QPushButton("2D 기반 계산")
        self.btn_calc_2d.setToolTip("이미 검출된 결과(형상 보정 포함)로 2D 기반 픽 포인트만 다시 계산")
        self.btn_calc_2d.clicked.connect(lambda _=False: self._on_calc_2d())
        self.chk_depth_correct = QCheckBox("표면→중심 깊이 보정")
        self.chk_depth_correct.setChecked(True)
        self.chk_depth_correct.setToolTip(
            "보이는 물체 표면 depth에 (축 기울기에 따른) 반두께/반길이를 더해 CAD 중심 depth로 만든다. 끄면 표면 depth 그대로.")
        self.chk_depth_correct.toggled.connect(lambda _=False: self._on_calc_2d(quiet=True))
        for i, w in enumerate((self.btn_run_2d, self.btn_calc_2d, self.chk_depth_correct)):
            self._top_row.insertWidget(1 + i, w)

        # 표시/로그 좌표계: 13번 탭에서 저장한 변환으로 카메라 -> 로봇 (5-2 서버의 같은 체크박스와 동일한 변환/기준)
        self._output = SavedTransformOutput()
        self.chk_robot = QCheckBox("로봇 좌표로 변환")
        self.chk_robot.setToolTip(
            "켜면 pick X,Y,Z 표시/로그를 13번 탭에서 [저장]한 좌표 변환(카메라→로봇)으로 바꿉니다 (ICP/2D 기반 모두).\n"
            "회전(R/P/Y)과 '기존 vs pcd 직접 읽기' 비교의 Δ는 카메라 기준 그대로입니다.\n"
            "카메라가 로봇 손에 달려 있으면 보정 때와 같은 촬영 자세에서만 맞습니다.")
        self.chk_robot.toggled.connect(self._on_robot_toggled)
        # 상태 문구는 폭에 맞춰 잘라 보인다 (길어지면 상단 줄/창 폭이 같이 커지던 문제 방지; 전체 문구는 툴팁)
        self.lbl_robot = ElidedLabel("")
        self.lbl_robot.setStyleSheet("color: #666; font-size: 11px;")
        self.lbl_robot.setMaximumWidth(260)
        self._top_row.insertWidget(4, self.chk_robot)
        self._top_row.insertWidget(5, self.lbl_robot)

        # 검출/보정 옵션을 5-2 서버가 쓰는 '12. 설정' 값과 맞춘다 (저장할 때마다 다시 맞춤)
        self._sync_from_settings()
        get_settings().changed.connect(self._on_settings_changed)

    # ------------------------------------------------------------------
    # 5-2와 같은 검출/보정 옵션 (12. 설정 탭 값)
    # ------------------------------------------------------------------
    def _sync_from_settings(self) -> None:
        """신뢰도 / 형상 보정 / 의심 마스크 제외 / 깊이 보정 위젯을 저장된 설정값으로 채운다 (5-2가 쓰는 값과 동일)."""
        snap = get_settings().snapshot()
        r = snap.refine
        pairs = ((self.chk_refine, r.enabled), (self.chk_depth_phi, r.depth_phi), (self.chk_overlap, r.resolve_overlap),
                 (self.chk_suspect, r.hide_suspect), (self.chk_depth_correct, r.depth_correct))
        widgets = [w for w, _v in pairs] + [self.spin_depth_tol, self.spin_min_score, self.thresh_slider]
        for w in widgets:
            w.blockSignals(True)
        try:
            for w, v in pairs:
                w.setChecked(bool(v))
            self.spin_depth_tol.setValue(float(r.depth_tol_mm))
            self.spin_min_score.setValue(float(r.min_score))
            self.thresh_slider.setValue(int(round(snap.detection.conf_threshold * 100)))
        finally:
            for w in widgets:
                w.blockSignals(False)
        self.thresh_label.setText(f"{self.thresh_slider.value() / 100:.2f}")

    def _on_settings_changed(self) -> None:
        self._sync_from_settings()
        self._refresh_robot_label()
        if self._raw_detections:
            self._on_refine_option_changed(force=True)   # 이미 검출한 결과도 새 보정/제외 기준으로 다시 만든다
        else:
            self._render_result_panel(self._icp_results)

    # ------------------------------------------------------------------
    # 좌표계 (카메라 / 로봇)
    # ------------------------------------------------------------------
    def _robot_active(self) -> bool:
        return self.chk_robot.isChecked() and not self._output.check()[0]

    def _out_xyz(self, xyz) -> tuple[float, float, float]:
        if self._robot_active():
            v = self._output.convert(xyz)
            if v is not None:
                return v
        return super()._out_xyz(xyz)

    def _frame_tag(self) -> str:
        return " (로봇 좌표)" if self._robot_active() else ""

    def _refresh_robot_label(self) -> None:
        if not self.chk_robot.isChecked():
            self.lbl_robot.setText("")   # 상단 줄이 좁아 꺼져 있을 땐 문구를 두지 않는다 (카메라 좌표가 기본)
            return
        code, text = self._output.check()
        self.lbl_robot.setText(("⚠ 카메라 좌표로 표시 중: " + text) if code else ("로봇 좌표로 표시 · " + text))
        self.lbl_robot.setStyleSheet(f"color: {'#c0392b' if code else '#2a8a2a'}; font-size: 11px;")

    @_guarded
    def _on_robot_toggled(self, on: bool) -> None:
        if on:
            code, text = self._output.check()
            if code:
                QMessageBox.warning(self, "로봇 좌표로 변환할 수 없음",
                                    f"{text}\n\n13번 탭에서 점 짝을 모아 [변환 계산] 후 [저장]하세요.")
                self.chk_robot.blockSignals(True)
                self.chk_robot.setChecked(False)
                self.chk_robot.blockSignals(False)
                on = False
        self._output.set_enabled(on)
        self._refresh_robot_label()
        self._log("표시 좌표계: " + ("로봇 (저장된 변환 적용)" if on else "카메라"))
        self._render_result_panel(self._icp_results)
        self._show_pick_overlays(self._icp_results)

    # ------------------------------------------------------------------
    # 상태 초기화 (검출이 바뀌면 2D 결과도 무효)
    # ------------------------------------------------------------------
    def _clear_detection_state(self) -> None:
        self._pick2d = []
        self._pick2d_pcd = []
        super()._clear_detection_state()

    def _apply_refine_and_show(self) -> None:
        self._pick2d = []
        self._pick2d_pcd = []
        super()._apply_refine_and_show()

    # ------------------------------------------------------------------
    # 실행
    # ------------------------------------------------------------------
    @_guarded
    def _on_run_all(self) -> None:
        if not self._run_detection():
            return
        self._on_calc_2d(quiet=True)
        self._run_icp()  # 결과 패널 / 오버레이를 다시 그리며 2D 카드도 같이 그려진다

    @_guarded
    def _on_run_2d_only(self) -> None:
        if self._run_detection():
            self._on_calc_2d()

    @_guarded
    def _on_calc_2d(self, quiet: bool = False) -> None:
        if not self._detections:
            if not quiet:
                QMessageBox.warning(self, "알림", self._no_detection_message())
            return
        if not self._refine_applied or not self._refine_results:
            if not quiet:
                QMessageBox.warning(self, "알림", "2D 기반 계산은 형상 보정이 켜져 있어야 합니다 (CAD 템플릿이 필요).")
            return
        paths = load_settings_paths()   # CAD는 '12. 설정' 탭 저장값
        problem = paths.problem(need_detector=False, need_cad=True)
        if problem or self._pcd is None or self._valid is None:
            if not quiet:
                QMessageBox.warning(self, "알림", problem or "이 프레임에 점군/유효마스크가 없습니다.")
            return

        cad_path = str(paths.cad)
        params = self._collect_params()
        bank = get_bank(cad_path)
        intr = estimate_intrinsics_from_organized_pcd(self._pcd, self._valid)

        # ICP와 같은 정의의 CAD 중심(축보정된 CAD 점군 중심)을 쓰기 위해 CAD를 준비한다.
        cad_center_mm = None
        try:
            self._ensure_cad_loaded(cad_path, params)
            cad_center_mm = np.asarray(self._cad_pcd.get_center(), dtype=np.float64) * 1000.0
        except Exception as exc:  # noqa: BLE001
            self._log(f"CAD 중심을 ICP 기준으로 못 구해 정점 평균을 씁니다: {exc}")

        self._bank, self._intr, self._axis_deg = bank, intr, params.cad_axis_correction_deg
        depth_correct = self.chk_depth_correct.isChecked()
        self._pick2d = []
        for i, rr in enumerate(self._refine_results):
            p = pose_from_template.pose_from_refine(
                rr, bank, intr, cad_axis_deg=self._axis_deg, cad_center_mm=cad_center_mm,
                depth_correct=depth_correct)
            self._pick2d.append(p)
            if p.ok:
                e = p.pose["euler_deg"]
                px = self._out_xyz(p.pick_point_mm)
                self._log(f"  obj{i} 2D 기반 pick{self._frame_tag()}=({px[0]:+.1f}, {px[1]:+.1f}, "
                          f"{px[2]:+.1f}) mm  R{e['roll_deg']:+.1f} P{e['pitch_deg']:+.1f} "
                          f"Y{e['yaw_deg']:+.1f} deg  (φ{p.phi_deg:+.0f} θ{p.theta_deg:.0f}, "
                          f"Z 표면 {p.z_surface_mm:.1f} → 중심 {p.z_center_mm:.1f}, roll 고정)")
            else:
                self._log(f"  obj{i} 2D 기반 불가: {p.error}")
        self._log(f"2D 기반 계산 완료: 성공 {sum(p.ok for p in self._pick2d)}/{len(self._pick2d)}"
                  f" (깊이 보정 {'ON' if depth_correct else 'OFF'})")

        # 같은 검출을 '점군 직접 읽기' 방식으로도 계산해 기존(평면 + 핀홀 역투영) 방식과의 차이를 mm로 본다.
        # 회전/깊이 보정은 같고 위치(X,Y,Z)를 얻는 방법만 다르다 (app/core/pose_from_template.py).
        self._pick2d_pcd = []
        diffs: list[dict] = []
        for i, rr in enumerate(self._refine_results):
            q = pose_from_template.pose_from_refine_pcd(
                rr, bank, intr, self._pcd, self._valid, cad_axis_deg=self._axis_deg,
                cad_center_mm=cad_center_mm, depth_correct=depth_correct)
            self._pick2d_pcd.append(q)
            if not q.ok:
                self._log(f"  obj{i} pcd 직접 읽기 불가: {q.error}")
                continue
            c = pose_from_template.compare_methods(self._pick2d[i], q, rr.center_uv, intr, self._pcd.shape)
            if c is None:   # 기존 방식이 실패한 인스턴스는 비교할 대상이 없다
                continue
            diffs.append(c)
            self._log(f"  obj{i} pcd 직접 읽기 [{q.source}] 기존 대비 Δ(X,Y,Z)=({c['dx']:+.2f}, {c['dy']:+.2f}, "
                      f"{c['dz']:+.2f}) mm  |Δ| 3D {c['d3_mm']:.2f} / XY {c['dxy_mm']:.2f} mm  "
                      f"(화면 중심에서 {c['r_px']:.0f}px, {c['r_pct']:.0f}%)")
        if diffs:
            d3 = [c["d3_mm"] for c in diffs]
            self._log(f"기존 vs pcd 직접 읽기: {len(diffs)}건, 3D 차이 평균 {sum(d3) / len(d3):.2f} / 최대 {max(d3):.2f} mm")

        # 화면 갱신 (이미 ICP 결과가 있으면 같이 다시 그린다)
        self._render_result_panel(self._icp_results)
        self._show_pick_overlays(self._icp_results)
        self.btn_open_viewer.setEnabled(len(self._icp_results) > 0)

    # ------------------------------------------------------------------
    # 화면: 오버레이 / 카드
    # ------------------------------------------------------------------
    def _extra_pick_overlays(self) -> list[dict]:
        out: list[dict] = []
        if not self._pick2d or self._intr is None:
            return out
        extent_mm = 60.0
        if self._cad_pcd is not None:
            extent_mm = float(self._cad_pcd.get_axis_aligned_bounding_box().get_max_extent()) * 1000.0
        length = float(np.clip(extent_mm * 0.6, 15.0, 120.0))
        for i, p in enumerate(self._pick2d):
            if not p.ok:
                continue
            origin = np.asarray(p.pick_point_mm, dtype=np.float64)
            rot = p.T[:3, :3]
            pts = np.vstack([origin] + [origin + rot[:, k] * length for k in range(3)])
            if np.any(pts[:, 2] <= 1.0):
                continue
            uv = project_points(pts, self._intr)
            e = p.pose["euler_deg"]
            ox = self._out_xyz(origin)
            label = (f"2D obj{i} (roll 고정)\n"
                     f"X{ox[0]:+.1f} Y{ox[1]:+.1f} Z{ox[2]:+.1f} mm{self._frame_tag()}\n"
                     f"R{e['roll_deg']:+.1f} P{e['pitch_deg']:+.1f} Y{e['yaw_deg']:+.1f} deg")
            out.append({"origin": tuple(uv[0]), "axes": [tuple(q) for q in uv[1:]], "label": label, "dot": DOT_2D, "label_above": True})
        return out

    def _render_result_panel(self, results: list[ICPResult]) -> None:
        super()._render_result_panel(results)  # 패널을 비우고 ICP 카드를 그린다
        icp_by_id = {r.instance_id: r for r in results}
        for i, p in enumerate(self._pick2d):
            card = QFrame()
            card.setObjectName("resultCard")
            card.setFrameShape(QFrame.Shape.StyledPanel)
            card.setStyleSheet("QFrame#resultCard { border: 1px solid #7cc; border-radius: 6px; }")
            lay = QVBoxLayout(card)
            lay.setContentsMargins(8, 6, 8, 6)
            title = QLabel(f"obj{i} · 2D 기반 (roll 고정)")
            title.setStyleSheet("font-weight: 600; color: #168;")
            lay.addWidget(title)
            if not p.ok:
                err = QLabel(p.error)
                err.setStyleSheet("color: #c0392b;")
                err.setWordWrap(True)
                lay.addWidget(err)
            else:
                e = p.pose["euler_deg"]
                pos = self._out_xyz(p.pick_point_mm)
                lay.addWidget(QLabel(f"pick{self._frame_tag()} X{pos[0]:+.1f} Y{pos[1]:+.1f} Z{pos[2]:+.1f} mm"))
                lay.addWidget(QLabel(f"R{e['roll_deg']:+.1f} P{e['pitch_deg']:+.1f} Y{e['yaw_deg']:+.1f} deg"))
                info = QLabel(f"φ{p.phi_deg:+.0f}° θ{p.theta_deg:.0f}°  Z 표면 {p.z_surface_mm:.1f} → 중심 {p.z_center_mm:.1f}")
                info.setStyleSheet("color: #888; font-size: 10px;")
                info.setWordWrap(True)
                lay.addWidget(info)
                cmp_ = pose_from_template.compare_with_icp(p, icp_by_id.get(i), self._bank, self._axis_deg)
                if cmp_ is not None:
                    c = QLabel(f"ICP 대비: 위치 {cmp_['dpos_mm']:.1f}mm (Z {cmp_['dz_mm']:+.1f})\n"
                               f"축 방향 {cmp_['axis_deg']:.1f}° / 전체 회전 {cmp_['rot_deg']:.1f}° (roll 포함)")
                    c.setStyleSheet("color: #555; font-size: 10px;")
                    c.setWordWrap(True)
                    lay.addWidget(c)
                    if cmp_["axis_deg"] > 90:
                        flip = QLabel("⚠ 축 방향이 반대 (머리/꼬리)")
                        flip.setStyleSheet("color: #c0392b; font-size: 10px;")
                        lay.addWidget(flip)
                self._add_pcd_compare_label(lay, i, p)
            self.result_layout.insertWidget(self.result_layout.count() - 1, card)

    def _add_pcd_compare_label(self, lay, i: int, base) -> None:
        """2D 기반 카드에 '점군 직접 읽기' 방식의 pick 좌표와 기존 방식 대비 차이(mm)를 덧붙인다."""
        q = self._pick2d_pcd[i] if i < len(self._pick2d_pcd) else None
        if q is None or self._pcd is None or self._intr is None:
            return
        if not q.ok:
            text, color = f"pcd 직접 읽기 불가: {q.error}", "#c0392b"
        else:
            c = pose_from_template.compare_methods(
                base, q, self._refine_results[i].center_uv, self._intr, self._pcd.shape)
            pos = q.pick_point_mm   # 이 카드는 '기존 대비 Δ'와 함께 보므로 항상 카메라 좌표
            text = (f"pcd 직접 읽기 [{q.source}] (카메라 좌표)\n"
                    f"pick X{pos[0]:+.1f} Y{pos[1]:+.1f} Z{pos[2]:+.1f} mm\n"
                    f"기존 대비 ΔX{c['dx']:+.2f} ΔY{c['dy']:+.2f} ΔZ{c['dz']:+.2f} mm\n"
                    f"|Δ| 3D {c['d3_mm']:.2f} / XY {c['dxy_mm']:.2f} mm\n"
                    f"표면 Z 차이 {c['dz_surf_mm']:+.2f} mm · 화면 중심에서 {c['r_px']:.0f}px ({c['r_pct']:.0f}%)")
            color = "#555"
        label = QLabel(text)
        label.setStyleSheet(f"color: {color}; font-size: 10px;")
        label.setWordWrap(True)
        lay.addWidget(label)
