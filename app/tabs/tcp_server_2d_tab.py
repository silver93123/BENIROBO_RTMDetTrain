"""탭 5-2: 2D 기반 TCP 서버 (로봇 요청 대기형, ICP 없음).

5번 탭과 같은 서버인데, 요청이 오면 촬영 -> 2D 검출 -> 형상 보정(CAD 템플릿)까지만 하고
ICP 정합 대신 템플릿에서 바로 픽 포인트/회전을 구해 응답한다 (계산: app/core/pose_from_template.py).

    로봇  --"PICK"-->  [이 탭]  촬영 -> 검출 -> 형상 보정 -> 2D 기반 pose  --"OK,1,x,y,z,roll,pitch,yaw"-->  로봇

5번 탭과 다른 점
    * 포트: 설정의 '2D 기반 서버 포트'(기본 5001). 5번 서버(기본 5000)와 동시에 켤 수 있다.
    * 형상 보정이 켜져 있어야 한다 (꺼져 있으면 NG,CONFIG_ERROR). 보정이 성공한 검출만 후보가 된다.
    * roll(축 둘레 회전)은 영상으로 알 수 없어 CAD 원래 상태로 고정이다. 축 방향(pitch/yaw 성분)만 의미가 있다.
    * 'fitness' 정렬은 형상 보정 score 기준이다.
    * 나머지 값(모델/CAD, 카메라, ROI, 명령어, 응답 형식, 깊이 보정)은 5번 탭과 같이 '12. 설정' 탭에서 바꾼다.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QColor

from app.core.camera_intrinsics import project_points
from app.core.pick_pipeline import PickRunResult
from app.tabs.tcp_server_tab import TcpServerTab

DOT_2D = (0, 200, 220)   # 2D 기반 픽 포인트 원 색


class TcpServer2DTab(TcpServerTab):
    LOG_PREFIX = "2D 기반 TCP 서버 탭"
    MODE = "2d"
    USE_PORT_2D = True
    RUN_DESC = "촬영 -> 검출 -> 형상 보정 -> 2D 기반 pose"

    # ------------------------------------------------------------------
    # 결과 표시
    # ------------------------------------------------------------------
    def _pick_overlays(self, res: PickRunResult) -> list[dict]:
        out: list[dict] = []
        if res.intrinsics is None:
            return out
        length = float(np.clip(res.cad_extent_mm * 0.6, 15.0, 120.0))
        for i, p in enumerate(res.pose2d_results):
            if not p.ok:
                continue
            origin = np.asarray(p.pick_point_mm, dtype=np.float64)
            rot = p.T[:3, :3]
            pts = np.vstack([origin] + [origin + rot[:, k] * length for k in range(3)])
            if np.any(pts[:, 2] <= 1.0):
                continue
            uv = project_points(pts, res.intrinsics)
            e = p.pose["euler_deg"]
            label = (f"obj{i} (roll 고정)\n"
                     f"X{origin[0]:+.1f} Y{origin[1]:+.1f} Z{origin[2]:+.1f} mm\n"
                     f"R{e['roll_deg']:+.1f} P{e['pitch_deg']:+.1f} Y{e['yaw_deg']:+.1f} deg")
            out.append({"origin": tuple(uv[0]), "axes": [tuple(q) for q in uv[1:]], "label": label,
                        "dot": QColor(*DOT_2D)})
        return out

    def _show_result(self, res: PickRunResult) -> None:
        if res.image_path:
            self.image_viewer.load_image(res.image_path)
            self.image_viewer.set_detections(res.detections)
            self.image_viewer.set_pick_overlays(self._pick_overlays(res))
        ok_color = "#2a8a2a"
        self.run_label.setText(
            f"{res.started_at.replace('T', ' ')}   {res.summary()}\n"
            + "   ".join(f"{k} {v:.0f}ms" for k, v in res.timings_ms.items()))
        self.run_label.setStyleSheet(f"color: {ok_color};" if res.ok else "color: #c0392b;")

        self._clear_cards()
        max_picks = max(1, self._store.snapshot().tcp.max_picks)
        for rank, c in enumerate(res.candidates):
            sent = "  [응답 포함]" if rank < max_picks else ""
            p = res.pose2d_results[c.instance_id]
            self._add_card(
                f"#{rank + 1}  obj{c.instance_id}  {c.label} (conf {c.confidence:.2f}){sent}",
                [(f"pick  X{c.xyz_mm[0]:+.1f}  Y{c.xyz_mm[1]:+.1f}  Z{c.xyz_mm[2]:+.1f} mm", ""),
                 (f"R{c.rpy_deg[0]:+.1f}  P{c.rpy_deg[1]:+.1f}  Y{c.rpy_deg[2]:+.1f} deg  (roll 고정)", ""),
                 (f"보정 score {c.fitness:.2f}   φ{p.phi_deg:+.0f}° θ{p.theta_deg:.0f}°   "
                  f"Z 표면 {p.z_surface_mm:.1f} → 중심 {p.z_center_mm:.1f}", ok_color)],
                border="#9cc99c" if rank < max_picks else "#ddd")
        ok_ids = {c.instance_id for c in res.candidates}
        for i, p in enumerate(res.pose2d_results):
            if i in ok_ids:
                continue
            conf = res.detections[i].confidence if i < len(res.detections) else 0.0
            self._add_card(f"obj{i}  (conf {conf:.2f})  ✗ 계산 불가", [(p.error or "실패", "#c0392b")],
                           border="#e0b4ae")
        for orig_idx, why, conf in res.suspect_excluded:
            self._add_card(f"obj{orig_idx}  (conf {conf:.2f})  의심 마스크로 제외", [(why, "#a0702a")],
                           border="#e3cfa0")
        if not res.ok and not res.pose2d_results:
            self._add_card(f"[{res.error_code}]", [(res.error_message, "#c0392b")], border="#e0b4ae")

    # ------------------------------------------------------------------
    # 설정 요약
    # ------------------------------------------------------------------
    def _refresh_config_summary(self) -> None:
        s = self._store.snapshot()
        ckpt = s.detection.checkpoint_path.replace("\\", "/").split("/")[-1] or "(미지정)"
        cad = s.cad.cad_path.replace("\\", "/").split("/")[-1] or "(미지정)"
        self.config_label.setText(
            f"2D 기반 (ICP 없음)  |  포트 {s.tcp.port_2d}  |  카메라 {s.camera.camera_type} ×{s.camera.num_frames}프레임  |  "
            f"모델 {ckpt}  |  CAD {cad}  |  형상 보정 {'ON' if s.refine.enabled else 'OFF (켜야 동작)'}  |  "
            f"깊이 보정 {'ON' if s.refine.depth_correct else 'OFF'}  |  "
            f"요청 '{s.tcp.trigger_cmd}' -> 응답 최대 {s.tcp.max_picks}개  |  값 변경은 '12. 설정' 탭 (저장 후 다음 요청부터 적용)")
