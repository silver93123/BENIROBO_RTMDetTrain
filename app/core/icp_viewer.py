"""ICP 결과 3D 뷰어 (별도 프로세스로 실행됨).

open3d의 시각화 창(GLFW)은 자체 이벤트 루프를 돌기 때문에 PyQt6 메인
이벤트 루프와 한 프로세스 안에서 같이 쓰면 불안정하다. 그래서 ICP 탭에서는
결과를 PLY로 저장해두고, 이 스크립트를 QProcess로 별도 실행해서 보여준다.

사용:
    python -m app.core.icp_viewer <ply_path> [--title TITLE]
"""
from __future__ import annotations

import argparse
import sys

import open3d as o3d


def main() -> int:
    parser = argparse.ArgumentParser(description="ICP 결과 3D 뷰어")
    parser.add_argument("ply_path", help="결과 포인트클라우드 (.ply)")
    parser.add_argument("--title", default="ICP 정합 결과")
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.ply_path)
    if len(pcd.points) == 0:
        print(f"ERROR: 빈 포인트클라우드입니다: {args.ply_path}", flush=True)
        return 1

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    o3d.visualization.draw_geometries(
        [pcd, axis], window_name=args.title,
        width=1000, height=750,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())