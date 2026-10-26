"""ICP 파라미터 UI 명세 (탭 4 'ICP 검출 테스트'와 '설정' 탭이 같은 표를 쓴다).

각 항목: (ICPParams 필드명, 라벨, 종류, 최소, 최대, 스텝, 소수자리, 툴팁)
종류: "int" | "float" | "bool"

위젯 생성 / 값 읽기 / 기본값 복원이 전부 이 표 하나로 돌아가므로,
ICPParams에 필드를 추가하면 여기에 한 줄만 넣으면 두 탭에 같이 나타난다.
"""
from __future__ import annotations

# 탭 4와 설정 탭이 공통으로 노출하는 3D 정합 파라미터.
B_SECTIONS: list[tuple[str, list[tuple]]] = [
    ("점군 추출", [
        ("mask_erode_px", "마스크 침식 px", "int", 0, 10, 1, 0,
         "depth 경계(flying pixel) 노이즈 완충용 (0=끔). 침식 후 점이 너무 적으면 자동으로 원본 마스크 사용."),
    ]),
    ("이상치 제거", [
        ("outlier_nb_neighbors", "이웃 수", "int", 1, 200, 1, 0, "통계적 outlier 제거의 이웃 점 개수"),
        ("outlier_std_ratio", "σ 배수", "float", 0.1, 10.0, 0.1, 2, "평균 이웃 거리 + σ*이 값을 넘는 점 제거"),
    ]),
    ("CAD 준비 (로드 + 축 보정, 가시면 HPR)", [
        ("cad_hpr_ref_distance_m", "카메라~부품 거리(m)", "float", 0.05, 5.0, 0.01, 3,
         "CAD 가시면(보이는 면만 정합) 계산 기준 거리"),
        ("cad_axis_roll_deg", "CAD 축보정 roll", "float", -180.0, 180.0, 1.0, 1,
         "CAD가 실제 물체 방향과 안 맞으면 여기부터 조정 (CAD 바뀔 때마다 다시 맞춰야 함)"),
        ("cad_axis_pitch_deg", "CAD 축보정 pitch", "float", -180.0, 180.0, 1.0, 1, ""),
        ("cad_axis_yaw_deg", "CAD 축보정 yaw", "float", -180.0, 180.0, 1.0, 1, ""),
    ]),
    ("초기 pose (기존 방식: 중심 맞춤 + 고정 roll/pitch/yaw)", [
        ("init_roll_deg", "초기 roll", "float", -180.0, 180.0, 1.0, 1,
         "ICP 초기 자세 고정값. CAD 가시면 계산에도 같은 값이 쓰인다."),
        ("init_pitch_deg", "초기 pitch", "float", -180.0, 180.0, 1.0, 1, ""),
        ("init_yaw_deg", "초기 yaw", "float", -180.0, 180.0, 1.0, 1, ""),
    ]),
    ("게이트", [
        ("fitness_threshold", "fitness ≥", "float", 0.0, 1.0, 0.01, 2, "이 값 미만이면 정합 실패 처리"),
        ("xyz_max_m", "XYZ max (m)", "float", 0.1, 10.0, 0.1, 2, "결과 위치가 이 범위(±)를 벗어나면 실패"),
        ("roll_limit_deg", "roll ± deg", "float", 0.0, 180.0, 1.0, 1, "회전 구속 (180=사실상 제한 없음)"),
        ("pitch_limit_deg", "pitch ± deg", "float", 0.0, 180.0, 1.0, 1, ""),
        ("yaw_limit_deg", "yaw ± deg", "float", 0.0, 180.0, 1.0, 1, ""),
    ]),
]

FGR_SPECS: list[tuple] = [
    ("fgr_voxel_size_m", "voxel size (m)", "float", 0.0005, 0.05, 0.0005, 4, "부품 크기에 맞춰 조정 (작은 부품은 3mm 이하 권장)"),
    ("fgr_normal_radius_factor", "normal 반경 배수", "float", 0.5, 10.0, 0.5, 2, ""),
    ("fgr_fpfh_radius_factor", "FPFH 반경 배수", "float", 1.0, 20.0, 0.5, 2, ""),
    ("fgr_distance_threshold_factor", "대응거리 배수", "float", 0.5, 10.0, 0.5, 2, "이 값 * voxel size가 대응점 최대 거리"),
    ("fgr_refine_with_icp", "ICP로 정밀화", "bool", 0, 0, 0, 0, ""),
    ("fgr_refine_max_dist_m", "정밀화 max_dist (m)", "float", 0.0005, 0.02, 0.0005, 4, ""),
    ("fgr_use_rotation_prior", "회전 prior 검증", "bool", 0, 0, 0, 0, ""),
    ("fgr_max_rotation_deviation_deg", "최대 허용 편차 (deg)", "float", 0.0, 180.0, 5.0, 1, ""),
]

# 설정 탭에만 노출하는 추가 파라미터 (탭 4의 실험 UI에는 넣지 않음).
EXTRA_SECTIONS: list[tuple[str, list[tuple]]] = [
    ("다단계 ICP 공통 (open3d_multistage)", [
        ("normal_radius_factor", "normal 반경 배수", "float", 0.5, 10.0, 0.5, 2,
         "voxel이 있는 stage의 point-to-plane normal 반경 = voxel × 이 값"),
        ("normal_radius_final", "원본밀도 normal 반경(m)", "float", 0.001, 0.05, 0.0005, 4,
         "voxel=None(원본밀도) stage의 normal 반경"),
    ]),
    ("CAD 가시면(HPR) 상세", [
        ("cad_hpr_radius_factor", "HPR radius 배수", "float", 1.0, 1000.0, 10.0, 1,
         "Katz HPR radius = CAD 대각선 × 이 값. 클수록 더 많은 면을 '보인다'고 판정"),
    ]),
    ("점군 업샘플링", [
        ("pc_upsample_factor", "업샘플링 배수", "int", 1, 8, 1, 0,
         "1=끔. 실측 정보를 늘리는 게 아니라 기존 depth 사이를 보간해 ICP 대응점 밀도만 높인다."),
    ]),
]

# probabilistic: 다중 프레임 촬영(num_frames>1)의 픽셀별 표준편차에서 Monte Carlo 재샘플링
# (업샘플링 배수 = 점당 샘플 수). 표준편차가 없으면 자동으로 일반 추출로 폴백한다.
PC_UPSAMPLE_METHODS = ["linear", "cubic", "probabilistic"]