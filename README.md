# 3D Vision Bin-Picking Toolkit (오프라인 모드, 프로젝트 자체완결형)

이 폴더 하나로 전부 완결됩니다. **다른 프로젝트 폴더나 다른 PC의 절대경로를
전혀 참조하지 않습니다** — GUI, 학습 스크립트, config, 데이터, 결과물이
모두 이 폴더 기준 상대경로로 연결되어 있어서, 폴더째로 옮기거나 다른 PC에
복사해도 그대로 동작합니다.

## 폴더 구조

```
binpicking_project/                 <- 이 폴더 전체가 "프로젝트 루트"
├── main.py                         # GUI 실행 진입점
├── requirements.txt
├── app/                             # GUI 코드
│   ├── main_window.py
│   ├── tabs/
│   │   ├── data_session_tab.py     # 1. 세션 폴더 스캔/상태 확인
│   │   ├── training_tab.py         # 2. 학습 스크립트 실행/모니터링
│   │   └── inference_test_tab.py   # 3. 2D 검출 테스트
│   └── core/
│       ├── paths.py                # 프로젝트 루트 자동 계산 (이 폴더 위치 기준)
│       ├── session_manager.py
│       ├── config_patcher.py
│       ├── train_runner.py
│       └── detector.py
├── scripts/
│   └── 2_Train_rtmdet_model.py     # 실제 학습 스크립트 (경로를 프로젝트 루트 기준 상대경로로 패치함)
├── configs/
│   └── rtmdet-ins_bracket.py       # mmdet 학습 config (경로를 프로젝트 루트 기준 상대경로로 패치함)
├── data/dataset/                   # 세션 폴더를 여기에 채워 넣으세요
│   └── <YYYYMMDD_HHMMSS>/
│       ├── intensity/
│       ├── pointcloud_organized/
│       └── annotations/instances_Train.json
├── models/                         # COCO pretrained 체크포인트를 여기에 받아두세요
└── work_dirs/                      # 학습 결과(체크포인트, 로그)가 여기 쌓입니다
```

## 실행 방법

```bash
cd binpicking_project
pip install -r requirements.txt
python main.py
```

앱을 실행하면 `data/dataset`, `configs/rtmdet-ins_bracket.py`, 프로젝트 루트 경로가
**자동으로 채워집니다** (별도로 폴더를 찾아다닐 필요 없음).

## 원본 파일에서 바뀐 부분 (다른 프로젝트 경로 의존성 제거)

업로드해주신 두 파일에 특정 사용자/PC의 절대경로가 하드코딩되어 있어서, 이 프로젝트
폴더만 옮기면 깨지는 문제가 있었습니다. 아래 3곳을 프로젝트 루트 기준 상대경로로 고쳤습니다.

1. **`scripts/2_Train_rtmdet_model.py` - `DATASET_BASE`**
   - 이전: `/home/silver/binpicking_vision/FINE_RTMDet/data/dataset` (절대경로 하드코딩)
   - 이후: `ROOT / "data" / "dataset"` (스크립트 자신의 위치 기준 자동 계산)

2. **`configs/rtmdet-ins_bracket.py` - `work_dir`, `load_from`**
   - 이전: `/home/silver/binpicking_vision/FINE_RTMDet/work_dirs/...`, `.../models/...pth`
   - 이후: `work_dirs/rtmdet-ins_bracket_v1`, `models/....pth` (상대경로 —
     스크립트가 실행 시 `os.chdir(ROOT)`를 하기 때문에 프로젝트 루트 기준으로 정확히 풀립니다)

3. **`configs/rtmdet-ins_bracket.py` - `_base_`**
   - 이전: `/home/silver/miniconda3/envs/vision_env/.../rtmdet-ins_tiny_8xb32-300e_coco.py`
     (특정 사용자명 + conda 환경명이 박혀 있어서 다른 PC/환경에서는 무조건 깨짐)
   - 이후: `import mmdet`으로 설치 위치를 코드로 찾아서 조합 → 어떤 conda 환경/사용자든 동작

GUI(`app/core/paths.py`)도 같은 원칙으로, 자기 자신의 파일 위치를 기준으로
프로젝트 루트를 계산해서 세션 루트/config/작업 디렉토리 기본값을 자동으로 채웁니다.

## 학습 탭 사용법

실제 스크립트 인자 그대로 갑니다.
```bash
python scripts/2_Train_rtmdet_model.py --dataset 20260521_114500 [--config ...] [--epochs ...]
```

1. **데이터 세션 탭**에서 세션을 선택하면 폴더명이 학습 탭 `--dataset`에 자동으로 채워집니다.
2. **config 파일**: 기본값으로 `configs/rtmdet-ins_bracket.py`가 자동 채워집니다. 하이퍼파라미터는
   이 파일 안에서 관리하고, "에디터로 열기"로 직접 수정 가능합니다.
3. **최초 학습 시작 체크포인트 override (선택)**: `work_dirs/rtmdet-ins_bracket_v1/`에
   `best_*.pth`가 하나도 없을 때만 적용됩니다 (있으면 스크립트가 자동으로 그걸 우선 사용).
   "현재 시작점 확인" 버튼으로 미리 확인할 수 있습니다.
4. **작업 디렉토리**: 기본값으로 프로젝트 루트가 자동 채워집니다 (스크립트가 상대경로로
   실행되기 때문에 필요).
5. **학습 시작**을 누르면 위 커맨드가 그대로 subprocess로 실행되고, stdout이 로그창에
   스트리밍되며, mmengine 로그 포맷에서 epoch/loss/mAP를 뽑아 카드/그래프에 반영합니다.

## 검증 상태

- 프로젝트 전체를 이 구조로 만들어서, **세션 스캔 → `--dataset` 자동 연동 → config
  기본값 자동 채움 → work_dir 상대경로 기준 best 체크포인트 탐색**까지 실제로
  실행해서 확인했습니다.
- **실제 `2_Train_rtmdet_model.py`를 mmdet 설치 환경에서 돌려보지는 못했습니다**
  (이 개발 환경에 torch/mmdet이 없음). 문법 검증(`py_compile`)은 통과했습니다.
  실제 GPU 환경에서 한 번 실행해보시고, 안 맞는 부분 있으면 알려주세요.
- `app/core/detector.py`(3번 탭, 2D 검출)는 여전히 더미 결과입니다. 실제 추론
  스크립트를 보여주시면 그에 맞춰 교체하겠습니다.

## 남은 준비물 (사용자가 채워야 하는 것)

- `models/`에 COCO pretrained 체크포인트 (`rtmdet-ins_tiny_8xb32-300e_coco_...pth`) 다운로드
- `data/dataset/<세션폴더>/`에 `intensity/`, `annotations/instances_Train.json` 등 실제 데이터
- Python 환경에 `torch`, `mmengine`, `mmcv`, `mmdetection` 설치 (이 GUI 앱 자체의
  `requirements.txt`에는 포함하지 않았습니다 — GUI는 이들 없이도 실행되고, 학습을
  실제로 돌릴 때만 필요합니다)

## 3번 탭 (오프라인 검출 테스트) 업데이트

`3_Detect_and_PickPoint.py`의 [A] Detection 단계만 추출해서 `detector.py`를
다시 만들었습니다. PCD 분리 / ICP / 픽포인트 계산은 포함하지 않습니다.

- **더 이상 더미(랜덤) 결과를 반환하지 않습니다.** 이전 버전은 파일명을 해시해서
  무작위 박스를 그렸는데, 이번에 실제 스크립트를 반영하면서 제거했습니다.
- 대신 `<프로젝트 루트>/src/detection.py`의 `RTMDetInferencer` 클래스를 그대로 가져다
  씁니다. **이 파일은 업로드되지 않아서 이 zip에는 포함되어 있지 않습니다.**
  프로젝트 루트에 `src/detection.py`가 있어야 실제 추론이 됩니다. 없으면 앱이
  명확한 에러 메시지를 띄웁니다 (조용히 가짜 결과를 만들지 않음).
- 체크포인트를 고르면 같은 폴더에 `.py` config가 하나뿐일 때 자동으로 제안합니다
  (실제 스크립트도 `work_dirs/.../rtmdet-ins_bracket_v1/rtmdet-ins_bracket.py`처럼
  체크포인트와 같은 폴더의 config 사본을 씁니다).
- conf threshold 기본값을 실제 스크립트의 `SCORE_THRESHOLD = 0.3`에 맞췄습니다.
- `RTMDetInferencer.infer()`가 반환하는 각 결과의 `mask`(2D bool 배열)도
  `Detection.mask`에 그대로 보관해둡니다 (현재 UI에서 시각화까지는 안 하지만,
  나중에 마스크 오버레이를 붙이고 싶으면 여기서 바로 쓸 수 있습니다).

**검증 상태**: 가짜 `RTMDetInferencer`를 만들어서 실제 호출 흐름(이미지 로드 →
grayscale→BGR 변환 → `infer()` 호출 → score threshold 필터링)까지 end-to-end로
확인했습니다. 실제 `src/detection.py`로는 아직 테스트하지 못했으니, 그 파일도
있으시면 보여주세요 — `RTMDetInferencer`의 실제 생성자 인자/반환값이 다르면
바로 맞춰드리겠습니다.
