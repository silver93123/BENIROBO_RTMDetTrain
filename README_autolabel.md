# 2D 오토라벨링 (모델 → 검수 → 재학습 루프)

## 적용 (레포 루트 기준)
1. 이 폴더의 `src/autolabel/`, `app/tabs/autolabel_tab.py`, `scripts/test_autolabel_pipeline.py` 를 같은 경로에 복사
2. `app/main_window.py` 수정 (3줄)
   ```python
   from app.tabs.autolabel_tab import AutoLabelTab          # 다른 탭 import 옆
   self.autolabel_tab = AutoLabelTab()                       # 다른 탭 생성 옆
   self._add_leaf("11. 2D 오토라벨링", self.autolabel_tab)    # 번호/위치는 기존 트리에 맞게
   self.autolabel_tab.log_message.connect(<다른 탭과 같은 로그 연결 방식>)
   ```
3. `.gitignore` 에 추가: `__pycache__/`, `*.pyc`
4. labelme 설치 (PySide6 기반이라 앱 env 와 **별도 env 권장**)
   ```bash
   python -m venv ~/venv_labelme && ~/venv_labelme/bin/pip install labelme==7.8.1
   ```
   `app/tabs/autolabel_tab.py` 의 `LABELME_EXECUTABLE` 을 `~/venv_labelme/bin/labelme` 절대경로로 지정
5. 동작 확인(GPU 불필요): `python scripts/test_autolabel_pipeline.py` → `ALL OK`

## 한 라운드
탭에서 1) 추론+대기열 → 2) labelme 검수 → 3) 병합+COCO 내보내기.
CLI: `python -m src.autolabel {predict,queue,edit,merge,export,stats} -h`
산출물: `data/autolabel/{cache,rounds,dataset}/<라운드>/` (data/ 는 이미 gitignore)

## 검수 규칙 (중요)
- 우선순위 높은 순으로 urgent → normal, 확신 높은 것은 auto(검수 생략)
- **수정했거나 labelme 의 `reviewed` 를 체크한 이미지만** 학습셋에 들어갑니다. 손대지 않은 초안은 기본 제외 (모델 오류의 자기강화 방지)
- 객체가 정말 없는 이미지는 `reviewed` 만 체크
- val/test 는 사람이 처음부터 라벨링한 것만 쓰세요

## 알려진 한계
- 구멍(hole)은 COCO 폴리곤 export 에서 빠짐
- 파일이 1바이트라도 달라지면 "수정됨"으로 간주 (labelme 가 내용 변화 없이 재저장해도 포함됨)
- 세션 폴더 구조와의 자동 연동은 아직 없음 (이미지 폴더를 직접 지정)
