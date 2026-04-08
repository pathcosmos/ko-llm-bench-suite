# Interactive Wizard Design Spec

**Date:** 2026-04-08
**Status:** Approved
**Prereq:** Phase 1 리팩토링 + P0/P1 수정 완료

---

## 1. Purpose

사용자가 **코드를 읽지 않고도** 설치→설정→실행→결과 확인까지 인터랙티브 위저드를 통해 완료할 수 있도록 한다. 다중 머신 설정, 모델 선택, 트랙 선택, 에러 복구를 단계별 프롬프트로 안내한다.

### Success Criteria

- `python kobench.py setup` → 설치 완료까지 자동 안내
- `python kobench.py eval` → 모델/트랙 선택 → 실행 → 에러 복구 → 결과 표시
- `python kobench.py config` → 인터랙티브 YAML 생성
- `python kobench.py status` → 시스템 진단 한눈에
- 기존 `--config`, `--tracks`, `--models` 플래그 100% 호환 유지

---

## 2. Architecture

### 2.1 서브커맨드 구조

```
python kobench.py                 → 기존 동작 (하위호환)
python kobench.py setup           → 설치 위저드
python kobench.py eval            → 인터랙티브 평가
python kobench.py config          → YAML 설정 빌더
python kobench.py status          → 시스템 진단
python kobench.py --config X.yaml → 기존 배치 모드 (변경 없음)
```

### 2.2 파일 구조

```
kobench/wizard/
├── __init__.py          # wizard 패키지
├── cli.py               # 서브커맨드 디스패처 + argparse 확장
├── ui.py                # Rich UI 헬퍼 (공통 컴포넌트)
├── setup_wizard.py      # setup 서브커맨드
├── eval_wizard.py       # eval 서브커맨드
├── config_wizard.py     # config 서브커맨드
├── machine.py           # 머신 연결 테스트 + 설정
├── models.py            # 모델 발견/선택/다운로드
├── tracks.py            # 트랙/시나리오 선택
└── executor.py          # 실행 컨트롤러 (진행률 + 에러 복구)
```

### 2.3 의존성

- `rich>=13.7` (이미 설치됨, requirements.txt에 추가만)
- 신규 의존성 없음

---

## 3. 서브커맨드 상세

### 3.1 `setup` — 설치 위저드

```
Step 1: 시스템 확인
  - Python 버전 ✅/❌
  - pip 패키지 (부족하면 자동 설치 제안)
  - GPU 감지 (nvidia-smi)

Step 2: Ollama 설정
  - 로컬 Ollama 감지 → 없으면 설치 안내
  - "원격 Ollama도 사용하시겠습니까?" → URL 입력 + 연결 테스트

Step 3: Judge 모델
  - qwen2.5:7b-instruct 확인 → 없으면 pull 제안
  - exaone3.5:7.8b 확인 → 없으면 pull 제안
  - 진행률 바 표시

Step 4: 검증 + 요약
  - 전체 체크 결과 테이블
  - 설정 파일 저장 제안
```

### 3.2 `eval` — 인터랙티브 평가

```
Step 1: 설정 로드
  - 설정 파일 선택 (발견된 YAML 목록 표시)
  - 또는 빠른 설정 (모델 2~3개로 즉시 시작)

Step 2: 환경 확인
  - Ollama 연결
  - GPU 상태
  - 원격 서버 (설정에 있으면)

Step 3: 모델 선택
  - Ollama에서 사용 가능한 모델 목록 표시
  - 미설치 모델 → pull 제안
  - 체크박스식 선택 (Confirm per model)

Step 4: 트랙 + 시나리오
  - 추천 시나리오 3개 제안:
    (1) 빠른 평가: T1+T4+T6 (~30분)
    (2) 전체 평가: T1~T7 (~4시간)
    (3) 커스텀: 직접 선택
  - 선택 후 예상 시간 표시

Step 5: 실행
  - Rich Progress 바 (트랙별 + 모델별)
  - 에러 발생 시: [retry] [skip model] [skip track] [abort]
  - 체크포인트 자동 저장

Step 6: 결과
  - 종합 랭킹 테이블 (Rich Table)
  - 리포트 생성 → 경로 표시
  - HTML 리포트 열기 제안
```

### 3.3 `config` — YAML 빌더

```
Step 1: 프로젝트 설정
  - 이름 입력
  - 출력 경로 (기본값 제안)

Step 2: 백엔드
  - Ollama URL (기본: localhost:11434)
  - 원격 여부

Step 3: 모델 목록
  - Ollama에서 발견된 모델 선택
  - 태그 지정 (baseline/custom)

Step 4: 트랙
  - 활성화할 트랙 선택 (1~7)

Step 5: Judge
  - 이중 Judge 사용?
  - 모델 선택 + 가중치

Step 6: 저장
  - YAML 미리보기 (Rich Syntax)
  - 경로 지정 후 저장
```

### 3.4 `status` — 시스템 진단

```
┌─ 시스템 ──────────────────────────────────┐
│ Python: 3.12.12                           │
│ OS: Ubuntu 24.04                          │
│ GPU: RTX 5060 Ti 16GB (VRAM: 15.8GB free) │
│ Ollama: 0.17.7 (localhost:11434) ✅       │
└───────────────────────────────────────────┘

┌─ 모델 (18개) ────────────────────────────┐
│ qwen2.5:3b          1.9GB  ✅            │
│ gemma3:4b            3.3GB  ✅            │
│ ...                                       │
└───────────────────────────────────────────┘

┌─ 이전 평가 ──────────────────────────────┐
│ 2026-04-07 16모델×7트랙 (examples/)      │
│ 체크포인트: T4 3/16 완료                  │
└───────────────────────────────────────────┘
```

---

## 4. 에러 복구 체계

### 4.1 에러 유형별 대응

| 에러 | 감지 방법 | 자동 대응 | 사용자 선택 |
|:---|:---|:---|:---|
| Ollama 미실행 | HTTP 연결 실패 | 자동 시작 시도 | 재시도/수동 시작 안내 |
| 모델 미설치 | `/api/show` 404 | pull 제안 | 다운로드/건너뛰기 |
| 모델 로딩 실패 | `switch_model()` False | 3회 재시도 | 건너뛰기/중단 |
| 추론 타임아웃 | requests.Timeout | 백오프 재시도 | 재시도/건너뛰기/중단 |
| GPU OOM | CUDA 에러 | 모델 언로드 | 작은 모델로/CPU/중단 |
| Judge 오류 | judge.py 예외 | 재시도 3회 | 건너뛰기/중단 |
| 디스크 부족 | 저장 실패 | 경고 | 정리 안내/경로 변경 |

### 4.2 롤백 매커니즘

```python
# 체크포인트 기반 롤백
def rollback_to_checkpoint(track_name):
    """마지막 성공 체크포인트로 되돌리기"""
    ckpt = load_checkpoint(track_name)
    if ckpt:
        completed = [r["model"] for r in ckpt.get("results", [])]
        console.print(f"체크포인트 복원: {len(completed)}개 모델 완료 상태")
        return ckpt
    return None

# 결과 파일 롤백
def rollback_results(track_name, timestamp):
    """특정 시점 이후 결과 파일 삭제"""
    for f in results_dir.glob(f"{track_name}_*.json"):
        if f.stat().st_mtime > timestamp:
            f.unlink()
```

---

## 5. 다중 머신 설정

### 5.1 YAML 확장

```yaml
machines:
  - name: "local"
    url: "http://localhost:11434"
    type: local              # local | remote
    tracks: [1, 2, 3, 4, 5]  # 이 머신에서 실행할 트랙
    
  - name: "gpu-server"
    url: "http://192.168.1.10:11434"
    type: remote
    tracks: [6, 7]           # 성능/Elo는 원격에서
```

### 5.2 연결 테스트

```python
def test_machine(url: str) -> dict:
    """머신 연결 + 상태 확인"""
    result = {"url": url, "connected": False, "models": [], "gpu": None}
    try:
        r = requests.get(f"{url}/", timeout=5)
        result["connected"] = r.status_code == 200
        tags = requests.get(f"{url}/api/tags", timeout=10).json()
        result["models"] = [m["name"] for m in tags.get("models", [])]
    except Exception as e:
        result["error"] = str(e)
    return result
```

---

## 6. Verification

1. `python kobench.py setup` — 인터랙티브 설치 완료
2. `python kobench.py eval` — 모델 2개 + T1 트랙으로 빠른 평가 성공
3. `python kobench.py config` — YAML 파일 생성 확인
4. `python kobench.py status` — 시스템 정보 표시
5. 에러 시나리오: 없는 모델로 eval → skip 선택 → 나머지 모델 계속 실행
6. 기존 `python kobench.py --config X.yaml` 배치 모드 하위호환 유지
7. `pytest tests/ -x` 전체 통과

---

## 7. Scope

### In Scope
- 4개 서브커맨드 (setup, eval, config, status)
- Rich UI (프롬프트, 테이블, 프로그레스 바, 패널)
- 에러 복구 UI (retry/skip/abort)
- 다중 머신 연결 테스트
- 체크포인트 기반 롤백
- 기존 배치 모드 100% 호환

### Out of Scope
- SSH 자동 접속 (사용자가 수동으로 터널 설정)
- Web UI
- 실시간 원격 모니터링
