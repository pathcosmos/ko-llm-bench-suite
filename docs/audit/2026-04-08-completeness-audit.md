# ko-llm-bench-suite 종합 완성도 감사 보고서

**감사일:** 2026-04-08
**감사 대상:** ko-llm-bench-suite v0.1.0 (커밋 8698c51)
**감사 방법:** 2개 병렬 에이전트를 투입하여 (1) 전체 도구 완성도 감사 + (2) YAML→실행 파이프라인 코드 추적을 동시 수행

---

## 1. 감사 방법론

### 1.1 접근 방식

두 가지 관점에서 병렬 감사를 실시했다:

**에이전트 A — End-to-End 완성도 감사:**
- 신규 사용자 관점에서 설치→설정→실행→결과 확인 전 과정 추적
- 10개 영역(First-Run, Ollama 설정, 멀티머신, 모델관리, 트랙, 리포트, 체크포인트, 설정, 테스트, 누락 기능) 점검
- 각 영역에 PASS / PARTIAL / FAIL 등급 부여

**에이전트 B — YAML 파이프라인 코드 추적:**
- `configs/default.yaml` → `kobench/config.py` → `kobench.py` → `runner.py` → `tracks/*.py` 경로를 코드 레벨에서 추적
- 6개 설정 카테고리(Backend, Models, Judge, Sampling, Output, Retry)별로 YAML 값이 실제 실행에 반영되는지 확인
- 각 카테고리에 CONNECTED / PARTIAL / DISCONNECTED 등급 부여

### 1.2 감사 범위

| 항목 | 대상 파일 | 검사 내용 |
|:---|:---|:---|
| README | `README.md`, `README_EN.md` | 설치 가이드 완전성, 기능 문서화 |
| CLI | `kobench.py` | 옵션 완전성, 에러 메시지 |
| 설정 | `kobench/config.py`, `configs/*.yaml` | YAML 로딩→적용 파이프라인 |
| 백엔드 | `kobench/backends/`, `kobench/runner.py` | 추상화 연결, Ollama API |
| Judge | `kobench/judge.py` | YAML judge 설정 반영 |
| 트랙 | `kobench/tracks/*.py` (7개) | 독립성, 데이터 완전성 |
| 데이터 | `benchmarks/*.json` | 파일 존재, 내용 유효성 |
| 테스트 | `tests/` (515개) | 통과율, 자체 완결성 |
| 인프라 | `ollama_watchdog.sh`, `requirements.txt` | 자동화 수준 |

---

## 2. 감사 결과 요약

### 2.1 전체 점수

| 영역 | 등급 | 세부 |
|:---|:---:|:---|
| 코어 평가 엔진 (7트랙) | **A** | 7,300줄, 모듈화 우수, 전 트랙 독립 실행 |
| 테스트 인프라 | **A** | 515개 전부 통과, Ollama 불필요, 11초 |
| 체크포인트/재개 | **A** | 모든 트랙에서 동작, 병렬 실행 지원 |
| 원격 서버 지원 | **A-** | 환경변수로 동작, SSH 터널 미문서화 |
| 리포트 생성 | **B+** | HTML/MD 자동, 26개 차트 중 11개만 자동 |
| README/문서 | **B+** | 한영 상세, 차트 삽입, 일부 기능 미문서화 |
| **YAML 설정 시스템** | **F** | **모델 목록만 작동, 나머지 6개 카테고리 미연결** |
| 자동 설치/설정 | **F** | setup.sh 없음, 모델 자동 다운로드 없음 |
| 설정 검증 | **F** | YAML 유효성 검사 없음 |
| Docker/CI | **F** | 없음 |

### 2.2 총평

> **코어 엔진은 연구/내부 사용 수준으로 우수(A)하나, YAML 설정 파이프라인이 미완성(F)이어서 "설정만으로 새 평가 실행"이라는 핵심 약속을 아직 지키지 못함.**

---

## 3. 핵심 발견: YAML 파이프라인 미연결

### 3.1 문제 설명

`configs/default.yaml`에 정의된 7개 설정 카테고리 중 **1개만 실제 실행에 반영**된다.

### 3.2 카테고리별 추적 결과

#### (1) models — ✅ CONNECTED

```
YAML models: [{name: "qwen2.5:3b"}, ...] 
  → kobench.py:175-179 load_yaml_config()
  → args.models = [m["name"] for m in yaml_cfg["models"]]
  → run_tracks(args.tracks, args.models)
  → track.run(models=args.models)
  ✅ 실행에 반영됨
```

**조건:** `--config` 플래그 필수, `--models` 플래그 미지정 시에만 적용.

#### (2) backend.url — ❌ DISCONNECTED

```
YAML backend.url: "http://192.168.1.100:11434"
  → kobench.py에서 load_yaml_config()으로 로드
  → ❌ config.OLLAMA_BASE_URL에 적용하는 코드 없음
  → config.py:20 하드코딩: OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
  → runner.py가 config.OLLAMA_BASE_URL 사용
  ❌ YAML 값 무시됨, 환경변수 또는 기본값만 사용
```

#### (3) backend.remote — ❌ DISCONNECTED

```
YAML backend.remote: true
  → 로드됨
  → ❌ config.OLLAMA_REMOTE에 적용 안 됨
  → config.py:21: OLLAMA_REMOTE = os.getenv("OLLAMA_REMOTE", "").lower() in ("1", "true", "yes")
  ❌ 환경변수만 인식
```

#### (4) judge 설정 — ❌ DISCONNECTED

```
YAML judge.primary.model: "qwen2.5:7b-instruct"
     judge.secondary.model: "exaone3.5:7.8b"
     judge.timeout: 120
  → 로드됨
  → ❌ config.JUDGE_MODELS, config.JUDGE_TIMEOUT에 적용 안 됨
  → judge.py:27 항상 config.JUDGE_MODEL 사용 (하드코딩)
  → judge.py:248 항상 config.JUDGE_MODELS dict 사용 (하드코딩)
  ❌ YAML judge 값 완전히 무시됨
```

#### (5) sampling 파라미터 — ❌ DISCONNECTED

```
YAML sampling.default.temperature: 0.7
  → 로드됨
  → ❌ config.SAMPLING_PARAMS에 적용 안 됨
  → config.py:122-138 하드코딩:
    SAMPLING_PARAMS = {"temperature": 0.7, "repeat_penalty": 1.2, ...}
    BENCHMARK_SAMPLING = {"temperature": 0.0, ...}
  ❌ YAML sampling 값 무시됨
```

#### (6) project.output_dir — ❌ DISCONNECTED

```
YAML project.output_dir: "./examples/frankenstallm/results"
  → 로드됨
  → ❌ config.RESULTS_DIR에 적용 안 됨
  → config.py:13: RESULTS_DIR = PROJECT_ROOT / "results"
  ❌ 항상 ./results/에 저장됨
```

#### (7) retry 설정 — ❌ DISCONNECTED

```
YAML retry.max_retries: 3
     retry.backoff_base: 5
  → 로드됨
  → ❌ config.MAX_RETRIES, config.RETRY_BACKOFF_BASE에 적용 안 됨
  → config.py:142-143 하드코딩
  ❌ YAML retry 값 무시됨
```

### 3.3 근본 원인

`kobench.py` 175~179라인에서 YAML을 로드하지만, **모델 목록만 추출**하고 나머지 설정을 `config` 모듈에 적용하지 않는다:

```python
# 현재 코드 (kobench.py:175-179)
if args.config:
    from kobench.config import load_yaml_config
    yaml_cfg = load_yaml_config(args.config)
    if yaml_cfg.get("models") and not args.models:
        args.models = [m["name"] for m in yaml_cfg["models"]]
    # ← 여기서 끝. backend, judge, sampling, output_dir 등은 적용 안 됨
```

**필요한 코드:**
```python
# 누락된 적용 로직
if args.config:
    yaml_cfg = load_yaml_config(args.config)
    if yaml_cfg.get("models") and not args.models:
        args.models = [m["name"] for m in yaml_cfg["models"]]
    # ↓ 이 부분이 전부 누락됨
    if "backend" in yaml_cfg:
        config.OLLAMA_BASE_URL = yaml_cfg["backend"].get("url", config.OLLAMA_BASE_URL)
        config.OLLAMA_REMOTE = yaml_cfg["backend"].get("remote", False)
        # API endpoint 재생성
        config.OLLAMA_API_GENERATE = f"{config.OLLAMA_BASE_URL}/api/generate"
        config.OLLAMA_API_CHAT = f"{config.OLLAMA_BASE_URL}/api/chat"
        ...
    if "judge" in yaml_cfg:
        j = yaml_cfg["judge"]
        config.JUDGE_DUAL_ENABLED = j.get("dual_enabled", True)
        config.JUDGE_MODELS["primary"] = j.get("primary", {}).get("model", ...)
        ...
    if "sampling" in yaml_cfg:
        config.SAMPLING_PARAMS.update(yaml_cfg["sampling"].get("default", {}))
        config.BENCHMARK_SAMPLING.update(yaml_cfg["sampling"].get("benchmark", {}))
    if "project" in yaml_cfg:
        config.RESULTS_DIR = Path(yaml_cfg["project"].get("output_dir", "results"))
        config.REPORTS_DIR = Path(yaml_cfg["project"].get("reports_dir", "reports"))
    if "retry" in yaml_cfg:
        config.MAX_RETRIES = yaml_cfg["retry"].get("max_retries", 3)
        config.RETRY_BACKOFF_BASE = yaml_cfg["retry"].get("backoff_base", 5)
        ...
```

---

## 4. 영역별 상세 감사 결과

### 4.1 신규 사용자 경험 (First-Run)

**등급: PARTIAL**

| 항목 | 결과 | 비고 |
|:---|:---:|:---|
| README 설치 가이드 | ✅ | 명확, 단계별, 한영 모두 |
| requirements.txt 완전성 | ⚠️ | pyproject.toml과 불일치 (pandas, seaborn 누락) |
| `kobench.py --help` | ✅ | 5개 옵션 표시 |
| 예시 설정 파일 | ✅ | `configs/examples/frankenstallm.yaml` |
| 자동 설치 스크립트 | ❌ | `setup.sh` 없음 |
| CLI 언어 | ⚠️ | 한국어 전용 (영어 미지원) |

### 4.2 Ollama 설정 자동화

**등급: FAIL**

| 항목 | 결과 | 비고 |
|:---|:---:|:---|
| Ollama 자동 설치 | ❌ | 수동 `curl` 필요 |
| 모델 자동 다운로드 | ❌ | 수동 `ollama pull` 필요 |
| Ollama 실행 확인 | ✅ | `wait_for_ollama()` 30초 대기 + 재시작 3회 |
| 에러 메시지 | ✅ | "Ollama 서버에 연결할 수 없습니다" 안내 |
| ollama_watchdog.sh | ✅ | 존재, 자동 재시작, 로그 기록 |
| 모델 미존재 시 에러 | ❌ | HTTP 500 + 트랙 크래시 (안내 없음) |

### 4.3 멀티머신/원격 설정

**등급: PASS**

| 항목 | 결과 | 비고 |
|:---|:---:|:---|
| 원격 Ollama URL 설정 | ✅ | `OLLAMA_BASE_URL` 환경변수 |
| `OLLAMA_REMOTE` 모드 | ✅ | 로컬 프로세스 관리 비활성화 |
| SSH 터널 문서 | ❌ | README에 미기재 |
| 병렬 실행 (`EVAL_CHECKPOINT_SUFFIX`) | ✅ | 코드에 구현, 실전 검증 완료 |
| 병렬 실행 문서 | ❌ | README에 미기재 |

### 4.4 트랙 완성도

**등급: PASS (A)**

| 트랙 | LOC | 독립 실행 | 체크포인트 | 데이터 파일 |
|:---|---:|:---:|:---:|:---:|
| T1 Korean Bench | 649 | ✅ | ✅ | ✅ |
| T2 Ko-Bench | 723 | ✅ | ✅ | ✅ (337줄) |
| T3 Korean Deep | 323 | ✅ | ✅ | ✅ (801줄) |
| T4 Code & Math | 597 | ✅ | ✅ | ✅ (432+184줄) |
| T5 Consistency | 881 | ✅ | ✅ | 내장 |
| T6 Performance | 581 | ✅ | ✅ | 내장 |
| T7 Pairwise | 491 | ✅ | ✅ | ✅ (101줄) |

### 4.5 테스트 인프라

**등급: PASS (A)**

```
$ pytest tests/ -q
========================= 515 passed in 11.19s =========================
```

| 항목 | 결과 |
|:---|:---|
| 총 테스트 수 | 515개 |
| 통과율 | 100% |
| 실행 시간 | 11초 |
| Ollama/GPU 필요 | 불필요 (mock 기반) |
| 단위 테스트 | 12 파일 |
| 통합 테스트 | 3 파일 |

---

## 5. 우선 수정 권고

### P0: 즉시 수정 필요

| # | 항목 | 예상 소요 | 영향도 |
|:---|:---|:---|:---|
| 1 | **YAML→config 적용 파이프라인 완성** | 30분 | YAML 설정의 핵심 약속 |
| 2 | **setup.sh 자동 설치 스크립트** | 15분 | 신규 사용자 30분 절약 |
| 3 | **모델 사전 검증 (`--check-models`)** | 20분 | 실행 중 크래시 방지 |
| 4 | **YAML 스키마 검증** | 20분 | 설정 오류 사전 방지 |

### P1: 중요 개선

| # | 항목 | 예상 소요 |
|:---|:---|:---|
| 5 | requirements.txt ↔ pyproject.toml 동기화 | 5분 |
| 6 | SSH 터널 + 병렬 실행 README 문서화 | 15분 |
| 7 | 체크포인트 기능 README 문서화 | 10분 |
| 8 | YAML 인라인 주석 추가 | 10분 |
| 9 | `--list-models` CLI 명령 추가 | 15분 |

### P2: 향후 개선

| # | 항목 |
|:---|:---|
| 10 | Dockerfile + docker-compose |
| 11 | GitHub Actions CI/CD |
| 12 | CLI 영어 지원 (i18n) |
| 13 | 시각화 차트 자동 생성 통합 |

---

## 6. 결론

ko-llm-bench-suite의 **코어 평가 엔진은 연구/내부 사용 수준으로 우수**하다 (7트랙 완전 구현, 515 테스트 통과, 체크포인트/원격 지원). 그러나 **YAML 설정 파이프라인이 사실상 미완성**이어서, 프로젝트가 약속하는 "YAML만으로 새 평가 실행"은 현재 불가능하다.

**P0 4건(~1.5시간)을 수정하면 한국어 LLM 전문 평가 도구로서의 핵심 기능이 완성**되며, 오픈소스 공개 수준에 도달한다.

---

*감사 수행: Claude Opus 4.6 (병렬 에이전트 2대)*
*검증 환경: Ubuntu 24.04, Python 3.12, RTX 5060 Ti 16GB*
