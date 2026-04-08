# ko-llm-bench-suite 심층 문제 분석 + 수정 계획

**분석일:** 2026-04-08
**감사 방법:** 병렬 에이전트 2대 (YAML 파이프라인 코드 추적 + 설정/자동화 갭 분석)
**대상 커밋:** b0643bb (v0.1.0)

---

## Part 1: YAML 파이프라인 완전 코드 추적

### 1.1 전체 설정 변수 매핑 (39개)

`kobench/config.py`의 런타임 변수 39개를 YAML 필드와 1:1 매핑한 결과:

| 카테고리 | 총 변수 | ✅ 연결됨 | ❌ 미연결 | N/A |
|:---|---:|---:|---:|---:|
| 경로 (RESULTS_DIR 등) | 4 | 0 | 3 | 1 |
| Ollama 백엔드 | 6 | 0 | **6** | 0 |
| Judge 설정 | 7 | 0 | **7** | 0 |
| 모델 목록 | 6 | **1** (조건부) | 0 | 5 |
| GPU/타임아웃 | 5 | 0 | 5 | 0 |
| 샘플링 파라미터 | 2 | 0 | **2** | 0 |
| 재시도/쿨다운 | 4 | 0 | **4** | 0 |
| 트랙별 설정 | 5 | 0 | 0 | 5 |
| **합계** | **39** | **1** | **27** | **11** |

### 1.2 카테고리별 미연결 상세

#### (A) Ollama 백엔드 — 6개 변수 전부 미연결

| 변수 | config.py 라인 | 현재 소스 | YAML 필드 | 사용처 |
|:---|---:|:---|:---|:---|
| `OLLAMA_BASE_URL` | L49 | `os.getenv()` | `backend.url` | runner.py:50,220,231,250,314,393 / judge.py:30 |
| `OLLAMA_REMOTE` | L50 | `os.getenv()` | `backend.remote` | runner.py:63,82,108,126,166 |
| `OLLAMA_API_GENERATE` | L51 | 파생 | 파생 | runner.py:220,250,314 / judge.py:30 / performance.py:183 |
| `OLLAMA_API_CHAT` | L52 | 파생 | 파생 | runner.py:393 |
| `OLLAMA_API_SHOW` | L53 | 파생 | 파생 | (미사용) |
| `OLLAMA_API_PS` | L54 | 파생 | 파생 | runner.py:231 |

**영향:** YAML에 `backend.url: "http://remote:11434"`를 설정해도 **항상 localhost:11434에 연결됨**.

**근본 원인:** `kobench.py:175-179`에서 YAML 로드 후 `config.OLLAMA_BASE_URL`에 대입하는 코드 없음.

#### (B) Judge 설정 — 7개 변수 전부 미연결

| 변수 | config.py 라인 | 하드코딩 값 | YAML 필드 |
|:---|---:|:---|:---|
| `JUDGE_MODEL` | L58 | `"qwen2.5:7b-instruct"` | `judge.primary.model` |
| `JUDGE_MODELS` | L59-62 | `{primary: "qwen2.5:7b-instruct", secondary: "exaone3.5:7.8b"}` | `judge.primary/secondary.model` |
| `JUDGE_WEIGHTS` | L63 | `{primary: 0.6, secondary: 0.4}` | `judge.primary/secondary.weight` |
| `JUDGE_DISAGREEMENT_THRESHOLD` | L64 | `3` | default.yaml에 미정의 |
| `JUDGE_DUAL_ENABLED` | L65 | `True` | `judge.dual_enabled` |
| `JUDGE_TIMEOUT` | L66 | `120` | `judge.timeout` |
| `JUDGE_SAMPLING` | L67-72 | 고정 dict | default.yaml에 미정의 |

**영향:** YAML에서 judge 모델을 변경해도 **항상 qwen2.5:7b + exaone3.5:7.8b를 사용**.

#### (C) 샘플링 — 2개 변수 미연결

| 변수 | config.py 라인 | YAML 필드 | 사용 빈도 |
|:---|---:|:---|:---|
| `SAMPLING_PARAMS` | L152-158 | `sampling.default.*` | runner.py + 전체 트랙 (8곳+) |
| `BENCHMARK_SAMPLING` | L161-167 | `sampling.benchmark.*` | korean_bench, korean_deep, code_math |

**영향:** `temperature: 0.5` 등 사용자 지정 값이 **전혀 반영되지 않음**.

#### (D) 경로 — 3개 미연결

| 변수 | config.py 라인 | YAML 필드 |
|:---|---:|:---|
| `RESULTS_DIR` | L42 | `project.output_dir` |
| `REPORTS_DIR` | L43 | `project.reports_dir` |
| `DATA_DIR` | L44 | (해당 없음) |

**영향:** `output_dir: "./exp1/results"`를 설정해도 **항상 `./results/`에 저장됨**.

#### (E) 재시도/쿨다운 — 4개 미연결

| 변수 | config.py 라인 | YAML 필드 | 사용 빈도 |
|:---|---:|:---|:---|
| `MAX_RETRIES` | L171 | `retry.max_retries` | runner.py 3곳 + judge.py 7곳 |
| `RETRY_BACKOFF_BASE` | L172 | `retry.backoff_base` | runner.py 2곳 |
| `COOLDOWN_BETWEEN_MODELS` | L173 | `retry.cooldown_between_models` | runner.py 3곳 + kobench.py 2곳 |
| `COOLDOWN_BETWEEN_TESTS` | L174 | `retry.cooldown_between_tests` | 전체 트랙 (20곳+) |

### 1.3 현재 vs 필요 실행 흐름

```
현재 (BROKEN):
kobench.py --config my.yaml
  └─ load_yaml_config() → dict 반환
  └─ models만 추출 (L178-179)
  └─ ❌ 나머지 6개 카테고리 적용 코드 없음
  └─ run_tracks() → config.* (하드코딩 기본값) 사용

필요 (FIXED):
kobench.py --config my.yaml
  └─ load_yaml_config() → dict 반환
  └─ models 추출 ✅
  └─ apply_yaml_to_config(yaml_cfg) ← 신규 함수
      ├─ backend → config.OLLAMA_BASE_URL, REMOTE, API_* 재생성
      ├─ judge → config.JUDGE_MODELS, WEIGHTS, TIMEOUT, DUAL_ENABLED
      ├─ sampling → config.SAMPLING_PARAMS, BENCHMARK_SAMPLING
      ├─ project → config.RESULTS_DIR, REPORTS_DIR (+ mkdir)
      └─ retry → config.MAX_RETRIES, BACKOFF, COOLDOWNS
  └─ run_tracks() → config.* (YAML 값 적용됨) 사용
```

---

## Part 2: 자동화/설정 갭 분석

### 2.1 모델 존재 확인 부재

**현재 동작 추적:**

```
kobench.py main() → run_tracks()
  → track.run(models=["nonexistent-model"])
    → runner.switch_model("nonexistent-model")  # runner.py:422
      → warmup_model("nonexistent-model")        # runner.py:246
        → requests.post(OLLAMA_API_GENERATE, ...)  # runner.py:250
          → Ollama 404 or empty response
        → return False                             # runner.py:259
      → 3회 재시도 (runner.py:471-473)
      → print("❌ 모델 전환 실패") & return False    # runner.py:475
    → 트랙이 해당 모델 건너뜀 (에러 결과 기록)
```

**문제점:**
1. 사전 검증 없이 추론 시도 → 모델당 30~60초 낭비
2. "모델이 없다"와 "네트워크 오류"를 구분하지 않음
3. Ollama `/api/show` 엔드포인트가 **정의만 되고 사용되지 않음** (config.py L53)

**해결책:**
```python
# runner.py에 추가
def check_model_exists(model: str) -> bool:
    """Ollama /api/show로 모델 존재 여부 경량 확인 (로드 없이)"""
    try:
        r = requests.post(config.OLLAMA_API_SHOW, json={"name": model}, timeout=5)
        return r.status_code == 200
    except Exception:
        return False
```

### 2.2 의존성 동기화 문제

| 패키지 | requirements.txt | pyproject.toml | 상태 |
|:---|:---|:---|:---:|
| matplotlib | `>=3.8` | `>=3.7` | ⚠️ 버전 불일치 |
| requests | `>=2.31` | `>=2.28` | ⚠️ 버전 불일치 |
| numpy | `>=1.26` | `>=1.24` | ⚠️ 버전 불일치 |
| scipy | `>=1.12` | `>=1.10` | ⚠️ 버전 불일치 |
| pyyaml | `>=6.0` | `>=6.0` | ✅ 일치 |
| **pandas** | **❌ 누락** | `>=2.0` | 🔴 report.py에서 import 실패 |
| **seaborn** | **❌ 누락** | `>=0.12` | 🔴 report.py에서 import 실패 |

**영향:** `pip install -r requirements.txt`로 설치 시 리포트 생성이 실패함.

### 2.3 YAML 인라인 문서 부재

`configs/default.yaml`에 **주석이 단 하나도 없음**. 각 필드의 의미, 유효 범위, 기본값 근거가 문서화되지 않아 사용자가 코드를 읽어야 함.

### 2.4 설정 검증 부재 — 에지 케이스 테스트

| 시나리오 | 현재 동작 | 기대 동작 |
|:---|:---|:---|
| `models: []` (빈 목록) | 기본 13모델로 폴백 ✅ | OK (단, 미문서화) |
| `tracks.enabled: [99]` | `ValueError` 크래시 ❌ | 유효하지 않은 트랙 번호 안내 |
| `temperature: -5` | Ollama에 전달 → 오류 ❌ | 유효 범위 [0, 2] 검증 |
| `judge.primary.model: "없는모델"` | 4시간 추론 후 크래시 ❌ | 사전 검증 후 안내 |
| `backend.type: "vllm"` | 무시, Ollama 사용 ❌ | 미지원 백엔드 경고 |

### 2.5 CLI 누락 기능

| 기능 | 현재 | 필요성 | 구현 복잡도 |
|:---|:---:|:---|:---:|
| `--version` | ❌ | CI/CD 통합 | 1줄 |
| `--list-models` | ❌ | 사용 가능 모델 확인 | 5줄 |
| `--check-models` | ❌ | 실행 전 모델 검증 | 10줄 |
| `--validate-config` | ❌ | YAML 오류 사전 감지 | 3줄 (validate 함수 호출) |
| `--show-config` | ❌ | 최종 설정 확인 (YAML + CLI 병합) | 5줄 |

### 2.6 README 미문서화 기능

| 기능 | 코드에 존재 | README에 문서화 |
|:---|:---:|:---:|
| 체크포인트/재개 | ✅ runner.py:488-504 | ❌ |
| `EVAL_CHECKPOINT_SUFFIX` 병렬 | ✅ runner.py:490,499 | ❌ |
| SSH 터널 원격 Ollama | ✅ config.py:49-50 | ❌ |
| 멀티머신 병렬 실행 | ✅ 실전 검증 완료 | ❌ |
| Judge 모델 필수 요건 | ✅ judge.py:27 | ⚠️ 언급만, 명확하지 않음 |

---

## Part 3: 수정 계획

### Phase A: YAML 파이프라인 완성 (P0, ~45분)

#### Task A-1: `apply_yaml_to_config()` 함수 작성

**파일:** `kobench/config.py`

```python
def apply_yaml_to_config(yaml_cfg: dict) -> None:
    """YAML 설정을 런타임 config 변수에 적용한다.
    
    기존 하드코딩 값을 YAML 값으로 오버라이드.
    YAML에 없는 필드는 기존 기본값 유지.
    """
    global OLLAMA_BASE_URL, OLLAMA_REMOTE, OLLAMA_API_GENERATE, OLLAMA_API_CHAT
    global OLLAMA_API_SHOW, OLLAMA_API_PS
    global JUDGE_MODEL, JUDGE_MODELS, JUDGE_WEIGHTS, JUDGE_DUAL_ENABLED, JUDGE_TIMEOUT
    global JUDGE_DISAGREEMENT_THRESHOLD
    global RESULTS_DIR, REPORTS_DIR
    global MAX_RETRIES, RETRY_BACKOFF_BASE, COOLDOWN_BETWEEN_MODELS, COOLDOWN_BETWEEN_TESTS
    global EVAFRILL_GPU_STRATEGY
    
    # Backend
    if "backend" in yaml_cfg:
        b = yaml_cfg["backend"]
        OLLAMA_BASE_URL = b.get("url", OLLAMA_BASE_URL)
        OLLAMA_REMOTE = b.get("remote", OLLAMA_REMOTE)
        OLLAMA_API_GENERATE = f"{OLLAMA_BASE_URL}/api/generate"
        OLLAMA_API_CHAT = f"{OLLAMA_BASE_URL}/api/chat"
        OLLAMA_API_SHOW = f"{OLLAMA_BASE_URL}/api/show"
        OLLAMA_API_PS = f"{OLLAMA_BASE_URL}/api/ps"
    
    # Judge
    if "judge" in yaml_cfg:
        j = yaml_cfg["judge"]
        JUDGE_DUAL_ENABLED = j.get("dual_enabled", JUDGE_DUAL_ENABLED)
        JUDGE_TIMEOUT = j.get("timeout", JUDGE_TIMEOUT)
        if "primary" in j:
            JUDGE_MODEL = j["primary"].get("model", JUDGE_MODEL)
            JUDGE_MODELS["primary"] = j["primary"].get("model", JUDGE_MODELS["primary"])
            JUDGE_WEIGHTS["primary"] = j["primary"].get("weight", JUDGE_WEIGHTS["primary"])
        if "secondary" in j:
            JUDGE_MODELS["secondary"] = j["secondary"].get("model", JUDGE_MODELS["secondary"])
            JUDGE_WEIGHTS["secondary"] = j["secondary"].get("weight", JUDGE_WEIGHTS["secondary"])
        if "disagreement_threshold" in j:
            JUDGE_DISAGREEMENT_THRESHOLD = j["disagreement_threshold"]
    
    # Sampling
    if "sampling" in yaml_cfg:
        if "default" in yaml_cfg["sampling"]:
            SAMPLING_PARAMS.update(yaml_cfg["sampling"]["default"])
        if "benchmark" in yaml_cfg["sampling"]:
            BENCHMARK_SAMPLING.update(yaml_cfg["sampling"]["benchmark"])
    
    # Project paths
    if "project" in yaml_cfg:
        p = yaml_cfg["project"]
        RESULTS_DIR = Path(p["output_dir"]) if "output_dir" in p else RESULTS_DIR
        REPORTS_DIR = Path(p["reports_dir"]) if "reports_dir" in p else REPORTS_DIR
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Retry
    if "retry" in yaml_cfg:
        r = yaml_cfg["retry"]
        MAX_RETRIES = r.get("max_retries", MAX_RETRIES)
        RETRY_BACKOFF_BASE = r.get("backoff_base", RETRY_BACKOFF_BASE)
        COOLDOWN_BETWEEN_MODELS = r.get("cooldown_between_models", COOLDOWN_BETWEEN_MODELS)
        COOLDOWN_BETWEEN_TESTS = r.get("cooldown_between_tests", COOLDOWN_BETWEEN_TESTS)
```

#### Task A-2: `kobench.py`에서 `apply_yaml_to_config()` 호출

**파일:** `kobench.py` (L175-179 수정)

```python
if args.config:
    from kobench.config import load_yaml_config, apply_yaml_to_config
    yaml_cfg = load_yaml_config(args.config)
    apply_yaml_to_config(yaml_cfg)  # ← 핵심 추가
    if yaml_cfg.get("models") and not args.models:
        args.models = [m["name"] for m in yaml_cfg["models"]]
```

#### Task A-3: 테스트 작성

`tests/unit/test_config.py`에 추가:
- `test_apply_yaml_backend()` — URL/REMOTE 변경 확인
- `test_apply_yaml_judge()` — Judge 모델/가중치 변경 확인
- `test_apply_yaml_sampling()` — Temperature 등 변경 확인
- `test_apply_yaml_paths()` — RESULTS_DIR 변경 확인
- `test_apply_yaml_retry()` — MAX_RETRIES 변경 확인
- `test_apply_yaml_partial()` — 일부 필드만 있을 때 나머지 기본값 유지

### Phase B: 설정 검증 + 모델 사전 체크 (P0, ~30분)

#### Task B-1: `validate_config()` 함수

**파일:** `kobench/config.py`

```python
def validate_config(yaml_cfg: dict) -> list[str]:
    """YAML 설정 유효성 검증. 오류 목록 반환 (빈 리스트 = 유효)."""
    errors = []
    
    # backend.type
    valid_backends = ("ollama",)
    bt = yaml_cfg.get("backend", {}).get("type", "ollama")
    if bt not in valid_backends:
        errors.append(f"backend.type '{bt}' 미지원. 가능: {valid_backends}")
    
    # tracks.enabled
    for t in yaml_cfg.get("tracks", {}).get("enabled", []):
        if t not in range(1, 8):
            errors.append(f"tracks.enabled에 유효하지 않은 트랙: {t} (1~7만 가능)")
    
    # sampling ranges
    for profile in ("default", "benchmark"):
        s = yaml_cfg.get("sampling", {}).get(profile, {})
        if "temperature" in s and not (0 <= s["temperature"] <= 2):
            errors.append(f"sampling.{profile}.temperature {s['temperature']}은 0~2 범위 밖")
        if "top_p" in s and not (0 <= s["top_p"] <= 1):
            errors.append(f"sampling.{profile}.top_p {s['top_p']}은 0~1 범위 밖")
    
    # judge weights
    j = yaml_cfg.get("judge", {})
    if j.get("dual_enabled", True):
        w1 = j.get("primary", {}).get("weight", 0.6)
        w2 = j.get("secondary", {}).get("weight", 0.4)
        if abs((w1 + w2) - 1.0) > 0.01:
            errors.append(f"judge weights 합 {w1+w2:.2f} ≠ 1.0")
    
    return errors
```

#### Task B-2: 모델 사전 검증 함수

**파일:** `kobench/runner.py`

```python
def check_models_available(models: list[str]) -> tuple[list[str], list[str]]:
    """모델 존재 여부 사전 확인. (available, missing) 반환."""
    available, missing = [], []
    for model in models:
        try:
            r = requests.post(config.OLLAMA_API_SHOW, json={"name": model}, timeout=5)
            (available if r.status_code == 200 else missing).append(model)
        except Exception:
            missing.append(model)
    return available, missing
```

#### Task B-3: `kobench.py` 통합

```python
# main()에서 실행 전 검증
if args.config:
    yaml_cfg = load_yaml_config(args.config)
    errors = validate_config(yaml_cfg)
    if errors:
        print("❌ 설정 오류:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    apply_yaml_to_config(yaml_cfg)
    ...

# 모델 사전 검증
models_to_eval = args.models or config.ALL_MODELS
available, missing = runner.check_models_available(models_to_eval)
if missing:
    print(f"⚠️ {len(missing)}개 모델 미설치: {missing}")
    print("  설치: " + " && ".join(f"ollama pull {m}" for m in missing))
    if not args.skip_model_check:
        sys.exit(1)
```

### Phase C: 자동화 + 문서 (P1, ~1시간)

#### Task C-1: `setup.sh` 스크립트

```bash
#!/bin/bash
# ko-llm-bench-suite 자동 설치 스크립트

echo "=== ko-llm-bench-suite 설치 ==="

# 1. Python 의존성
pip install -r requirements.txt

# 2. Ollama 설치 (미설치 시)
if ! command -v ollama &>/dev/null; then
    echo "Ollama 설치 중..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# 3. Judge 모델 다운로드
echo "Judge 모델 다운로드..."
ollama pull qwen2.5:7b-instruct
ollama pull exaone3.5:7.8b

# 4. 검증
python kobench.py --version
echo "✅ 설치 완료"
```

#### Task C-2: `requirements.txt` 동기화

pandas, seaborn 추가, 버전을 pyproject.toml과 일치시킴.

#### Task C-3: `default.yaml` 인라인 주석 추가

모든 필드에 한줄 주석 (용도, 유효 범위, 기본값 근거).

#### Task C-4: CLI 옵션 추가

- `--version`: `kobench.__version__` 출력
- `--list-models`: Ollama `/api/tags` 조회
- `--check-models`: 모델 존재 검증 dry-run
- `--validate-config`: YAML 검증만 수행

#### Task C-5: README 미문서화 기능 추가

- 체크포인트/재개 섹션
- `EVAL_CHECKPOINT_SUFFIX` 병렬 실행 가이드
- SSH 터널 원격 설정 예시
- 멀티머신 분산 실행 가이드
- Judge 모델 필수 요건 명확화

### Phase D: 최종 검증 (P1, ~15분)

- `python kobench.py --config configs/default.yaml --validate-config` → 오류 없음
- `python kobench.py --config configs/examples/frankenstallm.yaml --check-models` → 모델 목록 출력
- YAML에서 `backend.url` 변경 → 실제 연결 대상 변경 확인
- YAML에서 `judge.primary.model` 변경 → 실제 judge 모델 변경 확인
- YAML에서 `sampling.default.temperature: 0.5` → 실제 추론에 반영 확인
- `python -m pytest tests/ -x` → 전체 통과

---

## 수정 일정 요약

| Phase | 내용 | 소요 | 우선도 |
|:---|:---|---:|:---:|
| **A** | YAML→config 파이프라인 완성 (apply 함수 + 테스트) | 45분 | P0 |
| **B** | 설정 검증 + 모델 사전 체크 | 30분 | P0 |
| **C** | 자동화(setup.sh) + 문서 + CLI + YAML 주석 | 60분 | P1 |
| **D** | 최종 검증 | 15분 | P1 |
| **합계** | | **~2.5시간** | |

---

*분석 수행: Claude Opus 4.6 (병렬 에이전트 2대, 코드 추적 + 갭 분석)*
