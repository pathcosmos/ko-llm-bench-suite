# ko-llm-bench-suite Design Spec

**Date:** 2026-04-07
**Status:** Approved
**Approach:** Incremental refactoring of frankenstallm_test → ko-llm-bench-suite

---

## 1. Purpose

한국어 LLM 전문 평가 도구. 어떤 한국어 SLM/LLM이든 YAML 설정만으로 7개 트랙에 걸친 종합 벤치마크를 실행하고, 시각화 리포트를 자동 생성할 수 있는 올인원 평가 프레임워크.

### Success Criteria

- YAML 설정 파일 하나로 새 모델 평가 실행 가능 (코드 수정 불필요)
- Ollama / vLLM 두 가지 추론 백엔드 지원
- `pip install ko-llm-bench-suite` 또는 `python kobench.py --config my.yaml`로 즉시 사용
- 기존 7개 트랙 + frankenstallm 평가 결과가 examples/로 보존

---

## 2. Architecture

### 2.1 Directory Structure

```
ko-llm-bench-suite/
├── kobench/                          # Core package
│   ├── __init__.py                   # version, public API
│   ├── config.py                     # YAML loader + defaults + validation
│   ├── backends/                     # Inference backends
│   │   ├── __init__.py               # get_backend() factory
│   │   ├── base.py                   # InferenceBackend ABC
│   │   ├── ollama.py                 # OllamaBackend (from runner.py)
│   │   └── vllm.py                   # VLLMBackend (new)
│   ├── runner.py                     # Backend router + checkpoint + retry logic
│   ├── judge.py                      # Dual-Judge system (retain as-is)
│   ├── scoring.py                    # Aggregation + Bradley-Terry (retain as-is)
│   ├── report.py                     # HTML/MD report generation (retain as-is)
│   └── tracks/                       # Evaluation tracks
│       ├── __init__.py               # track registry
│       ├── base.py                   # BaseTrack ABC
│       ├── korean_bench.py           # Track 1 (from track1_korean_bench.py)
│       ├── ko_bench.py               # Track 2
│       ├── korean_deep.py            # Track 3
│       ├── code_math.py              # Track 4
│       ├── consistency.py            # Track 5
│       ├── performance.py            # Track 6
│       └── pairwise.py              # Track 7
├── benchmarks/                       # Datasets (from data/)
│   ├── korean_bench/
│   ├── ko_bench/
│   ├── korean_deep/
│   ├── code_problems/
│   ├── math_problems/
│   └── pairwise_prompts/
├── configs/                          # YAML configs
│   ├── default.yaml                  # Full default config
│   └── examples/
│       ├── frankenstallm.yaml        # FS eval config
│       ├── quick_eval.yaml           # 3-model quick test
│       └── remote_ollama.yaml        # Remote server setup
├── examples/                         # Example projects + legacy data
│   └── frankenstallm/
│       ├── README.md
│       ├── results/                  # All 112-cell results
│       ├── reports/                  # 26 charts + full report
│       └── visualizations/
├── tests/                            # Unit/integration tests
├── kobench.py                        # CLI entry point
├── pyproject.toml                    # Package metadata
├── README.md                         # Project README (Korean + English)
└── LICENSE                           # MIT
```

### 2.2 Core Components

#### InferenceBackend ABC (`kobench/backends/base.py`)

```python
class InferenceBackend(ABC):
    @abstractmethod
    def generate(self, model: str, prompt: str, system: str = "",
                 options: dict = None, timeout: int = None) -> dict:
        """Returns: {response, eval_count, tokens_per_sec, wall_time_s, error}"""

    @abstractmethod
    def chat(self, model: str, messages: list[dict],
             options: dict = None, timeout: int = None) -> dict:
        """Multi-turn chat. Same return format."""

    @abstractmethod
    def load_model(self, model: str) -> bool:
        """Load/warm-up model. Returns success."""

    @abstractmethod
    def unload_model(self, model: str) -> None:
        """Release model from memory."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """Available models on this backend."""

    @abstractmethod
    def health_check(self) -> bool:
        """Backend reachable and healthy."""
```

#### OllamaBackend (`kobench/backends/ollama.py`)

기존 `runner.py`의 Ollama 관련 로직을 이 클래스로 이동:
- `generate()` → 기존 `runner.generate()`
- `chat()` → 기존 `runner.chat()`
- `load_model()` → 기존 `runner.warmup_model()`
- `unload_model()` → 기존 `runner.unload_model()`
- `list_models()` → Ollama `/api/tags` 호출
- `health_check()` → 기존 `runner.wait_for_ollama()`
- Retry logic, backoff, timeout 전부 여기로

#### VLLMBackend (`kobench/backends/vllm.py`)

vLLM OpenAI-compatible API를 통한 추론:
- `generate()` → `POST /v1/completions`
- `chat()` → `POST /v1/chat/completions`
- `load_model()` → vLLM은 서버 시작 시 모델 로드, 여기선 health check
- `unload_model()` → no-op (vLLM은 서버 단위 모델 관리)
- `list_models()` → `GET /v1/models`
- `health_check()` → `GET /health`

#### Runner (`kobench/runner.py`)

경량화된 라우터. 백엔드 팩토리 + 체크포인트 + 모델 전환 로직만 담당:

```python
class Runner:
    def __init__(self, config: KoBenchConfig):
        self.backend = get_backend(config.backend)
        self.config = config

    def generate(self, model, prompt, **kwargs) -> dict:
        return self.backend.generate(model, prompt, **kwargs)

    def switch_model(self, new_model, current_model=None) -> bool:
        if current_model:
            self.backend.unload_model(current_model)
        return self.backend.load_model(new_model)

    def save_checkpoint(self, data, track_name) -> Path: ...
    def load_checkpoint(self, track_name) -> dict | None: ...
```

#### BaseTrack (`kobench/tracks/base.py`)

```python
class BaseTrack(ABC):
    name: str           # "korean_bench"
    display_name: str   # "한국어 표준 벤치마크"
    version: str        # "1.0"

    @abstractmethod
    def run(self, runner: Runner, models: list[str]) -> dict:
        """Returns: {"track": name, "summary": {...}, "results": [...]}"""
```

기존 7개 트랙은 이 ABC를 구현하도록 리팩토링. `run()` 시그니처가 `runner` 인스턴스를 받도록 변경 (현재는 모듈-레벨 `runner` import에 의존).

### 2.3 YAML Config Schema

```yaml
# configs/default.yaml
project:
  name: "Korean LLM Evaluation"
  version: "1.0"
  output_dir: "./results"
  reports_dir: "./reports"

backend:
  type: ollama                    # ollama | vllm
  url: "http://localhost:11434"   # Ollama or vLLM server URL
  remote: false                   # Skip local process management
  gpu_strategy: "auto"            # auto | ollama_suspend | evafrill_cpu

models:
  - name: "gemma3:4b"
    tags: [baseline]
  - name: "qwen2.5:3b"
    tags: [baseline]

tracks:
  enabled: [1, 2, 3, 4, 5, 6, 7]
  order: [6, 1, 3, 2, 4, 5, 7]    # Execution order

judge:
  dual_enabled: true
  primary:
    model: "qwen2.5:7b-instruct"
    weight: 0.6
  secondary:
    model: "exaone3.5:7.8b"
    weight: 0.4
  timeout: 120
  disagreement_threshold: 3

sampling:
  default:
    temperature: 0.7
    top_p: 0.9
    num_predict: 512
    num_ctx: 4096
  benchmark:                       # Greedy for deterministic tasks
    temperature: 0.0
    top_p: 1.0
    num_predict: 256

retry:
  max_retries: 3
  backoff_base: 5
  cooldown_between_models: 10
  cooldown_between_tests: 1

benchmarks:
  data_dir: "./benchmarks"         # Override benchmark data location
```

### 2.4 CLI Interface

```bash
# 기본 실행
python kobench.py --config configs/default.yaml

# 빠른 실행 (트랙/모델 선택)
python kobench.py --config my.yaml --tracks 1 4 7 --models "gemma3:4b" "qwen2.5:3b"

# 리포트만 재생성
python kobench.py --config my.yaml --report-only

# 모델 목록 확인
python kobench.py --list-models

# 시각화 생성
python kobench.py --config my.yaml --visualize
```

---

## 3. Migration Plan (Current → New)

### Phase 1: 구조 재편 (파일 이동 + 리네임)

| 현재 | 이동 후 |
|:---|:---|
| `eval_framework/` | `kobench/` |
| `eval_framework/tracks/track1_korean_bench.py` | `kobench/tracks/korean_bench.py` |
| `eval_framework/tracks/track2_ko_bench.py` | `kobench/tracks/ko_bench.py` |
| ... (나머지 5개 트랙도 동일) | ... |
| `eval_framework/evafrill_runner.py` | `kobench/backends/evafrill.py` (선택적 유지) |
| `eval_framework/evafrill_server.py` | `kobench/backends/evafrill_server.py` (선택적) |
| `data/` | `benchmarks/` |
| `run_evaluation.py` | `kobench.py` |
| `results/`, `reports/` | `examples/frankenstallm/results/`, `examples/frankenstallm/reports/` |
| `generate_visualizations*.py` | `examples/frankenstallm/` |

### Phase 2: 코드 리팩토링

1. **config.py**: YAML 로더 추가, 하드코딩된 모델 목록 제거
2. **backends/**: `runner.py`에서 Ollama 로직 분리 → `backends/ollama.py`
3. **backends/vllm.py**: vLLM OpenAI API 클라이언트 신규 작성
4. **tracks/base.py**: BaseTrack ABC 작성, 7개 트랙이 상속하도록 수정
5. **runner.py**: 경량 라우터로 축소, Backend 인스턴스 사용
6. **judge.py**: config에서 judge 모델을 읽도록 수정 (기존 하드코딩 제거)
7. **scoring.py**: `build_scorecard()`에 models 파라미터 추가

### Phase 3: 마무리

1. `pyproject.toml` 작성 (pip installable)
2. `README.md` 작성 (한국어 + 영어)
3. `configs/examples/` 예시 YAML 파일들
4. 새 GitHub repo `ko-llm-bench-suite` 생성 + push
5. 기존 `frankenstallm_test` remote 연결 해제

---

## 4. Scope Boundaries

### In Scope
- 기존 7개 트랙 + 26개 시각화 보존
- Ollama + vLLM 백엔드
- YAML 설정 기반 실행
- 패키지명/디렉토리 리네임
- CLI 인터페이스 개선
- README/문서화

### Out of Scope (향후)
- pip 패키지 PyPI 배포
- Web UI / 대시보드
- HuggingFace 백엔드
- 새로운 벤치마크 트랙 추가
- Docker 컨테이너
- CI/CD 자동 평가

---

## 5. Verification

1. `python kobench.py --config configs/examples/frankenstallm.yaml --report-only` → 기존 결과로 리포트 재생성 성공
2. `python kobench.py --config configs/examples/quick_eval.yaml --tracks 1 --models "qwen2.5:3b"` → 단일 모델 T1 평가 성공
3. 기존 7개 트랙 `run()` 함수가 BaseTrack 인터페이스로 정상 동작
4. `examples/frankenstallm/` 디렉토리에 112셀 결과 + 26개 차트 + 보고서 보존 확인
5. GitHub `ko-llm-bench-suite` public repo에 정상 push
