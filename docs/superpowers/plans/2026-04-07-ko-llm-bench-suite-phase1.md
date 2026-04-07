# ko-llm-bench-suite Phase 1: 구조 재편 + 코어 리팩토링

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** frankenstallm_test를 ko-llm-bench-suite로 리네임하고, eval_framework → kobench 패키지로 재편, YAML 설정 도입, Backend 추상화 레이어 구축

**Architecture:** 점진적 리팩토링. 기존 코드 80%+ 보존하면서 패키지명/디렉토리 변경 → import 일괄 치환 → YAML config 도입 → Backend ABC 추출 순서로 진행. 각 태스크 후 테스트 통과 확인.

**Tech Stack:** Python 3.12, PyYAML, Ollama API, pytest, git

**Spec:** `docs/superpowers/specs/2026-04-07-ko-llm-bench-suite-design.md`

---

## File Structure (Phase 1 최종 상태)

```
ko-llm-bench-suite/                   # renamed from frankenstallm_test
├── kobench/                           # renamed from eval_framework
│   ├── __init__.py
│   ├── config.py                      # YAML loader + validation
│   ├── backends/                      # NEW: inference backend abstraction
│   │   ├── __init__.py                # get_backend() factory
│   │   ├── base.py                    # InferenceBackend ABC
│   │   └── ollama.py                  # OllamaBackend (extracted from runner.py)
│   ├── runner.py                      # Slimmed: backend router + checkpoint
│   ├── judge.py                       # Retained, config refs updated
│   ├── scoring.py                     # Retained, hardcoded model list removed
│   ├── report.py                      # Retained
│   └── tracks/                        # Retained, imports updated
│       ├── __init__.py
│       ├── korean_bench.py            # renamed from track1_korean_bench.py
│       ├── ko_bench.py                # renamed from track2_ko_bench.py
│       ├── korean_deep.py             # renamed from track3_korean_deep.py
│       ├── code_math.py               # renamed from track4_code_math.py
│       ├── consistency.py             # renamed from track5_consistency.py
│       ├── performance.py             # renamed from track6_performance.py
│       └── pairwise.py               # renamed from track7_pairwise.py
├── benchmarks/                        # renamed from data/
├── configs/                           # NEW
│   ├── default.yaml
│   └── examples/
│       └── frankenstallm.yaml
├── examples/                          # NEW: legacy data
│   └── frankenstallm/
│       ├── results/
│       ├── reports/
│       └── visualizations/
├── tests/                             # Retained, imports updated
├── kobench.py                         # renamed from run_evaluation.py
├── pyproject.toml                     # NEW
└── README.md                          # Rewritten
```

---

### Task 1: 레거시 데이터를 examples/frankenstallm/로 이동

**Files:**
- Create: `examples/frankenstallm/README.md`
- Move: `results/` → `examples/frankenstallm/results/`
- Move: `reports/` → `examples/frankenstallm/reports/`
- Move: `generate_visualizations*.py` → `examples/frankenstallm/`
- Move: `COMPARISON_*.md`, `SUMMARY_*.md` → `examples/frankenstallm/`

- [ ] **Step 1: examples 디렉토리 생성 + 레거시 파일 이동**

```bash
mkdir -p examples/frankenstallm

# 결과/리포트 이동
git mv results examples/frankenstallm/results
git mv reports examples/frankenstallm/reports

# 시각화 스크립트 이동
git mv generate_visualizations.py examples/frankenstallm/
git mv generate_visualizations_detailed.py examples/frankenstallm/
git mv generate_visualizations_extra.py examples/frankenstallm/

# 레거시 문서 이동
git mv COMPARISON_BEST_CASES.md examples/frankenstallm/ 2>/dev/null; true
git mv COMPARISON_NEW_BEST_CASES.md examples/frankenstallm/ 2>/dev/null; true
git mv SUMMARY_*.md examples/frankenstallm/ 2>/dev/null; true

# 빈 results/reports 디렉토리 재생성 (새 프로젝트용)
mkdir -p results reports
```

- [ ] **Step 2: examples README 작성**

```markdown
# Frankenstallm Evaluation (Legacy)

16모델 × 7트랙 평가 결과 (2026-03~04).
ko-llm-bench-suite의 사용 예시로 보존.

- `results/` — 112셀 전체 결과 JSON
- `reports/` — 26개 시각화 차트 + 종합 보고서
- `reports/EVALUATION_REPORT_FULL.md` — 최종 보고서
```

- [ ] **Step 3: 커밋**

```bash
git add -A
git commit -m "refactor: move legacy frankenstallm data to examples/"
```

---

### Task 2: eval_framework → kobench 패키지 리네임

**Files:**
- Move: `eval_framework/` → `kobench/`
- Move: `eval_framework/tracks/track1_korean_bench.py` → `kobench/tracks/korean_bench.py`
- Move: (나머지 6개 트랙도 동일 패턴)

- [ ] **Step 1: 패키지 디렉토리 리네임**

```bash
git mv eval_framework kobench

# 트랙 파일 리네임 (track1_ prefix 제거)
cd kobench/tracks
git mv track1_korean_bench.py korean_bench.py
git mv track2_ko_bench.py ko_bench.py
git mv track3_korean_deep.py korean_deep.py
git mv track4_code_math.py code_math.py
git mv track5_consistency.py consistency.py
git mv track6_performance.py performance.py
git mv track7_pairwise.py pairwise.py
cd ../..
```

- [ ] **Step 2: 모든 Python 파일에서 import 경로 일괄 치환**

```bash
# eval_framework → kobench
find . -name "*.py" -not -path "./examples/*" \
  -exec sed -i 's/from eval_framework/from kobench/g; s/import eval_framework/import kobench/g' {} \;

# 트랙 모듈 경로 치환 (run_evaluation.py 내부의 track_map)
# track1_korean_bench → korean_bench 등
sed -i 's/track1_korean_bench/korean_bench/g; s/track2_ko_bench/ko_bench/g; s/track3_korean_deep/korean_deep/g; s/track4_code_math/code_math/g; s/track5_consistency/consistency/g; s/track6_performance/performance/g; s/track7_pairwise/pairwise/g' run_evaluation.py
```

- [ ] **Step 3: run_evaluation.py → kobench.py 리네임**

```bash
git mv run_evaluation.py kobench.py
```

- [ ] **Step 4: data/ → benchmarks/ 리네임**

```bash
git mv data benchmarks
# config.py에서 DATA_DIR 경로 업데이트
sed -i 's/DATA_DIR = PROJECT_ROOT \/ "data"/DATA_DIR = PROJECT_ROOT \/ "benchmarks"/' kobench/config.py
```

- [ ] **Step 5: 테스트 실행 확인**

```bash
python -m pytest tests/ -x -q 2>&1 | tail -5
```

Expected: 대부분 테스트 통과 (Ollama 미실행 시 integration 테스트는 skip 가능)

- [ ] **Step 6: 커밋**

```bash
git add -A
git commit -m "refactor: rename eval_framework → kobench, data → benchmarks, track files simplified"
```

---

### Task 3: YAML 설정 시스템 도입

**Files:**
- Create: `configs/default.yaml`
- Create: `configs/examples/frankenstallm.yaml`
- Modify: `kobench/config.py` — YAML 로더 추가
- Test: `tests/unit/test_config.py` — YAML 로딩 테스트

- [ ] **Step 1: PyYAML 의존성 추가**

```bash
pip install pyyaml
echo "pyyaml>=6.0" >> requirements.txt
```

- [ ] **Step 2: default.yaml 작성**

Create `configs/default.yaml`:

```yaml
project:
  name: "Korean LLM Evaluation"
  output_dir: "./results"
  reports_dir: "./reports"

backend:
  type: ollama
  url: "http://localhost:11434"
  remote: false

models: []  # Must be specified by user

tracks:
  enabled: [1, 2, 3, 4, 5, 6, 7]
  order: [6, 1, 3, 2, 4, 5, 7]

judge:
  dual_enabled: true
  primary:
    model: "qwen2.5:7b-instruct"
    weight: 0.6
  secondary:
    model: "exaone3.5:7.8b"
    weight: 0.4
  timeout: 120

sampling:
  default:
    temperature: 0.7
    top_p: 0.9
    repeat_penalty: 1.2
    num_predict: 512
    num_ctx: 4096
  benchmark:
    temperature: 0.0
    top_p: 1.0
    repeat_penalty: 1.0
    num_predict: 256
    num_ctx: 4096

retry:
  max_retries: 3
  backoff_base: 5
  cooldown_between_models: 10
  cooldown_between_tests: 1
```

- [ ] **Step 3: frankenstallm.yaml 예시 작성**

Create `configs/examples/frankenstallm.yaml`:

```yaml
project:
  name: "Frankenstallm 3B Evaluation"
  output_dir: "./examples/frankenstallm/results"
  reports_dir: "./examples/frankenstallm/reports"

backend:
  type: ollama
  url: "http://localhost:11434"

models:
  - name: "frankenstallm-3b:latest"
    tags: [custom, v1]
  - name: "frankenstallm-3b:Q8_0"
    tags: [custom, v1]
  - name: "frankenstallm-3b-v2:latest"
    tags: [custom, v2]
  - name: "frankenstallm-3b-v2:Q8_0"
    tags: [custom, v2]
  - name: "qwen2.5:3b"
    tags: [baseline]
  - name: "gemma3:4b"
    tags: [baseline]
  - name: "phi4-mini"
    tags: [baseline]
  - name: "exaone3.5:2.4b"
    tags: [baseline]
  - name: "llama3.2:3b"
    tags: [baseline]
  - name: "llama3.1:8b-instruct-q8_0"
    tags: [baseline, 8b]
  - name: "ingu627/exaone4.0:1.2b"
    tags: [baseline]
  - name: "deepseek-r1:1.5b"
    tags: [baseline]
  - name: "evafrill-mo-3b-slerp"
    tags: [custom, evafrill]
```

- [ ] **Step 4: config.py에 YAML 로더 추가**

`kobench/config.py` 상단에 추가:

```python
import yaml
from pathlib import Path

def load_config(config_path: str = None) -> dict:
    """YAML 설정 파일 로드. 없으면 기본값 사용."""
    defaults_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    
    config = {}
    if defaults_path.exists():
        with open(defaults_path) as f:
            config = yaml.safe_load(f) or {}
    
    if config_path:
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}
        _deep_merge(config, user_config)
    
    return config

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
```

기존 `config.py`의 하드코딩된 값들은 그대로 두되, `load_config()`로 오버라이드 가능하게 함. 기존 코드와의 호환성 유지.

- [ ] **Step 5: kobench.py에 --config 옵션 추가**

```python
parser.add_argument("--config", type=str, default=None,
                    help="YAML 설정 파일 경로 (예: configs/default.yaml)")
```

`main()`에서 `args.config`가 주어지면 `load_config(args.config)` 호출하여 모델 목록 등을 오버라이드.

- [ ] **Step 6: 테스트**

```bash
python -m pytest tests/unit/test_config.py -v
```

- [ ] **Step 7: 커밋**

```bash
git add -A
git commit -m "feat: add YAML configuration system with default.yaml and examples"
```

---

### Task 4: Backend 추상화 레이어

**Files:**
- Create: `kobench/backends/__init__.py`
- Create: `kobench/backends/base.py`
- Create: `kobench/backends/ollama.py`
- Modify: `kobench/runner.py` — Backend 사용으로 전환

- [ ] **Step 1: backends 패키지 생성**

```bash
mkdir -p kobench/backends
```

- [ ] **Step 2: base.py — InferenceBackend ABC 작성**

Create `kobench/backends/base.py`:

```python
"""추론 백엔드 추상 베이스 클래스."""

from abc import ABC, abstractmethod


class InferenceBackend(ABC):
    """모든 추론 백엔드가 구현해야 하는 인터페이스."""

    def __init__(self, url: str, remote: bool = False, **kwargs):
        self.url = url
        self.remote = remote

    @abstractmethod
    def generate(self, model: str, prompt: str, system: str = "",
                 options: dict = None, timeout: int = None) -> dict:
        """텍스트 생성. Returns: {response, eval_count, tokens_per_sec, wall_time_s, error}"""

    @abstractmethod
    def chat(self, model: str, messages: list[dict],
             options: dict = None, timeout: int = None) -> dict:
        """멀티턴 대화. Same return format as generate()."""

    @abstractmethod
    def load_model(self, model: str) -> bool:
        """모델 로드/웜업. Returns success."""

    @abstractmethod
    def unload_model(self, model: str) -> None:
        """모델 메모리 해제."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """사용 가능한 모델 목록."""

    @abstractmethod
    def health_check(self) -> bool:
        """백엔드 연결 상태 확인."""
```

- [ ] **Step 3: ollama.py — OllamaBackend 작성**

Create `kobench/backends/ollama.py`:

기존 `runner.py`에서 Ollama API 호출 로직 (`generate()`, `chat()`, `warmup_model()`, `unload_model()`, `wait_for_ollama()`, `get_loaded_models()`)을 `OllamaBackend` 클래스로 이동.

```python
"""Ollama 추론 백엔드."""

import time
import requests
from .base import InferenceBackend
from kobench import config


class OllamaBackend(InferenceBackend):
    """Ollama API 기반 추론 백엔드."""

    def __init__(self, url: str = None, remote: bool = False, **kwargs):
        super().__init__(url or config.OLLAMA_BASE_URL, remote)
        self.api_generate = f"{self.url}/api/generate"
        self.api_chat = f"{self.url}/api/chat"
        self.api_tags = f"{self.url}/api/tags"
        self.api_ps = f"{self.url}/api/ps"

    def generate(self, model, prompt, system="", options=None, timeout=None):
        # 기존 runner.generate() 로직 그대로 이동
        ...

    def chat(self, model, messages, options=None, timeout=None):
        # 기존 runner.chat() 로직 그대로 이동
        ...

    def load_model(self, model):
        # 기존 runner.warmup_model() 로직
        ...

    def unload_model(self, model):
        # 기존 runner.unload_model() 로직
        ...

    def list_models(self):
        resp = requests.get(self.api_tags, timeout=10)
        return [m['name'] for m in resp.json().get('models', [])]

    def health_check(self):
        # 기존 runner.wait_for_ollama() 로직
        ...
```

- [ ] **Step 4: __init__.py — Backend factory**

Create `kobench/backends/__init__.py`:

```python
"""추론 백엔드 팩토리."""

from .base import InferenceBackend
from .ollama import OllamaBackend


def get_backend(backend_type: str = "ollama", **kwargs) -> InferenceBackend:
    """설정에 따라 적절한 백엔드 인스턴스 반환."""
    backends = {
        "ollama": OllamaBackend,
    }
    if backend_type not in backends:
        raise ValueError(f"Unknown backend: {backend_type}. Available: {list(backends.keys())}")
    return backends[backend_type](**kwargs)
```

- [ ] **Step 5: runner.py를 경량 라우터로 축소**

기존 `runner.py`에서 Ollama 직접 호출 코드를 제거하고, `OllamaBackend`를 사용하도록 전환. 체크포인트/모델전환/retry 로직은 runner.py에 유지.

핵심 변경: `generate()`, `chat()` 함수가 내부적으로 `backend.generate()`, `backend.chat()`를 호출.

```python
# runner.py (축소 버전)
from kobench.backends import get_backend

_backend = None

def _get_backend():
    global _backend
    if _backend is None:
        _backend = get_backend(config.BACKEND_TYPE, url=config.OLLAMA_BASE_URL, remote=config.OLLAMA_REMOTE)
    return _backend

def generate(model, prompt, system="", options=None, timeout=None):
    return _get_backend().generate(model, prompt, system, options, timeout)
```

이 방식으로 기존 트랙 코드(`runner.generate()` 호출)는 수정 불필요.

- [ ] **Step 6: 테스트**

```bash
python -m pytest tests/ -x -q 2>&1 | tail -10
```

- [ ] **Step 7: 커밋**

```bash
git add -A
git commit -m "refactor: extract OllamaBackend from runner.py with InferenceBackend ABC"
```

---

### Task 5: pyproject.toml + README + 새 GitHub repo

**Files:**
- Create: `pyproject.toml`
- Rewrite: `README.md`
- Create: `LICENSE`

- [ ] **Step 1: pyproject.toml 작성**

```toml
[project]
name = "ko-llm-bench-suite"
version = "0.1.0"
description = "한국어 LLM 종합 벤치마크 평가 도구"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
dependencies = [
    "requests>=2.28",
    "pyyaml>=6.0",
    "numpy>=1.24",
    "pandas>=2.0",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "scipy>=1.10",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-mock>=3.0"]

[project.scripts]
kobench = "kobench:main"
```

- [ ] **Step 2: README.md 작성**

한국어 + 영어 README. Quick start, 설치법, YAML 설정 예시, 트랙 설명, 결과 예시 포함.

- [ ] **Step 3: 새 GitHub repo 생성 + remote 변경**

```bash
# 기존 remote 확인 후 제거
git remote -v
git remote remove origin

# 새 public repo 생성
gh repo create pathcosmos/ko-llm-bench-suite --public --description "한국어 LLM 종합 벤치마크 평가 도구 (Korean LLM Benchmark Suite)"

# 새 remote 연결
git remote add origin https://github.com/pathcosmos/ko-llm-bench-suite.git

# push
git push -u origin master
```

- [ ] **Step 4: 커밋**

```bash
git add -A
git commit -m "feat: add pyproject.toml, README, prepare for ko-llm-bench-suite release"
git push origin master
```

---

## Verification Checklist

- [ ] `python kobench.py --help` 정상 출력
- [ ] `python -m pytest tests/ -x` 기존 테스트 통과
- [ ] `from kobench import config` import 성공
- [ ] `from kobench.backends import get_backend` import 성공
- [ ] `examples/frankenstallm/results/` 에 112셀 결과 보존
- [ ] `examples/frankenstallm/reports/visualizations/` 에 26개 차트 보존
- [ ] GitHub `ko-llm-bench-suite` repo에 정상 push
- [ ] 기존 `frankenstallm_test` remote 연결 해제 확인
