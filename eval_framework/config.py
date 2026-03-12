"""
평가 프레임워크 설정 — 모델 목록, API 설정, 타임아웃, 샘플링 파라미터
"""

import os
from pathlib import Path

# ── Ollama 모델 저장 경로 (반드시 프로세스 시작 전에 설정) ────────────────────
os.environ.setdefault("OLLAMA_MODELS", "/var/snap/ollama/common/models")

# ── 경로 ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ── Ollama API ────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_GENERATE = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_API_CHAT = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_API_SHOW = f"{OLLAMA_BASE_URL}/api/show"
OLLAMA_API_PS = f"{OLLAMA_BASE_URL}/api/ps"

# ── LLM-as-Judge ─────────────────────────────────────────────────────────────
# Claude Code CLI (`claude -p`)를 사용하므로 별도 API 키 불필요

# ── 모델 목록 ─────────────────────────────────────────────────────────────────
# v1 모델은 GGUF 토크나이저 결함(SPM byte_to_token 누락)으로 추론 불가
# llama.cpp 계열 엔진 모두에서 SIGABRT 발생 → v2만 평가
FRANKENSTALLM_V1_MODELS = [
    "frankenstallm-3b-Q4_K_M",
    "frankenstallm-3b-Q8_0",
    "frankenstallm-3b-f16",
]
FRANKENSTALLM_MODELS = [
    "frankenstallm-3b-v2-Q4_K_M",
    "frankenstallm-3b-v2-Q8_0",
    "frankenstallm-3b-v2-f16",
]

COMPARISON_MODELS = [
    "qwen2.5:3b",
    "gemma3:4b",
    "phi4-mini",
    "exaone3.5:2.4b",
    "llama3.2:3b",
    "llama3.1:8b-instruct-q8_0",
    "ingu627/exaone4.0:1.2b",
]

ALL_MODELS = FRANKENSTALLM_MODELS + COMPARISON_MODELS

# ── GPU 가용성 확인 ───────────────────────────────────────────────────────────
def _gpu_available() -> bool:
    """nvidia-smi로 GPU 사용 가능 여부 확인"""
    import subprocess as sp
    try:
        r = sp.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                   capture_output=True, text=True, timeout=5)
        return r.returncode == 0 and len(r.stdout.strip()) > 0
    except Exception:
        return False

GPU_AVAILABLE = _gpu_available()

# ── 모델별 타임아웃 (초) ──────────────────────────────────────────────────────
# CPU 모드에서는 타임아웃을 2배로 늘림
_TIMEOUT_MULTIPLIER = 1 if GPU_AVAILABLE else 2
MODEL_TIMEOUTS = {name: 120 * _TIMEOUT_MULTIPLIER for name in ALL_MODELS}
for name in ALL_MODELS:
    if "f16" in name:
        MODEL_TIMEOUTS[name] = 300 * _TIMEOUT_MULTIPLIER
    elif "Q8_0" in name:
        MODEL_TIMEOUTS[name] = 180 * _TIMEOUT_MULTIPLIER
# 8B 모델은 로딩/추론이 더 오래 걸림
for name in ALL_MODELS:
    if "8b" in name.lower():
        MODEL_TIMEOUTS[name] = 360 * _TIMEOUT_MULTIPLIER

WARMUP_TIMEOUT = 360 * _TIMEOUT_MULTIPLIER  # 모델 최초 로딩 시

# ── 샘플링 파라미터 (frankenstallm ORPO 최적) ────────────────────────────────
SAMPLING_PARAMS = {
    "temperature": 0.7,
    "repeat_penalty": 1.2,
    "top_p": 0.9,
    "num_predict": 512,
    "num_ctx": 4096,
}

# 벤치마크(정확도 측정)용 — greedy decoding
BENCHMARK_SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repeat_penalty": 1.0,
    "num_predict": 256,
    "num_ctx": 4096,
}


# ── 안정성 설정 ───────────────────────────────────────────────────────────────
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 5  # 초: 5, 10, 20 ...
COOLDOWN_BETWEEN_MODELS = 10  # 모델 전환 시 대기 (초)
COOLDOWN_BETWEEN_TESTS = 1  # 테스트 간 대기 (초)
HEALTH_CHECK_INTERVAL = 30  # Ollama health check 주기 (초)

# ── Track별 설정 ──────────────────────────────────────────────────────────────
TRACK1_TASKS = [
    "kobest_boolq",
    "kobest_copa",
    "kobest_sentineg",
    "kobest_hellaswag",
]

TRACK2_CATEGORIES = [
    "writing", "roleplay", "reasoning", "math",
    "coding", "extraction", "stem", "humanities",
]

TRACK6_INPUT_LENGTHS = [100, 500, 1000, 2000]
TRACK6_CONCURRENT_LEVELS = [1, 2, 4]

TRACK7_NUM_PROMPTS = 20  # 쌍대비교용 대표 프롬프트 수
