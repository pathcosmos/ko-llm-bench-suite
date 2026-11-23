"""추론 백엔드 팩토리.

ollama   : Ollama REST (기존)
vllm     : vLLM OpenAI 호환 서버 (`vllm serve`, BF16 — 최신 아키텍처·A3B)
llamacpp : llama.cpp `llama-server` (GGUF)
EVAFRILL은 runner가 모델 이름으로 evafrill_runner(HTTP/subprocess, Goal 1 증분 디코딩 엔진 사용)로 보낸다.
"""

from .base import InferenceBackend
from .ollama import OllamaBackend
from .openai_compat import LlamaCppBackend, OpenAICompatBackend, VllmBackend

__all__ = ["InferenceBackend", "OllamaBackend", "OpenAICompatBackend", "VllmBackend", "LlamaCppBackend", "get_backend",
           "BACKEND_TYPES"]

BACKEND_TYPES = {"ollama": OllamaBackend, "vllm": VllmBackend, "llamacpp": LlamaCppBackend}


def get_backend(backend_type: str = "ollama", **kwargs) -> InferenceBackend:
    """설정에 따라 백엔드 인스턴스를 돌려준다. backend_type: ollama | vllm | llamacpp"""
    if backend_type not in BACKEND_TYPES:
        raise ValueError(f"Unknown backend: {backend_type}. Available: {list(BACKEND_TYPES)}")
    return BACKEND_TYPES[backend_type](**kwargs)
