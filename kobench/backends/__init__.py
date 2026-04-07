"""추론 백엔드 팩토리."""

from .base import InferenceBackend
from .ollama import OllamaBackend

__all__ = ["InferenceBackend", "OllamaBackend", "get_backend"]


def get_backend(backend_type: str = "ollama", **kwargs) -> InferenceBackend:
    """설정에 따라 적절한 백엔드 인스턴스 반환.

    Args:
        backend_type: 백엔드 종류 ("ollama", 향후 "vllm" 등)
        **kwargs: 백엔드 생성자에 전달할 인자

    Returns:
        InferenceBackend 인스턴스
    """
    backends = {
        "ollama": OllamaBackend,
    }
    if backend_type not in backends:
        raise ValueError(
            f"Unknown backend: {backend_type}. "
            f"Available: {list(backends.keys())}"
        )
    return backends[backend_type](**kwargs)
