"""추론 백엔드 추상 베이스 클래스."""

from abc import ABC, abstractmethod
from typing import Optional


class InferenceBackend(ABC):
    """모든 추론 백엔드가 구현해야 하는 인터페이스.

    각 백엔드는 텍스트 생성, 대화, 모델 관리 기능을 제공한다.
    반환 딕셔너리 형식:
        {
            "response": str,
            "eval_count": int,
            "eval_duration_s": float,
            "prompt_eval_count": int,
            "prompt_eval_duration_s": float,
            "total_duration_s": float,
            "tokens_per_sec": float,
            "wall_time_s": float,
            "error": str | None,
        }
    """

    def __init__(self, url: str, remote: bool = False, **kwargs):
        self.url = url
        self.remote = remote

    @abstractmethod
    def generate(self, model: str, prompt: str, system: str = "",
                 options: Optional[dict] = None,
                 timeout: Optional[int] = None) -> dict:
        """텍스트 생성.

        Returns:
            dict with response, eval_count, eval_duration_s, tokens_per_sec,
            wall_time_s, error 등.
        """

    @abstractmethod
    def chat(self, model: str, messages: list[dict],
             options: Optional[dict] = None,
             timeout: Optional[int] = None) -> dict:
        """멀티턴 대화. generate()와 동일한 반환 형식."""

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

    @staticmethod
    def _error_result(error_msg: str) -> dict:
        """에러 발생 시 표준 반환 딕셔너리."""
        return {
            "response": "",
            "eval_count": 0,
            "eval_duration_s": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration_s": 0,
            "total_duration_s": 0,
            "wall_time_s": 0,
            "tokens_per_sec": 0,
            "error": error_msg,
        }
