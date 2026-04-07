"""Ollama 추론 백엔드 — Ollama REST API를 통한 텍스트 생성/대화."""

import time
from typing import Optional

import requests

from .base import InferenceBackend


# 기본 재시도/타임아웃 설정 (runner.py 와 동일한 기본값)
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BACKOFF_BASE = 5
_DEFAULT_TIMEOUT = 120
_DEFAULT_WARMUP_TIMEOUT = 360


class OllamaBackend(InferenceBackend):
    """Ollama REST API 기반 추론 백엔드.

    Parameters:
        url: Ollama 서버 base URL (예: http://localhost:11434)
        remote: 원격 서버 여부 (True이면 로컬 프로세스 제어 불가)
        max_retries: API 호출 최대 재시도 횟수
        retry_backoff_base: 재시도 백오프 기본 초 (지수 증가)
        default_timeout: 기본 요청 타임아웃 (초)
        warmup_timeout: 모델 로드 시 타임아웃 (초)
        default_options: 기본 샘플링 파라미터
    """

    def __init__(
        self,
        url: str = "http://localhost:11434",
        remote: bool = False,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff_base: int = _DEFAULT_RETRY_BACKOFF_BASE,
        default_timeout: int = _DEFAULT_TIMEOUT,
        warmup_timeout: int = _DEFAULT_WARMUP_TIMEOUT,
        default_options: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(url=url, remote=remote, **kwargs)
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.default_timeout = default_timeout
        self.warmup_timeout = warmup_timeout
        self.default_options = default_options or {}

        # 파생 URL
        self._api_generate = f"{self.url}/api/generate"
        self._api_chat = f"{self.url}/api/chat"
        self._api_ps = f"{self.url}/api/ps"
        self._api_tags = f"{self.url}/api/tags"

    # ── 핵심 추론 API ────────────────────────────────────────────────────────

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        options: Optional[dict] = None,
        timeout: Optional[int] = None,
    ) -> dict:
        """Ollama /api/generate 호출 (재시도 포함).

        Returns:
            dict with response, eval_count, eval_duration_s, tokens_per_sec 등.
        """
        if timeout is None:
            timeout = self.default_timeout
        if options is None:
            options = dict(self.default_options)

        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system

        last_error = None
        for attempt in range(self.max_retries):
            if attempt > 0:
                backoff = self.retry_backoff_base * (2 ** (attempt - 1))
                print(f"    \u21bb 재시도 {attempt + 1}/{self.max_retries} ({backoff}s 대기)")
                time.sleep(backoff)
                if not self.health_check():
                    last_error = "Ollama 서버 무응답"
                    continue

            try:
                start = time.time()
                resp = requests.post(
                    self._api_generate, json=payload, timeout=timeout,
                )
                wall_time = time.time() - start
                data = resp.json()

                eval_dur = data.get("eval_duration", 0)
                eval_cnt = data.get("eval_count", 0)

                return {
                    "response": data.get("response", ""),
                    "eval_count": eval_cnt,
                    "eval_duration_s": eval_dur / 1e9,
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "prompt_eval_duration_s": data.get("prompt_eval_duration", 0) / 1e9,
                    "total_duration_s": data.get("total_duration", 0) / 1e9,
                    "wall_time_s": wall_time,
                    "tokens_per_sec": eval_cnt / (eval_dur / 1e9) if eval_dur > 0 else 0,
                    "error": None,
                }
            except requests.exceptions.Timeout:
                last_error = f"타임아웃 ({timeout}s)"
            except requests.exceptions.ConnectionError:
                last_error = "Ollama 연결 실패"
            except Exception as e:
                last_error = str(e)

        return self._error_result(last_error)

    def chat(
        self,
        model: str,
        messages: list[dict],
        options: Optional[dict] = None,
        timeout: Optional[int] = None,
    ) -> dict:
        """Ollama /api/chat 호출 (멀티턴용, 재시도 포함)."""
        if timeout is None:
            timeout = self.default_timeout
        if options is None:
            options = dict(self.default_options)

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        last_error = None
        for attempt in range(self.max_retries):
            if attempt > 0:
                backoff = self.retry_backoff_base * (2 ** (attempt - 1))
                time.sleep(backoff)
                if not self.health_check():
                    last_error = "Ollama 서버 무응답"
                    continue

            try:
                start = time.time()
                resp = requests.post(
                    self._api_chat, json=payload, timeout=timeout,
                )
                wall_time = time.time() - start
                data = resp.json()

                msg = data.get("message", {})
                eval_dur = data.get("eval_duration", 0)
                eval_cnt = data.get("eval_count", 0)

                return {
                    "response": msg.get("content", ""),
                    "eval_count": eval_cnt,
                    "eval_duration_s": eval_dur / 1e9,
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "prompt_eval_duration_s": data.get("prompt_eval_duration", 0) / 1e9,
                    "total_duration_s": data.get("total_duration", 0) / 1e9,
                    "wall_time_s": wall_time,
                    "tokens_per_sec": eval_cnt / (eval_dur / 1e9) if eval_dur > 0 else 0,
                    "error": None,
                }
            except requests.exceptions.Timeout:
                last_error = f"타임아웃 ({timeout}s)"
            except requests.exceptions.ConnectionError:
                last_error = "Ollama 연결 실패"
            except Exception as e:
                last_error = str(e)

        return self._error_result(last_error)

    # ── 모델 관리 ────────────────────────────────────────────────────────────

    def load_model(self, model: str) -> bool:
        """모델 웜업 — 짧은 프롬프트로 모델을 VRAM에 올림."""
        try:
            r = requests.post(
                self._api_generate,
                json={
                    "model": model,
                    "prompt": "hello",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=self.warmup_timeout,
            )
            return r.status_code == 200
        except Exception as e:
            print(f"  \u26a0 Warmup 실패: {e}")
            return False

    def unload_model(self, model: str) -> None:
        """모델 언로드 — keep_alive=0으로 빈 요청."""
        try:
            requests.post(
                self._api_generate,
                json={"model": model, "keep_alive": 0},
                timeout=30,
            )
        except Exception:
            pass

    def list_models(self) -> list[str]:
        """현재 로딩된 모델 목록 (Ollama /api/ps)."""
        try:
            r = requests.get(self._api_ps, timeout=10)
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def list_available_models(self) -> list[str]:
        """설치된 전체 모델 목록 (Ollama /api/tags)."""
        try:
            r = requests.get(self._api_tags, timeout=10)
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def health_check(self) -> bool:
        """Ollama 서버 상태 확인."""
        try:
            r = requests.get(f"{self.url}/", timeout=5)
            return r.status_code == 200
        except Exception:
            return False
