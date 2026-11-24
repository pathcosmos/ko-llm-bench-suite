"""OpenAI 호환 서버 백엔드: vLLM(`vllm serve`)과 llama.cpp(`llama-server`).

기존 트랙(T2–T7)은 runner.generate/chat을 통해 Ollama REST를 직접 불렀다. backend.type이 vllm|llamacpp이면
runner가 이 백엔드로 보낸다. 반환 형식은 InferenceBackend 규약(Ollama와 같은 키)을 따른다.

- generate(): system + user 메시지로 /v1/chat/completions (서버가 모델 chat 템플릿 적용 = Ollama /api/generate의 템플릿 적용과 대응).
  raw_models에 든 모델(base 등 chat 템플릿 없는 모델)은 /v1/completions에 프롬프트를 그대로 넣는다.
- 옵션 변환 (Ollama → OpenAI): num_predict→max_tokens, repeat_penalty→repetition_penalty(vLLM) / repeat_penalty(llama-server),
  temperature, top_p, top_k, seed, stop. 처리 속도(tokens_per_sec)는 클라이언트 측 wall 시간 기준이다(엔진 보고값이 없으므로).
- 모델 적재·해제는 서버가 관리한다(서버 하나 = 모델 하나). load_model은 서버의 /v1/models에 그 이름이 있는지 확인한다.
"""

from __future__ import annotations

import time
from typing import Optional

import requests

from .base import InferenceBackend


class OpenAICompatBackend(InferenceBackend):
    flavor = "openai"

    def __init__(self, url: str = "http://localhost:8000", remote: bool = True, max_retries: int = 3,
                 retry_backoff_base: int = 5, default_timeout: int = 300, default_options: Optional[dict] = None,
                 raw_models: Optional[list[str]] = None, api_key: Optional[str] = None, **kwargs):
        super().__init__(url=url.rstrip("/"), remote=remote, **kwargs)
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.default_timeout = default_timeout
        self.default_options = default_options or {}
        self.raw_models = set(raw_models or [])
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    # ── 옵션 변환 ────────────────────────────────────────────────────────────
    def _params(self, options: Optional[dict]) -> dict:
        o = {**self.default_options, **(options or {})}
        p = {}
        if "temperature" in o:
            p["temperature"] = float(o["temperature"])
        if "top_p" in o:
            p["top_p"] = float(o["top_p"])
        if "top_k" in o:
            p["top_k"] = int(o["top_k"])
        if "seed" in o:
            p["seed"] = int(o["seed"])
        if "num_predict" in o and int(o["num_predict"]) > 0:
            p["max_tokens"] = int(o["num_predict"])
        if o.get("stop"):
            p["stop"] = list(o["stop"]) if not isinstance(o["stop"], str) else [o["stop"]]
        if "repeat_penalty" in o:
            p[self._rep_key] = float(o["repeat_penalty"])
        return p

    _rep_key = "repetition_penalty"

    # ── 요청 ────────────────────────────────────────────────────────────────
    def _post(self, path: str, body: dict, timeout: int) -> dict:
        last = None
        for attempt in range(self.max_retries):
            if attempt:
                time.sleep(self.retry_backoff_base * (2 ** (attempt - 1)))
            try:
                t0 = time.time()
                r = requests.post(self.url + path, json=body, timeout=timeout, headers=self.headers)
                wall = time.time() - t0
                if r.status_code != 200:
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                    if 400 <= r.status_code < 500:
                        break                       # 요청 오류는 재시도해도 같다
                    continue
                d = r.json()
                d["_wall"] = wall
                return d
            except requests.exceptions.Timeout:
                last = f"타임아웃 ({timeout}s)"
            except requests.exceptions.ConnectionError:
                last = f"{self.flavor} 서버 연결 실패"
            except Exception as e:  # noqa: BLE001
                last = str(e)
        return {"_error": last}

    def _result(self, d: dict, text_key: str) -> dict:
        if "_error" in d:
            return self._error_result(d["_error"])
        ch = (d.get("choices") or [{}])[0]
        text = ch["message"].get("content") or "" if text_key == "message" else ch.get("text", "")
        u = d.get("usage") or {}
        n_out, n_in, wall = int(u.get("completion_tokens", 0)), int(u.get("prompt_tokens", 0)), float(d["_wall"])
        return {"response": text, "eval_count": n_out, "eval_duration_s": wall, "prompt_eval_count": n_in,
                "prompt_eval_duration_s": 0.0, "total_duration_s": wall, "wall_time_s": wall,
                "tokens_per_sec": n_out / wall if wall > 0 else 0.0, "finish_reason": ch.get("finish_reason"), "error": None}

    def generate(self, model: str, prompt: str, system: str = "", options: Optional[dict] = None,
                 timeout: Optional[int] = None) -> dict:
        timeout = timeout or self.default_timeout
        if model in self.raw_models:
            body = {"model": model, "prompt": (system + "\n\n" + prompt) if system else prompt, **self._params(options)}
            return self._result(self._post("/v1/completions", body, timeout), "text")
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self.chat(model, msgs, options, timeout)

    def chat(self, model: str, messages: list[dict], options: Optional[dict] = None, timeout: Optional[int] = None) -> dict:
        timeout = timeout or self.default_timeout
        body = {"model": model, "messages": messages, **self._params(options)}
        return self._result(self._post("/v1/chat/completions", body, timeout), "message")

    # ── 관리 ────────────────────────────────────────────────────────────────
    def list_models(self) -> list[str]:
        try:
            r = requests.get(self.url + "/v1/models", timeout=10, headers=self.headers)
            return [m["id"] for m in r.json().get("data", [])]
        except Exception:  # noqa: BLE001
            return []

    def health_check(self) -> bool:
        try:
            return requests.get(self.url + "/health", timeout=5).status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def load_model(self, model: str) -> bool:
        if not self.health_check():
            return False
        served = self.list_models()
        return not served or model in served or self.flavor == "llamacpp"   # llama-server는 이름과 무관하게 적재된 모델로 답한다

    def unload_model(self, model: str) -> None:
        return None


class VllmBackend(OpenAICompatBackend):
    flavor = "vllm"
    _rep_key = "repetition_penalty"


class LlamaCppBackend(OpenAICompatBackend):
    flavor = "llamacpp"
    _rep_key = "repeat_penalty"
