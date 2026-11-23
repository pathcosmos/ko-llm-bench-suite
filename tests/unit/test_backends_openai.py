"""OpenAI 호환 백엔드(vLLM / llama-server)와 runner·judge·config 연결."""

from unittest.mock import MagicMock, patch

import pytest

from kobench import config, runner
from kobench.backends import BACKEND_TYPES, LlamaCppBackend, VllmBackend, get_backend


def _resp(json_body, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_body
    r.text = str(json_body)
    return r


CHAT_OK = {"choices": [{"message": {"content": "안녕하세요"}, "finish_reason": "stop"}],
           "usage": {"prompt_tokens": 12, "completion_tokens": 4}}
COMP_OK = {"choices": [{"text": " 이어쓰기", "finish_reason": "length"}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


def test_factory_has_all_backends():
    assert set(BACKEND_TYPES) == {"ollama", "vllm", "llamacpp"}
    assert isinstance(get_backend("vllm", url="http://x"), VllmBackend)
    with pytest.raises(ValueError):
        get_backend("tgi")


@pytest.mark.parametrize("cls,rep_key", [(VllmBackend, "repetition_penalty"), (LlamaCppBackend, "repeat_penalty")])
def test_option_mapping(cls, rep_key):
    be = cls(url="http://x")
    p = be._params({"temperature": 0.7, "top_p": 0.9, "top_k": 50, "repeat_penalty": 1.2, "num_predict": 256, "seed": 1,
                    "stop": "</s>"})
    assert p == {"temperature": 0.7, "top_p": 0.9, "top_k": 50, rep_key: 1.2, "max_tokens": 256, "seed": 1, "stop": ["</s>"]}


@patch("kobench.backends.openai_compat.requests.post")
def test_generate_uses_chat_with_system(mock_post):
    mock_post.return_value = _resp(CHAT_OK)
    r = VllmBackend(url="http://x/").generate("m", "질문", system="시스템", options={"temperature": 0})
    url, body = mock_post.call_args[0][0], mock_post.call_args[1]["json"]
    assert url == "http://x/v1/chat/completions"
    assert body["messages"] == [{"role": "system", "content": "시스템"}, {"role": "user", "content": "질문"}]
    assert r["response"] == "안녕하세요" and r["eval_count"] == 4 and r["prompt_eval_count"] == 12 and r["error"] is None


@patch("kobench.backends.openai_compat.requests.post")
def test_raw_models_use_completions(mock_post):
    mock_post.return_value = _resp(COMP_OK)
    r = LlamaCppBackend(url="http://x", raw_models=["base"]).generate("base", "옛날 옛적에")
    assert mock_post.call_args[0][0] == "http://x/v1/completions"
    assert mock_post.call_args[1]["json"]["prompt"] == "옛날 옛적에" and r["response"] == " 이어쓰기"


@patch("kobench.backends.openai_compat.time.sleep")
@patch("kobench.backends.openai_compat.requests.post")
def test_client_error_not_retried_server_error_retried(mock_post, _sleep):
    mock_post.return_value = _resp({"error": "bad"}, status=400)
    r = VllmBackend(url="http://x", max_retries=3).chat("m", [{"role": "user", "content": "q"}])
    assert r["error"].startswith("HTTP 400") and mock_post.call_count == 1
    mock_post.reset_mock()
    mock_post.side_effect = [_resp({}, status=503), _resp(CHAT_OK)]
    r = VllmBackend(url="http://x", max_retries=3).chat("m", [{"role": "user", "content": "q"}])
    assert r["error"] is None and mock_post.call_count == 2


def test_runner_routes_to_configured_backend(monkeypatch):
    fake = MagicMock()
    fake.generate.return_value = {"response": "ok", "error": None}
    fake.chat.return_value = {"response": "chat-ok", "error": None}
    monkeypatch.setattr(config, "BACKEND_TYPE", "vllm")
    monkeypatch.setattr(config, "BACKEND_URL", "http://x")
    monkeypatch.setattr(runner, "get_inference_backend", lambda: fake)
    runner._backend_cache.clear()
    try:
        assert runner.generate("qwen3-4b", "p")["response"] == "ok"
        assert runner.chat("qwen3-4b", [{"role": "user", "content": "q"}])["response"] == "chat-ok"
        runner.unload_all_models()                   # 외부 서버 백엔드에서는 아무것도 하지 않는다
        fake.health_check.return_value = True
        assert runner.wait_for_ollama(max_wait=1) is True
    finally:
        runner._backend_cache.clear()


def test_validate_config_accepts_new_backends():
    assert config.validate_config({"backend": {"type": "vllm", "url": "http://x"}}) == []
    assert any("url" in e for e in config.validate_config({"backend": {"type": "llamacpp"}}))
    assert any("미지원" in e for e in config.validate_config({"backend": {"type": "tgi", "url": "http://x"}}))


def test_custom_tracks_marked():
    assert {"track3", "track4", "track5"} <= set(config.CUSTOM_TRACKS)
