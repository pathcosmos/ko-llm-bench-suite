"""eval_framework/runner.py 단위 테스트"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import requests

from eval_framework import runner


# ═══════════════════════════════════════════════════════════════════════════════
# generate 테스트 (7 케이스)
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerate:

    @patch("eval_framework.runner.evafrill_runner.is_evafrill", return_value=False)
    @patch("eval_framework.runner.requests.post")
    def test_success(self, mock_post, mock_is_eva):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": "답변입니다",
            "eval_count": 20,
            "eval_duration": 2_000_000_000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 500_000_000,
            "total_duration": 2_500_000_000,
        }
        mock_post.return_value = mock_resp

        result = runner.generate("test-model", "테스트 프롬프트")
        assert result["response"] == "답변입니다"
        assert result["eval_count"] == 20
        assert result["eval_duration_s"] == 2.0
        assert result["tokens_per_sec"] == 10.0
        assert result["error"] is None

    @patch("eval_framework.runner.evafrill_runner.is_evafrill", return_value=False)
    @patch("eval_framework.runner.wait_for_ollama", return_value=True)
    @patch("eval_framework.runner.time.sleep")
    @patch("eval_framework.runner.requests.post")
    def test_timeout_then_success(self, mock_post, mock_sleep, mock_wait, mock_is_eva):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": "ok",
            "eval_count": 5,
            "eval_duration": 1_000_000_000,
            "prompt_eval_count": 3,
            "prompt_eval_duration": 200_000_000,
            "total_duration": 1_200_000_000,
        }
        mock_post.side_effect = [
            requests.exceptions.Timeout("timeout"),
            mock_resp,
        ]
        result = runner.generate("test-model", "prompt")
        assert result["response"] == "ok"
        assert result["error"] is None

    @patch("eval_framework.runner.evafrill_runner.is_evafrill", return_value=False)
    @patch("eval_framework.runner.wait_for_ollama", return_value=True)
    @patch("eval_framework.runner.time.sleep")
    @patch("eval_framework.runner.requests.post")
    def test_connection_error(self, mock_post, mock_sleep, mock_wait, mock_is_eva):
        mock_post.side_effect = requests.exceptions.ConnectionError("refused")
        result = runner.generate("test-model", "prompt")
        assert result["error"] is not None
        assert result["response"] == ""

    @patch("eval_framework.runner.evafrill_runner.is_evafrill", return_value=False)
    @patch("eval_framework.runner.wait_for_ollama", return_value=False)
    @patch("eval_framework.runner.time.sleep")
    @patch("eval_framework.runner.requests.post")
    def test_all_retries_exhausted(self, mock_post, mock_sleep, mock_wait, mock_is_eva):
        mock_post.side_effect = requests.exceptions.Timeout("timeout")
        result = runner.generate("test-model", "prompt")
        assert result["error"] is not None
        assert result["tokens_per_sec"] == 0

    @patch("eval_framework.runner.evafrill_runner.is_evafrill", return_value=False)
    @patch("eval_framework.runner.requests.post")
    def test_custom_options(self, mock_post, mock_is_eva):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": "ok", "eval_count": 1, "eval_duration": 100_000_000,
            "prompt_eval_count": 1, "prompt_eval_duration": 50_000_000,
            "total_duration": 150_000_000,
        }
        mock_post.return_value = mock_resp
        custom = {"temperature": 0.0, "num_predict": 100}
        runner.generate("test-model", "prompt", options=custom)
        payload = mock_post.call_args[1]["json"]
        assert payload["options"] == custom

    @patch("eval_framework.runner.evafrill_runner.is_evafrill", return_value=False)
    @patch("eval_framework.runner.requests.post")
    def test_system_prompt(self, mock_post, mock_is_eva):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": "ok", "eval_count": 1, "eval_duration": 100_000_000,
            "prompt_eval_count": 1, "prompt_eval_duration": 50_000_000,
            "total_duration": 150_000_000,
        }
        mock_post.return_value = mock_resp
        runner.generate("test-model", "prompt", system="You are helpful")
        payload = mock_post.call_args[1]["json"]
        assert payload["system"] == "You are helpful"

    @patch("eval_framework.runner.evafrill_runner.is_evafrill", return_value=True)
    @patch("eval_framework.runner.evafrill_runner.subprocess_generate")
    def test_evafrill_delegation(self, mock_eva_gen, mock_is_eva):
        mock_eva_gen.return_value = {
            "response": "evafrill response",
            "eval_count": 5, "eval_duration_s": 1.0,
            "prompt_eval_count": 3, "prompt_eval_duration_s": 0.5,
            "total_duration_s": 1.5, "wall_time_s": 1.5,
            "tokens_per_sec": 5.0, "error": None,
        }
        result = runner.generate("evafrill-mo-3b-slerp", "prompt")
        assert result["response"] == "evafrill response"
        mock_eva_gen.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# chat 테스트 (5 케이스)
# ═══════════════════════════════════════════════════════════════════════════════


class TestChat:

    @patch("eval_framework.runner.evafrill_runner.is_evafrill", return_value=False)
    @patch("eval_framework.runner.requests.post")
    def test_success(self, mock_post, mock_is_eva):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "채팅 응답"},
            "eval_count": 8,
            "eval_duration": 1_000_000_000,
            "prompt_eval_count": 5,
            "prompt_eval_duration": 300_000_000,
            "total_duration": 1_300_000_000,
        }
        mock_post.return_value = mock_resp
        messages = [{"role": "user", "content": "안녕"}]
        result = runner.chat("test-model", messages)
        assert result["response"] == "채팅 응답"
        assert result["error"] is None

    @patch("eval_framework.runner.evafrill_runner.is_evafrill", return_value=False)
    @patch("eval_framework.runner.wait_for_ollama", return_value=True)
    @patch("eval_framework.runner.time.sleep")
    @patch("eval_framework.runner.requests.post")
    def test_retry_on_timeout(self, mock_post, mock_sleep, mock_wait, mock_is_eva):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "ok"},
            "eval_count": 1, "eval_duration": 100_000_000,
            "prompt_eval_count": 1, "prompt_eval_duration": 50_000_000,
            "total_duration": 150_000_000,
        }
        mock_post.side_effect = [
            requests.exceptions.Timeout("timeout"),
            mock_resp,
        ]
        result = runner.chat("test-model", [{"role": "user", "content": "hi"}])
        assert result["response"] == "ok"

    @patch("eval_framework.runner.evafrill_runner.is_evafrill", return_value=True)
    @patch("eval_framework.runner.evafrill_runner.subprocess_generate")
    def test_evafrill_delegation(self, mock_eva_gen, mock_is_eva):
        mock_eva_gen.return_value = {
            "response": "eva chat", "eval_count": 3, "eval_duration_s": 0.5,
            "prompt_eval_count": 2, "prompt_eval_duration_s": 0.2,
            "total_duration_s": 0.7, "wall_time_s": 0.7,
            "tokens_per_sec": 6.0, "error": None,
        }
        messages = [
            {"role": "system", "content": "system msg"},
            {"role": "user", "content": "user msg"},
        ]
        result = runner.chat("evafrill-mo-3b-slerp", messages)
        assert result["response"] == "eva chat"
        # system 메시지가 분리되어 전달되는지 확인
        call_kwargs = mock_eva_gen.call_args
        assert call_kwargs[1]["system"] == "system msg" or call_kwargs.kwargs.get("system") == "system msg"

    @patch("eval_framework.runner.evafrill_runner.is_evafrill", return_value=True)
    @patch("eval_framework.runner.evafrill_runner.subprocess_generate")
    def test_evafrill_system_extraction(self, mock_eva_gen, mock_is_eva):
        mock_eva_gen.return_value = {
            "response": "ok", "eval_count": 1, "eval_duration_s": 0.1,
            "prompt_eval_count": 1, "prompt_eval_duration_s": 0.05,
            "total_duration_s": 0.15, "wall_time_s": 0.15,
            "tokens_per_sec": 10.0, "error": None,
        }
        messages = [
            {"role": "system", "content": "시스템 지시"},
            {"role": "user", "content": "사용자 질문"},
            {"role": "assistant", "content": "이전 답변"},
        ]
        runner.chat("evafrill-mo-3b-slerp", messages)
        _, kwargs = mock_eva_gen.call_args
        assert kwargs["system"] == "시스템 지시"
        assert "사용자 질문" in kwargs["prompt"]

    @patch("eval_framework.runner.evafrill_runner.is_evafrill", return_value=False)
    @patch("eval_framework.runner.wait_for_ollama", return_value=False)
    @patch("eval_framework.runner.time.sleep")
    @patch("eval_framework.runner.requests.post")
    def test_all_retries_fail(self, mock_post, mock_sleep, mock_wait, mock_is_eva):
        mock_post.side_effect = requests.exceptions.ConnectionError("refused")
        result = runner.chat("test-model", [{"role": "user", "content": "hi"}])
        assert result["error"] is not None


# ═══════════════════════════════════════════════════════════════════════════════
# switch_model 테스트 (5 케이스)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSwitchModel:

    @patch("eval_framework.runner.evafrill_runner.is_evafrill", return_value=False)
    @patch("eval_framework.runner.warmup_model", return_value=True)
    @patch("eval_framework.runner.unload_model")
    @patch("eval_framework.runner.ollama_health_check", return_value=True)
    @patch("eval_framework.runner.time.sleep")
    def test_success(self, mock_sleep, mock_health, mock_unload, mock_warmup, mock_is_eva):
        result = runner.switch_model("new-model", current_model="old-model")
        assert result is True
        mock_unload.assert_called_with("old-model")
        mock_warmup.assert_called_with("new-model")

    @patch("eval_framework.runner.evafrill_runner.is_evafrill", return_value=False)
    @patch("eval_framework.runner.warmup_model", return_value=True)
    @patch("eval_framework.runner.ollama_health_check", return_value=True)
    @patch("eval_framework.runner.time.sleep")
    def test_same_model_no_unload(self, mock_sleep, mock_health, mock_warmup, mock_is_eva):
        result = runner.switch_model("same-model", current_model="same-model")
        assert result is True

    @patch("eval_framework.runner.evafrill_runner.is_evafrill", return_value=False)
    @patch("eval_framework.runner._restart_ollama")
    @patch("eval_framework.runner.warmup_model", side_effect=[False, False, True])
    @patch("eval_framework.runner.ollama_health_check", return_value=True)
    @patch("eval_framework.runner.time.sleep")
    def test_warmup_fail_then_restart(self, mock_sleep, mock_health, mock_warmup, mock_restart, mock_is_eva):
        result = runner.switch_model("model", current_model=None)
        assert result is True

    @patch("eval_framework.runner.evafrill_runner.is_evafrill")
    @patch("eval_framework.runner.evafrill_runner.subprocess_unload_model")
    @patch("eval_framework.runner.warmup_model", return_value=True)
    @patch("eval_framework.runner.ollama_health_check", return_value=True)
    @patch("eval_framework.runner.time.sleep")
    def test_evafrill_to_ollama(self, mock_sleep, mock_health, mock_warmup, mock_eva_unload, mock_is_eva):
        # current_model is evafrill, new_model is not
        mock_is_eva.side_effect = lambda m: "evafrill" in m.lower()
        result = runner.switch_model("qwen2.5:3b", current_model="evafrill-mo-3b-slerp")
        assert result is True
        mock_eva_unload.assert_called_once()

    @patch("eval_framework.runner.evafrill_runner.is_evafrill")
    @patch("eval_framework.runner.evafrill_runner.subprocess_load_model")
    @patch("eval_framework.runner.unload_model")
    @patch("eval_framework.runner.time.sleep")
    def test_ollama_to_evafrill(self, mock_sleep, mock_unload, mock_eva_load, mock_is_eva):
        mock_is_eva.side_effect = lambda m: "evafrill" in m.lower()
        result = runner.switch_model("evafrill-mo-3b-slerp", current_model="qwen2.5:3b")
        assert result is True
        mock_unload.assert_called_with("qwen2.5:3b")
        mock_eva_load.assert_called_once()

    @patch("eval_framework.runner.evafrill_runner.is_evafrill")
    @patch("eval_framework.runner.evafrill_runner.subprocess_load_model", return_value=True)
    @patch("eval_framework.runner._stop_ollama")
    @patch("eval_framework.runner.unload_model")
    @patch("eval_framework.runner.time.sleep")
    @patch("eval_framework.runner.config")
    def test_ollama_suspend_stops_before_evafrill(
        self, mock_config, mock_sleep, mock_unload, mock_stop, mock_eva_load, mock_is_eva
    ):
        """ollama_suspend 전략: EVAFRILL 전환 시 Ollama 정지"""
        mock_config.EVAFRILL_GPU_STRATEGY = "ollama_suspend"
        mock_config.COOLDOWN_BETWEEN_MODELS = 10
        mock_is_eva.side_effect = lambda m: "evafrill" in m.lower()
        result = runner.switch_model("evafrill-mo-3b-slerp", current_model="qwen2.5:3b")
        assert result is True
        mock_stop.assert_called_once()

    @patch("eval_framework.runner.evafrill_runner.is_evafrill")
    @patch("eval_framework.runner.evafrill_runner.subprocess_unload_model")
    @patch("eval_framework.runner._restart_ollama", return_value=True)
    @patch("eval_framework.runner.warmup_model", return_value=True)
    @patch("eval_framework.runner.ollama_health_check", return_value=True)
    @patch("eval_framework.runner.time.sleep")
    @patch("eval_framework.runner.config")
    def test_ollama_restarts_after_evafrill(
        self, mock_config, mock_sleep, mock_health, mock_warmup,
        mock_restart, mock_eva_unload, mock_is_eva
    ):
        """ollama_suspend 전략: EVAFRILL→Ollama 전환 시 Ollama GPU 재시작"""
        mock_config.EVAFRILL_GPU_STRATEGY = "ollama_suspend"
        mock_config.COOLDOWN_BETWEEN_MODELS = 10
        mock_is_eva.side_effect = lambda m: "evafrill" in m.lower()
        result = runner.switch_model("qwen2.5:3b", current_model="evafrill-mo-3b-slerp")
        assert result is True
        mock_eva_unload.assert_called_once()
        mock_restart.assert_called_once()

    @patch("eval_framework.runner.evafrill_runner.is_evafrill")
    @patch("eval_framework.runner.evafrill_runner.subprocess_load_model", return_value=True)
    @patch("eval_framework.runner._stop_ollama")
    @patch("eval_framework.runner.unload_model")
    @patch("eval_framework.runner.time.sleep")
    def test_evafrill_cpu_no_ollama_stop(
        self, mock_sleep, mock_unload, mock_stop, mock_eva_load, mock_is_eva
    ):
        """evafrill_cpu 전략(기본): Ollama 정지하지 않음"""
        mock_is_eva.side_effect = lambda m: "evafrill" in m.lower()
        result = runner.switch_model("evafrill-mo-3b-slerp", current_model="qwen2.5:3b")
        assert result is True
        mock_stop.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# Health & Utility 테스트 (5 케이스)
# ═══════════════════════════════════════════════════════════════════════════════


class TestWaitForOllama:

    @patch("eval_framework.runner.ollama_health_check", return_value=True)
    def test_immediate_success(self, mock_health):
        assert runner.wait_for_ollama(max_wait=5) is True

    @patch("eval_framework.runner._restart_ollama", return_value=True)
    @patch("eval_framework.runner.ollama_health_check", return_value=False)
    @patch("eval_framework.runner.time.sleep")
    @patch("eval_framework.runner.time.time")
    def test_timeout_then_auto_restart(self, mock_time, mock_sleep, mock_health, mock_restart):
        # time.time() 시뮬레이션: 0, 0, 70 (> max_wait=60)
        mock_time.side_effect = [0, 0, 70]
        result = runner.wait_for_ollama(max_wait=60, auto_restart=True)
        assert result is True
        mock_restart.assert_called()

    @patch("eval_framework.runner._restart_ollama", return_value=False)
    @patch("eval_framework.runner.ollama_health_check", return_value=False)
    @patch("eval_framework.runner.time.sleep")
    @patch("eval_framework.runner.time.time")
    def test_all_restart_attempts_fail(self, mock_time, mock_sleep, mock_health, mock_restart):
        mock_time.side_effect = [0, 0, 70]
        result = runner.wait_for_ollama(max_wait=60, auto_restart=True)
        assert result is False


class TestUnloadModel:

    @patch("eval_framework.runner.requests.post")
    def test_unload_success(self, mock_post):
        runner.unload_model("test-model")
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        assert payload["keep_alive"] == 0

    @patch("eval_framework.runner.requests.post", side_effect=Exception("fail"))
    def test_unload_error_silent(self, mock_post):
        """unload_model은 에러를 무시"""
        runner.unload_model("test-model")  # should not raise


class TestHealthAndUtility:

    @patch("eval_framework.runner.requests.get")
    def test_health_check_success(self, mock_get):
        mock_get.return_value.status_code = 200
        assert runner.ollama_health_check() is True

    @patch("eval_framework.runner.requests.get")
    def test_health_check_failure(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("refused")
        assert runner.ollama_health_check() is False

    @patch("eval_framework.runner.requests.get")
    def test_get_loaded_models(self, mock_get):
        mock_get.return_value.json.return_value = {
            "models": [{"name": "m1"}, {"name": "m2"}]
        }
        models = runner.get_loaded_models()
        assert models == ["m1", "m2"]

    @patch("eval_framework.runner.subprocess.run")
    def test_get_vram_usage_success(self, mock_run):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "4096, 8192, 4096, 75"
        result = runner.get_vram_usage()
        assert result["vram_used_mb"] == 4096
        assert result["vram_total_mb"] == 8192
        assert result["vram_free_mb"] == 4096
        assert result["gpu_util_pct"] == 75

    @patch("eval_framework.runner.subprocess.run")
    def test_get_vram_usage_no_gpu(self, mock_run):
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
        result = runner.get_vram_usage()
        assert result["vram_used_mb"] == 0
        assert result["vram_total_mb"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint 테스트 (4 케이스)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckpoint:

    def test_save_checkpoint(self, tmp_results_dir):
        data = {"model": "test", "score": 0.8}
        path = runner.save_checkpoint(data, "track_test")
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["score"] == 0.8

    def test_load_checkpoint_exists(self, tmp_results_dir):
        data = {"model": "test", "results": [1, 2, 3]}
        runner.save_checkpoint(data, "track_load")
        loaded = runner.load_checkpoint("track_load")
        assert loaded is not None
        assert loaded["results"] == [1, 2, 3]

    def test_load_checkpoint_missing(self, tmp_results_dir):
        loaded = runner.load_checkpoint("nonexistent_track")
        assert loaded is None

    def test_save_results_incremental(self, tmp_results_dir):
        data = {"results": [{"score": 5}]}
        path = runner.save_results_incremental(data, "track_inc")
        assert path.exists()
        assert "track_inc_" in path.name
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["results"][0]["score"] == 5
