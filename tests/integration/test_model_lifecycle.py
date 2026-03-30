"""모델 전환 라이프사이클 통합 테스트

switch_model → unload → warmup → health check 흐름 검증
"""

import pytest
from unittest.mock import patch, MagicMock, call

from eval_framework import runner


class TestFullSwitchCycle:
    """모델 A→B→C 전환 시 unload/warmup 순서 검증"""

    @patch("eval_framework.runner.evafrill_runner.is_evafrill", return_value=False)
    @patch("eval_framework.runner.warmup_model", return_value=True)
    @patch("eval_framework.runner.unload_model")
    @patch("eval_framework.runner.ollama_health_check", return_value=True)
    @patch("eval_framework.runner.time.sleep")
    def test_sequential_switch(self, mock_sleep, mock_health, mock_unload, mock_warmup, mock_is_eva):
        # A → B
        assert runner.switch_model("model_b", current_model="model_a") is True
        mock_unload.assert_called_with("model_a")

        mock_unload.reset_mock()
        mock_warmup.reset_mock()

        # B → C
        assert runner.switch_model("model_c", current_model="model_b") is True
        mock_unload.assert_called_with("model_b")
        mock_warmup.assert_called_with("model_c")


class TestSwitchWithServerRestart:
    """health check 실패 → 자동 재시작 → 복구"""

    @patch("eval_framework.runner.evafrill_runner.is_evafrill", return_value=False)
    @patch("eval_framework.runner.warmup_model", side_effect=[False, True])
    @patch("eval_framework.runner._restart_ollama")
    @patch("eval_framework.runner.ollama_health_check", side_effect=[False, True, True])
    @patch("eval_framework.runner.wait_for_ollama", return_value=True)
    @patch("eval_framework.runner.time.sleep")
    def test_restart_recovery(self, mock_sleep, mock_wait, mock_health, mock_restart, mock_warmup, mock_is_eva):
        result = runner.switch_model("model_x", current_model=None)
        assert result is True


class TestEvafrillOllamaTransition:
    """evafrill ↔ ollama 전환 전체 사이클"""

    @patch("eval_framework.runner.evafrill_runner.is_evafrill")
    @patch("eval_framework.runner.evafrill_runner.subprocess_load_model")
    @patch("eval_framework.runner.evafrill_runner.subprocess_unload_model")
    @patch("eval_framework.runner.warmup_model", return_value=True)
    @patch("eval_framework.runner.unload_model")
    @patch("eval_framework.runner.ollama_health_check", return_value=True)
    @patch("eval_framework.runner.time.sleep")
    def test_full_cycle(self, mock_sleep, mock_health, mock_ollama_unload, mock_warmup,
                         mock_eva_unload, mock_eva_load, mock_is_eva):
        mock_is_eva.side_effect = lambda m: "evafrill" in m.lower()

        # Ollama → EVAFRILL
        result = runner.switch_model("evafrill-mo-3b-slerp", current_model="qwen2.5:3b")
        assert result is True
        mock_ollama_unload.assert_called_with("qwen2.5:3b")
        mock_eva_load.assert_called_once()

        mock_eva_load.reset_mock()
        mock_ollama_unload.reset_mock()

        # EVAFRILL → Ollama
        result = runner.switch_model("gemma3:4b", current_model="evafrill-mo-3b-slerp")
        assert result is True
        mock_eva_unload.assert_called()
