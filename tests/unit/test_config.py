"""kobench/config.py 단위 테스트"""

import pytest
from unittest.mock import patch, MagicMock
import subprocess


class TestGpuAvailable:
    """_gpu_available 함수 테스트"""

    def test_gpu_available_true(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce RTX 4090\n"
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            from kobench.config import _gpu_available
            assert _gpu_available() is True

    def test_gpu_not_available_no_nvidia_smi(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
            from kobench.config import _gpu_available
            assert _gpu_available() is False

    def test_gpu_not_available_nonzero_return(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            from kobench.config import _gpu_available
            assert _gpu_available() is False

    def test_gpu_not_available_empty_output(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            from kobench.config import _gpu_available
            assert _gpu_available() is False


class TestTimeoutCalculation:
    """타임아웃 계산 로직 테스트 — 현재 config 값 기반"""

    def test_default_timeout_exists_for_all_models(self):
        from kobench import config
        for model in config.ALL_MODELS:
            assert model in config.MODEL_TIMEOUTS
            assert config.MODEL_TIMEOUTS[model] > 0

    def test_q8_models_higher_timeout(self):
        from kobench import config
        for model in config.ALL_MODELS:
            if "Q8_0" in model or "q8_0" in model:
                base = 180 * (1 if config.GPU_AVAILABLE else 2)
                # Q8_0 모델이 8b가 아니면 180 * multiplier
                if "8b" not in model.lower():
                    assert config.MODEL_TIMEOUTS[model] >= base

    def test_8b_models_higher_timeout(self):
        from kobench import config
        for model in config.ALL_MODELS:
            if "8b" in model.lower():
                base = 360 * (1 if config.GPU_AVAILABLE else 2)
                assert config.MODEL_TIMEOUTS[model] >= base

    def test_evafrill_highest_timeout(self):
        from kobench import config
        for model in config.ALL_MODELS:
            if "evafrill" in model.lower():
                base = 600 * (1 if config.GPU_AVAILABLE else 2)
                assert config.MODEL_TIMEOUTS[model] == base

    def test_deepseek_r1_timeout(self):
        from kobench import config
        for model in config.ALL_MODELS:
            if "deepseek-r1" in model.lower():
                base = 240 * (1 if config.GPU_AVAILABLE else 2)
                assert config.MODEL_TIMEOUTS[model] >= base


class TestModelListConsistency:
    """모델 목록 일관성 검증"""

    def test_all_models_is_concatenation(self):
        from kobench import config
        expected = (
            config.FRANKENSTALLM_MODELS
            + config.COMPARISON_MODELS
            + config.EVAFRILL_MODELS
        )
        assert config.ALL_MODELS == expected

    def test_frankenstallm_models_is_v1_plus_v2(self):
        from kobench import config
        expected = config.FRANKENSTALLM_V1_MODELS + config.FRANKENSTALLM_V2_MODELS
        assert config.FRANKENSTALLM_MODELS == expected


class TestApplyYamlToConfig:
    """apply_yaml_to_config 함수 테스트"""

    def test_apply_yaml_backend(self):
        """YAML backend settings are applied to config."""
        from kobench.config import apply_yaml_to_config
        from kobench import config
        original_url = config.OLLAMA_BASE_URL
        try:
            apply_yaml_to_config({"backend": {"url": "http://test:9999", "remote": True}})
            assert config.OLLAMA_BASE_URL == "http://test:9999"
            assert config.OLLAMA_REMOTE == True
            assert config.OLLAMA_API_GENERATE == "http://test:9999/api/generate"
        finally:
            config.OLLAMA_BASE_URL = original_url
            config.OLLAMA_REMOTE = False
            config.OLLAMA_API_GENERATE = f"{original_url}/api/generate"
            config.OLLAMA_API_CHAT = f"{original_url}/api/chat"
            config.OLLAMA_API_SHOW = f"{original_url}/api/show"
            config.OLLAMA_API_PS = f"{original_url}/api/ps"

    def test_apply_yaml_judge(self):
        """YAML judge settings are applied to config."""
        from kobench.config import apply_yaml_to_config
        from kobench import config
        original_models = dict(config.JUDGE_MODELS)
        original_weights = dict(config.JUDGE_WEIGHTS)
        try:
            apply_yaml_to_config({"judge": {
                "dual_enabled": False,
                "primary": {"model": "custom-judge", "weight": 0.8},
                "secondary": {"model": "custom-judge-2", "weight": 0.2},
                "timeout": 60,
            }})
            assert config.JUDGE_DUAL_ENABLED == False
            assert config.JUDGE_MODELS["primary"] == "custom-judge"
            assert config.JUDGE_MODELS["secondary"] == "custom-judge-2"
            assert config.JUDGE_WEIGHTS["primary"] == 0.8
            assert config.JUDGE_TIMEOUT == 60
        finally:
            config.JUDGE_MODELS.update(original_models)
            config.JUDGE_WEIGHTS.update(original_weights)
            config.JUDGE_DUAL_ENABLED = True
            config.JUDGE_TIMEOUT = 120

    def test_apply_yaml_sampling(self):
        """YAML sampling settings are applied to config."""
        from kobench.config import apply_yaml_to_config
        from kobench import config
        orig_temp = config.SAMPLING_PARAMS["temperature"]
        orig_bench_temp = config.BENCHMARK_SAMPLING["temperature"]
        try:
            apply_yaml_to_config({"sampling": {
                "default": {"temperature": 0.3},
                "benchmark": {"temperature": 0.1},
            }})
            assert config.SAMPLING_PARAMS["temperature"] == 0.3
            assert config.BENCHMARK_SAMPLING["temperature"] == 0.1
        finally:
            config.SAMPLING_PARAMS["temperature"] = orig_temp
            config.BENCHMARK_SAMPLING["temperature"] = orig_bench_temp

    def test_apply_yaml_retry(self):
        """YAML retry settings are applied to config."""
        from kobench.config import apply_yaml_to_config
        from kobench import config
        orig = (config.MAX_RETRIES, config.RETRY_BACKOFF_BASE, config.COOLDOWN_BETWEEN_MODELS)
        try:
            apply_yaml_to_config({"retry": {"max_retries": 5, "backoff_base": 10, "cooldown_between_models": 20}})
            assert config.MAX_RETRIES == 5
            assert config.RETRY_BACKOFF_BASE == 10
            assert config.COOLDOWN_BETWEEN_MODELS == 20
        finally:
            config.MAX_RETRIES, config.RETRY_BACKOFF_BASE, config.COOLDOWN_BETWEEN_MODELS = orig

    def test_apply_yaml_partial(self):
        """Partial YAML (only some sections) preserves existing defaults."""
        from kobench.config import apply_yaml_to_config
        from kobench import config
        orig_url = config.OLLAMA_BASE_URL
        orig_retries = config.MAX_RETRIES
        try:
            apply_yaml_to_config({"retry": {"max_retries": 1}})
            assert config.OLLAMA_BASE_URL == orig_url  # unchanged
            assert config.MAX_RETRIES == 1  # changed
        finally:
            config.MAX_RETRIES = orig_retries

    def test_apply_yaml_empty(self):
        """Empty YAML dict changes nothing."""
        from kobench.config import apply_yaml_to_config
        from kobench import config
        orig_url = config.OLLAMA_BASE_URL
        apply_yaml_to_config({})
        assert config.OLLAMA_BASE_URL == orig_url
