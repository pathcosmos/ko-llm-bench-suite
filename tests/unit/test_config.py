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
