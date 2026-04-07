"""kobench/evafrill_runner.py 단위 테스트

evafrill_runner.py는 import 시 torch, 커스텀 모델 모듈을 로딩하므로
이 테스트 파일에서는 sys.modules를 패치하여 의존성 없이 테스트.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch


# ── evafrill_runner import를 위한 mock 설정 ───────────────────────────────────
# 모듈 import 전에 torch와 커스텀 모듈을 mock해야 함

def _setup_evafrill_mocks():
    """evafrill_runner import에 필요한 mock 모듈 설정"""
    mocks = {}
    # torch가 이미 설치되어 있으면 실제 모듈 사용
    try:
        import torch
    except ImportError:
        mock_torch = MagicMock()
        mock_torch.nn = MagicMock()
        mock_torch.nn.functional = MagicMock()
        mocks["torch"] = mock_torch
        mocks["torch.nn"] = mock_torch.nn
        mocks["torch.nn.functional"] = mock_torch.nn.functional

    # 커스텀 모델 모듈 mock
    for mod_name in [
        "model", "model.config", "model.transformer",
        "tokenizers", "safetensors", "safetensors.torch",
    ]:
        if mod_name not in sys.modules:
            mocks[mod_name] = MagicMock()

    return mocks


# ═══════════════════════════════════════════════════════════════════════════════
# is_evafrill 테스트 (4 케이스)
# ═══════════════════════════════════════════════════════════════════════════════


class TestIsEvafrill:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self._mocks = _setup_evafrill_mocks()
        with patch.dict(sys.modules, self._mocks):
            from kobench.evafrill_runner import is_evafrill
            self.is_evafrill = is_evafrill

    def test_exact_model_name(self):
        assert self.is_evafrill("evafrill-mo-3b-slerp") is True

    def test_uppercase(self):
        assert self.is_evafrill("EVAFRILL-Mo-3B") is True

    def test_non_evafrill(self):
        assert self.is_evafrill("qwen2.5:3b") is False

    def test_partial_match_still_true(self):
        """'evafrill'이 포함되면 True"""
        assert self.is_evafrill("my-evafrill-variant") is True


# ═══════════════════════════════════════════════════════════════════════════════
# _top_p_filtering 테스트 (5 케이스) — torch 필요
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def torch_available():
    """torch가 설치되어 있는지 확인"""
    try:
        import torch
        return True
    except ImportError:
        return False


class TestTopPFiltering:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self._mocks = _setup_evafrill_mocks()
        with patch.dict(sys.modules, self._mocks):
            try:
                import torch
                self.torch = torch
                from kobench.evafrill_runner import _top_p_filtering
                self.filter_fn = _top_p_filtering
                self.has_torch = True
            except (ImportError, AttributeError):
                self.has_torch = False

    def test_top_k_filtering(self):
        if not self.has_torch:
            pytest.skip("torch not available")
        logits = self.torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        filtered = self.filter_fn(logits, top_p=1.0, top_k=3)
        # 하위 2개는 -inf
        assert filtered.shape[-1] == 5
        assert filtered[0, 0] == float("-inf")
        assert filtered[0, 1] == float("-inf")
        assert filtered[0, 4] == 5.0

    def test_top_p_filtering(self):
        if not self.has_torch:
            pytest.skip("torch not available")
        logits = self.torch.tensor([1.0, 2.0, 10.0])
        filtered = self.filter_fn(logits, top_p=0.5, top_k=50)
        # top_p=0.5 → 가장 큰 값(10.0)만 남을 수 있음
        assert filtered[0, 2] == 10.0

    def test_both_active(self):
        if not self.has_torch:
            pytest.skip("torch not available")
        logits = self.torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        filtered = self.filter_fn(logits, top_p=0.9, top_k=3)
        # top_k=3으로 하위 2개 제거, 그 후 top_p 적용
        assert filtered[0, 0] == float("-inf")
        assert filtered[0, 1] == float("-inf")

    def test_1d_input_unsqueezed(self):
        if not self.has_torch:
            pytest.skip("torch not available")
        logits = self.torch.tensor([1.0, 2.0, 3.0])
        filtered = self.filter_fn(logits, top_p=0.9, top_k=2)
        assert filtered.dim() == 2
        assert filtered.shape[0] == 1

    def test_output_shape_preserved(self):
        if not self.has_torch:
            pytest.skip("torch not available")
        logits = self.torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        filtered = self.filter_fn(logits, top_p=0.9, top_k=3)
        assert filtered.shape == (1, 4)
