"""공유 fixtures — Ollama API mock, 샘플 데이터, 임시 디렉토리"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# ── EVAFRILL 커스텀 모듈 mock ─────────────────────────────────────────────────
# evafrill_runner.py가 /home/lanco/models/EVAFRILL-Mo 의 커스텀 모듈을 import하는데,
# 해당 모듈이 yaml 등 추가 의존성을 요구함. 테스트에서는 mock으로 대체.
_EVAFRILL_MOCK_MODULES = [
    "model", "model.config", "model.transformer",
    "tokenizers", "safetensors", "safetensors.torch",
    "yaml",
]
for _mod in _EVAFRILL_MOCK_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


# ── Ollama API mock fixtures ────────────────────────────────────────────────


@pytest.fixture
def mock_ollama_post():
    """requests.post 패치 — Ollama generate/chat API 응답 반환"""
    with patch("kobench.runner.requests.post") as mock_post:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "response": "테스트 응답입니다.",
            "eval_count": 10,
            "eval_duration": 1_000_000_000,  # 1s in nanoseconds
            "prompt_eval_count": 5,
            "prompt_eval_duration": 500_000_000,
            "total_duration": 1_500_000_000,
            "message": {"content": "테스트 채팅 응답입니다."},
        }
        mock_post.return_value = response
        yield mock_post


@pytest.fixture
def mock_ollama_get():
    """requests.get 패치 — health check, /api/ps 용"""
    with patch("kobench.runner.requests.get") as mock_get:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "models": [{"name": "test-model:latest"}],
        }
        mock_get.return_value = response
        yield mock_get


# ── 샘플 데이터 fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def sample_generate_response():
    """runner.generate() 반환 형식 샘플"""
    return {
        "response": "테스트 응답입니다.",
        "eval_count": 10,
        "eval_duration_s": 1.0,
        "prompt_eval_count": 5,
        "prompt_eval_duration_s": 0.5,
        "total_duration_s": 1.5,
        "wall_time_s": 1.6,
        "tokens_per_sec": 10.0,
        "error": None,
    }


@pytest.fixture
def sample_judge_json():
    """judge 응답 JSON 문자열"""
    return '{"score": 8, "reasoning": "정확하고 자연스러운 답변입니다."}'


@pytest.fixture
def sample_pairwise_json():
    """pairwise judge 응답 JSON 문자열"""
    return '{"winner": "A", "reasoning": "A가 더 정확합니다."}'


@pytest.fixture
def sample_criteria_json():
    """multi-criteria judge 응답 JSON 문자열"""
    return '{"scores": {"정확성": 8, "유용성": 7}, "reasoning": "전반적으로 좋음"}'


@pytest.fixture
def sample_comparisons():
    """Bradley-Terry 테스트용 비교 데이터"""
    return [
        {"model_a": "model_x", "model_b": "model_y", "winner": "A"},
        {"model_a": "model_x", "model_b": "model_y", "winner": "A"},
        {"model_a": "model_y", "model_b": "model_x", "winner": "B"},
        {"model_a": "model_x", "model_b": "model_z", "winner": "A"},
        {"model_a": "model_y", "model_b": "model_z", "winner": "A"},
        {"model_a": "model_z", "model_b": "model_y", "winner": "B"},
    ]


@pytest.fixture
def sample_accuracy_results():
    """aggregate_accuracy 테스트용 데이터"""
    return [
        {"model": "model_a", "correct": True},
        {"model": "model_a", "correct": True},
        {"model": "model_a", "correct": False},
        {"model": "model_b", "correct": False},
        {"model": "model_b", "correct": True},
    ]


@pytest.fixture
def sample_judge_results():
    """aggregate_judge_scores 테스트용 데이터"""
    return [
        {"model": "model_a", "judge_score": 8, "category": "writing"},
        {"model": "model_a", "judge_score": 6, "category": "reasoning"},
        {"model": "model_a", "judge_score": 9, "category": "writing"},
        {"model": "model_b", "score": 7, "category": "writing"},
        {"model": "model_b", "score": 5, "category": "reasoning"},
    ]


# ── 임시 디렉토리 fixtures ───────────────────────────────────────────────────


@pytest.fixture
def tmp_results_dir(tmp_path):
    """config.RESULTS_DIR를 임시 경로로 패치"""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    with patch("kobench.config.RESULTS_DIR", results_dir):
        with patch("kobench.runner.config.RESULTS_DIR", results_dir):
            yield results_dir


# ── time.sleep mock ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_time_sleep():
    """time.sleep 패치 — 테스트 속도 향상"""
    with patch("kobench.runner.time.sleep"):
        with patch("kobench.judge.time.sleep"):
            yield
