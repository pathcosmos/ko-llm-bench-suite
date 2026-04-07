"""kobench/scoring.py 단위 테스트"""

import math
import pytest
import numpy as np
from unittest.mock import patch

from kobench.scoring import (
    aggregate_accuracy,
    aggregate_judge_scores,
    aggregate_performance,
    fit_bradley_terry,
    build_scorecard,
    save_scorecard,
)


# ═══════════════════════════════════════════════════════════════════════════════
# aggregate_accuracy 테스트 (4 케이스)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAggregateAccuracy:

    def test_basic(self, sample_accuracy_results):
        result = aggregate_accuracy(sample_accuracy_results)
        assert abs(result["model_a"] - 2 / 3) < 1e-6
        assert abs(result["model_b"] - 1 / 2) < 1e-6

    def test_empty_list(self):
        result = aggregate_accuracy([])
        assert result == {}

    def test_all_correct(self):
        results = [
            {"model": "m1", "correct": True},
            {"model": "m1", "correct": True},
        ]
        result = aggregate_accuracy(results)
        assert result["m1"] == 1.0

    def test_custom_model_key(self):
        results = [
            {"name": "m1", "correct": True},
            {"name": "m1", "correct": False},
        ]
        result = aggregate_accuracy(results, model_key="name")
        assert abs(result["m1"] - 0.5) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# aggregate_judge_scores 테스트 (3 케이스)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAggregateJudgeScores:

    def test_basic(self, sample_judge_results):
        result = aggregate_judge_scores(sample_judge_results)
        assert "model_a" in result
        assert "model_b" in result
        # model_a: scores [8, 6, 9] → mean ~7.67
        assert abs(result["model_a"]["mean"] - np.mean([8, 6, 9])) < 1e-6
        assert result["model_a"]["n"] == 3

    def test_by_category(self, sample_judge_results):
        result = aggregate_judge_scores(sample_judge_results)
        cats = result["model_a"]["by_category"]
        assert "writing" in cats
        assert "reasoning" in cats
        assert cats["writing"]["n"] == 2
        assert abs(cats["writing"]["mean"] - np.mean([8, 9])) < 1e-6

    def test_skip_zero_scores(self):
        results = [
            {"model": "m1", "judge_score": 0, "category": "test"},
            {"model": "m1", "judge_score": 5, "category": "test"},
        ]
        result = aggregate_judge_scores(results)
        assert result["m1"]["n"] == 1
        assert result["m1"]["mean"] == 5.0


# ═══════════════════════════════════════════════════════════════════════════════
# aggregate_performance 테스트 (2 케이스)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAggregatePerformance:

    def test_basic(self):
        results = [
            {"model": "m1", "tokens_per_sec": 10.0, "ttft_s": 0.5},
            {"model": "m1", "tokens_per_sec": 20.0, "ttft_s": 0.3},
        ]
        result = aggregate_performance(results)
        assert abs(result["m1"]["tokens_per_sec"]["mean"] - 15.0) < 1e-6
        assert abs(result["m1"]["ttft_s"]["min"] - 0.3) < 1e-6

    def test_missing_keys_graceful(self):
        results = [
            {"model": "m1", "tokens_per_sec": 10.0},
            {"model": "m1"},
        ]
        result = aggregate_performance(results)
        assert "tokens_per_sec" in result["m1"]
        assert "prefill_tok_s" not in result["m1"]


# ═══════════════════════════════════════════════════════════════════════════════
# fit_bradley_terry 테스트 (5 케이스)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFitBradleyTerry:

    def test_clear_winner(self):
        """model_x가 항상 이기면 elo가 더 높아야 함"""
        comps = [
            {"model_a": "winner", "model_b": "loser", "winner": "A"},
        ] * 10
        result = fit_bradley_terry(comps)
        assert result["winner"]["elo"] > result["loser"]["elo"]

    def test_tie_similar_elo(self):
        """전부 동점이면 elo가 비슷해야 함"""
        comps = [
            {"model_a": "m1", "model_b": "m2", "winner": "TIE"},
        ] * 10
        result = fit_bradley_terry(comps)
        assert abs(result["m1"]["elo"] - result["m2"]["elo"]) < 50

    def test_single_model(self):
        result = fit_bradley_terry([], models=["only_one"])
        assert result["only_one"]["elo"] == 1000

    def test_three_models_transitive(self, sample_comparisons):
        """model_x > model_y > model_z"""
        result = fit_bradley_terry(sample_comparisons)
        assert result["model_x"]["elo"] > result["model_y"]["elo"]
        assert "wins" in result["model_x"]
        assert "losses" in result["model_x"]
        assert "ci_lower" in result["model_x"]
        assert "ci_upper" in result["model_x"]

    def test_empty_comparisons(self):
        result = fit_bradley_terry([], models=["a", "b"])
        # 비교 없으면 기본값
        assert "a" in result
        assert "b" in result


# ═══════════════════════════════════════════════════════════════════════════════
# build_scorecard 테스트 (2 케이스)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildScorecard:

    @patch("kobench.scoring.config.ALL_MODELS", ["m1", "m2"])
    def test_integrates_tracks(self):
        track_results = {
            "track1": {"m1": 0.85, "m2": 0.72},
            "track2": {"m1": {"mean": 7.5}, "m2": {"mean": 6.0}},
        }
        sc = build_scorecard(track_results)
        assert sc["m1"]["track1"] == 0.85
        assert sc["m1"]["track2"] == 7.5
        assert sc["m2"]["track1"] == 0.72

    @patch("kobench.scoring.config.ALL_MODELS", ["m1", "m2", "m3"])
    def test_missing_model_handled(self):
        track_results = {
            "track1": {"m1": 0.9},
        }
        sc = build_scorecard(track_results)
        assert "track1" in sc["m1"]
        assert "track1" not in sc["m3"]


# ═══════════════════════════════════════════════════════════════════════════════
# save_scorecard 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestSaveScorecard:

    def test_save_and_verify(self, tmp_path):
        scorecard = {"m1": {"model": "m1", "track1": 0.9}}
        path = tmp_path / "scorecard.json"
        result_path = save_scorecard(scorecard, path=path)
        assert result_path.exists()
        import json
        with open(result_path) as f:
            loaded = json.load(f)
        assert loaded["m1"]["track1"] == 0.9
