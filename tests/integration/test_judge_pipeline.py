"""Judge 채점 파이프라인 통합 테스트

score_response → aggregate_judge_scores
score_pairwise → fit_bradley_terry
score_with_criteria → 카테고리별 집계
"""

import pytest
from unittest.mock import patch

from kobench.judge import score_response, score_pairwise, score_with_criteria
from kobench.scoring import aggregate_judge_scores, fit_bradley_terry


class TestScoreThenAggregate:
    """score_response N회 → aggregate_judge_scores → 유효한 summary"""

    @patch("kobench.judge._call_judge")
    def test_score_then_aggregate(self, mock_call):
        """여러 모델의 score_response 결과를 집계"""
        mock_call.side_effect = [
            '{"score": 8, "reasoning": "good"}',
            '{"score": 6, "reasoning": "ok"}',
            '{"score": 9, "reasoning": "great"}',
            '{"score": 7, "reasoning": "decent"}',
            '{"score": 5, "reasoning": "fair"}',
        ]
        results = []
        models = ["model_a", "model_a", "model_a", "model_b", "model_b"]
        categories = ["writing", "reasoning", "writing", "writing", "reasoning"]

        for model, cat in zip(models, categories):
            score = score_response("질문", "답변", cat)
            results.append({
                "model": model,
                "judge_score": score["score"],
                "category": cat,
            })

        summary = aggregate_judge_scores(results)
        assert "model_a" in summary
        assert "model_b" in summary
        assert summary["model_a"]["n"] == 3
        assert summary["model_b"]["n"] == 2
        assert "by_category" in summary["model_a"]
        assert "writing" in summary["model_a"]["by_category"]


class TestPairwiseToBradleyTerry:
    """score_pairwise 다수 쌍 → fit_bradley_terry → Elo 랭킹"""

    @patch("kobench.judge._call_judge")
    def test_pairwise_to_elo(self, mock_call):
        """3모델 6쌍 비교 → BT Elo 산출"""
        # model_a가 대부분 이김
        responses = [
            '{"winner": "A", "reasoning": "A wins"}',
            '{"winner": "A", "reasoning": "A wins"}',
            '{"winner": "A", "reasoning": "A wins"}',
            '{"winner": "B", "reasoning": "B wins"}',
            '{"winner": "A", "reasoning": "A wins"}',
            '{"winner": "tie", "reasoning": "close"}',
        ]
        mock_call.side_effect = responses

        pairs = [
            ("model_a", "model_b"),
            ("model_a", "model_b"),
            ("model_a", "model_c"),
            ("model_b", "model_c"),
            ("model_a", "model_c"),
            ("model_b", "model_c"),
        ]

        comparisons = []
        for model_a, model_b in pairs:
            result = score_pairwise("질문", f"{model_a} 응답", f"{model_b} 응답")
            winner = result["winner"]
            comparisons.append({
                "model_a": model_a,
                "model_b": model_b,
                "winner": winner,
            })

        elo = fit_bradley_terry(comparisons)
        assert "model_a" in elo
        assert "model_b" in elo
        assert "model_c" in elo
        # model_a가 가장 많이 이겼으므로 elo가 가장 높아야 함
        assert elo["model_a"]["elo"] > elo["model_c"]["elo"]


class TestCriteriaScoringAcrossCategories:
    """score_with_criteria로 여러 카테고리 채점 후 집계"""

    @patch("kobench.judge._call_judge")
    def test_multi_category_scoring(self, mock_call):
        mock_call.side_effect = [
            '{"scores": {"정확성": 8, "창의성": 7}, "reasoning": "writing ok"}',
            '{"scores": {"정확성": 6, "논리성": 9}, "reasoning": "reasoning ok"}',
            '{"scores": {"정확성": 9, "창의성": 8}, "reasoning": "writing great"}',
        ]

        all_results = []
        tests = [
            ("model_a", "writing", {"정확성": "사실관계", "창의성": "독창성"}),
            ("model_a", "reasoning", {"정확성": "사실관계", "논리성": "추론 능력"}),
            ("model_b", "writing", {"정확성": "사실관계", "창의성": "독창성"}),
        ]

        for model, cat, criteria in tests:
            result = score_with_criteria("질문", "답변", criteria)
            if result["scores"]:
                avg_score = sum(result["scores"].values()) / len(result["scores"])
                all_results.append({
                    "model": model,
                    "judge_score": avg_score,
                    "category": cat,
                })

        summary = aggregate_judge_scores(all_results)
        assert "model_a" in summary
        assert "model_b" in summary
        assert summary["model_a"]["by_category"]["writing"]["n"] == 1
        assert summary["model_a"]["by_category"]["reasoning"]["n"] == 1
