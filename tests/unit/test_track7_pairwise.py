"""
Track 7 (Pairwise Comparison) 단위 테스트

_resolve_winner, _collect_responses, _run_comparisons, _build_summary, run
"""

import pytest
from unittest.mock import patch, MagicMock

from kobench.tracks import pairwise as t7


# ── _resolve_winner ──────────────────────────────────────────────────────────


class TestResolveWinner:
    """위치 편향 제거: 정방향/역방향 결과 종합"""

    def test_both_agree_a_wins(self):
        """정방향 A 승 + 역방향 B 승 (=정방향 A 승) → A"""
        assert t7._resolve_winner("A", "B") == "A"

    def test_both_agree_b_wins(self):
        """정방향 B 승 + 역방향 A 승 (=정방향 B 승) → B"""
        assert t7._resolve_winner("B", "A") == "B"

    def test_both_agree_tie(self):
        """정방향 TIE + 역방향 TIE → TIE"""
        assert t7._resolve_winner("TIE", "TIE") == "TIE"

    def test_disagree_a_vs_a(self):
        """정방향 A + 역방향 A(=정방향 B) → 불일치 → TIE"""
        assert t7._resolve_winner("A", "A") == "TIE"

    def test_disagree_b_vs_b(self):
        """정방향 B + 역방향 B(=정방향 A) → 불일치 → TIE"""
        assert t7._resolve_winner("B", "B") == "TIE"

    def test_disagree_a_vs_tie(self):
        """정방향 A + 역방향 TIE → 불일치 → TIE"""
        assert t7._resolve_winner("A", "TIE") == "TIE"

    def test_disagree_b_vs_tie(self):
        """정방향 B + 역방향 TIE → 불일치 → TIE"""
        assert t7._resolve_winner("B", "TIE") == "TIE"

    def test_disagree_tie_vs_a(self):
        """정방향 TIE + 역방향 A(=정방향 B) → 불일치 → TIE"""
        assert t7._resolve_winner("TIE", "A") == "TIE"

    def test_disagree_tie_vs_b(self):
        """정방향 TIE + 역방향 B(=정방향 A) → 불일치 → TIE"""
        assert t7._resolve_winner("TIE", "B") == "TIE"


# ── _collect_responses ───────────────────────────────────────────────────────


class TestCollectResponses:
    """모델별 프롬프트 응답 수집"""

    @patch.object(t7, "PROMPTS", [
        {"id": "p1", "category": "test", "prompt": "질문1"},
        {"id": "p2", "category": "test", "prompt": "질문2"},
    ])
    @patch("kobench.tracks.pairwise.runner")
    @patch("kobench.tracks.pairwise.time")
    def test_normal_collection(self, mock_time, mock_runner):
        """정상 수집: 2모델 x 2프롬프트"""
        mock_runner.switch_model.return_value = True
        mock_runner.generate.side_effect = [
            {"response": "답변1-1"},
            {"response": "답변1-2"},
            {"response": "답변2-1"},
            {"response": "답변2-2"},
        ]

        result = t7._collect_responses(["m1", "m2"], {})

        assert set(result.keys()) == {"m1", "m2"}
        assert result["m1"]["p1"] == "답변1-1"
        assert result["m2"]["p2"] == "답변2-2"

    @patch.object(t7, "PROMPTS", [
        {"id": "p1", "category": "test", "prompt": "질문1"},
    ])
    @patch("kobench.tracks.pairwise.runner")
    @patch("kobench.tracks.pairwise.time")
    def test_checkpoint_skip(self, mock_time, mock_runner):
        """체크포인트에 이미 수집된 모델은 스킵"""
        checkpoint = {"responses": {"m1": {"p1": "기존답변"}}}
        result = t7._collect_responses(["m1"], checkpoint)

        assert result["m1"]["p1"] == "기존답변"
        mock_runner.generate.assert_not_called()

    @patch.object(t7, "PROMPTS", [
        {"id": "p1", "category": "test", "prompt": "질문1"},
    ])
    @patch("kobench.tracks.pairwise.runner")
    @patch("kobench.tracks.pairwise.time")
    def test_model_load_failure(self, mock_time, mock_runner):
        """모델 로딩 실패 시 스킵"""
        mock_runner.switch_model.return_value = False
        result = t7._collect_responses(["bad_model"], {})

        assert "bad_model" not in result
        mock_runner.generate.assert_not_called()

    @patch.object(t7, "PROMPTS", [
        {"id": "p1", "category": "test", "prompt": "질문1"},
    ])
    @patch("kobench.tracks.pairwise.runner")
    @patch("kobench.tracks.pairwise.time")
    def test_generation_error(self, mock_time, mock_runner):
        """생성 오류 시 빈 문자열 저장"""
        mock_runner.switch_model.return_value = True
        mock_runner.generate.return_value = {"error": "timeout"}

        result = t7._collect_responses(["m1"], {})
        assert result["m1"]["p1"] == ""


# ── _run_comparisons ─────────────────────────────────────────────────────────


class TestRunComparisons:
    """쌍대비교 실행"""

    @patch.object(t7, "PROMPTS", [
        {"id": "p1", "category": "test", "prompt": "질문1"},
    ])
    @patch("kobench.tracks.pairwise.judge")
    @patch("kobench.tracks.pairwise.runner")
    @patch("kobench.tracks.pairwise.time")
    def test_normal_comparison(self, mock_time, mock_runner, mock_judge):
        """정상 비교: 2모델 1프롬프트 → 정방향+역방향"""
        mock_judge.score_pairwise.side_effect = [
            {"winner": "A", "reasoning": "A가 더 좋음"},
            {"winner": "B", "reasoning": "B가 더 좋음"},  # 역방향 B = 정방향 A
        ]
        responses = {
            "m1": {"p1": "답변1"},
            "m2": {"p1": "답변2"},
        }

        result = t7._run_comparisons(["m1", "m2"], responses, {})

        assert len(result) == 1
        assert result[0]["winner"] == "A"  # A 합의
        assert result[0]["model_a"] == "m1"
        assert result[0]["model_b"] == "m2"

    @patch.object(t7, "PROMPTS", [
        {"id": "p1", "category": "test", "prompt": "질문1"},
    ])
    @patch("kobench.tracks.pairwise.runner")
    @patch("kobench.tracks.pairwise.time")
    def test_both_empty_responses(self, mock_time, mock_runner):
        """양쪽 모두 빈 응답 → TIE"""
        responses = {"m1": {"p1": ""}, "m2": {"p1": ""}}
        result = t7._run_comparisons(["m1", "m2"], responses, {})

        assert result[0]["winner"] == "TIE"

    @patch.object(t7, "PROMPTS", [
        {"id": "p1", "category": "test", "prompt": "질문1"},
    ])
    @patch("kobench.tracks.pairwise.runner")
    @patch("kobench.tracks.pairwise.time")
    def test_model_a_empty(self, mock_time, mock_runner):
        """model_a만 빈 응답 → B 승"""
        responses = {"m1": {"p1": ""}, "m2": {"p1": "답변"}}
        result = t7._run_comparisons(["m1", "m2"], responses, {})

        assert result[0]["winner"] == "B"

    @patch.object(t7, "PROMPTS", [
        {"id": "p1", "category": "test", "prompt": "질문1"},
    ])
    @patch("kobench.tracks.pairwise.runner")
    @patch("kobench.tracks.pairwise.time")
    def test_checkpoint_skip(self, mock_time, mock_runner):
        """이미 비교 완료된 키는 스킵"""
        checkpoint = {
            "comparisons": [{"model_a": "m1", "model_b": "m2", "prompt_id": "p1", "winner": "A"}],
            "comparison_keys": ["m1|m2|p1"],
        }
        responses = {"m1": {"p1": "답변1"}, "m2": {"p1": "답변2"}}

        result = t7._run_comparisons(["m1", "m2"], responses, checkpoint)
        assert len(result) == 1  # 기존 1건 유지, 추가 비교 없음


# ── _build_summary ───────────────────────────────────────────────────────────


class TestBuildSummary:
    """Elo 기반 모델 순위 요약"""

    def test_basic_ranking(self):
        """Elo 점수 기반 순위 생성"""
        elo_scores = {
            "m1": {"elo": 1100, "ci_lower": 1050, "ci_upper": 1150, "wins": 5, "losses": 1},
            "m2": {"elo": 900, "ci_lower": 850, "ci_upper": 950, "wins": 1, "losses": 5},
        }
        summary = t7._build_summary(elo_scores, [], ["m1", "m2"])

        assert summary["m1"]["rank"] == 1
        assert summary["m2"]["rank"] == 2
        assert summary["m1"]["elo"] == 1100

    def test_field_completeness(self):
        """요약에 필수 필드가 모두 포함"""
        elo_scores = {
            "m1": {"elo": 1000, "ci_lower": 950, "ci_upper": 1050, "wins": 3, "losses": 3},
        }
        summary = t7._build_summary(elo_scores, [], ["m1"])

        required_fields = {"elo", "ci_lower", "ci_upper", "wins", "losses", "rank"}
        assert required_fields == set(summary["m1"].keys())

    def test_three_models_ordering(self):
        """3개 모델 순위 정렬"""
        elo_scores = {
            "m1": {"elo": 800, "ci_lower": 750, "ci_upper": 850, "wins": 0, "losses": 4},
            "m2": {"elo": 1200, "ci_lower": 1150, "ci_upper": 1250, "wins": 4, "losses": 0},
            "m3": {"elo": 1000, "ci_lower": 950, "ci_upper": 1050, "wins": 2, "losses": 2},
        }
        summary = t7._build_summary(elo_scores, [], ["m1", "m2", "m3"])

        assert summary["m2"]["rank"] == 1
        assert summary["m3"]["rank"] == 2
        assert summary["m1"]["rank"] == 3


# ── run ──────────────────────────────────────────────────────────────────────


class TestRun:
    """Track 7 메인 실행 흐름"""

    @patch.object(t7, "PROMPTS", [
        {"id": "p1", "category": "test", "prompt": "질문1"},
    ])
    @patch("kobench.tracks.pairwise.runner")
    @patch("kobench.tracks.pairwise.judge")
    @patch("kobench.tracks.pairwise.scoring")
    @patch("kobench.tracks.pairwise.time")
    def test_full_flow(self, mock_time, mock_scoring, mock_judge, mock_runner):
        """전체 흐름: 응답 수집 → 비교 → Elo 산출"""
        mock_runner.load_checkpoint.return_value = None
        mock_runner.wait_for_ollama.return_value = True
        mock_runner.switch_model.return_value = True
        mock_runner.generate.side_effect = [
            {"response": "답변1"},
            {"response": "답변2"},
        ]
        mock_judge.score_pairwise.side_effect = [
            {"winner": "A", "reasoning": "good"},
            {"winner": "B", "reasoning": "good"},  # reverse
        ]
        mock_scoring.fit_bradley_terry.return_value = {
            "m1": {"elo": 1100, "ci_lower": 1050, "ci_upper": 1150, "wins": 1, "losses": 0},
            "m2": {"elo": 900, "ci_lower": 850, "ci_upper": 950, "wins": 0, "losses": 1},
        }

        result = t7.run(["m1", "m2"])

        assert result["track"] == "pairwise"
        assert "error" not in result
        assert result["summary"]["m1"]["rank"] == 1

    @patch.object(t7, "PROMPTS", [
        {"id": "p1", "category": "test", "prompt": "질문1"},
    ])
    @patch("kobench.tracks.pairwise.runner")
    def test_ollama_unavailable(self, mock_runner):
        """Ollama 연결 실패 시 에러 반환"""
        mock_runner.load_checkpoint.return_value = None
        mock_runner.wait_for_ollama.return_value = False

        result = t7.run(["m1", "m2"])
        assert "error" in result

    @patch.object(t7, "PROMPTS", [
        {"id": "p1", "category": "test", "prompt": "질문1"},
    ])
    @patch("kobench.tracks.pairwise.runner")
    @patch("kobench.tracks.pairwise.time")
    def test_insufficient_models(self, mock_time, mock_runner):
        """유효 모델 부족 시 에러 반환"""
        mock_runner.load_checkpoint.return_value = None
        mock_runner.wait_for_ollama.return_value = True
        mock_runner.switch_model.return_value = False  # 모든 모델 로딩 실패

        result = t7.run(["m1", "m2"])
        assert "error" in result
