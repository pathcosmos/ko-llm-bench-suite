"""Track 실행 흐름 통합 테스트

Track 7 (Pairwise) 최소 실행 + 체크포인트 이어하기
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from kobench.scoring import fit_bradley_terry
from kobench.judge import score_pairwise
from kobench import runner


class TestTrack7MinimalExecution:
    """Track 7 최소 실행 — 2모델, 2프롬프트 → elo 결과"""

    @patch("kobench.judge._call_judge")
    def test_minimal_pairwise_to_elo(self, mock_call):
        """2개 모델 × 2개 프롬프트 × 2방향 = 8개 judge 호출"""
        models = ["model_a", "model_b"]
        prompts = [
            {"id": "p1", "category": "test", "prompt": "질문1"},
            {"id": "p2", "category": "test", "prompt": "질문2"},
        ]

        # forward: A wins, reverse: B wins (=A wins both directions for p1)
        # forward: tie, reverse: tie for p2
        mock_call.side_effect = [
            '{"winner": "A", "reasoning": "A가 낫다"}',  # p1 forward
            '{"winner": "B", "reasoning": "B가 낫다"}',  # p1 reverse (=A wins)
            '{"winner": "tie", "reasoning": "비슷"}',    # p2 forward
            '{"winner": "tie", "reasoning": "비슷"}',    # p2 reverse
        ]

        comparisons = []
        mock_responses = {
            ("model_a", "p1"): "A의 답변1",
            ("model_b", "p1"): "B의 답변1",
            ("model_a", "p2"): "A의 답변2",
            ("model_b", "p2"): "B의 답변2",
        }

        for p in prompts:
            resp_a = mock_responses[("model_a", p["id"])]
            resp_b = mock_responses[("model_b", p["id"])]

            # Forward
            forward = score_pairwise(p["prompt"], resp_a, resp_b)
            # Reverse
            reverse = score_pairwise(p["prompt"], resp_b, resp_a)

            # Winner resolution
            if forward["winner"] == "A" and reverse["winner"] == "B":
                winner = "A"
            elif forward["winner"] == "B" and reverse["winner"] == "A":
                winner = "A"  # reversed means B→A
            elif forward["winner"] == reverse["winner"]:
                winner = forward["winner"]
            else:
                winner = "TIE"

            comparisons.append({
                "model_a": "model_a",
                "model_b": "model_b",
                "winner": winner,
            })

        elo = fit_bradley_terry(comparisons)
        assert "model_a" in elo
        assert "model_b" in elo
        assert "elo" in elo["model_a"]
        assert "ci_lower" in elo["model_a"]
        assert "ci_upper" in elo["model_a"]


class TestCheckpointResume:
    """체크포인트 저장/로드 → 이어하기 시뮬레이션"""

    def test_checkpoint_save_and_resume(self, tmp_results_dir):
        # 첫 실행: 부분 결과 저장
        partial_data = {
            "completed_pairs": [("model_a", "model_b")],
            "comparisons": [
                {"model_a": "model_a", "model_b": "model_b", "winner": "A"},
            ],
            "prompt_idx": 1,
        }
        runner.save_checkpoint(partial_data, "pairwise")

        # 재개: 체크포인트 로드
        loaded = runner.load_checkpoint("pairwise")
        assert loaded is not None
        assert loaded["prompt_idx"] == 1
        assert len(loaded["comparisons"]) == 1

        # 추가 결과 병합
        loaded["comparisons"].append(
            {"model_a": "model_a", "model_b": "model_b", "winner": "B"},
        )
        loaded["prompt_idx"] = 2

        # 최종 저장
        runner.save_checkpoint(loaded, "pairwise")
        final = runner.load_checkpoint("pairwise")
        assert final["prompt_idx"] == 2
        assert len(final["comparisons"]) == 2
