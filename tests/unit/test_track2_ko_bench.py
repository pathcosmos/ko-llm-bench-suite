"""eval_framework/tracks/track2_ko_bench.py 단위 테스트"""

import json
import pytest
from unittest.mock import patch, MagicMock

from eval_framework.tracks.track2_ko_bench import (
    _scores_mean,
    _make_error_entry,
    _make_partial_entry,
    _perf_summary,
    _build_summary,
    _load_questions,
    run,
    TRACK_NAME,
)


# ═══════════════════════════════════════════════════════════════════════════════
# _scores_mean 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoresMean:
    """_scores_mean: score_with_criteria 결과에서 평균 점수 계산"""

    def test_normal_scores(self):
        """여러 점수의 평균을 올바르게 계산"""
        result = {"scores": {"정확성": 8, "유용성": 6, "자연스러움": 10}}
        assert _scores_mean(result) == 8.0

    def test_single_score(self):
        """점수가 하나일 때"""
        result = {"scores": {"정확성": 7}}
        assert _scores_mean(result) == 7.0

    def test_empty_scores(self):
        """scores 딕셔너리가 비어 있을 때 0.0 반환"""
        result = {"scores": {}}
        assert _scores_mean(result) == 0.0

    def test_missing_scores_key(self):
        """scores 키가 없을 때 0.0 반환"""
        result = {"reasoning": "좋음"}
        assert _scores_mean(result) == 0.0

    def test_fractional_mean(self):
        """소수점 평균"""
        result = {"scores": {"a": 7, "b": 8}}
        assert _scores_mean(result) == 7.5


# ═══════════════════════════════════════════════════════════════════════════════
# _perf_summary 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerfSummary:
    """_perf_summary: 러너 결과에서 성능 지표 추출"""

    def test_full_result(self):
        """모든 성능 필드가 있는 경우"""
        result = {
            "tokens_per_sec": 25.5,
            "eval_count": 100,
            "wall_time_s": 4.0,
            "response": "답변",
        }
        summary = _perf_summary(result)
        assert summary == {
            "tokens_per_sec": 25.5,
            "eval_count": 100,
            "wall_time_s": 4.0,
        }

    def test_missing_fields_default_zero(self):
        """성능 필드가 없으면 0으로 기본값"""
        result = {"response": "답변"}
        summary = _perf_summary(result)
        assert summary == {
            "tokens_per_sec": 0,
            "eval_count": 0,
            "wall_time_s": 0,
        }

    def test_partial_fields(self):
        """일부 필드만 있는 경우"""
        result = {"tokens_per_sec": 10.0}
        summary = _perf_summary(result)
        assert summary["tokens_per_sec"] == 10.0
        assert summary["eval_count"] == 0
        assert summary["wall_time_s"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# _make_error_entry 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestMakeErrorEntry:
    """_make_error_entry: Turn 1 오류 시 결과 엔트리 생성"""

    def _sample_question(self):
        return {"turn1": "질문1", "turn2": "질문2"}

    def test_basic_structure(self):
        """기본 구조 검증"""
        q = self._sample_question()
        entry = _make_error_entry("model-a", "writing", 0, q, "timeout")

        assert entry["model"] == "model-a"
        assert entry["category"] == "writing"
        assert entry["question_idx"] == 0
        assert entry["turn1_question"] == "질문1"
        assert entry["turn2_question"] == "질문2"
        assert entry["turn1_answer"] == ""
        assert entry["turn2_answer"] == ""
        assert entry["error"] == "timeout"

    def test_scores_are_empty(self):
        """오류 엔트리의 점수는 빈 딕셔너리"""
        q = self._sample_question()
        entry = _make_error_entry("model-a", "math", 3, q, "connection error")

        assert entry["turn1_scores"]["scores"] == {}
        assert entry["turn2_scores"]["scores"] == {}
        assert entry["turn1_mean"] == 0.0
        assert entry["turn2_mean"] == 0.0

    def test_turn2_error_message(self):
        """Turn 2 스킵 메시지 확인"""
        q = self._sample_question()
        entry = _make_error_entry("model-a", "coding", 1, q, "OOM")
        assert "turn1 실패로 스킵" in entry["turn2_scores"]["error"]

    def test_perf_empty(self):
        """성능 데이터가 빈 딕셔너리"""
        q = self._sample_question()
        entry = _make_error_entry("model-a", "stem", 0, q, "err")
        assert entry["turn1_perf"] == {}
        assert entry["turn2_perf"] == {}


# ═══════════════════════════════════════════════════════════════════════════════
# _make_partial_entry 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestMakePartialEntry:
    """_make_partial_entry: Turn 2 오류 시 결과 엔트리 생성"""

    def _sample_args(self):
        q = {"turn1": "질문1", "turn2": "질문2"}
        turn1_result = {
            "tokens_per_sec": 15.0,
            "eval_count": 50,
            "wall_time_s": 3.0,
        }
        turn1_score = {"scores": {"정확성": 8, "유용성": 6}, "reasoning": "좋음"}
        return q, turn1_result, turn1_score

    def test_basic_structure(self):
        """기본 구조 — Turn 1 정상, Turn 2 오류"""
        q, t1_result, t1_score = self._sample_args()
        entry = _make_partial_entry(
            "model-b", "reasoning", 2, q,
            "Turn1 답변", t1_result, t1_score,
            "timeout",
        )

        assert entry["model"] == "model-b"
        assert entry["category"] == "reasoning"
        assert entry["question_idx"] == 2
        assert entry["turn1_answer"] == "Turn1 답변"
        assert entry["turn2_answer"] == ""

    def test_turn1_scores_preserved(self):
        """Turn 1 점수가 그대로 유지"""
        q, t1_result, t1_score = self._sample_args()
        entry = _make_partial_entry(
            "model-b", "writing", 0, q,
            "답변", t1_result, t1_score,
            "err",
        )
        assert entry["turn1_scores"] == t1_score
        assert entry["turn1_mean"] == 7.0  # (8+6)/2

    def test_turn2_scores_error(self):
        """Turn 2 점수는 오류 상태"""
        q, t1_result, t1_score = self._sample_args()
        entry = _make_partial_entry(
            "model-b", "math", 0, q,
            "답변", t1_result, t1_score,
            "network error",
        )
        assert entry["turn2_scores"]["scores"] == {}
        assert entry["turn2_scores"]["error"] == "network error"
        assert entry["turn2_mean"] == 0.0

    def test_error_field_format(self):
        """error 필드에 'turn2:' 접두사"""
        q, t1_result, t1_score = self._sample_args()
        entry = _make_partial_entry(
            "model-b", "coding", 0, q,
            "답변", t1_result, t1_score,
            "500 error",
        )
        assert entry["error"] == "turn2: 500 error"

    def test_turn1_perf_populated(self):
        """Turn 1 성능 지표 포함"""
        q, t1_result, t1_score = self._sample_args()
        entry = _make_partial_entry(
            "model-b", "stem", 0, q,
            "답변", t1_result, t1_score,
            "err",
        )
        assert entry["turn1_perf"]["tokens_per_sec"] == 15.0
        assert entry["turn2_perf"] == {}


# ═══════════════════════════════════════════════════════════════════════════════
# _build_summary 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildSummary:
    """_build_summary: 모델 x 카테고리별 요약 통계 생성"""

    def test_single_model_single_category(self):
        """단일 모델, 단일 카테고리"""
        results = [
            {"model": "m1", "category": "writing", "turn1_mean": 8.0, "turn2_mean": 6.0},
            {"model": "m1", "category": "writing", "turn1_mean": 7.0, "turn2_mean": 9.0},
        ]
        summary = _build_summary(results)
        assert "m1" in summary
        assert "writing" in summary["m1"]
        assert summary["m1"]["writing"]["turn1_mean"] == 7.5
        assert summary["m1"]["writing"]["turn2_mean"] == 7.5
        assert summary["m1"]["writing"]["overall_mean"] == 7.5

    def test_multiple_models(self):
        """여러 모델"""
        results = [
            {"model": "m1", "category": "math", "turn1_mean": 8.0, "turn2_mean": 6.0},
            {"model": "m2", "category": "math", "turn1_mean": 9.0, "turn2_mean": 7.0},
        ]
        summary = _build_summary(results)
        assert "m1" in summary
        assert "m2" in summary
        assert summary["m1"]["math"]["turn1_mean"] == 8.0
        assert summary["m2"]["math"]["turn1_mean"] == 9.0

    def test_multiple_categories(self):
        """한 모델에 여러 카테고리"""
        results = [
            {"model": "m1", "category": "writing", "turn1_mean": 8.0, "turn2_mean": 7.0},
            {"model": "m1", "category": "coding", "turn1_mean": 6.0, "turn2_mean": 5.0},
        ]
        summary = _build_summary(results)
        assert "writing" in summary["m1"]
        assert "coding" in summary["m1"]

    def test_zero_scores_excluded(self):
        """turn_mean이 0인 항목은 평균 계산에서 제외"""
        results = [
            {"model": "m1", "category": "writing", "turn1_mean": 8.0, "turn2_mean": 0.0},
            {"model": "m1", "category": "writing", "turn1_mean": 6.0, "turn2_mean": 4.0},
        ]
        summary = _build_summary(results)
        assert summary["m1"]["writing"]["turn1_mean"] == 7.0
        assert summary["m1"]["writing"]["turn2_mean"] == 4.0

    def test_empty_results(self):
        """결과가 비어 있을 때"""
        summary = _build_summary([])
        assert summary == {}

    def test_overall_mean_calculation(self):
        """overall_mean은 t1과 t2 점수 전체의 평균"""
        results = [
            {"model": "m1", "category": "stem", "turn1_mean": 10.0, "turn2_mean": 6.0},
        ]
        summary = _build_summary(results)
        assert summary["m1"]["stem"]["overall_mean"] == 8.0

    def test_values_are_rounded(self):
        """결과값이 소수점 둘째 자리로 반올림"""
        results = [
            {"model": "m1", "category": "math", "turn1_mean": 7.0, "turn2_mean": 8.0},
            {"model": "m1", "category": "math", "turn1_mean": 7.0, "turn2_mean": 9.0},
            {"model": "m1", "category": "math", "turn1_mean": 8.0, "turn2_mean": 7.0},
        ]
        summary = _build_summary(results)
        s = summary["m1"]["math"]
        assert s["turn1_mean"] == 7.33
        assert s["turn2_mean"] == 8.0


# ═══════════════════════════════════════════════════════════════════════════════
# _load_questions 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadQuestions:
    """_load_questions: JSON 파일 로드"""

    def test_returns_none_when_file_missing(self, tmp_path):
        """파일이 없으면 None 반환"""
        with patch("eval_framework.tracks.track2_ko_bench.config.DATA_DIR", tmp_path):
            result = _load_questions()
            assert result is None

    def test_loads_json_successfully(self, tmp_path):
        """JSON 파일이 있으면 정상 로드"""
        ko_bench_dir = tmp_path / "ko_bench"
        ko_bench_dir.mkdir()
        data = {"writing": [{"turn1": "q1", "turn2": "q2"}]}
        (ko_bench_dir / "questions.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8",
        )
        with patch("eval_framework.tracks.track2_ko_bench.config.DATA_DIR", tmp_path):
            result = _load_questions()
            assert result == data


# ═══════════════════════════════════════════════════════════════════════════════
# run 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestRun:
    """run: 전체 평가 실행 흐름"""

    @pytest.fixture(autouse=True)
    def _setup_patches(self, tmp_path):
        """공통 패치 설정"""
        patches = [
            patch("eval_framework.tracks.track2_ko_bench.runner.load_checkpoint", return_value=None),
            patch("eval_framework.tracks.track2_ko_bench.runner.switch_model", return_value=True),
            patch("eval_framework.tracks.track2_ko_bench.runner.unload_all_models"),
            patch("eval_framework.tracks.track2_ko_bench.runner.save_checkpoint"),
            patch("eval_framework.tracks.track2_ko_bench.runner.save_results_incremental", return_value="results/test.json"),
            patch("eval_framework.tracks.track2_ko_bench.time.sleep"),
            patch("eval_framework.tracks.track2_ko_bench.config.TRACK2_CATEGORIES", ["writing"]),
            patch(
                "eval_framework.tracks.track2_ko_bench._load_questions",
                return_value=None,
            ),
            patch(
                "eval_framework.tracks.track2_ko_bench.QUESTIONS",
                {"writing": [{"turn1": "질문1", "turn2": "질문2"}]},
            ),
        ]
        for p in patches:
            p.start()

        yield
        patch.stopall()

    def test_basic_run_returns_structure(self):
        """기본 실행 시 올바른 출력 구조"""
        chat_result = {
            "response": "답변입니다",
            "error": None,
            "tokens_per_sec": 10.0,
            "eval_count": 50,
            "wall_time_s": 2.0,
        }
        judge_result = {
            "scores": {"문체": 8, "구조": 7, "창의성": 9},
            "reasoning": "좋음",
        }

        with patch("eval_framework.tracks.track2_ko_bench.runner.chat", return_value=chat_result):
            with patch("eval_framework.tracks.track2_ko_bench.judge.score_with_criteria", return_value=judge_result):
                output = run(["test-model"])

        assert output["track"] == TRACK_NAME
        assert "results" in output
        assert "summary" in output
        assert len(output["results"]) == 1

    def test_turn1_error_creates_error_entry(self):
        """Turn 1 오류 시 error entry 생성"""
        error_result = {
            "response": "",
            "error": "model not found",
            "tokens_per_sec": 0,
            "eval_count": 0,
            "wall_time_s": 0,
        }

        with patch("eval_framework.tracks.track2_ko_bench.runner.chat", return_value=error_result):
            output = run(["test-model"])

        assert len(output["results"]) == 1
        entry = output["results"][0]
        assert entry["error"] == "model not found"
        assert entry["turn1_mean"] == 0.0
        assert entry["turn2_mean"] == 0.0

    def test_turn2_error_creates_partial_entry(self):
        """Turn 2 오류 시 partial entry 생성 (Turn 1은 정상)"""
        call_count = 0

        def mock_chat(model, messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "response": "Turn1 답변",
                    "error": None,
                    "tokens_per_sec": 10.0,
                    "eval_count": 50,
                    "wall_time_s": 2.0,
                }
            else:
                return {
                    "response": "",
                    "error": "timeout",
                    "tokens_per_sec": 0,
                    "eval_count": 0,
                    "wall_time_s": 0,
                }

        judge_result = {
            "scores": {"문체": 8, "구조": 7, "창의성": 9},
            "reasoning": "좋음",
        }

        with patch("eval_framework.tracks.track2_ko_bench.runner.chat", side_effect=mock_chat):
            with patch("eval_framework.tracks.track2_ko_bench.judge.score_with_criteria", return_value=judge_result):
                output = run(["test-model"])

        assert len(output["results"]) == 1
        entry = output["results"][0]
        assert "turn2" in entry["error"]
        assert entry["turn1_answer"] == "Turn1 답변"
        assert entry["turn2_answer"] == ""

    def test_model_switch_failure_skips_model(self):
        """모델 전환 실패 시 해당 모델 스킵"""
        with patch("eval_framework.tracks.track2_ko_bench.runner.switch_model", return_value=False):
            output = run(["bad-model"])

        assert len(output["results"]) == 0

    def test_checkpoint_restores_completed(self):
        """체크포인트에서 이미 완료된 항목은 스킵"""
        checkpoint_data = {
            "results": [{
                "model": "test-model",
                "category": "writing",
                "question_idx": 0,
                "turn1_mean": 8.0,
                "turn2_mean": 7.0,
                "error": None,
            }],
        }
        with patch("eval_framework.tracks.track2_ko_bench.runner.load_checkpoint", return_value=checkpoint_data):
            with patch("eval_framework.tracks.track2_ko_bench.runner.chat") as mock_chat:
                output = run(["test-model"])

        mock_chat.assert_not_called()
        assert len(output["results"]) == 1

    def test_successful_both_turns_full_entry(self):
        """양쪽 턴 모두 성공 시 완전한 엔트리 생성"""
        chat_result = {
            "response": "답변입니다",
            "error": None,
            "tokens_per_sec": 12.0,
            "eval_count": 60,
            "wall_time_s": 2.5,
        }
        judge_result = {
            "scores": {"문체": 8, "구조": 7, "창의성": 9},
            "reasoning": "좋음",
        }

        with patch("eval_framework.tracks.track2_ko_bench.runner.chat", return_value=chat_result):
            with patch("eval_framework.tracks.track2_ko_bench.judge.score_with_criteria", return_value=judge_result):
                output = run(["test-model"])

        entry = output["results"][0]
        assert entry["error"] is None
        assert entry["turn1_mean"] == 8.0  # (8+7+9)/3
        assert entry["turn2_mean"] == 8.0
        assert entry["turn1_perf"]["tokens_per_sec"] == 12.0
        assert entry["turn2_perf"]["tokens_per_sec"] == 12.0
