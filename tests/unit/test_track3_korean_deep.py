"""eval_framework/tracks/track3_korean_deep.py 단위 테스트"""

import json
import pytest
from unittest.mock import patch, MagicMock, call

from eval_framework.tracks.track3_korean_deep import (
    _normalize,
    _score_exact,
    _score_contains,
    _score_llm_judge,
    _load_questions,
    _build_summary,
    _print_summary,
    run,
    TRACK_NAME,
    CATEGORY_CRITERIA,
)


# ═══════════════════════════════════════════════════════════════════════════════
# _normalize 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalize:
    """_normalize: 채점용 텍스트 정규화"""

    def test_strip_whitespace(self):
        """앞뒤 공백 제거"""
        assert _normalize("  hello  ") == "hello"

    def test_remove_spaces(self):
        """내부 공백 제거"""
        assert _normalize("hello world") == "helloworld"

    def test_remove_punctuation(self):
        """구두점 제거"""
        assert _normalize("안녕하세요!") == "안녕하세요"
        assert _normalize("테스트.결과") == "테스트결과"

    def test_remove_various_punctuation(self):
        """다양한 구두점 문자 제거"""
        assert _normalize("a,b;c:d'e") == "abcde"
        assert _normalize("(test)") == "test"
        assert _normalize("[hello]") == "hello"
        assert _normalize("{world}") == "world"
        assert _normalize("a~b·c…d") == "abcd"

    def test_lowercase(self):
        """소문자 변환"""
        assert _normalize("HELLO") == "hello"
        assert _normalize("Hello World") == "helloworld"

    def test_nfc_normalization(self):
        """유니코드 NFC 정규화 — NFD 한글이 NFC로 변환"""
        # NFD로 분해된 '가' = 'ㄱ' + 'ㅏ'
        nfd_ga = "\u1100\u1161"  # NFD 형태의 '가'
        assert _normalize(nfd_ga) == "가"

    def test_empty_string(self):
        """빈 문자열"""
        assert _normalize("") == ""

    def test_tabs_and_newlines(self):
        """탭과 개행 제거"""
        assert _normalize("hello\tworld\n") == "helloworld"

    def test_ascii_quotes_removed(self):
        """ASCII 따옴표/큰따옴표 제거"""
        assert _normalize('"안녕"') == "안녕"
        assert _normalize("'테스트'") == "테스트"


# ═══════════════════════════════════════════════════════════════════════════════
# _score_exact 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreExact:
    """_score_exact: 정규화 후 문자열 일치 채점"""

    def test_exact_match(self):
        """정확히 일치하면 1.0"""
        assert _score_exact("안녕하세요", "안녕하세요") == 1.0

    def test_match_with_whitespace(self):
        """공백 차이는 무시"""
        assert _score_exact("안녕 하세요", "안녕하세요") == 1.0

    def test_match_with_punctuation(self):
        """구두점 차이는 무시"""
        assert _score_exact("안녕하세요!", "안녕하세요") == 1.0

    def test_match_case_insensitive(self):
        """대소문자 차이는 무시"""
        assert _score_exact("Hello", "hello") == 1.0

    def test_no_match(self):
        """불일치하면 0.0"""
        assert _score_exact("안녕하세요", "감사합니다") == 0.0

    def test_empty_strings(self):
        """빈 문자열끼리 비교"""
        assert _score_exact("", "") == 1.0

    def test_partial_match(self):
        """부분 일치는 0.0"""
        assert _score_exact("안녕", "안녕하세요") == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# _score_contains 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreContains:
    """_score_contains: 키워드 포함 여부 채점"""

    def test_keyword_present(self):
        """키워드가 포함되면 1.0"""
        assert _score_contains("오늘 날씨가 좋습니다", ["날씨"]) == 1.0

    def test_multiple_keywords_one_match(self):
        """여러 키워드 중 하나만 포함되어도 1.0"""
        assert _score_contains("오늘 날씨가 좋습니다", ["비", "날씨", "눈"]) == 1.0

    def test_no_keyword_match(self):
        """키워드가 하나도 없으면 0.0"""
        assert _score_contains("오늘 날씨가 좋습니다", ["비", "눈", "바람"]) == 0.0

    def test_empty_keywords(self):
        """키워드 목록이 비어 있으면 0.0"""
        assert _score_contains("아무 텍스트", []) == 0.0

    def test_empty_response(self):
        """응답이 비어 있으면 0.0"""
        assert _score_contains("", ["키워드"]) == 0.0

    def test_whitespace_stripped(self):
        """응답 앞뒤 공백 제거 후 검색"""
        assert _score_contains("  날씨  ", ["날씨"]) == 1.0

    def test_keyword_not_normalized(self):
        """키워드 매칭은 원본 텍스트(strip만) 기준"""
        # _score_contains는 _normalize를 사용하지 않음
        assert _score_contains("Hello World", ["Hello"]) == 1.0
        assert _score_contains("Hello World", ["hello"]) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# _score_llm_judge 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreLlmJudge:
    """_score_llm_judge: LLM Judge 채점 (mock)"""

    @patch("eval_framework.tracks.track3_korean_deep.judge.score_response")
    def test_normal_score(self, mock_score):
        """정상 점수 반환 — 1-10을 0.0-1.0으로 정규화"""
        mock_score.return_value = {"score": 8, "reasoning": "좋은 답변", "error": None}
        result = _score_llm_judge("질문", "응답", "한국 문화 상식")

        assert result["score"] == 0.8
        assert result["score_raw"] == 8
        assert result["reasoning"] == "좋은 답변"
        assert result["error"] is None

    @patch("eval_framework.tracks.track3_korean_deep.judge.score_response")
    def test_zero_score(self, mock_score):
        """0점 처리"""
        mock_score.return_value = {"score": 0, "reasoning": "부적절", "error": None}
        result = _score_llm_judge("질문", "응답", "맞춤법/문법")

        assert result["score"] == 0.0
        assert result["score_raw"] == 0

    @patch("eval_framework.tracks.track3_korean_deep.judge.score_response")
    def test_perfect_score(self, mock_score):
        """만점 처리"""
        mock_score.return_value = {"score": 10, "reasoning": "완벽", "error": None}
        result = _score_llm_judge("질문", "응답", "사자성어/관용구")

        assert result["score"] == 1.0
        assert result["score_raw"] == 10

    @patch("eval_framework.tracks.track3_korean_deep.judge.score_response")
    def test_missing_score_defaults_zero(self, mock_score):
        """score 키가 없으면 0으로 처리"""
        mock_score.return_value = {"reasoning": "오류", "error": "API 오류"}
        result = _score_llm_judge("질문", "응답", "감정/뉘앙스")

        assert result["score"] == 0.0
        assert result["score_raw"] == 0
        assert result["error"] == "API 오류"

    @patch("eval_framework.tracks.track3_korean_deep.judge.score_response")
    def test_calls_judge_with_correct_criteria(self, mock_score):
        """카테고리에 맞는 criteria 전달"""
        mock_score.return_value = {"score": 7, "reasoning": "", "error": None}
        _score_llm_judge("질문텍스트", "응답텍스트", "존댓말/반말 전환")

        mock_score.assert_called_once_with(
            prompt="질문텍스트",
            response="응답텍스트",
            category="존댓말/반말 전환",
            criteria=CATEGORY_CRITERIA["존댓말/반말 전환"],
        )

    @patch("eval_framework.tracks.track3_korean_deep.judge.score_response")
    def test_unknown_category_empty_criteria(self, mock_score):
        """알 수 없는 카테고리 — 빈 criteria"""
        mock_score.return_value = {"score": 5, "reasoning": "", "error": None}
        _score_llm_judge("질문", "응답", "알수없는카테고리")

        mock_score.assert_called_once_with(
            prompt="질문",
            response="응답",
            category="알수없는카테고리",
            criteria="",
        )

    @patch("eval_framework.tracks.track3_korean_deep.judge.score_response")
    def test_missing_reasoning(self, mock_score):
        """reasoning 키가 없으면 빈 문자열"""
        mock_score.return_value = {"score": 6, "error": None}
        result = _score_llm_judge("질문", "응답", "숫자/단위")

        assert result["reasoning"] == ""


# ═══════════════════════════════════════════════════════════════════════════════
# _load_questions 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadQuestions:
    """_load_questions: questions.json 로드"""

    def test_loads_json_successfully(self, tmp_path):
        """JSON 파일이 있으면 정상 로드"""
        korean_deep_dir = tmp_path / "korean_deep"
        korean_deep_dir.mkdir()
        data = [{"id": 1, "question": "테스트", "category": "문화"}]
        (korean_deep_dir / "questions.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8",
        )
        with patch(
            "eval_framework.tracks.track3_korean_deep.QUESTIONS_PATH",
            korean_deep_dir / "questions.json",
        ):
            result = _load_questions()
            assert result == data

    def test_file_not_found_raises(self, tmp_path):
        """파일이 없으면 FileNotFoundError 발생"""
        with patch(
            "eval_framework.tracks.track3_korean_deep.QUESTIONS_PATH",
            tmp_path / "nonexistent" / "questions.json",
        ):
            with pytest.raises(FileNotFoundError):
                _load_questions()


# ═══════════════════════════════════════════════════════════════════════════════
# _build_summary 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildSummary:
    """_build_summary: 모델별 · 카테고리별 요약 통계 생성"""

    def test_single_model_single_category(self):
        """단일 모델, 단일 카테고리"""
        results = [
            {"model": "m1", "category": "문화", "score": 1.0},
            {"model": "m1", "category": "문화", "score": 0.5},
        ]
        summary = _build_summary(results)
        assert "m1" in summary
        assert "문화" in summary["m1"]
        assert summary["m1"]["문화"]["avg_score"] == 0.75
        assert summary["m1"]["문화"]["n"] == 2

    def test_accuracy_calculation(self):
        """accuracy: score >= 0.5 인 비율"""
        results = [
            {"model": "m1", "category": "문화", "score": 1.0},
            {"model": "m1", "category": "문화", "score": 0.3},
            {"model": "m1", "category": "문화", "score": 0.5},
            {"model": "m1", "category": "문화", "score": 0.0},
        ]
        summary = _build_summary(results)
        # 1.0 >= 0.5, 0.3 < 0.5, 0.5 >= 0.5, 0.0 < 0.5 → 2/4 = 0.5
        assert summary["m1"]["문화"]["accuracy"] == 0.5

    def test_multiple_models(self):
        """여러 모델"""
        results = [
            {"model": "m1", "category": "문화", "score": 1.0},
            {"model": "m2", "category": "문화", "score": 0.5},
        ]
        summary = _build_summary(results)
        assert "m1" in summary
        assert "m2" in summary

    def test_multiple_categories(self):
        """한 모델에 여러 카테고리"""
        results = [
            {"model": "m1", "category": "문화", "score": 1.0},
            {"model": "m1", "category": "문법", "score": 0.7},
        ]
        summary = _build_summary(results)
        assert "문화" in summary["m1"]
        assert "문법" in summary["m1"]

    def test_overall_statistics(self):
        """_overall 전체 통계 포함"""
        results = [
            {"model": "m1", "category": "문화", "score": 1.0},
            {"model": "m1", "category": "문법", "score": 0.5},
        ]
        summary = _build_summary(results)
        overall = summary["m1"]["_overall"]
        assert overall["n"] == 2
        assert overall["avg_score"] == 0.75
        assert overall["accuracy"] == 1.0  # 둘 다 >= 0.5

    def test_empty_results(self):
        """결과가 비어 있을 때"""
        summary = _build_summary([])
        assert summary == {}

    def test_values_are_rounded(self):
        """결과값이 소수점 넷째 자리로 반올림"""
        results = [
            {"model": "m1", "category": "문화", "score": 0.333},
            {"model": "m1", "category": "문화", "score": 0.333},
            {"model": "m1", "category": "문화", "score": 0.334},
        ]
        summary = _build_summary(results)
        # avg = 0.33333... → round to 4 places = 0.3333
        assert summary["m1"]["문화"]["avg_score"] == 0.3333

    def test_all_below_threshold(self):
        """모든 점수가 0.5 미만 — accuracy 0.0"""
        results = [
            {"model": "m1", "category": "문화", "score": 0.1},
            {"model": "m1", "category": "문화", "score": 0.2},
        ]
        summary = _build_summary(results)
        assert summary["m1"]["문화"]["accuracy"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# _print_summary 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestPrintSummary:
    """_print_summary: 요약 테이블 출력 (출력 내용 검증)"""

    def test_prints_model_name(self, capsys):
        """모델 이름 출력"""
        summary = {
            "test-model": {
                "_overall": {"accuracy": 0.8, "avg_score": 0.75, "n": 10},
                "문화": {"accuracy": 0.9, "avg_score": 0.85, "n": 5},
            }
        }
        _print_summary(summary)
        captured = capsys.readouterr()
        assert "test-model" in captured.out

    def test_prints_overall_stats(self, capsys):
        """전체 통계 출력"""
        summary = {
            "m1": {
                "_overall": {"accuracy": 0.75, "avg_score": 0.6, "n": 20},
            }
        }
        _print_summary(summary)
        captured = capsys.readouterr()
        assert "75.0%" in captured.out
        assert "0.600" in captured.out
        assert "20" in captured.out

    def test_prints_category_stats(self, capsys):
        """카테고리별 통계 출력"""
        summary = {
            "m1": {
                "_overall": {"accuracy": 0.5, "avg_score": 0.5, "n": 2},
                "문화": {"accuracy": 0.5, "avg_score": 0.5, "n": 2},
            }
        }
        _print_summary(summary)
        captured = capsys.readouterr()
        assert "문화" in captured.out

    def test_empty_summary(self, capsys):
        """빈 요약 — 오류 없이 출력"""
        _print_summary({})
        captured = capsys.readouterr()
        assert "Track 3" in captured.out

    def test_multiple_models(self, capsys):
        """여러 모델 출력"""
        summary = {
            "m1": {"_overall": {"accuracy": 0.8, "avg_score": 0.7, "n": 5}},
            "m2": {"_overall": {"accuracy": 0.6, "avg_score": 0.5, "n": 5}},
        }
        _print_summary(summary)
        captured = capsys.readouterr()
        assert "m1" in captured.out
        assert "m2" in captured.out


# ═══════════════════════════════════════════════════════════════════════════════
# run 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestRun:
    """run: 전체 평가 실행 흐름"""

    SAMPLE_QUESTIONS = [
        {
            "id": "q1",
            "category": "한국 문화 상식",
            "question": "김치의 주재료는?",
            "answer_type": "exact",
            "expected_answer": "배추",
            "keywords": [],
        },
        {
            "id": "q2",
            "category": "사자성어/관용구",
            "question": "일석이조의 뜻은?",
            "answer_type": "contains",
            "expected_answer": "",
            "keywords": ["돌", "새"],
        },
        {
            "id": "q3",
            "category": "뉴스 스타일 요약",
            "question": "다음 기사를 요약하세요.",
            "answer_type": "llm_judge",
            "expected_answer": "",
            "keywords": [],
        },
    ]

    @pytest.fixture(autouse=True)
    def _setup_patches(self):
        """공통 패치 설정"""
        patches = [
            patch(
                "eval_framework.tracks.track3_korean_deep.runner.load_checkpoint",
                return_value=None,
            ),
            patch(
                "eval_framework.tracks.track3_korean_deep.runner.wait_for_ollama",
                return_value=True,
            ),
            patch(
                "eval_framework.tracks.track3_korean_deep.runner.switch_model",
                return_value=True,
            ),
            patch("eval_framework.tracks.track3_korean_deep.runner.unload_all_models"),
            patch("eval_framework.tracks.track3_korean_deep.runner.save_checkpoint"),
            patch(
                "eval_framework.tracks.track3_korean_deep.runner.save_results_incremental",
                return_value="results/test.json",
            ),
            patch("eval_framework.tracks.track3_korean_deep.time.sleep"),
            patch(
                "eval_framework.tracks.track3_korean_deep._load_questions",
                return_value=self.SAMPLE_QUESTIONS,
            ),
        ]
        for p in patches:
            p.start()
        yield
        patch.stopall()

    def test_basic_run_returns_structure(self):
        """기본 실행 시 올바른 출력 구조"""
        gen_result = {
            "response": "배추",
            "error": None,
            "tokens_per_sec": 10.0,
            "wall_time_s": 2.0,
        }
        judge_result = {"score": 8, "reasoning": "좋음", "error": None}

        with patch(
            "eval_framework.tracks.track3_korean_deep.runner.generate",
            return_value=gen_result,
        ):
            with patch(
                "eval_framework.tracks.track3_korean_deep.judge.score_response",
                return_value=judge_result,
            ):
                output = run(["test-model"])

        assert output["track"] == TRACK_NAME
        assert "results" in output
        assert "summary" in output
        assert "timestamp" in output
        assert len(output["results"]) == 3

    def test_exact_scoring(self):
        """exact 타입 문제 — 정확한 답이면 score=1.0"""
        gen_result = {
            "response": "배추",
            "error": None,
            "tokens_per_sec": 10.0,
            "wall_time_s": 1.0,
        }
        judge_result = {"score": 7, "reasoning": "", "error": None}

        with patch(
            "eval_framework.tracks.track3_korean_deep.runner.generate",
            return_value=gen_result,
        ):
            with patch(
                "eval_framework.tracks.track3_korean_deep.judge.score_response",
                return_value=judge_result,
            ):
                output = run(["test-model"])

        exact_result = [r for r in output["results"] if r["answer_type"] == "exact"][0]
        assert exact_result["score"] == 1.0

    def test_contains_scoring(self):
        """contains 타입 문제 — 키워드 포함 시 score=1.0"""
        gen_result = {
            "response": "하나의 돌로 두 마리 새를 잡는다",
            "error": None,
            "tokens_per_sec": 10.0,
            "wall_time_s": 1.0,
        }
        judge_result = {"score": 7, "reasoning": "", "error": None}

        with patch(
            "eval_framework.tracks.track3_korean_deep.runner.generate",
            return_value=gen_result,
        ):
            with patch(
                "eval_framework.tracks.track3_korean_deep.judge.score_response",
                return_value=judge_result,
            ):
                output = run(["test-model"])

        contains_result = [r for r in output["results"] if r["answer_type"] == "contains"][0]
        assert contains_result["score"] == 1.0

    def test_llm_judge_scoring(self):
        """llm_judge 타입 문제 — Phase 2에서 Judge 채점"""
        gen_result = {
            "response": "요약 결과입니다",
            "error": None,
            "tokens_per_sec": 10.0,
            "wall_time_s": 1.0,
        }
        judge_result = {"score": 8, "reasoning": "잘 요약함", "error": None}

        with patch(
            "eval_framework.tracks.track3_korean_deep.runner.generate",
            return_value=gen_result,
        ):
            with patch(
                "eval_framework.tracks.track3_korean_deep.judge.score_response",
                return_value=judge_result,
            ):
                output = run(["test-model"])

        judge_item = [r for r in output["results"] if r["answer_type"] == "llm_judge"][0]
        assert judge_item["score"] == 0.8
        assert judge_item["judge_score_raw"] == 8
        assert judge_item["judge_reasoning"] == "잘 요약함"

    def test_error_response_scores_zero(self):
        """오류 응답 — score=0.0"""
        gen_result = {
            "response": "",
            "error": "model not found",
            "tokens_per_sec": 0,
            "wall_time_s": 0,
        }

        with patch(
            "eval_framework.tracks.track3_korean_deep.runner.generate",
            return_value=gen_result,
        ):
            output = run(["test-model"])

        for r in output["results"]:
            assert r["score"] == 0.0
            assert r["error"] == "model not found"

    def test_ollama_not_responding_skips_model(self):
        """Ollama 서버 무응답 시 모델 건너뜀"""
        with patch(
            "eval_framework.tracks.track3_korean_deep.runner.wait_for_ollama",
            return_value=False,
        ):
            output = run(["bad-model"])

        assert len(output["results"]) == 0

    def test_model_switch_failure_skips_model(self):
        """모델 전환 실패 시 해당 모델 스킵"""
        with patch(
            "eval_framework.tracks.track3_korean_deep.runner.switch_model",
            return_value=False,
        ):
            output = run(["bad-model"])

        assert len(output["results"]) == 0

    def test_checkpoint_restores_completed(self):
        """체크포인트에서 이미 완료된 모델은 스킵"""
        checkpoint_data = {
            "results": [
                {"model": "test-model", "id": "q1", "category": "문화", "score": 1.0},
            ],
        }
        with patch(
            "eval_framework.tracks.track3_korean_deep.runner.load_checkpoint",
            return_value=checkpoint_data,
        ):
            with patch(
                "eval_framework.tracks.track3_korean_deep.runner.generate",
            ) as mock_gen:
                output = run(["test-model"])

        mock_gen.assert_not_called()
        assert len(output["results"]) == 1

    def test_defaults_to_all_models(self):
        """models가 None이면 config.ALL_MODELS 사용"""
        gen_result = {
            "response": "답변",
            "error": None,
            "tokens_per_sec": 10.0,
            "wall_time_s": 1.0,
        }
        judge_result = {"score": 5, "reasoning": "", "error": None}

        with patch(
            "eval_framework.tracks.track3_korean_deep.config.ALL_MODELS",
            ["default-model"],
        ):
            with patch(
                "eval_framework.tracks.track3_korean_deep.runner.generate",
                return_value=gen_result,
            ):
                with patch(
                    "eval_framework.tracks.track3_korean_deep.judge.score_response",
                    return_value=judge_result,
                ):
                    output = run(None)

        assert any(r["model"] == "default-model" for r in output["results"])

    def test_unload_called_for_llm_judge(self):
        """llm_judge 문항이 있으면 모델 언로드 후 채점"""
        gen_result = {
            "response": "답변",
            "error": None,
            "tokens_per_sec": 10.0,
            "wall_time_s": 1.0,
        }
        judge_result = {"score": 7, "reasoning": "", "error": None}

        with patch(
            "eval_framework.tracks.track3_korean_deep.runner.generate",
            return_value=gen_result,
        ):
            with patch(
                "eval_framework.tracks.track3_korean_deep.judge.score_response",
                return_value=judge_result,
            ):
                with patch(
                    "eval_framework.tracks.track3_korean_deep.runner.unload_all_models",
                ) as mock_unload:
                    output = run(["test-model"])

        mock_unload.assert_called_once()

    def test_checkpoint_saved_per_model(self):
        """모델별 체크포인트 저장"""
        gen_result = {
            "response": "답변",
            "error": None,
            "tokens_per_sec": 10.0,
            "wall_time_s": 1.0,
        }
        judge_result = {"score": 7, "reasoning": "", "error": None}

        with patch(
            "eval_framework.tracks.track3_korean_deep.runner.generate",
            return_value=gen_result,
        ):
            with patch(
                "eval_framework.tracks.track3_korean_deep.judge.score_response",
                return_value=judge_result,
            ):
                with patch(
                    "eval_framework.tracks.track3_korean_deep.runner.save_checkpoint",
                ) as mock_save:
                    output = run(["model-a", "model-b"])

        assert mock_save.call_count == 2

    def test_results_saved_incrementally(self):
        """최종 결과 저장 호출"""
        gen_result = {
            "response": "답변",
            "error": None,
            "tokens_per_sec": 10.0,
            "wall_time_s": 1.0,
        }
        judge_result = {"score": 5, "reasoning": "", "error": None}

        with patch(
            "eval_framework.tracks.track3_korean_deep.runner.generate",
            return_value=gen_result,
        ):
            with patch(
                "eval_framework.tracks.track3_korean_deep.judge.score_response",
                return_value=judge_result,
            ):
                with patch(
                    "eval_framework.tracks.track3_korean_deep.runner.save_results_incremental",
                ) as mock_save:
                    mock_save.return_value = "results/test.json"
                    output = run(["test-model"])

        mock_save.assert_called_once()
        args = mock_save.call_args[0]
        assert args[1] == TRACK_NAME
