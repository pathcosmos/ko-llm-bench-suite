"""eval_framework/tracks/track5_consistency.py 단위 테스트"""

import json
import pytest
from unittest.mock import patch, MagicMock, call

from eval_framework.tracks.track5_consistency import (
    jaccard_similarity,
    edit_distance_ratio,
    detect_korean_ratio,
    check_instruction_compliance,
    _test_repetition_consistency,
    _test_paraphrase_robustness,
    _test_length_sensitivity,
    _test_language_mixing,
    _test_instruction_following,
    _test_hallucination_detection,
    _build_summary,
    run,
    TRACK_NAME,
    REPETITION_PROMPTS,
    REPETITION_COUNT,
    PARAPHRASE_QUESTIONS,
    LENGTH_SENSITIVITY_DATA,
    LANGUAGE_MIXING_PROMPTS,
    INSTRUCTION_FOLLOWING_DATA,
    HALLUCINATION_PROMPTS,
    REFUSAL_PATTERNS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# jaccard_similarity 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestJaccardSimilarity:
    """jaccard_similarity: 두 집합 간 Jaccard 유사도 계산"""

    def test_identical_sets(self):
        """동일한 집합이면 1.0"""
        assert jaccard_similarity({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    def test_disjoint_sets(self):
        """겹치지 않는 집합이면 0.0"""
        assert jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        """부분 겹침: {a,b,c} & {b,c,d} → 교집합 2, 합집합 4 → 0.5"""
        assert jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"}) == 0.5

    def test_both_empty(self):
        """두 집합 모두 비어 있으면 1.0"""
        assert jaccard_similarity(set(), set()) == 1.0

    def test_one_empty(self):
        """한쪽만 비어 있으면 0.0"""
        assert jaccard_similarity(set(), {"a"}) == 0.0
        assert jaccard_similarity({"a"}, set()) == 0.0

    def test_subset(self):
        """부분집합 관계: {a} & {a,b} → 1/2"""
        assert jaccard_similarity({"a"}, {"a", "b"}) == 0.5

    def test_single_element_same(self):
        """단일 원소 동일"""
        assert jaccard_similarity({"x"}, {"x"}) == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# edit_distance_ratio 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestEditDistanceRatio:
    """edit_distance_ratio: 편집 거리 비율 (0=동일, 1=완전히 다름)"""

    def test_identical_strings(self):
        """동일 문자열이면 0.0"""
        assert edit_distance_ratio("hello", "hello") == 0.0

    def test_completely_different(self):
        """완전히 다른 문자열 (같은 길이) → 1.0"""
        assert edit_distance_ratio("abc", "xyz") == 1.0

    def test_both_empty(self):
        """빈 문자열 둘 다이면 0.0"""
        assert edit_distance_ratio("", "") == 0.0

    def test_one_empty(self):
        """한쪽만 비어 있으면 1.0"""
        assert edit_distance_ratio("", "abc") == 1.0
        assert edit_distance_ratio("abc", "") == 1.0

    def test_one_char_diff(self):
        """한 글자만 다른 경우: 거리 1 / 최대길이 3 ≈ 0.333"""
        ratio = edit_distance_ratio("abc", "axc")
        assert abs(ratio - 1.0 / 3.0) < 1e-9

    def test_insertion(self):
        """삽입: 'ab' → 'abc' → 거리 1 / 3"""
        ratio = edit_distance_ratio("ab", "abc")
        assert abs(ratio - 1.0 / 3.0) < 1e-9

    def test_korean_strings(self):
        """한글 문자열 비교"""
        ratio = edit_distance_ratio("서울", "서울시")
        assert abs(ratio - 1.0 / 3.0) < 1e-9

    def test_symmetry(self):
        """대칭성: edit_distance_ratio(a, b) == edit_distance_ratio(b, a)"""
        assert edit_distance_ratio("abc", "abcd") == edit_distance_ratio("abcd", "abc")


# ═══════════════════════════════════════════════════════════════════════════════
# detect_korean_ratio 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestDetectKoreanRatio:
    """detect_korean_ratio: 텍스트에서 한국어 문자 비율"""

    def test_all_korean(self):
        """순수 한글이면 1.0"""
        assert detect_korean_ratio("안녕하세요") == 1.0

    def test_all_english(self):
        """순수 영어면 0.0"""
        assert detect_korean_ratio("hello world") == 0.0

    def test_empty_string(self):
        """빈 문자열이면 0.0"""
        assert detect_korean_ratio("") == 0.0

    def test_only_punctuation(self):
        """구두점만 있으면 0.0 (알파벳/숫자 문자 없음)"""
        assert detect_korean_ratio("!@#$%^&*()") == 0.0

    def test_mixed_korean_english(self):
        """한영 혼용: '안녕hello' → 한글 2자, 영어 5자 → 2/7"""
        ratio = detect_korean_ratio("안녕hello")
        # 안(1) 녕(1) h e l l o → 2/7
        assert abs(ratio - 2.0 / 7.0) < 1e-9

    def test_korean_with_spaces(self):
        """공백은 무시: '서울 부산' → 한글 4자 / 4자 = 1.0"""
        assert detect_korean_ratio("서울 부산") == 1.0

    def test_jamo(self):
        """한글 자모 (ㄱ, ㅎ 등) 인식"""
        ratio = detect_korean_ratio("ㄱㄴㄷ")
        assert ratio == 1.0

    def test_numbers_not_korean(self):
        """숫자는 한국어로 분류되지 않음"""
        ratio = detect_korean_ratio("123")
        assert ratio == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# check_instruction_compliance 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckInstructionCompliance:
    """check_instruction_compliance: 지시 준수 여부 규칙 기반 검사"""

    # ── count_items ──────────────────────────────────────────────────────

    def test_count_items_numbered_match(self):
        """번호 매기기로 정확한 항목 수 일치"""
        response = "1. 사과\n2. 바나나\n3. 포도"
        result = check_instruction_compliance(response, "count_items", 3)
        assert result["compliant"] is True

    def test_count_items_numbered_mismatch(self):
        """번호 매기기 항목 수 불일치"""
        response = "1. 사과\n2. 바나나"
        result = check_instruction_compliance(response, "count_items", 3)
        assert result["compliant"] is False

    def test_count_items_bulleted(self):
        """글머리 기호 항목 카운트"""
        response = "- 사과\n- 바나나\n- 포도"
        result = check_instruction_compliance(response, "count_items", 3)
        assert result["compliant"] is True

    def test_count_items_plain_lines(self):
        """번호/글머리 없으면 줄 단위 카운트"""
        response = "사과\n바나나\n포도"
        result = check_instruction_compliance(response, "count_items", 3)
        assert result["compliant"] is True

    # ── max_chars ────────────────────────────────────────────────────────

    def test_max_chars_within_limit(self):
        """글자 수 제한 이내"""
        response = "짧은 답변"
        result = check_instruction_compliance(response, "max_chars", 50)
        assert result["compliant"] is True

    def test_max_chars_exceeded(self):
        """글자 수 제한 초과"""
        response = "a" * 51
        result = check_instruction_compliance(response, "max_chars", 50)
        assert result["compliant"] is False

    def test_max_chars_exact(self):
        """정확히 제한과 같은 글자 수"""
        response = "a" * 50
        result = check_instruction_compliance(response, "max_chars", 50)
        assert result["compliant"] is True

    # ── json_format ──────────────────────────────────────────────────────

    def test_json_format_valid(self):
        """유효한 JSON"""
        response = '{"수도": "서울", "인구": "5100만"}'
        result = check_instruction_compliance(response, "json_format", None)
        assert result["compliant"] is True

    def test_json_format_invalid(self):
        """유효하지 않은 JSON"""
        response = "이것은 JSON이 아닙니다"
        result = check_instruction_compliance(response, "json_format", None)
        assert result["compliant"] is False

    def test_json_format_in_code_block(self):
        """코드블록 안에 유효한 JSON"""
        response = '```json\n{"key": "value"}\n```'
        result = check_instruction_compliance(response, "json_format", None)
        assert result["compliant"] is True

    def test_json_format_invalid_in_code_block(self):
        """코드블록 안에 유효하지 않은 JSON"""
        response = '```json\n{invalid json}\n```'
        result = check_instruction_compliance(response, "json_format", None)
        assert result["compliant"] is False

    # ── numbered_list ────────────────────────────────────────────────────

    def test_numbered_list_compliant(self):
        """번호 항목 2개 이상"""
        response = "1. 첫째\n2. 둘째\n3. 셋째"
        result = check_instruction_compliance(response, "numbered_list", None)
        assert result["compliant"] is True

    def test_numbered_list_insufficient(self):
        """번호 항목 1개만 — 비준수"""
        response = "1. 유일한 항목"
        result = check_instruction_compliance(response, "numbered_list", None)
        assert result["compliant"] is False

    def test_numbered_list_zero(self):
        """번호 항목 없음"""
        response = "그냥 텍스트입니다."
        result = check_instruction_compliance(response, "numbered_list", None)
        assert result["compliant"] is False

    # ── table_format ─────────────────────────────────────────────────────

    def test_table_format_compliant(self):
        """올바른 Markdown 표"""
        response = (
            "| 나라 | 수도 |\n"
            "|---|---|\n"
            "| 한국 | 서울 |\n"
            "| 일본 | 도쿄 |"
        )
        result = check_instruction_compliance(response, "table_format", None)
        assert result["compliant"] is True

    def test_table_format_no_separator(self):
        """구분선 없는 표 — 비준수"""
        response = "| 나라 | 수도 |\n| 한국 | 서울 |\n| 일본 | 도쿄 |"
        result = check_instruction_compliance(response, "table_format", None)
        assert result["compliant"] is False

    def test_table_format_too_few_rows(self):
        """행이 2개 미만 — 비준수"""
        response = "| 나라 | 수도 |\n|---|---|"
        result = check_instruction_compliance(response, "table_format", None)
        assert result["compliant"] is False

    # ── unknown type ─────────────────────────────────────────────────────

    def test_unknown_instruction_type(self):
        """알 수 없는 지시 유형"""
        result = check_instruction_compliance("답변", "unknown_type", None)
        assert result["compliant"] is False
        assert "알 수 없는 지시 유형" in result["detail"]


# ═══════════════════════════════════════════════════════════════════════════════
# _test_repetition_consistency 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestRepetitionConsistency:
    """_test_repetition_consistency: 동일 프롬프트 반복 일관성 측정"""

    @patch("eval_framework.tracks.track5_consistency.time.sleep")
    @patch("eval_framework.tracks.track5_consistency.runner.generate")
    @patch("eval_framework.tracks.track5_consistency.config.SAMPLING_PARAMS", {"temperature": 0.7})
    @patch("eval_framework.tracks.track5_consistency.config.COOLDOWN_BETWEEN_TESTS", 0)
    @patch("eval_framework.tracks.track5_consistency.REPETITION_PROMPTS", ["프롬프트1"])
    @patch("eval_framework.tracks.track5_consistency.REPETITION_COUNT", 3)
    def test_basic_structure(self, mock_generate, mock_sleep):
        """반환 결과의 기본 구조 검증"""
        mock_generate.return_value = {"response": "동일한 답변"}
        results = _test_repetition_consistency("test-model")

        assert len(results) == 1
        r = results[0]
        assert r["model"] == "test-model"
        assert r["test_type"] == "repetition_consistency"
        assert r["prompt_index"] == 0
        assert r["num_trials"] == 3
        assert "avg_edit_distance_ratio" in r
        assert "avg_jaccard_similarity" in r

    @patch("eval_framework.tracks.track5_consistency.time.sleep")
    @patch("eval_framework.tracks.track5_consistency.runner.generate")
    @patch("eval_framework.tracks.track5_consistency.config.SAMPLING_PARAMS", {"temperature": 0.7})
    @patch("eval_framework.tracks.track5_consistency.config.COOLDOWN_BETWEEN_TESTS", 0)
    @patch("eval_framework.tracks.track5_consistency.REPETITION_PROMPTS", ["프롬프트1"])
    @patch("eval_framework.tracks.track5_consistency.REPETITION_COUNT", 3)
    def test_identical_responses(self, mock_generate, mock_sleep):
        """모든 응답이 동일하면 edit_distance=0, jaccard=1"""
        mock_generate.return_value = {"response": "동일 답변"}
        results = _test_repetition_consistency("test-model")

        assert results[0]["avg_edit_distance_ratio"] == 0.0
        assert results[0]["avg_jaccard_similarity"] == 1.0

    @patch("eval_framework.tracks.track5_consistency.time.sleep")
    @patch("eval_framework.tracks.track5_consistency.runner.generate")
    @patch("eval_framework.tracks.track5_consistency.config.SAMPLING_PARAMS", {"temperature": 0.7})
    @patch("eval_framework.tracks.track5_consistency.config.COOLDOWN_BETWEEN_TESTS", 0)
    @patch("eval_framework.tracks.track5_consistency.REPETITION_PROMPTS", ["프롬프트1"])
    @patch("eval_framework.tracks.track5_consistency.REPETITION_COUNT", 2)
    def test_different_responses(self, mock_generate, mock_sleep):
        """서로 다른 응답이면 edit_distance > 0"""
        mock_generate.side_effect = [
            {"response": "aaa"},
            {"response": "bbb"},
        ]
        results = _test_repetition_consistency("test-model")

        assert results[0]["avg_edit_distance_ratio"] > 0.0

    @patch("eval_framework.tracks.track5_consistency.time.sleep")
    @patch("eval_framework.tracks.track5_consistency.runner.generate")
    @patch("eval_framework.tracks.track5_consistency.config.SAMPLING_PARAMS", {"temperature": 0.7})
    @patch("eval_framework.tracks.track5_consistency.config.COOLDOWN_BETWEEN_TESTS", 0)
    @patch("eval_framework.tracks.track5_consistency.REPETITION_PROMPTS", ["프롬프트1"])
    @patch("eval_framework.tracks.track5_consistency.REPETITION_COUNT", 3)
    def test_generate_called_correct_times(self, mock_generate, mock_sleep):
        """generate가 프롬프트 수 x 반복 횟수만큼 호출"""
        mock_generate.return_value = {"response": "답변"}
        _test_repetition_consistency("test-model")

        assert mock_generate.call_count == 3  # 1 prompt x 3 reps


# ═══════════════════════════════════════════════════════════════════════════════
# _test_paraphrase_robustness 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestParaphraseRobustness:
    """_test_paraphrase_robustness: 패러프레이즈 강건성 측정"""

    @patch("eval_framework.tracks.track5_consistency.time.sleep")
    @patch("eval_framework.tracks.track5_consistency.runner.generate")
    @patch("eval_framework.tracks.track5_consistency.config.SAMPLING_PARAMS", {"temperature": 0.7})
    @patch("eval_framework.tracks.track5_consistency.config.COOLDOWN_BETWEEN_TESTS", 0)
    @patch(
        "eval_framework.tracks.track5_consistency.PARAPHRASE_QUESTIONS",
        [{"answer_keywords": ["서울"], "variants": ["수도는?", "capital은?"]}],
    )
    def test_all_correct(self, mock_generate, mock_sleep):
        """모든 변형에서 정답 키워드 포함 시 hit_rate=1.0"""
        mock_generate.return_value = {"response": "서울입니다"}
        results = _test_paraphrase_robustness("test-model")

        assert len(results) == 1
        assert results[0]["keyword_hit_rate"] == 1.0
        assert results[0]["all_consistent"] is True

    @patch("eval_framework.tracks.track5_consistency.time.sleep")
    @patch("eval_framework.tracks.track5_consistency.runner.generate")
    @patch("eval_framework.tracks.track5_consistency.config.SAMPLING_PARAMS", {"temperature": 0.7})
    @patch("eval_framework.tracks.track5_consistency.config.COOLDOWN_BETWEEN_TESTS", 0)
    @patch(
        "eval_framework.tracks.track5_consistency.PARAPHRASE_QUESTIONS",
        [{"answer_keywords": ["서울"], "variants": ["질문1", "질문2"]}],
    )
    def test_none_correct(self, mock_generate, mock_sleep):
        """키워드가 전혀 없으면 hit_rate=0.0, 여전히 일관적"""
        mock_generate.return_value = {"response": "잘 모르겠습니다"}
        results = _test_paraphrase_robustness("test-model")

        assert results[0]["keyword_hit_rate"] == 0.0
        assert results[0]["all_consistent"] is True

    @patch("eval_framework.tracks.track5_consistency.time.sleep")
    @patch("eval_framework.tracks.track5_consistency.runner.generate")
    @patch("eval_framework.tracks.track5_consistency.config.SAMPLING_PARAMS", {"temperature": 0.7})
    @patch("eval_framework.tracks.track5_consistency.config.COOLDOWN_BETWEEN_TESTS", 0)
    @patch(
        "eval_framework.tracks.track5_consistency.PARAPHRASE_QUESTIONS",
        [{"answer_keywords": ["서울"], "variants": ["질문1", "질문2"]}],
    )
    def test_inconsistent(self, mock_generate, mock_sleep):
        """변형 간 결과가 다르면 all_consistent=False"""
        mock_generate.side_effect = [
            {"response": "서울입니다"},
            {"response": "잘 모르겠습니다"},
        ]
        results = _test_paraphrase_robustness("test-model")

        assert results[0]["keyword_hit_rate"] == 0.5
        assert results[0]["all_consistent"] is False

    @patch("eval_framework.tracks.track5_consistency.time.sleep")
    @patch("eval_framework.tracks.track5_consistency.runner.generate")
    @patch("eval_framework.tracks.track5_consistency.config.SAMPLING_PARAMS", {"temperature": 0.7})
    @patch("eval_framework.tracks.track5_consistency.config.COOLDOWN_BETWEEN_TESTS", 0)
    @patch(
        "eval_framework.tracks.track5_consistency.PARAPHRASE_QUESTIONS",
        [{"answer_keywords": ["서울"], "variants": ["질문1", "질문2"]}],
    )
    def test_result_structure(self, mock_generate, mock_sleep):
        """반환 구조 검증"""
        mock_generate.return_value = {"response": "서울"}
        results = _test_paraphrase_robustness("test-model")

        r = results[0]
        assert r["test_type"] == "paraphrase_robustness"
        assert r["num_variants"] == 2
        assert "keyword_hits" in r


# ═══════════════════════════════════════════════════════════════════════════════
# _test_length_sensitivity 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestLengthSensitivity:
    """_test_length_sensitivity: 짧은/중간/긴 프롬프트에서 일관성"""

    @patch("eval_framework.tracks.track5_consistency.time.sleep")
    @patch("eval_framework.tracks.track5_consistency.runner.generate")
    @patch("eval_framework.tracks.track5_consistency.config.SAMPLING_PARAMS", {"temperature": 0.7})
    @patch("eval_framework.tracks.track5_consistency.config.COOLDOWN_BETWEEN_TESTS", 0)
    @patch(
        "eval_framework.tracks.track5_consistency.LENGTH_SENSITIVITY_DATA",
        [{"answer_keywords": ["서울"], "short": "짧은", "medium": "중간", "long": "긴"}],
    )
    def test_all_correct(self, mock_generate, mock_sleep):
        """모든 길이에서 정답이면 all_correct=True, consistent=True"""
        mock_generate.return_value = {"response": "서울입니다"}
        results = _test_length_sensitivity("test-model")

        assert len(results) == 1
        assert results[0]["all_correct"] is True
        assert results[0]["consistent_across_lengths"] is True

    @patch("eval_framework.tracks.track5_consistency.time.sleep")
    @patch("eval_framework.tracks.track5_consistency.runner.generate")
    @patch("eval_framework.tracks.track5_consistency.config.SAMPLING_PARAMS", {"temperature": 0.7})
    @patch("eval_framework.tracks.track5_consistency.config.COOLDOWN_BETWEEN_TESTS", 0)
    @patch(
        "eval_framework.tracks.track5_consistency.LENGTH_SENSITIVITY_DATA",
        [{"answer_keywords": ["서울"], "short": "짧은", "medium": "중간", "long": "긴"}],
    )
    def test_inconsistent_across_lengths(self, mock_generate, mock_sleep):
        """길이별로 결과가 다르면 consistent=False"""
        mock_generate.side_effect = [
            {"response": "서울입니다"},   # short: correct
            {"response": "잘 모르겠습니다"},  # medium: wrong
            {"response": "서울입니다"},   # long: correct
        ]
        results = _test_length_sensitivity("test-model")

        assert results[0]["consistent_across_lengths"] is False
        assert results[0]["any_correct"] is True
        assert results[0]["all_correct"] is False

    @patch("eval_framework.tracks.track5_consistency.time.sleep")
    @patch("eval_framework.tracks.track5_consistency.runner.generate")
    @patch("eval_framework.tracks.track5_consistency.config.SAMPLING_PARAMS", {"temperature": 0.7})
    @patch("eval_framework.tracks.track5_consistency.config.COOLDOWN_BETWEEN_TESTS", 0)
    @patch(
        "eval_framework.tracks.track5_consistency.LENGTH_SENSITIVITY_DATA",
        [{"answer_keywords": ["서울"], "short": "짧은", "medium": "중간", "long": "긴"}],
    )
    def test_generate_called_three_times(self, mock_generate, mock_sleep):
        """각 항목당 short/medium/long 3회 호출"""
        mock_generate.return_value = {"response": "답변"}
        _test_length_sensitivity("test-model")

        assert mock_generate.call_count == 3


# ═══════════════════════════════════════════════════════════════════════════════
# _test_language_mixing 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestLanguageMixing:
    """_test_language_mixing: 한영 혼용 질문에서 응답 언어 일관성"""

    @patch("eval_framework.tracks.track5_consistency.time.sleep")
    @patch("eval_framework.tracks.track5_consistency.runner.generate")
    @patch("eval_framework.tracks.track5_consistency.config.SAMPLING_PARAMS", {"temperature": 0.7})
    @patch("eval_framework.tracks.track5_consistency.config.COOLDOWN_BETWEEN_TESTS", 0)
    @patch(
        "eval_framework.tracks.track5_consistency.LANGUAGE_MIXING_PROMPTS",
        ["Python에서 list comprehension을 explain해주세요."],
    )
    def test_korean_response(self, mock_generate, mock_sleep):
        """한글 응답이면 korean_ratio 가 높음"""
        mock_generate.return_value = {"response": "리스트 컴프리헨션은 파이썬의 기능입니다"}
        results = _test_language_mixing("test-model")

        assert len(results) == 1
        assert results[0]["test_type"] == "language_mixing"
        assert results[0]["korean_ratio"] > 0.0

    @patch("eval_framework.tracks.track5_consistency.time.sleep")
    @patch("eval_framework.tracks.track5_consistency.runner.generate")
    @patch("eval_framework.tracks.track5_consistency.config.SAMPLING_PARAMS", {"temperature": 0.7})
    @patch("eval_framework.tracks.track5_consistency.config.COOLDOWN_BETWEEN_TESTS", 0)
    @patch(
        "eval_framework.tracks.track5_consistency.LANGUAGE_MIXING_PROMPTS",
        ["질문1"],
    )
    def test_result_structure(self, mock_generate, mock_sleep):
        """반환 구조 검증"""
        mock_generate.return_value = {"response": "답변 response"}
        results = _test_language_mixing("test-model")

        r = results[0]
        assert r["model"] == "test-model"
        assert "korean_ratio" in r
        assert "response_length" in r
        assert "prompt" in r


# ═══════════════════════════════════════════════════════════════════════════════
# _test_instruction_following 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestInstructionFollowing:
    """_test_instruction_following: 형식 지시 준수 여부"""

    @patch("eval_framework.tracks.track5_consistency.time.sleep")
    @patch("eval_framework.tracks.track5_consistency.runner.generate")
    @patch("eval_framework.tracks.track5_consistency.config.SAMPLING_PARAMS", {"temperature": 0.7})
    @patch("eval_framework.tracks.track5_consistency.config.COOLDOWN_BETWEEN_TESTS", 0)
    @patch(
        "eval_framework.tracks.track5_consistency.INSTRUCTION_FOLLOWING_DATA",
        [("3개 나열하세요.", "count_items", 3)],
    )
    def test_compliant(self, mock_generate, mock_sleep):
        """지시 준수 시 compliant=True"""
        mock_generate.return_value = {"response": "1. 하나\n2. 둘\n3. 셋"}
        results = _test_instruction_following("test-model")

        assert len(results) == 1
        assert results[0]["compliant"] is True

    @patch("eval_framework.tracks.track5_consistency.time.sleep")
    @patch("eval_framework.tracks.track5_consistency.runner.generate")
    @patch("eval_framework.tracks.track5_consistency.config.SAMPLING_PARAMS", {"temperature": 0.7})
    @patch("eval_framework.tracks.track5_consistency.config.COOLDOWN_BETWEEN_TESTS", 0)
    @patch(
        "eval_framework.tracks.track5_consistency.INSTRUCTION_FOLLOWING_DATA",
        [("3개 나열하세요.", "count_items", 3)],
    )
    def test_non_compliant(self, mock_generate, mock_sleep):
        """지시 불이행 시 compliant=False"""
        mock_generate.return_value = {"response": "1. 하나\n2. 둘"}
        results = _test_instruction_following("test-model")

        assert results[0]["compliant"] is False

    @patch("eval_framework.tracks.track5_consistency.time.sleep")
    @patch("eval_framework.tracks.track5_consistency.runner.generate")
    @patch("eval_framework.tracks.track5_consistency.config.SAMPLING_PARAMS", {"temperature": 0.7})
    @patch("eval_framework.tracks.track5_consistency.config.COOLDOWN_BETWEEN_TESTS", 0)
    @patch(
        "eval_framework.tracks.track5_consistency.INSTRUCTION_FOLLOWING_DATA",
        [("JSON으로 출력하세요.", "json_format", None)],
    )
    def test_result_structure(self, mock_generate, mock_sleep):
        """반환 구조 검증"""
        mock_generate.return_value = {"response": '{"key": "value"}'}
        results = _test_instruction_following("test-model")

        r = results[0]
        assert r["test_type"] == "instruction_following"
        assert "instruction_type" in r
        assert "detail" in r
        assert "response_length" in r


# ═══════════════════════════════════════════════════════════════════════════════
# _test_hallucination_detection 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestHallucinationDetection:
    """_test_hallucination_detection: 허구적 질문에 대한 환각/거부 판정"""

    @patch("eval_framework.tracks.track5_consistency.time.sleep")
    @patch("eval_framework.tracks.track5_consistency.runner.generate")
    @patch("eval_framework.tracks.track5_consistency.config.SAMPLING_PARAMS", {"temperature": 0.7})
    @patch("eval_framework.tracks.track5_consistency.config.COOLDOWN_BETWEEN_TESTS", 0)
    @patch(
        "eval_framework.tracks.track5_consistency.HALLUCINATION_PROMPTS",
        ["존재하지 않는 질문"],
    )
    def test_refused(self, mock_generate, mock_sleep):
        """거부 패턴이 포함되면 refused=True"""
        mock_generate.return_value = {"response": "해당 정보를 확인할 수 없습니다."}
        results = _test_hallucination_detection("test-model")

        assert len(results) == 1
        assert results[0]["refused"] is True

    @patch("eval_framework.tracks.track5_consistency.time.sleep")
    @patch("eval_framework.tracks.track5_consistency.runner.generate")
    @patch("eval_framework.tracks.track5_consistency.config.SAMPLING_PARAMS", {"temperature": 0.7})
    @patch("eval_framework.tracks.track5_consistency.config.COOLDOWN_BETWEEN_TESTS", 0)
    @patch(
        "eval_framework.tracks.track5_consistency.HALLUCINATION_PROMPTS",
        ["존재하지 않는 질문"],
    )
    def test_hallucinated(self, mock_generate, mock_sleep):
        """거부 패턴이 없으면 refused=False (환각)"""
        mock_generate.return_value = {"response": "네, 그 논문의 주요 주장은 경제 성장입니다."}
        results = _test_hallucination_detection("test-model")

        assert results[0]["refused"] is False

    @patch("eval_framework.tracks.track5_consistency.time.sleep")
    @patch("eval_framework.tracks.track5_consistency.runner.generate")
    @patch("eval_framework.tracks.track5_consistency.config.SAMPLING_PARAMS", {"temperature": 0.7})
    @patch("eval_framework.tracks.track5_consistency.config.COOLDOWN_BETWEEN_TESTS", 0)
    @patch(
        "eval_framework.tracks.track5_consistency.HALLUCINATION_PROMPTS",
        ["질문1"],
    )
    def test_result_structure(self, mock_generate, mock_sleep):
        """반환 구조 검증"""
        mock_generate.return_value = {"response": "답변 텍스트입니다"}
        results = _test_hallucination_detection("test-model")

        r = results[0]
        assert r["test_type"] == "hallucination_detection"
        assert "refused" in r
        assert "response_preview" in r
        assert "response_length" in r


# ═══════════════════════════════════════════════════════════════════════════════
# _build_summary 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildSummary:
    """_build_summary: 모델별 6개 차원 요약 통계 생성"""

    def test_empty_results(self):
        """빈 결과 목록이면 빈 요약"""
        assert _build_summary([]) == {}

    def test_repetition_score(self):
        """repetition_consistency = 1 - avg_edit_distance_ratio"""
        results = [
            {
                "model": "m1",
                "test_type": "repetition_consistency",
                "avg_edit_distance_ratio": 0.2,
            },
        ]
        summary = _build_summary(results)
        assert abs(summary["m1"]["repetition_consistency"] - 0.8) < 1e-4

    def test_paraphrase_score(self):
        """paraphrase_robustness = keyword_hit_rate 평균"""
        results = [
            {"model": "m1", "test_type": "paraphrase_robustness", "keyword_hit_rate": 0.8},
            {"model": "m1", "test_type": "paraphrase_robustness", "keyword_hit_rate": 1.0},
        ]
        summary = _build_summary(results)
        assert abs(summary["m1"]["paraphrase_robustness"] - 0.9) < 1e-4

    def test_length_sensitivity_score(self):
        """length_sensitivity: consistent=True → 1.0, False → 0.0"""
        results = [
            {"model": "m1", "test_type": "length_sensitivity", "consistent_across_lengths": True},
            {"model": "m1", "test_type": "length_sensitivity", "consistent_across_lengths": False},
        ]
        summary = _build_summary(results)
        assert abs(summary["m1"]["length_sensitivity"] - 0.5) < 1e-4

    def test_language_consistency_score(self):
        """language_consistency = korean_ratio 평균"""
        results = [
            {"model": "m1", "test_type": "language_mixing", "korean_ratio": 0.9},
            {"model": "m1", "test_type": "language_mixing", "korean_ratio": 0.7},
        ]
        summary = _build_summary(results)
        assert abs(summary["m1"]["language_consistency"] - 0.8) < 1e-4

    def test_instruction_following_score(self):
        """instruction_following: compliant=True → 1.0, False → 0.0"""
        results = [
            {"model": "m1", "test_type": "instruction_following", "compliant": True},
            {"model": "m1", "test_type": "instruction_following", "compliant": False},
            {"model": "m1", "test_type": "instruction_following", "compliant": True},
        ]
        summary = _build_summary(results)
        assert abs(summary["m1"]["instruction_following"] - 2.0 / 3.0) < 1e-4

    def test_hallucination_detection_score(self):
        """hallucination_detection: refused=True → 1.0, False → 0.0"""
        results = [
            {"model": "m1", "test_type": "hallucination_detection", "refused": True},
            {"model": "m1", "test_type": "hallucination_detection", "refused": False},
        ]
        summary = _build_summary(results)
        assert abs(summary["m1"]["hallucination_detection"] - 0.5) < 1e-4

    def test_multiple_models(self):
        """여러 모델 결과가 분리되어 요약"""
        results = [
            {"model": "m1", "test_type": "paraphrase_robustness", "keyword_hit_rate": 1.0},
            {"model": "m2", "test_type": "paraphrase_robustness", "keyword_hit_rate": 0.5},
        ]
        summary = _build_summary(results)
        assert "m1" in summary
        assert "m2" in summary
        assert summary["m1"]["paraphrase_robustness"] == 1.0
        assert summary["m2"]["paraphrase_robustness"] == 0.5

    def test_missing_dimensions_default_zero(self):
        """데이터가 없는 차원은 0.0"""
        results = [
            {"model": "m1", "test_type": "paraphrase_robustness", "keyword_hit_rate": 1.0},
        ]
        summary = _build_summary(results)
        assert summary["m1"]["repetition_consistency"] == 0.0
        assert summary["m1"]["length_sensitivity"] == 0.0
        assert summary["m1"]["language_consistency"] == 0.0
        assert summary["m1"]["instruction_following"] == 0.0
        assert summary["m1"]["hallucination_detection"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# run 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestRun:
    """run: Track 5 전체 실행"""

    @patch("eval_framework.tracks.track5_consistency.runner.save_results_incremental")
    @patch("eval_framework.tracks.track5_consistency.runner.save_checkpoint")
    @patch("eval_framework.tracks.track5_consistency.runner.load_checkpoint", return_value=None)
    @patch("eval_framework.tracks.track5_consistency.runner.wait_for_ollama", return_value=True)
    @patch("eval_framework.tracks.track5_consistency.runner.switch_model", return_value=True)
    @patch("eval_framework.tracks.track5_consistency._test_hallucination_detection", return_value=[])
    @patch("eval_framework.tracks.track5_consistency._test_instruction_following", return_value=[])
    @patch("eval_framework.tracks.track5_consistency._test_language_mixing", return_value=[])
    @patch("eval_framework.tracks.track5_consistency._test_length_sensitivity", return_value=[])
    @patch("eval_framework.tracks.track5_consistency._test_paraphrase_robustness", return_value=[])
    @patch("eval_framework.tracks.track5_consistency._test_repetition_consistency", return_value=[])
    def test_basic_flow(
        self, mock_rep, mock_para, mock_len, mock_lang, mock_inst, mock_hall,
        mock_switch, mock_wait, mock_load_ckpt, mock_save_ckpt, mock_save_res,
    ):
        """기본 실행 흐름: 모든 서브테스트가 호출됨"""
        result = run(models=["test-model"])

        assert result["track"] == TRACK_NAME
        assert "results" in result
        assert "summary" in result
        mock_rep.assert_called_once_with("test-model")
        mock_para.assert_called_once_with("test-model")
        mock_len.assert_called_once_with("test-model")
        mock_lang.assert_called_once_with("test-model")
        mock_inst.assert_called_once_with("test-model")
        mock_hall.assert_called_once_with("test-model")

    @patch("eval_framework.tracks.track5_consistency.runner.save_results_incremental")
    @patch("eval_framework.tracks.track5_consistency.runner.save_checkpoint")
    @patch("eval_framework.tracks.track5_consistency.runner.load_checkpoint", return_value=None)
    @patch("eval_framework.tracks.track5_consistency.runner.wait_for_ollama", return_value=False)
    def test_ollama_unavailable(self, mock_wait, mock_load_ckpt, mock_save_ckpt, mock_save_res):
        """Ollama 연결 실패 시 에러 포함 결과 반환"""
        result = run(models=["test-model"])

        assert "error" in result
        assert "Ollama" in result["error"]

    @patch("eval_framework.tracks.track5_consistency.runner.save_results_incremental")
    @patch("eval_framework.tracks.track5_consistency.runner.save_checkpoint")
    @patch("eval_framework.tracks.track5_consistency.runner.load_checkpoint", return_value=None)
    @patch("eval_framework.tracks.track5_consistency.runner.wait_for_ollama", return_value=True)
    @patch("eval_framework.tracks.track5_consistency.runner.switch_model", return_value=False)
    @patch("eval_framework.tracks.track5_consistency._test_hallucination_detection")
    @patch("eval_framework.tracks.track5_consistency._test_instruction_following")
    @patch("eval_framework.tracks.track5_consistency._test_language_mixing")
    @patch("eval_framework.tracks.track5_consistency._test_length_sensitivity")
    @patch("eval_framework.tracks.track5_consistency._test_paraphrase_robustness")
    @patch("eval_framework.tracks.track5_consistency._test_repetition_consistency")
    def test_model_load_failure(
        self, mock_rep, mock_para, mock_len, mock_lang, mock_inst, mock_hall,
        mock_switch, mock_wait, mock_load_ckpt, mock_save_ckpt, mock_save_res,
    ):
        """모델 로딩 실패 시 서브테스트 호출 안 됨"""
        result = run(models=["bad-model"])

        mock_rep.assert_not_called()
        mock_para.assert_not_called()
        # 결과에 model_load_failed 엔트리 존재
        has_fail = any(
            r.get("test_type") == "model_load_failed" for r in result["results"]
        )
        assert has_fail

    @patch("eval_framework.tracks.track5_consistency.runner.save_results_incremental")
    @patch("eval_framework.tracks.track5_consistency.runner.save_checkpoint")
    @patch(
        "eval_framework.tracks.track5_consistency.runner.load_checkpoint",
        return_value={
            "results": [{"model": "m1", "test_type": "repetition_consistency", "avg_edit_distance_ratio": 0.1}],
            "completed_keys": [
                "model_loaded:m1",
                "repetition:m1",
                "paraphrase:m1",
                "length:m1",
                "language:m1",
                "instruction:m1",
                "hallucination:m1",
            ],
        },
    )
    @patch("eval_framework.tracks.track5_consistency.runner.wait_for_ollama", return_value=True)
    @patch("eval_framework.tracks.track5_consistency.runner.switch_model", return_value=True)
    @patch("eval_framework.tracks.track5_consistency._test_repetition_consistency")
    def test_checkpoint_skip(
        self, mock_rep, mock_switch, mock_wait, mock_load_ckpt,
        mock_save_ckpt, mock_save_res,
    ):
        """체크포인트에서 이미 완료된 작업은 건너뜀"""
        result = run(models=["m1"])

        mock_rep.assert_not_called()
        mock_switch.assert_not_called()

    @patch("eval_framework.tracks.track5_consistency.runner.save_results_incremental")
    @patch("eval_framework.tracks.track5_consistency.runner.save_checkpoint")
    @patch("eval_framework.tracks.track5_consistency.runner.load_checkpoint", return_value=None)
    @patch("eval_framework.tracks.track5_consistency.runner.wait_for_ollama", return_value=True)
    @patch("eval_framework.tracks.track5_consistency.runner.switch_model", return_value=True)
    @patch("eval_framework.tracks.track5_consistency._test_hallucination_detection", return_value=[])
    @patch("eval_framework.tracks.track5_consistency._test_instruction_following", return_value=[])
    @patch("eval_framework.tracks.track5_consistency._test_language_mixing", return_value=[])
    @patch("eval_framework.tracks.track5_consistency._test_length_sensitivity", return_value=[])
    @patch("eval_framework.tracks.track5_consistency._test_paraphrase_robustness", return_value=[])
    @patch("eval_framework.tracks.track5_consistency._test_repetition_consistency", return_value=[])
    def test_uses_config_models_when_none(
        self, mock_rep, mock_para, mock_len, mock_lang, mock_inst, mock_hall,
        mock_switch, mock_wait, mock_load_ckpt, mock_save_ckpt, mock_save_res,
    ):
        """models=None이면 config.ALL_MODELS 사용"""
        with patch("eval_framework.tracks.track5_consistency.config.ALL_MODELS", ["cfg-model"]):
            result = run(models=None)

        mock_switch.assert_called_once_with("cfg-model", None)

    @patch("eval_framework.tracks.track5_consistency.runner.save_results_incremental")
    @patch("eval_framework.tracks.track5_consistency.runner.save_checkpoint")
    @patch("eval_framework.tracks.track5_consistency.runner.load_checkpoint", return_value=None)
    @patch("eval_framework.tracks.track5_consistency.runner.wait_for_ollama", return_value=True)
    @patch("eval_framework.tracks.track5_consistency.runner.switch_model", return_value=True)
    @patch("eval_framework.tracks.track5_consistency._test_hallucination_detection", return_value=[])
    @patch("eval_framework.tracks.track5_consistency._test_instruction_following", return_value=[])
    @patch("eval_framework.tracks.track5_consistency._test_language_mixing", return_value=[])
    @patch("eval_framework.tracks.track5_consistency._test_length_sensitivity", return_value=[])
    @patch("eval_framework.tracks.track5_consistency._test_paraphrase_robustness", return_value=[])
    @patch("eval_framework.tracks.track5_consistency._test_repetition_consistency", return_value=[])
    def test_save_results_called(
        self, mock_rep, mock_para, mock_len, mock_lang, mock_inst, mock_hall,
        mock_switch, mock_wait, mock_load_ckpt, mock_save_ckpt, mock_save_res,
    ):
        """실행 완료 후 save_results_incremental 호출"""
        run(models=["test-model"])

        mock_save_res.assert_called_once()
        args = mock_save_res.call_args
        assert args[0][1] == TRACK_NAME
