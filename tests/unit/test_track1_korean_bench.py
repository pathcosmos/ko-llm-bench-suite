"""kobench/tracks/korean_bench.py 단위 테스트"""

import json
import pytest
from unittest.mock import patch, MagicMock, call

from kobench.tracks.korean_bench import (
    _build_kobest_boolq,
    _build_kobest_copa,
    _build_kobest_sentineg,
    _build_kobest_hellaswag,
    _build_kmmlu,
    _build_all_questions,
    _parse_answer,
    _lm_eval_available,
    _run_lm_eval,
    _run_standalone,
    run,
    TRACK_NAME,
)


# ═══════════════════════════════════════════════════════════════════════════════
# _build_kobest_boolq 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildKobestBoolq:
    """_build_kobest_boolq: 예/아니오 독해 문항 생성"""

    def test_returns_20_items(self):
        """20개 문항을 반환"""
        items = _build_kobest_boolq()
        assert len(items) == 20

    def test_item_structure(self):
        """각 문항의 필수 키 검증"""
        items = _build_kobest_boolq()
        for item in items:
            assert "id" in item
            assert "benchmark" in item
            assert "question" in item
            assert "choices" in item
            assert "answer" in item
            assert "subject" in item

    def test_benchmark_label(self):
        """benchmark 필드가 'kobest_boolq'"""
        items = _build_kobest_boolq()
        for item in items:
            assert item["benchmark"] == "kobest_boolq"

    def test_choices_are_yes_no(self):
        """선택지가 ['예', '아니오']"""
        items = _build_kobest_boolq()
        for item in items:
            assert item["choices"] == ["예", "아니오"]

    def test_answer_is_0_or_1(self):
        """정답이 0 또는 1"""
        items = _build_kobest_boolq()
        for item in items:
            assert item["answer"] in (0, 1)

    def test_id_format(self):
        """ID 형식이 boolq_000 ~ boolq_019"""
        items = _build_kobest_boolq()
        for i, item in enumerate(items):
            assert item["id"] == f"boolq_{i:03d}"

    def test_question_contains_passage(self):
        """질문 텍스트에 지문과 질문이 포함"""
        items = _build_kobest_boolq()
        for item in items:
            assert "지문:" in item["question"]
            assert "질문:" in item["question"]

    def test_subject_is_empty(self):
        """subject 필드는 빈 문자열"""
        items = _build_kobest_boolq()
        for item in items:
            assert item["subject"] == ""


# ═══════════════════════════════════════════════════════════════════════════════
# _build_kobest_copa 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildKobestCopa:
    """_build_kobest_copa: 2지선다 인과관계 추론 문항 생성"""

    def test_returns_20_items(self):
        """20개 문항을 반환"""
        items = _build_kobest_copa()
        assert len(items) == 20

    def test_benchmark_label(self):
        """benchmark 필드가 'kobest_copa'"""
        items = _build_kobest_copa()
        for item in items:
            assert item["benchmark"] == "kobest_copa"

    def test_two_choices(self):
        """선택지가 2개"""
        items = _build_kobest_copa()
        for item in items:
            assert len(item["choices"]) == 2

    def test_answer_is_0_or_1(self):
        """정답이 0 또는 1"""
        items = _build_kobest_copa()
        for item in items:
            assert item["answer"] in (0, 1)

    def test_id_format(self):
        """ID 형식이 copa_000 ~ copa_019"""
        items = _build_kobest_copa()
        for i, item in enumerate(items):
            assert item["id"] == f"copa_{i:03d}"

    def test_question_contains_ab(self):
        """질문에 A, B 선택지 포함"""
        items = _build_kobest_copa()
        for item in items:
            assert "A." in item["question"]
            assert "B." in item["question"]


# ═══════════════════════════════════════════════════════════════════════════════
# _build_kobest_sentineg 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildKobestSentineg:
    """_build_kobest_sentineg: 부정어 포함 감성 분석 문항 생성"""

    def test_returns_20_items(self):
        """20개 문항을 반환"""
        items = _build_kobest_sentineg()
        assert len(items) == 20

    def test_benchmark_label(self):
        """benchmark 필드가 'kobest_sentineg'"""
        items = _build_kobest_sentineg()
        for item in items:
            assert item["benchmark"] == "kobest_sentineg"

    def test_choices_are_positive_negative(self):
        """선택지가 ['긍정', '부정']"""
        items = _build_kobest_sentineg()
        for item in items:
            assert item["choices"] == ["긍정", "부정"]

    def test_answer_is_0_or_1(self):
        """정답이 0(긍정) 또는 1(부정)"""
        items = _build_kobest_sentineg()
        for item in items:
            assert item["answer"] in (0, 1)

    def test_id_format(self):
        """ID 형식이 sentineg_000 ~ sentineg_019"""
        items = _build_kobest_sentineg()
        for i, item in enumerate(items):
            assert item["id"] == f"sentineg_{i:03d}"

    def test_question_contains_sentiment_prompt(self):
        """질문에 감성 판단 프롬프트 포함"""
        items = _build_kobest_sentineg()
        for item in items:
            assert "감성을 판단" in item["question"]


# ═══════════════════════════════════════════════════════════════════════════════
# _build_kobest_hellaswag 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildKobestHellaswag:
    """_build_kobest_hellaswag: 4지선다 상식 추론 문항 생성"""

    def test_returns_20_items(self):
        """20개 문항을 반환"""
        items = _build_kobest_hellaswag()
        assert len(items) == 20

    def test_benchmark_label(self):
        """benchmark 필드가 'kobest_hellaswag'"""
        items = _build_kobest_hellaswag()
        for item in items:
            assert item["benchmark"] == "kobest_hellaswag"

    def test_four_choices(self):
        """선택지가 4개"""
        items = _build_kobest_hellaswag()
        for item in items:
            assert len(item["choices"]) == 4

    def test_answer_range(self):
        """정답 인덱스가 0~3"""
        items = _build_kobest_hellaswag()
        for item in items:
            assert 0 <= item["answer"] <= 3

    def test_id_format(self):
        """ID 형식이 hellaswag_000 ~ hellaswag_019"""
        items = _build_kobest_hellaswag()
        for i, item in enumerate(items):
            assert item["id"] == f"hellaswag_{i:03d}"

    def test_question_contains_abcd(self):
        """질문에 A, B, C, D 선택지 포함"""
        items = _build_kobest_hellaswag()
        for item in items:
            assert "A." in item["question"]
            assert "B." in item["question"]
            assert "C." in item["question"]
            assert "D." in item["question"]


# ═══════════════════════════════════════════════════════════════════════════════
# _build_kmmlu 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildKmmlu:
    """_build_kmmlu: KMMLU subset 문항 생성"""

    def test_returns_50_items(self):
        """10과목 x 5문항 = 50개"""
        items = _build_kmmlu()
        assert len(items) == 50

    def test_benchmark_label(self):
        """benchmark 필드가 'kmmlu'"""
        items = _build_kmmlu()
        for item in items:
            assert item["benchmark"] == "kmmlu"

    def test_four_choices(self):
        """선택지가 4개"""
        items = _build_kmmlu()
        for item in items:
            assert len(item["choices"]) == 4

    def test_answer_range(self):
        """정답 인덱스가 0~3"""
        items = _build_kmmlu()
        for item in items:
            assert 0 <= item["answer"] <= 3

    def test_id_format(self):
        """ID 형식이 kmmlu_000 ~ kmmlu_049"""
        items = _build_kmmlu()
        for i, item in enumerate(items):
            assert item["id"] == f"kmmlu_{i:03d}"

    def test_subject_is_non_empty(self):
        """subject 필드가 비어 있지 않음"""
        items = _build_kmmlu()
        for item in items:
            assert item["subject"] != ""

    def test_ten_subjects(self):
        """10개 과목이 존재"""
        items = _build_kmmlu()
        subjects = set(item["subject"] for item in items)
        assert len(subjects) == 10

    def test_five_items_per_subject(self):
        """과목당 5문항"""
        items = _build_kmmlu()
        from collections import Counter
        counts = Counter(item["subject"] for item in items)
        for subject, count in counts.items():
            assert count == 5, f"{subject}: {count}개"

    def test_question_contains_subject(self):
        """질문에 과목명 포함"""
        items = _build_kmmlu()
        for item in items:
            assert f"[{item['subject']}]" in item["question"]


# ═══════════════════════════════════════════════════════════════════════════════
# _build_all_questions 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildAllQuestions:
    """_build_all_questions: 전체 standalone 문항 통합"""

    def test_total_count(self):
        """총 문항 수: 20+20+20+20+50 = 130"""
        questions = _build_all_questions()
        assert len(questions) == 130

    def test_contains_all_benchmarks(self):
        """모든 벤치마크 포함"""
        questions = _build_all_questions()
        benchmarks = set(q["benchmark"] for q in questions)
        assert benchmarks == {
            "kobest_boolq",
            "kobest_copa",
            "kobest_sentineg",
            "kobest_hellaswag",
            "kmmlu",
        }

    def test_order_boolq_first(self):
        """boolq가 첫 번째"""
        questions = _build_all_questions()
        assert questions[0]["benchmark"] == "kobest_boolq"

    def test_order_kmmlu_last(self):
        """kmmlu가 마지막"""
        questions = _build_all_questions()
        assert questions[-1]["benchmark"] == "kmmlu"

    def test_unique_ids(self):
        """모든 ID가 고유"""
        questions = _build_all_questions()
        ids = [q["id"] for q in questions]
        assert len(ids) == len(set(ids))


# ═══════════════════════════════════════════════════════════════════════════════
# _parse_answer 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseAnswer:
    """_parse_answer: 모델 응답에서 정답 인덱스 추출"""

    # --- BoolQ ---

    def test_boolq_yes(self):
        """BoolQ: '예' -> 0"""
        assert _parse_answer("예", "kobest_boolq", 2) == 0

    def test_boolq_no(self):
        """BoolQ: '아니오' -> 1"""
        assert _parse_answer("아니오", "kobest_boolq", 2) == 1

    def test_boolq_yes_with_explanation(self):
        """BoolQ: '예, 맞습니다' -> 0"""
        assert _parse_answer("예, 맞습니다.", "kobest_boolq", 2) == 0

    def test_boolq_no_with_explanation(self):
        """BoolQ: '아니오, 틀립니다' -> 1"""
        assert _parse_answer("아니오, 그렇지 않습니다.", "kobest_boolq", 2) == 1

    # --- COPA / SentiNeg (2지선다) ---

    def test_copa_a(self):
        """COPA: 'A' -> 0"""
        assert _parse_answer("A", "kobest_copa", 2) == 0

    def test_copa_b(self):
        """COPA: 'B' -> 1"""
        assert _parse_answer("B", "kobest_copa", 2) == 1

    def test_sentineg_a(self):
        """SentiNeg: 'A' -> 0"""
        assert _parse_answer("A", "kobest_sentineg", 2) == 0

    def test_sentineg_b_lowercase(self):
        """SentiNeg: 'b' -> 1 (소문자 허용)"""
        assert _parse_answer("b", "kobest_sentineg", 2) == 1

    # --- HellaSwag / KMMLU (4지선다) ---

    def test_hellaswag_a(self):
        """HellaSwag: 'A' -> 0"""
        assert _parse_answer("A", "kobest_hellaswag", 4) == 0

    def test_hellaswag_d(self):
        """HellaSwag: 'D' -> 3"""
        assert _parse_answer("D", "kobest_hellaswag", 4) == 3

    def test_kmmlu_c(self):
        """KMMLU: 'C' -> 2"""
        assert _parse_answer("C", "kmmlu", 4) == 2

    def test_kmmlu_with_prefix(self):
        """KMMLU: '정답은 B입니다' -> 1"""
        assert _parse_answer("정답은 B입니다", "kmmlu", 4) == 1

    # --- 경계 조건 ---

    def test_out_of_range_choice(self):
        """선택지 수를 초과하는 답: 2지선다에서 C -> None"""
        assert _parse_answer("C", "kobest_copa", 2) is None

    def test_empty_response(self):
        """빈 응답 -> None"""
        assert _parse_answer("", "kobest_boolq", 2) is None

    def test_no_matching_pattern(self):
        """패턴 없는 응답 -> None"""
        assert _parse_answer("잘 모르겠습니다", "kmmlu", 4) is None

    def test_lowercase_letter(self):
        """소문자 'd' -> 3"""
        assert _parse_answer("d", "kobest_hellaswag", 4) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# _lm_eval_available 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestLmEvalAvailable:
    """_lm_eval_available: lm_eval CLI 설치 여부 확인"""

    def test_available_when_found(self):
        """lm_eval이 PATH에 있으면 True"""
        with patch("kobench.tracks.korean_bench.shutil.which", return_value="/usr/bin/lm_eval"):
            assert _lm_eval_available() is True

    def test_unavailable_when_not_found(self):
        """lm_eval이 PATH에 없으면 False"""
        with patch("kobench.tracks.korean_bench.shutil.which", return_value=None):
            assert _lm_eval_available() is False


# ═══════════════════════════════════════════════════════════════════════════════
# _run_lm_eval 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunLmEval:
    """_run_lm_eval: lm-evaluation-harness subprocess 실행"""

    def test_returns_parsed_scores_on_success(self, tmp_path):
        """성공 시 파싱된 점수 반환"""
        # 결과 파일 준비
        results_dir = tmp_path / "lm_eval_test-model"
        results_dir.mkdir(parents=True)
        raw_results = {
            "results": {
                "kobest_boolq": {"acc,none": 0.85},
                "kobest_copa": {"acc_norm,none": 0.72},
            }
        }
        (results_dir / "results.json").write_text(json.dumps(raw_results))

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("kobench.tracks.korean_bench.subprocess.run", return_value=mock_result):
            with patch("kobench.tracks.korean_bench.config.RESULTS_DIR", tmp_path):
                scores = _run_lm_eval("test-model")

        assert scores == {"kobest_boolq": 0.85, "kobest_copa": 0.72}

    def test_returns_none_on_nonzero_exit(self):
        """비정상 종료 시 None 반환"""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error message"

        with patch("kobench.tracks.korean_bench.subprocess.run", return_value=mock_result):
            result = _run_lm_eval("test-model")

        assert result is None

    def test_returns_none_on_timeout(self):
        """타임아웃 시 None 반환"""
        import subprocess
        with patch(
            "kobench.tracks.korean_bench.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="lm_eval", timeout=1800),
        ):
            result = _run_lm_eval("test-model")

        assert result is None

    def test_returns_none_on_exception(self):
        """일반 예외 시 None 반환"""
        with patch(
            "kobench.tracks.korean_bench.subprocess.run",
            side_effect=OSError("file not found"),
        ):
            result = _run_lm_eval("test-model")

        assert result is None

    def test_returns_none_when_no_results_file(self, tmp_path):
        """결과 파일이 없으면 None 반환"""
        results_dir = tmp_path / "lm_eval_test-model"
        results_dir.mkdir(parents=True)
        # 결과 파일 없음

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("kobench.tracks.korean_bench.subprocess.run", return_value=mock_result):
            with patch("kobench.tracks.korean_bench.config.RESULTS_DIR", tmp_path):
                result = _run_lm_eval("test-model")

        assert result is None

    def test_missing_acc_defaults_to_zero(self, tmp_path):
        """acc 메트릭이 없으면 0.0으로 처리"""
        results_dir = tmp_path / "lm_eval_test-model"
        results_dir.mkdir(parents=True)
        raw_results = {
            "results": {
                "kobest_boolq": {"f1,none": 0.9},  # acc 없음
            }
        }
        (results_dir / "results.json").write_text(json.dumps(raw_results))

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("kobench.tracks.korean_bench.subprocess.run", return_value=mock_result):
            with patch("kobench.tracks.korean_bench.config.RESULTS_DIR", tmp_path):
                scores = _run_lm_eval("test-model")

        assert scores == {"kobest_boolq": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# _run_standalone 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunStandalone:
    """_run_standalone: standalone 모드 단일 모델 평가"""

    @pytest.fixture(autouse=True)
    def _patch_sleep(self):
        """time.sleep 무력화"""
        with patch("kobench.tracks.korean_bench.time.sleep"):
            yield

    def _make_questions(self, n=2, benchmark="kobest_copa"):
        """테스트용 간단한 문항 생성"""
        return [
            {
                "id": f"test_{i:03d}",
                "benchmark": benchmark,
                "question": f"질문 {i}",
                "choices": ["A선택", "B선택"],
                "answer": 0,
                "subject": "",
            }
            for i in range(n)
        ]

    def test_all_correct(self):
        """모든 문항 정답 시 accuracy 1.0"""
        questions = self._make_questions(2, "kobest_copa")
        gen_result = {
            "response": "A",
            "error": None,
        }

        with patch("kobench.tracks.korean_bench.runner.generate", return_value=gen_result):
            result = _run_standalone("test-model", questions)

        assert result["model"] == "test-model"
        assert result["scores"]["kobest_copa"] == 1.0
        assert len(result["details"]) == 2
        assert all(d["correct"] for d in result["details"])

    def test_all_wrong(self):
        """모든 문항 오답 시 accuracy 0.0"""
        questions = self._make_questions(2, "kobest_copa")
        gen_result = {
            "response": "B",  # 정답은 0(A)이므로 오답
            "error": None,
        }

        with patch("kobench.tracks.korean_bench.runner.generate", return_value=gen_result):
            result = _run_standalone("test-model", questions)

        assert result["scores"]["kobest_copa"] == 0.0
        assert all(not d["correct"] for d in result["details"])

    def test_error_response(self):
        """에러 응답 시 오답 처리"""
        questions = self._make_questions(1, "kobest_copa")
        gen_result = {
            "response": "",
            "error": "model not found",
        }

        with patch("kobench.tracks.korean_bench.runner.generate", return_value=gen_result):
            result = _run_standalone("test-model", questions)

        assert result["scores"]["kobest_copa"] == 0.0
        assert result["details"][0]["correct"] is False
        assert result["details"][0]["error"] == "model not found"

    def test_multiple_benchmarks(self):
        """여러 벤치마크 문항 혼합 시 각각 별도 accuracy 계산"""
        questions = [
            {
                "id": "copa_000",
                "benchmark": "kobest_copa",
                "question": "질문1",
                "choices": ["A", "B"],
                "answer": 0,
                "subject": "",
            },
            {
                "id": "boolq_000",
                "benchmark": "kobest_boolq",
                "question": "질문2",
                "choices": ["예", "아니오"],
                "answer": 0,
                "subject": "",
            },
        ]
        gen_result = {
            "response": "A",
            "error": None,
        }

        with patch("kobench.tracks.korean_bench.runner.generate", return_value=gen_result):
            result = _run_standalone("test-model", questions)

        assert "kobest_copa" in result["scores"]
        assert "kobest_boolq" in result["scores"]

    def test_detail_structure(self):
        """details 항목의 필수 키 검증"""
        questions = self._make_questions(1, "kobest_copa")
        gen_result = {
            "response": "A",
            "error": None,
        }

        with patch("kobench.tracks.korean_bench.runner.generate", return_value=gen_result):
            result = _run_standalone("test-model", questions)

        detail = result["details"][0]
        assert "id" in detail
        assert "benchmark" in detail
        assert "expected" in detail
        assert "parsed" in detail
        assert "correct" in detail
        assert "raw_response" in detail
        assert "error" in detail

    def test_unparseable_response(self):
        """파싱 불가능한 응답 시 parsed=None, correct=False"""
        questions = self._make_questions(1, "kobest_copa")
        gen_result = {
            "response": "잘 모르겠습니다",
            "error": None,
        }

        with patch("kobench.tracks.korean_bench.runner.generate", return_value=gen_result):
            result = _run_standalone("test-model", questions)

        detail = result["details"][0]
        assert detail["parsed"] is None
        assert detail["correct"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# run 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestRun:
    """run: 전체 평가 실행 흐름"""

    @pytest.fixture(autouse=True)
    def _setup_patches(self):
        """공통 패치 설정"""
        patches = [
            patch("kobench.tracks.korean_bench.runner.wait_for_ollama", return_value=True),
            patch("kobench.tracks.korean_bench.runner.load_checkpoint", return_value=None),
            patch("kobench.tracks.korean_bench.runner.switch_model", return_value=True),
            patch("kobench.tracks.korean_bench.runner.save_checkpoint"),
            patch("kobench.tracks.korean_bench.runner.save_results_incremental", return_value="results/test.json"),
            patch("kobench.tracks.korean_bench.time.sleep"),
            patch("kobench.tracks.korean_bench._lm_eval_available", return_value=False),
        ]
        for p in patches:
            p.start()
        yield
        patch.stopall()

    def test_basic_run_returns_structure(self):
        """기본 실행 시 올바른 출력 구조"""
        gen_result = {
            "response": "A",
            "error": None,
        }

        with patch("kobench.tracks.korean_bench.runner.generate", return_value=gen_result):
            output = run(["test-model"])

        assert output["track"] == TRACK_NAME
        assert "results" in output
        assert "summary" in output
        assert "timestamp" in output
        assert output["num_models"] == 1
        assert output["mode"] == "standalone"

    def test_ollama_not_available_raises(self):
        """Ollama 서버 연결 실패 시 RuntimeError"""
        with patch("kobench.tracks.korean_bench.runner.wait_for_ollama", return_value=False):
            with pytest.raises(RuntimeError, match="Ollama"):
                run(["test-model"])

    def test_model_switch_failure(self):
        """모델 전환 실패 시 에러 결과 포함"""
        with patch("kobench.tracks.korean_bench.runner.switch_model", return_value=False):
            output = run(["bad-model"])

        assert len(output["results"]) == 1
        assert output["results"][0]["error"] == "모델 로딩 실패"
        assert output["results"][0]["scores"] == {}

    def test_checkpoint_restores_completed(self):
        """체크포인트에서 이미 완료된 모델은 스킵"""
        checkpoint_data = {
            "results": [{
                "model": "test-model",
                "scores": {"kobest_boolq": 0.85},
                "details": [],
            }],
        }
        with patch("kobench.tracks.korean_bench.runner.load_checkpoint", return_value=checkpoint_data):
            with patch("kobench.tracks.korean_bench.runner.generate") as mock_gen:
                output = run(["test-model"])

        mock_gen.assert_not_called()
        assert len(output["results"]) == 1
        assert output["results"][0]["scores"]["kobest_boolq"] == 0.85

    def test_lm_eval_mode_success(self):
        """lm_eval 모드 성공 시 lm_eval 결과 사용"""
        lm_scores = {"kobest_boolq": 0.9, "kobest_copa": 0.8}

        with patch("kobench.tracks.korean_bench._lm_eval_available", return_value=True):
            with patch("kobench.tracks.korean_bench._run_lm_eval", return_value=lm_scores):
                output = run(["test-model"])

        assert len(output["results"]) == 1
        assert output["results"][0]["scores"] == lm_scores
        assert output["results"][0]["mode"] == "lm_eval"

    def test_lm_eval_fallback_to_standalone(self):
        """lm_eval 실패 시 standalone으로 fallback"""
        gen_result = {
            "response": "A",
            "error": None,
        }

        with patch("kobench.tracks.korean_bench._lm_eval_available", return_value=True):
            with patch("kobench.tracks.korean_bench._run_lm_eval", return_value=None):
                with patch("kobench.tracks.korean_bench.runner.generate", return_value=gen_result):
                    output = run(["test-model"])

        assert len(output["results"]) == 1
        assert output["results"][0]["mode"] == "standalone"

    def test_summary_populated(self):
        """summary에 모델별 점수가 포함"""
        gen_result = {
            "response": "A",
            "error": None,
        }

        with patch("kobench.tracks.korean_bench.runner.generate", return_value=gen_result):
            output = run(["test-model"])

        assert "test-model" in output["summary"]
        assert isinstance(output["summary"]["test-model"], dict)

    def test_default_models_from_config(self):
        """models 인자 없을 시 config.ALL_MODELS 사용"""
        gen_result = {
            "response": "A",
            "error": None,
        }

        with patch("kobench.tracks.korean_bench.config.ALL_MODELS", ["model-a"]):
            with patch("kobench.tracks.korean_bench.runner.generate", return_value=gen_result):
                output = run()

        assert output["num_models"] == 1

    def test_multiple_models(self):
        """여러 모델 평가"""
        gen_result = {
            "response": "A",
            "error": None,
        }

        with patch("kobench.tracks.korean_bench.runner.generate", return_value=gen_result):
            output = run(["model-a", "model-b"])

        assert output["num_models"] == 2
        models_in_results = [r["model"] for r in output["results"]]
        assert "model-a" in models_in_results
        assert "model-b" in models_in_results
