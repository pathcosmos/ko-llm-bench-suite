"""kobench/tracks/code_math.py 단위 테스트"""

import json
import pytest
from unittest.mock import patch, MagicMock

from kobench.tracks.code_math import (
    _extract_python_code,
    _extract_sql,
    _extract_numeric_answer,
    _check_math_answer,
    _build_test_harness,
    _run_python_code,
    _run_sql_test,
    _evaluate_debug,
    _eval_python,
    _eval_sql,
    _eval_debug,
    _eval_math,
    run,
)


# ═══════════════════════════════════════════════════════════════════════════════
# _extract_python_code 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractPythonCode:
    """_extract_python_code: 응답에서 Python 코드 블록 추출"""

    def test_python_fenced_block(self):
        """```python 블록에서 코드 추출"""
        response = "설명입니다.\n```python\ndef foo():\n    return 42\n```\n끝."
        result = _extract_python_code(response)
        assert result == "def foo():\n    return 42"

    def test_generic_fenced_block(self):
        """언어 지정 없는 ``` 블록에서 코드 추출"""
        response = "코드:\n```\ndef bar():\n    pass\n```"
        result = _extract_python_code(response)
        assert result == "def bar():\n    pass"

    def test_python_block_preferred_over_generic(self):
        """```python 블록이 ``` 블록보다 우선"""
        response = "```\ngeneric\n```\n```python\ndef real():\n    pass\n```"
        result = _extract_python_code(response)
        assert result == "def real():\n    pass"

    def test_def_fallback(self):
        """코드 블록 없으면 def로 시작하는 부분 추출"""
        response = "설명입니다.\ndef solve(x):\n    return x + 1\n"
        result = _extract_python_code(response)
        assert "def solve(x):" in result
        assert "return x + 1" in result

    def test_no_code_returns_stripped(self):
        """코드 블록도 def도 없으면 전체 텍스트 반환"""
        response = "  그냥 텍스트입니다  "
        result = _extract_python_code(response)
        assert result == "그냥 텍스트입니다"

    def test_empty_response(self):
        """빈 응답"""
        result = _extract_python_code("")
        assert result == ""

    def test_multiline_python_block(self):
        """여러 줄 Python 코드 블록"""
        response = "```python\ndef add(a, b):\n    s = a + b\n    return s\n```"
        result = _extract_python_code(response)
        assert "def add(a, b):" in result
        assert "return s" in result


# ═══════════════════════════════════════════════════════════════════════════════
# _extract_sql 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractSql:
    """_extract_sql: 응답에서 SQL 쿼리 추출"""

    def test_sql_fenced_block(self):
        """```sql 블록에서 SQL 추출"""
        response = "쿼리:\n```sql\nSELECT * FROM users;\n```"
        result = _extract_sql(response)
        assert result == "SELECT * FROM users;"

    def test_generic_fenced_block(self):
        """언어 지정 없는 ``` 블록에서 SQL 추출"""
        response = "```\nSELECT id FROM items;\n```"
        result = _extract_sql(response)
        assert result == "SELECT id FROM items;"

    def test_select_fallback(self):
        """코드 블록 없으면 SELECT로 시작하는 줄 추출"""
        response = "답변입니다.\nSELECT name\nFROM users\nWHERE age > 20;"
        result = _extract_sql(response)
        assert "SELECT name" in result
        assert "WHERE age > 20;" in result

    def test_no_sql_returns_stripped(self):
        """SQL도 블록도 없으면 전체 텍스트 반환"""
        response = "  텍스트  "
        result = _extract_sql(response)
        assert result == "텍스트"

    def test_sql_block_preferred_over_generic(self):
        """```sql 블록이 우선"""
        response = "```\ngeneric\n```\n```sql\nSELECT 1;\n```"
        result = _extract_sql(response)
        assert result == "SELECT 1;"

    def test_empty_response(self):
        """빈 응답"""
        result = _extract_sql("")
        assert result == ""


# ═══════════════════════════════════════════════════════════════════════════════
# _extract_numeric_answer 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractNumericAnswer:
    """_extract_numeric_answer: 응답에서 숫자 답 추출"""

    def test_answer_pattern_korean(self):
        """'답: 숫자' 패턴"""
        assert _extract_numeric_answer("따라서 답: 42") == 42.0

    def test_answer_pattern_equals(self):
        """'정답= 숫자' 패턴"""
        assert _extract_numeric_answer("정답= 3.14") == 3.14

    def test_bold_pattern(self):
        """**숫자** 볼드 패턴"""
        assert _extract_numeric_answer("최종 결과는 **100** 입니다") == 100.0

    def test_therefore_pattern(self):
        """'따라서' 패턴"""
        assert _extract_numeric_answer("따라서 결과는 25입니다") == 25.0

    def test_equals_pattern(self):
        """= 숫자 패턴"""
        assert _extract_numeric_answer("3 + 4 = 7") == 7.0

    def test_last_number_fallback(self):
        """패턴 매칭 실패 시 마지막 숫자 추출"""
        assert _extract_numeric_answer("계산하면 5, 10, 15") == 15.0

    def test_negative_number(self):
        """음수 추출"""
        assert _extract_numeric_answer("답: -5") == -5.0

    def test_decimal_number(self):
        """소수 추출"""
        assert _extract_numeric_answer("답: 3.14159") == 3.14159

    def test_no_number(self):
        """숫자가 없으면 None 반환"""
        assert _extract_numeric_answer("숫자 없는 텍스트") is None

    def test_empty_response(self):
        """빈 응답"""
        assert _extract_numeric_answer("") is None


# ═══════════════════════════════════════════════════════════════════════════════
# _check_math_answer 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckMathAnswer:
    """_check_math_answer: 수학 답 비교"""

    def test_exact_match(self):
        """정확히 일치"""
        assert _check_math_answer(42.0, 42.0) is True

    def test_within_tolerance(self):
        """허용 오차 내"""
        assert _check_math_answer(42.005, 42.0) is True

    def test_outside_tolerance(self):
        """허용 오차 초과"""
        assert _check_math_answer(42.02, 42.0) is False

    def test_none_extracted(self):
        """추출 값이 None이면 False"""
        assert _check_math_answer(None, 42.0) is False

    def test_custom_tolerance(self):
        """사용자 정의 허용 오차"""
        assert _check_math_answer(42.5, 42.0, tolerance=1.0) is True
        assert _check_math_answer(44.0, 42.0, tolerance=1.0) is False

    def test_negative_numbers(self):
        """음수 비교"""
        assert _check_math_answer(-5.0, -5.0) is True
        assert _check_math_answer(-5.02, -5.0) is False

    def test_zero(self):
        """0 비교"""
        assert _check_math_answer(0.0, 0.0) is True
        assert _check_math_answer(0.005, 0.0) is True


# ═══════════════════════════════════════════════════════════════════════════════
# _build_test_harness 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildTestHarness:
    """_build_test_harness: 테스트 하네스 코드 생성"""

    def test_single_test_case(self):
        """단일 테스트 케이스"""
        test_cases = [{"input": [1, 2], "expected_output": 3}]
        code = _build_test_harness("add", test_cases)
        assert "import json" in code
        assert "add(*[1, 2])" in code
        assert "print(json.dumps(results))" in code

    def test_multiple_test_cases(self):
        """여러 테스트 케이스"""
        test_cases = [
            {"input": [1, 2], "expected_output": 3},
            {"input": [0, 0], "expected_output": 0},
        ]
        code = _build_test_harness("add", test_cases)
        assert code.count("try:") == 2

    def test_generated_code_is_executable(self):
        """생성된 코드가 실행 가능한 파이썬 구문"""
        test_cases = [{"input": ["hello"], "expected_output": 5}]
        code = _build_test_harness("strlen", test_cases)
        # 구문 오류 없이 컴파일 가능해야 함
        compile(code, "<string>", "exec")

    def test_empty_test_cases(self):
        """빈 테스트 케이스 목록"""
        code = _build_test_harness("func", [])
        assert "results = []" in code
        assert "print(json.dumps(results))" in code

    def test_exception_handling(self):
        """예외 처리 코드 포함"""
        test_cases = [{"input": [1], "expected_output": 1}]
        code = _build_test_harness("f", test_cases)
        assert "except Exception as e:" in code
        assert "'pass': False" in code


# ═══════════════════════════════════════════════════════════════════════════════
# _run_python_code 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunPythonCode:
    """_run_python_code: Python 코드 샌드박스 실행"""

    def test_successful_execution(self):
        """정상 실행"""
        result = _run_python_code("x = 1 + 1", "print(x)")
        assert result["returncode"] == 0
        assert result["stdout"] == "2"
        assert result["error"] is None

    def test_syntax_error(self):
        """구문 오류"""
        result = _run_python_code("def broken(:", "")
        assert result["returncode"] != 0
        assert result["stderr"] != ""

    def test_timeout(self):
        """타임아웃"""
        result = _run_python_code("import time; time.sleep(10)", "", timeout=1)
        assert result["error"] == "timeout"
        assert result["returncode"] == -1

    def test_runtime_error(self):
        """런타임 오류"""
        result = _run_python_code("x = 1/0", "")
        assert result["returncode"] != 0

    def test_empty_code(self):
        """빈 코드"""
        result = _run_python_code("", "print('ok')")
        assert result["stdout"] == "ok"

    def test_temp_file_cleanup(self):
        """임시 파일 정리 확인 (subprocess mock)"""
        with patch("kobench.tracks.code_math.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="", stderr="", returncode=0
            )
            result = _run_python_code("pass", "")
            assert result["error"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# _run_sql_test 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunSqlTest:
    """_run_sql_test: SQL 쿼리 실행 및 결과 비교"""

    def test_matching_results(self):
        """결과가 일치하는 경우"""
        schema = "CREATE TABLE t (id INT, name TEXT); INSERT INTO t VALUES (1, 'a'); INSERT INTO t VALUES (2, 'b')"
        result = _run_sql_test(schema, "SELECT * FROM t", "SELECT * FROM t")
        assert result["correct"] is True
        assert result["error"] is None

    def test_different_results(self):
        """결과가 다른 경우"""
        schema = "CREATE TABLE t (id INT); INSERT INTO t VALUES (1); INSERT INTO t VALUES (2)"
        result = _run_sql_test(schema, "SELECT * FROM t WHERE id=1", "SELECT * FROM t")
        assert result["correct"] is False

    def test_order_independent_match(self):
        """순서 무시 비교"""
        schema = "CREATE TABLE t (id INT); INSERT INTO t VALUES (1); INSERT INTO t VALUES (2)"
        result = _run_sql_test(
            schema,
            "SELECT * FROM t ORDER BY id DESC",
            "SELECT * FROM t ORDER BY id ASC",
        )
        assert result["correct"] is True

    def test_invalid_query_returns_error(self):
        """잘못된 SQL 쿼리"""
        schema = "CREATE TABLE t (id INT)"
        result = _run_sql_test(schema, "INVALID SQL", "SELECT * FROM t")
        assert result["correct"] is False
        assert result["error"] is not None

    def test_invalid_schema_returns_error(self):
        """잘못된 스키마"""
        result = _run_sql_test("INVALID SCHEMA", "SELECT 1", "SELECT 1")
        assert result["correct"] is False
        assert result["error"] is not None

    def test_trailing_semicolon_handled(self):
        """세미콜론 제거 처리"""
        schema = "CREATE TABLE t (id INT); INSERT INTO t VALUES (1)"
        result = _run_sql_test(schema, "SELECT * FROM t;", "SELECT * FROM t")
        assert result["correct"] is True

    def test_empty_table(self):
        """빈 테이블"""
        schema = "CREATE TABLE t (id INT)"
        result = _run_sql_test(schema, "SELECT * FROM t", "SELECT * FROM t")
        assert result["correct"] is True
        assert result["expected_rows"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# _evaluate_debug 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluateDebug:
    """_evaluate_debug: 디버깅 응답 평가"""

    def _make_problem(self, bug_description="off-by-one 인덱스 오류"):
        return {
            "bug_description": bug_description,
            "test_cases": [
                {"input": [[1, 2, 3]], "expected_output": 6},
            ],
        }

    def test_bug_identified_and_fix_works(self):
        """버그 식별 + 수정 성공"""
        problem = self._make_problem()
        response = (
            "인덱스가 range(0 부터 시작해야 합니다.\n"
            "```python\n"
            "def solve(lst):\n"
            "    return sum(lst)\n"
            "```"
        )
        with patch(
            "kobench.tracks.code_math._run_python_code",
            return_value={
                "stdout": json.dumps([{"pass": True, "got": 6}]),
                "stderr": "",
                "returncode": 0,
                "error": None,
            },
        ):
            result = _evaluate_debug(response, problem)
        assert result["bug_identified"] is True
        assert result["fix_works"] is True
        assert result["correct"] is True

    def test_bug_not_identified(self):
        """버그 미식별"""
        problem = self._make_problem()
        response = (
            "코드가 이상합니다.\n"
            "```python\n"
            "def solve(lst):\n"
            "    return sum(lst)\n"
            "```"
        )
        with patch(
            "kobench.tracks.code_math._run_python_code",
            return_value={
                "stdout": json.dumps([{"pass": True, "got": 6}]),
                "stderr": "",
                "returncode": 0,
                "error": None,
            },
        ):
            result = _evaluate_debug(response, problem)
        assert result["bug_identified"] is False
        assert result["correct"] is False

    def test_fix_fails(self):
        """수정 코드 실행 실패"""
        problem = self._make_problem()
        response = (
            "range(0 부터 시작해야 합니다.\n"
            "```python\n"
            "def solve(lst):\n"
            "    return 0\n"
            "```"
        )
        with patch(
            "kobench.tracks.code_math._run_python_code",
            return_value={
                "stdout": json.dumps([{"pass": False, "got": 0}]),
                "stderr": "",
                "returncode": 0,
                "error": None,
            },
        ):
            result = _evaluate_debug(response, problem)
        assert result["bug_identified"] is True
        assert result["fix_works"] is False
        assert result["correct"] is False

    def test_no_code_in_response(self):
        """응답에 코드가 없는 경우"""
        problem = self._make_problem()
        response = "range(0 으로 시작해야 합니다."
        # No def in response, _extract_python_code returns stripped text
        # No func_match, so fix_works=False
        result = _evaluate_debug(response, problem)
        assert result["fix_works"] is False

    def test_runtime_error_in_fix(self):
        """수정 코드 런타임 오류"""
        problem = self._make_problem()
        response = (
            "range(0 부터.\n"
            "```python\n"
            "def solve(lst):\n"
            "    return sum(lst)\n"
            "```"
        )
        with patch(
            "kobench.tracks.code_math._run_python_code",
            return_value={
                "stdout": "",
                "stderr": "Error",
                "returncode": 1,
                "error": "runtime error",
            },
        ):
            result = _evaluate_debug(response, problem)
        assert result["fix_works"] is False

    def test_no_bug_keywords_defaults_identified(self):
        """키워드 목록이 비면 bug_identified=True"""
        problem = self._make_problem(bug_description="알 수 없는 버그")
        response = (
            "수정합니다.\n"
            "```python\n"
            "def solve(lst):\n"
            "    return sum(lst)\n"
            "```"
        )
        with patch(
            "kobench.tracks.code_math._run_python_code",
            return_value={
                "stdout": json.dumps([{"pass": True, "got": 6}]),
                "stderr": "",
                "returncode": 0,
                "error": None,
            },
        ):
            result = _evaluate_debug(response, problem)
        assert result["bug_identified"] is True

    def test_comparison_operator_bug(self):
        """비교 연산자 관련 버그 키워드"""
        problem = self._make_problem(bug_description="비교 연산자가 반대")
        response = (
            "> 연산자가 반대입니다.\n"
            "```python\n"
            "def solve(lst):\n"
            "    return max(lst)\n"
            "```"
        )
        with patch(
            "kobench.tracks.code_math._run_python_code",
            return_value={
                "stdout": json.dumps([{"pass": False, "got": 3}]),
                "stderr": "",
                "returncode": 0,
                "error": None,
            },
        ):
            result = _evaluate_debug(response, problem)
        assert result["bug_identified"] is True

    def test_invalid_json_stdout(self):
        """stdout이 유효한 JSON이 아닌 경우"""
        problem = self._make_problem()
        response = (
            "range(0 부터.\n"
            "```python\n"
            "def solve(lst):\n"
            "    return sum(lst)\n"
            "```"
        )
        with patch(
            "kobench.tracks.code_math._run_python_code",
            return_value={
                "stdout": "not valid json",
                "stderr": "",
                "returncode": 0,
                "error": None,
            },
        ):
            result = _evaluate_debug(response, problem)
        assert result["fix_works"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# _eval_python 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvalPython:
    """_eval_python: Python 코딩 문제 평가"""

    def _sample_problem(self):
        return {
            "id": "py1",
            "description": "두 수를 더하는 함수를 작성하세요",
            "function_signature": "def add(a, b):",
            "test_cases": [
                {"input": [1, 2], "expected_output": 3},
                {"input": [0, 0], "expected_output": 0},
            ],
        }

    @patch("kobench.tracks.code_math.time.sleep")
    @patch("kobench.tracks.code_math._run_python_code")
    @patch("kobench.tracks.code_math.runner.generate")
    def test_all_tests_pass(self, mock_generate, mock_run, mock_sleep):
        """모든 테스트 통과"""
        mock_generate.return_value = {
            "response": "```python\ndef add(a, b):\n    return a + b\n```",
            "error": None,
        }
        mock_run.return_value = {
            "stdout": json.dumps([{"pass": True}, {"pass": True}]),
            "stderr": "",
            "returncode": 0,
            "error": None,
        }
        results = _eval_python("model-a", [self._sample_problem()])
        assert len(results) == 1
        assert results[0]["pass_at_1"] == 1.0
        assert results[0]["passed"] == 2

    @patch("kobench.tracks.code_math.time.sleep")
    @patch("kobench.tracks.code_math.runner.generate")
    def test_generate_error(self, mock_generate, mock_sleep):
        """모델 응답 오류"""
        mock_generate.return_value = {
            "response": "",
            "error": "timeout",
        }
        results = _eval_python("model-a", [self._sample_problem()])
        assert results[0]["pass_at_1"] == 0.0
        assert results[0]["error"] == "timeout"

    @patch("kobench.tracks.code_math.time.sleep")
    @patch("kobench.tracks.code_math._run_python_code")
    @patch("kobench.tracks.code_math.runner.generate")
    def test_partial_pass(self, mock_generate, mock_run, mock_sleep):
        """일부 테스트만 통과"""
        mock_generate.return_value = {
            "response": "```python\ndef add(a, b):\n    return a + b\n```",
            "error": None,
        }
        mock_run.return_value = {
            "stdout": json.dumps([{"pass": True}, {"pass": False}]),
            "stderr": "",
            "returncode": 0,
            "error": None,
        }
        results = _eval_python("model-a", [self._sample_problem()])
        assert results[0]["pass_at_1"] == 0.0
        assert results[0]["passed"] == 1

    @patch("kobench.tracks.code_math.time.sleep")
    @patch("kobench.tracks.code_math._run_python_code")
    @patch("kobench.tracks.code_math.runner.generate")
    def test_execution_error(self, mock_generate, mock_run, mock_sleep):
        """코드 실행 오류"""
        mock_generate.return_value = {
            "response": "```python\ndef add(a, b):\n    return a + b\n```",
            "error": None,
        }
        mock_run.return_value = {
            "stdout": "",
            "stderr": "SyntaxError",
            "returncode": 1,
            "error": "runtime error",
        }
        results = _eval_python("model-a", [self._sample_problem()])
        assert results[0]["passed"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# _eval_sql 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvalSql:
    """_eval_sql: SQL 문제 평가"""

    def _sample_problem(self):
        return {
            "id": "sql1",
            "description": "모든 사용자를 조회하세요",
            "schema": "CREATE TABLE users (id INT, name TEXT)",
            "test_query": "SELECT * FROM users",
        }

    @patch("kobench.tracks.code_math.time.sleep")
    @patch("kobench.tracks.code_math._run_sql_test")
    @patch("kobench.tracks.code_math.runner.generate")
    def test_correct_sql(self, mock_generate, mock_run_sql, mock_sleep):
        """올바른 SQL 생성"""
        mock_generate.return_value = {
            "response": "```sql\nSELECT * FROM users;\n```",
            "error": None,
        }
        mock_run_sql.return_value = {
            "correct": True,
            "expected_rows": [],
            "generated_rows": [],
            "error": None,
        }
        results = _eval_sql("model-a", [self._sample_problem()])
        assert results[0]["correct"] is True

    @patch("kobench.tracks.code_math.time.sleep")
    @patch("kobench.tracks.code_math.runner.generate")
    def test_generate_error(self, mock_generate, mock_sleep):
        """모델 응답 오류"""
        mock_generate.return_value = {
            "response": "",
            "error": "timeout",
        }
        results = _eval_sql("model-a", [self._sample_problem()])
        assert results[0]["correct"] is False
        assert results[0]["error"] == "timeout"

    @patch("kobench.tracks.code_math.time.sleep")
    @patch("kobench.tracks.code_math._run_sql_test")
    @patch("kobench.tracks.code_math.runner.generate")
    def test_incorrect_sql(self, mock_generate, mock_run_sql, mock_sleep):
        """잘못된 SQL 결과"""
        mock_generate.return_value = {
            "response": "```sql\nSELECT id FROM users;\n```",
            "error": None,
        }
        mock_run_sql.return_value = {
            "correct": False,
            "expected_rows": [(1, "a")],
            "generated_rows": [(1,)],
            "error": None,
        }
        results = _eval_sql("model-a", [self._sample_problem()])
        assert results[0]["correct"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# _eval_debug 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvalDebug:
    """_eval_debug: 디버깅 문제 평가"""

    def _sample_problem(self):
        return {
            "id": "dbg1",
            "description": "인덱스 오류가 있는 합계 함수를 수정하세요",
            "buggy_code": "def solve(lst):\n    total = 0\n    for i in range(1, len(lst)):\n        total += lst[i]\n    return total",
            "bug_description": "off-by-one 인덱스 오류",
            "test_cases": [
                {"input": [[1, 2, 3]], "expected_output": 6},
            ],
        }

    @patch("kobench.tracks.code_math.time.sleep")
    @patch("kobench.tracks.code_math._evaluate_debug")
    @patch("kobench.tracks.code_math.runner.generate")
    def test_successful_debug(self, mock_generate, mock_eval_debug, mock_sleep):
        """디버깅 성공"""
        mock_generate.return_value = {
            "response": "range(0 부터 시작해야 합니다.\n```python\ndef solve(lst):\n    return sum(lst)\n```",
            "error": None,
        }
        mock_eval_debug.return_value = {
            "bug_identified": True,
            "fix_works": True,
            "correct": True,
        }
        results = _eval_debug("model-a", [self._sample_problem()])
        assert results[0]["correct"] is True
        assert results[0]["id"] == "dbg1"

    @patch("kobench.tracks.code_math.time.sleep")
    @patch("kobench.tracks.code_math.runner.generate")
    def test_generate_error(self, mock_generate, mock_sleep):
        """모델 응답 오류"""
        mock_generate.return_value = {
            "response": "",
            "error": "connection error",
        }
        results = _eval_debug("model-a", [self._sample_problem()])
        assert results[0]["correct"] is False
        assert results[0]["error"] == "connection error"


# ═══════════════════════════════════════════════════════════════════════════════
# _eval_math 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvalMath:
    """_eval_math: 수학 문제 평가"""

    def _sample_problem(self):
        return {
            "id": "math1",
            "question": "1 + 1은 얼마입니까?",
            "answer": 2.0,
        }

    @patch("kobench.tracks.code_math.time.sleep")
    @patch("kobench.tracks.code_math.runner.generate")
    def test_correct_answer(self, mock_generate, mock_sleep):
        """올바른 답"""
        mock_generate.return_value = {
            "response": "1 + 1 = 2이므로 답: 2",
            "error": None,
        }
        results = _eval_math("model-a", [self._sample_problem()])
        assert results[0]["correct"] is True
        assert results[0]["extracted"] == 2.0

    @patch("kobench.tracks.code_math.time.sleep")
    @patch("kobench.tracks.code_math.runner.generate")
    def test_wrong_answer(self, mock_generate, mock_sleep):
        """잘못된 답"""
        mock_generate.return_value = {
            "response": "답: 5",
            "error": None,
        }
        results = _eval_math("model-a", [self._sample_problem()])
        assert results[0]["correct"] is False

    @patch("kobench.tracks.code_math.time.sleep")
    @patch("kobench.tracks.code_math.runner.generate")
    def test_generate_error(self, mock_generate, mock_sleep):
        """모델 응답 오류"""
        mock_generate.return_value = {
            "response": "",
            "error": "OOM",
        }
        results = _eval_math("model-a", [self._sample_problem()])
        assert results[0]["correct"] is False
        assert results[0]["error"] == "OOM"
        assert results[0]["extracted"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# run 테스트
# ═══════════════════════════════════════════════════════════════════════════════


class TestRun:
    """run: Track 4 전체 실행 흐름"""

    @pytest.fixture(autouse=True)
    def _setup_patches(self):
        """공통 패치 설정"""
        code_data = {
            "python_problems": [
                {
                    "id": "py1",
                    "description": "두 수를 더하는 함수",
                    "function_signature": "def add(a, b):",
                    "test_cases": [{"input": [1, 2], "expected_output": 3}],
                }
            ],
            "sql_problems": [
                {
                    "id": "sql1",
                    "description": "모든 사용자 조회",
                    "schema": "CREATE TABLE t (id INT)",
                    "test_query": "SELECT * FROM t",
                }
            ],
            "debugging_problems": [
                {
                    "id": "dbg1",
                    "description": "인덱스 오류 수정",
                    "buggy_code": "def f(x): return x",
                    "bug_description": "off-by-one",
                    "test_cases": [{"input": [1], "expected_output": 1}],
                }
            ],
        }
        math_data = [
            {"id": "math1", "question": "1+1은?", "answer": 2.0}
        ]

        patches = [
            patch(
                "kobench.tracks.code_math._load_code_problems",
                return_value=code_data,
            ),
            patch(
                "kobench.tracks.code_math._load_math_problems",
                return_value=math_data,
            ),
            patch("kobench.tracks.code_math.runner.load_checkpoint", return_value=None),
            patch("kobench.tracks.code_math.runner.switch_model", return_value=True),
            patch("kobench.tracks.code_math.runner.save_checkpoint"),
            patch("kobench.tracks.code_math.runner.save_results_incremental"),
            patch("kobench.tracks.code_math.time.sleep"),
        ]
        for p in patches:
            p.start()

        yield
        patch.stopall()

    @patch("kobench.tracks.code_math._eval_math")
    @patch("kobench.tracks.code_math._eval_debug")
    @patch("kobench.tracks.code_math._eval_sql")
    @patch("kobench.tracks.code_math._eval_python")
    def test_basic_run_returns_structure(self, mock_py, mock_sql, mock_dbg, mock_math):
        """기본 실행 시 올바른 출력 구조"""
        mock_py.return_value = [{"id": "py1", "pass_at_1": 1.0, "passed": 1, "total": 1}]
        mock_sql.return_value = [{"id": "sql1", "correct": True}]
        mock_dbg.return_value = [{"id": "dbg1", "correct": True}]
        mock_math.return_value = [{"id": "math1", "correct": True}]

        output = run(["test-model"])

        assert output["track"] == "code_math"
        assert "results" in output
        assert "summary" in output
        assert len(output["results"]) == 1
        assert output["results"][0]["model"] == "test-model"

    @patch("kobench.tracks.code_math._eval_math")
    @patch("kobench.tracks.code_math._eval_debug")
    @patch("kobench.tracks.code_math._eval_sql")
    @patch("kobench.tracks.code_math._eval_python")
    def test_scores_computed(self, mock_py, mock_sql, mock_dbg, mock_math):
        """점수 계산 검증"""
        mock_py.return_value = [{"id": "py1", "pass_at_1": 1.0}]
        mock_sql.return_value = [{"id": "sql1", "correct": True}]
        mock_dbg.return_value = [{"id": "dbg1", "correct": False}]
        mock_math.return_value = [{"id": "math1", "correct": True}]

        output = run(["test-model"])
        scores = output["summary"]["test-model"]

        assert scores["python_pass1"] == 1.0
        assert scores["sql_accuracy"] == 1.0
        assert scores["debug_accuracy"] == 0.0
        assert scores["math_accuracy"] == 1.0

    def test_model_switch_failure(self):
        """모델 전환 실패 시 오류 기록"""
        with patch("kobench.tracks.code_math.runner.switch_model", return_value=False):
            output = run(["bad-model"])

        assert len(output["results"]) == 1
        assert output["results"][0]["error"] == "모델 로딩 실패"

    @patch("kobench.tracks.code_math._eval_math")
    @patch("kobench.tracks.code_math._eval_debug")
    @patch("kobench.tracks.code_math._eval_sql")
    @patch("kobench.tracks.code_math._eval_python")
    def test_checkpoint_skips_completed(self, mock_py, mock_sql, mock_dbg, mock_math):
        """체크포인트에서 완료된 모델 스킵"""
        checkpoint_data = {
            "results": [{
                "model": "test-model",
                "scores": {"python_pass1": 0.5, "sql_accuracy": 0.5, "debug_accuracy": 0.5, "math_accuracy": 0.5},
            }],
        }
        with patch("kobench.tracks.code_math.runner.load_checkpoint", return_value=checkpoint_data):
            output = run(["test-model"])

        mock_py.assert_not_called()
        mock_sql.assert_not_called()
        mock_dbg.assert_not_called()
        mock_math.assert_not_called()
        assert output["summary"]["test-model"]["python_pass1"] == 0.5

    @patch("kobench.tracks.code_math._eval_math")
    @patch("kobench.tracks.code_math._eval_debug")
    @patch("kobench.tracks.code_math._eval_sql")
    @patch("kobench.tracks.code_math._eval_python")
    def test_uses_config_models_when_none(self, mock_py, mock_sql, mock_dbg, mock_math):
        """models=None이면 config.ALL_MODELS 사용"""
        mock_py.return_value = []
        mock_sql.return_value = []
        mock_dbg.return_value = []
        mock_math.return_value = []

        with patch("kobench.tracks.code_math.config.ALL_MODELS", ["default-model"]):
            output = run(None)

        assert any(r["model"] == "default-model" for r in output["results"])

    @patch("kobench.tracks.code_math._eval_math")
    @patch("kobench.tracks.code_math._eval_debug")
    @patch("kobench.tracks.code_math._eval_sql")
    @patch("kobench.tracks.code_math._eval_python")
    def test_multiple_models(self, mock_py, mock_sql, mock_dbg, mock_math):
        """여러 모델 평가"""
        mock_py.return_value = [{"id": "py1", "pass_at_1": 1.0}]
        mock_sql.return_value = [{"id": "sql1", "correct": True}]
        mock_dbg.return_value = [{"id": "dbg1", "correct": True}]
        mock_math.return_value = [{"id": "math1", "correct": True}]

        output = run(["model-a", "model-b"])

        assert len(output["results"]) == 2
        models = {r["model"] for r in output["results"]}
        assert models == {"model-a", "model-b"}
