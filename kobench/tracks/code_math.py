"""
Track 4: 코드 및 수학 문제 평가
- Python 코딩 (Pass@1)
- SQL 쿼리 생성
- 디버깅 능력
- 수학 문제 풀이
"""

import json
import re
import sqlite3
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Optional

from kobench import config
from kobench import runner


# ── 데이터 로드 ─────────────────────────────────────────────────────────────

def _load_code_problems() -> dict:
    path = config.DATA_DIR / "code_problems" / "problems.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_math_problems() -> list[dict]:
    path = config.DATA_DIR / "math_problems" / "problems.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)["math_problems"]


# ── Python 코드 실행 (샌드박스) ──────────────────────────────────────────────

def _extract_python_code(response: str) -> str:
    """응답에서 Python 코드 블록을 추출"""
    # ```python ... ``` 블록 탐색
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # ``` ... ``` 블록 (언어 지정 없음)
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # 코드 블록 없으면 def로 시작하는 부분 추출
    lines = response.split("\n")
    code_lines = []
    in_func = False
    for line in lines:
        if line.strip().startswith("def "):
            in_func = True
        if in_func:
            code_lines.append(line)
            # 빈 줄이 나오고 다음 줄이 비함수면 종료
    if code_lines:
        return "\n".join(code_lines).strip()
    return response.strip()


def _run_python_code(code: str, test_code: str, timeout: int = 5) -> dict:
    """Python 코드를 subprocess에서 실행 (샌드박스)"""
    full_code = f"{code}\n\n{test_code}"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "", "returncode": -1, "error": "timeout"}
    except Exception as e:
        return {"stdout": "", "stderr": "", "returncode": -1, "error": str(e)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _build_test_harness(func_name: str, test_cases: list[dict]) -> str:
    """테스트 케이스를 실행하는 테스트 하네스 코드 생성"""
    lines = ["import json", "results = []"]
    for i, tc in enumerate(test_cases):
        args = tc["input"]
        expected = tc["expected_output"]
        lines.append(f"try:")
        lines.append(f"    _result = {func_name}(*{json.dumps(args)})")
        lines.append(f"    _expected = {json.dumps(expected)}")
        lines.append(f"    results.append({{'pass': _result == _expected, 'got': _result}})")
        lines.append(f"except Exception as e:")
        lines.append(f"    results.append({{'pass': False, 'got': str(e)}})")
    lines.append("print(json.dumps(results))")
    return "\n".join(lines)


# ── SQL 실행 ─────────────────────────────────────────────────────────────────

def _extract_sql(response: str) -> str:
    """응답에서 SQL 쿼리를 추출"""
    pattern = r"```sql\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # SELECT로 시작하는 줄 찾기
    lines = response.split("\n")
    sql_lines = []
    capture = False
    for line in lines:
        stripped = line.strip().upper()
        if stripped.startswith("SELECT"):
            capture = True
        if capture:
            sql_lines.append(line.strip())
            if ";" in line:
                break
    if sql_lines:
        return "\n".join(sql_lines)
    return response.strip()


def _run_sql_test(schema: str, generated_query: str, test_query: str) -> dict:
    """sqlite3로 SQL 쿼리를 실행하고 결과 비교"""
    try:
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()

        # 스키마 및 데이터 삽입
        for statement in schema.split(";"):
            stmt = statement.strip()
            if stmt:
                cursor.execute(stmt)
        conn.commit()

        # 기대 결과 실행
        cursor.execute(test_query)
        expected_rows = cursor.fetchall()
        expected_cols = [desc[0] for desc in cursor.description] if cursor.description else []

        # 생성된 쿼리 실행
        # 세미콜론 제거 후 실행
        gen_q = generated_query.rstrip(";").strip()
        cursor.execute(gen_q)
        generated_rows = cursor.fetchall()
        generated_cols = [desc[0] for desc in cursor.description] if cursor.description else []

        conn.close()

        # 결과 비교: 행 내용이 같은지 (순서 무시 가능하도록 set 비교도 시도)
        rows_match = (expected_rows == generated_rows)
        if not rows_match:
            # 순서 무시 비교
            rows_match = (set(expected_rows) == set(generated_rows) and
                         len(expected_rows) == len(generated_rows))

        return {
            "correct": rows_match,
            "expected_rows": expected_rows,
            "generated_rows": generated_rows,
            "error": None,
        }
    except Exception as e:
        return {
            "correct": False,
            "expected_rows": [],
            "generated_rows": [],
            "error": str(e),
        }


# ── 수학 답 추출 ─────────────────────────────────────────────────────────────

def _extract_numeric_answer(response: str) -> Optional[float]:
    """응답에서 최종 숫자 답을 추출"""
    # "답: 숫자" 또는 "정답: 숫자" 패턴
    patterns = [
        r"(?:답|정답|답은|결과|answer)\s*[:=은는]\s*(-?\d+\.?\d*)",
        r"(?:따라서|그러므로|그래서|결론적으로)\s*.*?(-?\d+\.?\d*)",
        r"\*\*(-?\d+\.?\d*)\*\*",  # **숫자** 볼드 패턴
        r"=\s*(-?\d+\.?\d*)\s*(?:$|[.\s원개명도cm²³])",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue

    # 마지막 수단: 응답에서 마지막 숫자 추출
    all_numbers = re.findall(r"-?\d+\.?\d*", response)
    if all_numbers:
        try:
            return float(all_numbers[-1])
        except ValueError:
            return None
    return None


def _check_math_answer(extracted: Optional[float], expected: float, tolerance: float = 0.01) -> bool:
    """수학 답 비교 (허용 오차 포함)"""
    if extracted is None:
        return False
    return abs(extracted - expected) <= tolerance


# ── 디버깅 평가 ──────────────────────────────────────────────────────────────

def _evaluate_debug(response: str, problem: dict) -> dict:
    """디버깅 응답 평가: 버그 언급 + 수정 코드 실행"""
    bug_desc = problem["bug_description"]
    test_cases = problem["test_cases"]

    # 버그 식별 여부: 핵심 키워드 확인
    bug_keywords = []
    if "off-by-one" in bug_desc or "인덱스" in bug_desc or "range(1" in bug_desc:
        bug_keywords = ["0", "range(0", "인덱스", "첫 번째", "off-by-one", "시작"]
    elif "반대" in bug_desc or "비교 연산자" in bug_desc:
        bug_keywords = [">", "반대", "최대", "크", "비교"]
    elif "초기값" in bug_desc:
        bug_keywords = ["lst[0]", "초기", "음수", "첫 번째"]
    elif "IndexError" in bug_desc:
        bug_keywords = ["len(s) - 1", "len(s)-1", "인덱스", "IndexError", "범위"]
    elif "홀수" in bug_desc or "짝수" in bug_desc:
        bug_keywords = ["0", "== 0", "짝수", "홀수", "반대"]
    elif "len(lst) + 1" in bug_desc:
        bug_keywords = ["+1", "len(lst)", "나누", "제거"]
    elif "무한 루프" in bug_desc:
        bug_keywords = ["-= 1", "-=1", "감소", "무한", "n -= 1"]
    elif "pop" in bug_desc or "크기가 변" in bug_desc:
        bug_keywords = ["컴프리헨션", "새 리스트", "pop", "크기", "인덱스"]
    elif "key" in bug_desc and "값" in bug_desc:
        bug_keywords = ["d[key]", "values", "값", ".values()"]
    elif "append" in bug_desc:
        bug_keywords = ["extend", "+", "append", "원소"]
    else:
        bug_keywords = []

    bug_identified = False
    response_lower = response.lower()
    if bug_keywords:
        bug_identified = any(kw.lower() in response_lower for kw in bug_keywords)
    else:
        # 키워드 없으면 수정 코드 실행 결과로만 판단
        bug_identified = True

    # 수정 코드 추출 및 실행
    fixed_code = _extract_python_code(response)
    if not fixed_code:
        return {"bug_identified": bug_identified, "fix_works": False, "correct": False}

    # 함수 이름 추출
    func_match = re.search(r"def\s+(\w+)\s*\(", fixed_code)
    if not func_match:
        return {"bug_identified": bug_identified, "fix_works": False, "correct": False}
    func_name = func_match.group(1)

    test_harness = _build_test_harness(func_name, test_cases)
    result = _run_python_code(fixed_code, test_harness)

    if result["error"] or result["returncode"] != 0:
        return {"bug_identified": bug_identified, "fix_works": False, "correct": False}

    try:
        test_results = json.loads(result["stdout"])
        all_pass = all(t["pass"] for t in test_results)
    except (json.JSONDecodeError, KeyError):
        all_pass = False

    return {
        "bug_identified": bug_identified,
        "fix_works": all_pass,
        "correct": bug_identified and all_pass,
    }


# ── 메인 평가 함수들 ─────────────────────────────────────────────────────────

def _eval_python(model: str, problems: list[dict]) -> list[dict]:
    """Python 코딩 문제 평가"""
    results = []
    for prob in problems:
        print(f"    [Python] {prob['id']}: {prob['description'][:40]}...")
        prompt = (
            f"다음 Python 함수를 구현하세요.\n\n"
            f"설명: {prob['description']}\n"
            f"함수 시그니처: {prob['function_signature']}\n\n"
            f"함수 구현만 작성하세요. 코드 블록(```python)으로 감싸세요."
        )
        resp = runner.generate(
            model=model,
            prompt=prompt,
            options=dict(config.SAMPLING_PARAMS),
        )
        if resp["error"]:
            results.append({
                "id": prob["id"],
                "passed": 0,
                "total": len(prob["test_cases"]),
                "pass_at_1": 0.0,
                "error": resp["error"],
            })
            continue

        code = _extract_python_code(resp["response"])

        # 함수 이름 추출
        sig_match = re.search(r"def\s+(\w+)\s*\(", prob["function_signature"])
        func_name = sig_match.group(1) if sig_match else "solution"

        test_harness = _build_test_harness(func_name, prob["test_cases"])
        exec_result = _run_python_code(code, test_harness)

        passed = 0
        total = len(prob["test_cases"])
        if exec_result["error"] is None and exec_result["returncode"] == 0:
            try:
                test_results = json.loads(exec_result["stdout"])
                passed = sum(1 for t in test_results if t["pass"])
            except (json.JSONDecodeError, KeyError):
                passed = 0

        results.append({
            "id": prob["id"],
            "passed": passed,
            "total": total,
            "pass_at_1": 1.0 if passed == total else 0.0,
            "response_preview": resp["response"][:200],
        })
        time.sleep(config.COOLDOWN_BETWEEN_TESTS)

    return results


def _eval_sql(model: str, problems: list[dict]) -> list[dict]:
    """SQL 문제 평가"""
    results = []
    for prob in problems:
        print(f"    [SQL] {prob['id']}: {prob['description'][:40]}...")
        prompt = (
            f"다음 SQL 문제를 풀어주세요.\n\n"
            f"테이블 스키마:\n{prob['schema']}\n\n"
            f"문제: {prob['description']}\n\n"
            f"SQL 쿼리만 작성하세요. 코드 블록(```sql)으로 감싸세요."
        )
        resp = runner.generate(
            model=model,
            prompt=prompt,
            options=dict(config.SAMPLING_PARAMS),
        )
        if resp["error"]:
            results.append({
                "id": prob["id"],
                "correct": False,
                "error": resp["error"],
            })
            continue

        generated_sql = _extract_sql(resp["response"])
        sql_result = _run_sql_test(prob["schema"], generated_sql, prob["test_query"])

        results.append({
            "id": prob["id"],
            "correct": sql_result["correct"],
            "error": sql_result["error"],
            "response_preview": resp["response"][:200],
        })
        time.sleep(config.COOLDOWN_BETWEEN_TESTS)

    return results


def _eval_debug(model: str, problems: list[dict]) -> list[dict]:
    """디버깅 문제 평가"""
    results = []
    for prob in problems:
        print(f"    [Debug] {prob['id']}: {prob['description'][:40]}...")
        prompt = (
            f"다음 Python 코드에 버그가 있습니다. 버그를 찾아서 설명하고, 수정된 코드를 제공하세요.\n\n"
            f"문제 설명: {prob['description']}\n\n"
            f"버그가 있는 코드:\n```python\n{prob['buggy_code']}\n```\n\n"
            f"1. 버그가 무엇인지 설명하세요.\n"
            f"2. 수정된 전체 함수를 ```python 코드 블록으로 제공하세요."
        )
        resp = runner.generate(
            model=model,
            prompt=prompt,
            options=dict(config.SAMPLING_PARAMS),
        )
        if resp["error"]:
            results.append({
                "id": prob["id"],
                "bug_identified": False,
                "fix_works": False,
                "correct": False,
                "error": resp["error"],
            })
            continue

        eval_result = _evaluate_debug(resp["response"], prob)
        eval_result["id"] = prob["id"]
        eval_result["response_preview"] = resp["response"][:200]
        results.append(eval_result)
        time.sleep(config.COOLDOWN_BETWEEN_TESTS)

    return results


def _eval_math(model: str, problems: list[dict]) -> list[dict]:
    """수학 문제 평가"""
    results = []
    for prob in problems:
        print(f"    [Math] {prob['id']}: {prob['question'][:40]}...")
        prompt = (
            f"다음 수학 문제를 풀어주세요. 풀이 과정을 보여주고, 최종 답을 '답: [숫자]' 형식으로 작성하세요.\n\n"
            f"문제: {prob['question']}"
        )
        resp = runner.generate(
            model=model,
            prompt=prompt,
            options=dict(config.BENCHMARK_SAMPLING),
        )
        if resp["error"]:
            results.append({
                "id": prob["id"],
                "correct": False,
                "expected": prob["answer"],
                "extracted": None,
                "error": resp["error"],
            })
            continue

        extracted = _extract_numeric_answer(resp["response"])
        correct = _check_math_answer(extracted, prob["answer"])

        results.append({
            "id": prob["id"],
            "correct": correct,
            "expected": prob["answer"],
            "extracted": extracted,
            "response_preview": resp["response"][:200],
        })
        time.sleep(config.COOLDOWN_BETWEEN_TESTS)

    return results


# ── run() ─────────────────────────────────────────────────────────────────────

def run(models: Optional[list[str]] = None) -> dict:
    """
    Track 4 실행: 코드 및 수학 문제 평가

    Args:
        models: 평가할 모델 목록 (None이면 config.ALL_MODELS 사용)

    Returns:
        dict with track results and summary
    """
    if models is None:
        models = config.ALL_MODELS

    # 데이터 로드
    code_data = _load_code_problems()
    python_problems = code_data["python_problems"]
    sql_problems = code_data["sql_problems"]
    debug_problems = code_data["debugging_problems"]
    math_problems = _load_math_problems()

    print(f"\n{'='*60}")
    print(f"Track 4: 코드 및 수학 평가")
    print(f"  Python: {len(python_problems)}문제 | SQL: {len(sql_problems)}문제")
    print(f"  디버깅: {len(debug_problems)}문제 | 수학: {len(math_problems)}문제")
    print(f"  모델 수: {len(models)}")
    print(f"{'='*60}\n")

    # 체크포인트 로드
    checkpoint = runner.load_checkpoint("code_math")
    all_results = checkpoint.get("results", []) if checkpoint else []
    completed_models = {r["model"] for r in all_results}

    summary = {}
    current_model = None

    for model in models:
        if model in completed_models:
            print(f"  ⏭ {model}: 체크포인트에서 건너뜀")
            # summary 복원
            for r in all_results:
                if r["model"] == model:
                    summary[model] = r.get("scores", {})
            continue

        print(f"\n── 모델: {model} ──")
        if not runner.switch_model(model, current_model):
            print(f"  ✗ 모델 로딩 실패: {model}")
            all_results.append({
                "model": model,
                "error": "모델 로딩 실패",
                "scores": {},
            })
            runner.save_checkpoint(
                {"results": all_results}, "code_math"
            )
            continue
        current_model = model

        # Python 평가
        print(f"  ▶ Python 코딩 평가...")
        py_results = _eval_python(model, python_problems)
        py_pass1 = (
            sum(r["pass_at_1"] for r in py_results) / len(py_results)
            if py_results else 0.0
        )

        # SQL 평가
        print(f"  ▶ SQL 쿼리 평가...")
        sql_results = _eval_sql(model, sql_problems)
        sql_acc = (
            sum(1 for r in sql_results if r["correct"]) / len(sql_results)
            if sql_results else 0.0
        )

        # 디버깅 평가
        print(f"  ▶ 디버깅 평가...")
        dbg_results = _eval_debug(model, debug_problems)
        dbg_acc = (
            sum(1 for r in dbg_results if r["correct"]) / len(dbg_results)
            if dbg_results else 0.0
        )

        # 수학 평가
        print(f"  ▶ 수학 문제 평가...")
        math_results = _eval_math(model, math_problems)
        math_acc = (
            sum(1 for r in math_results if r["correct"]) / len(math_results)
            if math_results else 0.0
        )

        scores = {
            "python_pass1": round(py_pass1, 4),
            "sql_accuracy": round(sql_acc, 4),
            "debug_accuracy": round(dbg_acc, 4),
            "math_accuracy": round(math_acc, 4),
        }
        summary[model] = scores

        print(f"  ✓ {model} 결과:")
        print(f"    Python Pass@1: {scores['python_pass1']:.2%}")
        print(f"    SQL 정확도:    {scores['sql_accuracy']:.2%}")
        print(f"    디버깅 정확도: {scores['debug_accuracy']:.2%}")
        print(f"    수학 정확도:   {scores['math_accuracy']:.2%}")

        all_results.append({
            "model": model,
            "scores": scores,
            "python_details": py_results,
            "sql_details": sql_results,
            "debug_details": dbg_results,
            "math_details": math_results,
        })

        # 체크포인트 저장
        runner.save_checkpoint({"results": all_results}, "code_math")

    # 최종 결과
    final = {
        "track": "code_math",
        "results": all_results,
        "summary": summary,
    }

    runner.save_results_incremental(final, "code_math")
    print(f"\n{'='*60}")
    print(f"Track 4 완료!")
    print(f"{'='*60}\n")

    return final
