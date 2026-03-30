"""
Track 3 — 한국어 심화 평가 (Korean Deep Evaluation)

8개 카테고리, 100문항:
  1. 존댓말/반말 전환 (10)
  2. 한국 문화 상식 (20)
  3. 사자성어/관용구 (15)
  4. 맞춤법/문법 (15)
  5. 뉴스 스타일 요약 (10)
  6. 감정/뉘앙스 (10)
  7. 숫자/단위 (10)
  8. 존칭 체계 (10)

채점 방식 (하이브리드):
  - exact  : 정규화 후 문자열 일치
  - contains: 키워드 포함 여부
  - llm_judge: Claude API를 통한 LLM 채점 (1-10)
"""

import json
import time
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from eval_framework import config
from eval_framework import runner
from eval_framework import judge

TRACK_NAME = "track3_korean_deep"
QUESTIONS_PATH = config.DATA_DIR / "korean_deep" / "questions.json"

# ── 카테고리별 LLM Judge 평가 기준 ────────────────────────────────────────────
CATEGORY_CRITERIA = {
    "존댓말/반말 전환": "존댓말/반말 전환이 정확한지, 자연스러운 한국어인지 평가하세요.",
    "한국 문화 상식": "한국 문화에 대한 사실 관계가 정확한지 평가하세요.",
    "사자성어/관용구": "사자성어/관용구의 뜻풀이가 정확하고 이해하기 쉬운지 평가하세요.",
    "맞춤법/문법": "맞춤법/문법 설명이 정확하고 올바른 답을 제시했는지 평가하세요.",
    "뉴스 스타일 요약": "핵심 내용을 빠짐없이 포함하면서도 간결하게 요약했는지 평가하세요.",
    "감정/뉘앙스": "한국어의 미묘한 감정과 뉘앙스를 정확히 파악하고 설명했는지 평가하세요.",
    "숫자/단위": "숫자 변환이나 단위 사용이 정확한지 평가하세요.",
    "존칭 체계": "한국어 존칭/경어법을 올바르게 이해하고 적절한 표현을 사용했는지 평가하세요.",
}


def _normalize(text: str) -> str:
    """채점용 텍스트 정규화 — 공백/구두점 제거, NFD→NFC, 소문자"""
    text = unicodedata.normalize("NFC", text.strip())
    # 구두점·공백 제거
    for ch in " \t\n.,!?;:'\"()[]{}~·…""''":
        text = text.replace(ch, "")
    return text.lower()


def _score_exact(response: str, expected: str) -> float:
    """exact match (정규화 후 비교). 일치하면 1.0, 아니면 0.0"""
    return 1.0 if _normalize(response) == _normalize(expected) else 0.0


def _score_contains(response: str, keywords: list[str]) -> float:
    """키워드 중 하나라도 포함되면 1.0, 아니면 0.0"""
    resp_norm = response.strip()
    for kw in keywords:
        if kw in resp_norm:
            return 1.0
    return 0.0


def _score_llm_judge(
    question: str,
    response: str,
    category: str,
) -> dict:
    """LLM Judge 채점 — 1-10 점수를 0.0-1.0으로 정규화하여 반환"""
    criteria = CATEGORY_CRITERIA.get(category, "")
    result = judge.score_response(
        prompt=question,
        response=response,
        category=category,
        criteria=criteria,
    )
    score_raw = result.get("score", 0)
    return {
        "score": score_raw / 10.0,
        "score_raw": score_raw,
        "reasoning": result.get("reasoning", ""),
        "error": result.get("error"),
    }


def _load_questions() -> list[dict]:
    """questions.json 로드"""
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def run(models: Optional[list[str]] = None) -> dict:
    """
    Track 3 실행

    Args:
        models: 평가할 모델 목록. None이면 config.ALL_MODELS 사용.

    Returns:
        {
            "track": "track3_korean_deep",
            "timestamp": "...",
            "results": [ ... per-model-per-question records ... ],
            "summary": {
                model: {
                    category: {"accuracy": float, "avg_score": float, "n": int},
                    ...
                },
                ...
            }
        }
    """
    if models is None:
        models = list(config.ALL_MODELS)

    questions = _load_questions()
    print(f"\n{'='*60}")
    print(f" Track 3: 한국어 심화 평가 ({len(questions)}문항 × {len(models)}모델)")
    print(f"{'='*60}\n")

    # 체크포인트 복원
    checkpoint = runner.load_checkpoint(TRACK_NAME)
    all_results: list[dict] = checkpoint.get("results", []) if checkpoint else []
    completed_models: set[str] = set()
    if checkpoint:
        completed_models = {r["model"] for r in all_results}
        print(f"  체크포인트 복원: {len(completed_models)}개 모델 완료")

    current_model = None

    for model in models:
        if model in completed_models:
            print(f"  [{model}] 이미 완료 — 건너뜀")
            continue

        # 모델 전환
        if not runner.wait_for_ollama():
            print(f"  [오류] Ollama 서버 무응답 — {model} 건너뜀")
            continue
        if not runner.switch_model(model, current_model):
            print(f"  [오류] 모델 로딩 실패 — {model} 건너뜀")
            continue
        current_model = model

        model_results = []
        pending_llm_judge: list[dict] = []
        print(f"\n  [{model}] 평가 시작 ({len(questions)}문항)")

        # ── Phase 1: 응답 수집 + 규칙 기반 채점 (모델 호출만) ────
        for idx, q in enumerate(questions):
            qid = q["id"]
            category = q["category"]
            question_text = q["question"]
            answer_type = q["answer_type"]
            expected = q.get("expected_answer", "")
            keywords = q.get("keywords", [])

            # 프롬프트 구성
            system_msg = "당신은 한국어에 능통한 AI 어시스턴트입니다. 질문에 정확하고 간결하게 답변하세요."
            gen_result = runner.generate(
                model=model,
                prompt=question_text,
                system=system_msg,
                options=dict(config.BENCHMARK_SAMPLING),
            )

            response_text = gen_result.get("response", "")
            error = gen_result.get("error")

            # 규칙 기반 채점 (exact/contains)은 즉시 처리
            score = 0.0
            judge_detail = None

            if error:
                score = 0.0
            elif answer_type == "exact":
                score = _score_exact(response_text, expected)
            elif answer_type == "contains":
                score = _score_contains(response_text, keywords)
            elif answer_type == "llm_judge":
                # Judge 채점은 Phase 2로 보류
                pending_llm_judge.append({
                    "result_idx": len(model_results),
                    "question_text": question_text,
                    "response_text": response_text,
                    "category": category,
                })

            record = {
                "model": model,
                "id": qid,
                "category": category,
                "question": question_text,
                "expected_answer": expected,
                "answer_type": answer_type,
                "response": response_text,
                "score": score,
                "tokens_per_sec": gen_result.get("tokens_per_sec", 0),
                "wall_time_s": gen_result.get("wall_time_s", 0),
                "error": error,
            }
            model_results.append(record)

            # 진행 표시 (10문항 단위)
            if (idx + 1) % 10 == 0 or idx == len(questions) - 1:
                done = idx + 1
                print(f"    {done}/{len(questions)} 응답 수집 완료")

            time.sleep(config.COOLDOWN_BETWEEN_TESTS)

        # ── Phase 2: 모델 언로드 → LLM Judge 일괄 채점 ───────────
        if pending_llm_judge:
            print(f"  [{model}] VRAM 확보: 모델 언로드 → Judge 채점 ({len(pending_llm_judge)}건)")
            runner.unload_all_models()

            for ji, pj in enumerate(pending_llm_judge):
                print(f"    Judge [{ji+1}/{len(pending_llm_judge)}]", end=" ", flush=True)
                judge_detail = _score_llm_judge(
                    pj["question_text"], pj["response_text"], pj["category"],
                )
                ridx = pj["result_idx"]
                model_results[ridx]["score"] = judge_detail["score"]
                model_results[ridx]["judge_score_raw"] = judge_detail["score_raw"]
                model_results[ridx]["judge_reasoning"] = judge_detail["reasoning"]
                model_results[ridx]["judge_error"] = judge_detail["error"]
                print(f"score={judge_detail['score_raw']}/10")

        # 최종 평균 표시
        if model_results:
            avg = sum(r["score"] for r in model_results) / len(model_results)
            print(f"  [{model}] 전체 평균: {avg:.3f}")

        all_results.extend(model_results)

        # 체크포인트 저장
        runner.save_checkpoint(
            {"results": all_results, "saved_at": datetime.now().isoformat()},
            TRACK_NAME,
        )
        print(f"  [{model}] 체크포인트 저장 완료")

    # ── 요약 생성 ──────────────────────────────────────────────────────────────
    summary = _build_summary(all_results)

    final_output = {
        "track": TRACK_NAME,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
        "summary": summary,
    }

    # 최종 저장
    saved_path = runner.save_results_incremental(final_output, TRACK_NAME)
    print(f"\n  결과 저장: {saved_path}")

    _print_summary(summary)

    return final_output


def _build_summary(results: list[dict]) -> dict:
    """
    모델별 · 카테고리별 요약 통계

    Returns:
        {model: {category: {"accuracy": float, "avg_score": float, "n": int}}}
    """
    # model → category → list of scores
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        buckets[r["model"]][r["category"]].append(r["score"])

    summary = {}
    for model, cats in buckets.items():
        summary[model] = {}
        for cat, scores in cats.items():
            n = len(scores)
            avg = sum(scores) / n if n else 0.0
            # accuracy = fraction with score >= 0.5 (for contains/exact: 1.0, for judge: 5+/10)
            acc = sum(1 for s in scores if s >= 0.5) / n if n else 0.0
            summary[model][cat] = {
                "accuracy": round(acc, 4),
                "avg_score": round(avg, 4),
                "n": n,
            }
        # 전체 통계
        all_scores = [s for cat_scores in cats.values() for s in cat_scores]
        total_n = len(all_scores)
        summary[model]["_overall"] = {
            "accuracy": round(sum(1 for s in all_scores if s >= 0.5) / total_n, 4) if total_n else 0.0,
            "avg_score": round(sum(all_scores) / total_n, 4) if total_n else 0.0,
            "n": total_n,
        }

    return summary


def _print_summary(summary: dict) -> None:
    """요약 테이블 출력"""
    print(f"\n{'='*60}")
    print(" Track 3 결과 요약")
    print(f"{'='*60}")

    for model, cats in summary.items():
        overall = cats.get("_overall", {})
        print(f"\n  {model}")
        print(f"    전체: 정확도={overall.get('accuracy', 0):.1%}  "
              f"평균={overall.get('avg_score', 0):.3f}  "
              f"N={overall.get('n', 0)}")
        for cat, stats in sorted(cats.items()):
            if cat == "_overall":
                continue
            print(f"    {cat:16s}: 정확도={stats['accuracy']:.1%}  "
                  f"평균={stats['avg_score']:.3f}  N={stats['n']}")
    print()
