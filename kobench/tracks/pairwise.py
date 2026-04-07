"""
Track 7: 쌍대비교 (Pairwise Comparison) — LLM-as-Judge로 모든 모델 쌍 비교 후 Bradley-Terry Elo 산출

11개 모델 기준: 11C2 = 55 쌍 x 20 프롬프트 = 1,100 비교
위치 편향 제거를 위해 순서 반전 비교 포함 → 총 2,200 Claude API 호출
"""

import json
import time
import itertools
from typing import Optional

from kobench import config
from kobench import runner
from kobench import judge
from kobench import scoring


def _load_prompts() -> list[dict] | None:
    """data/track7_prompts.json에서 프롬프트 로드 (없으면 None → inline fallback)"""
    path = config.DATA_DIR / "track7_prompts.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None

TRACK_NAME = "pairwise"

# ── 20개 대표 프롬프트 (다양한 능력 영역) ────────────────────────────────────

PROMPTS = [
    # --- 한국 지식 (3) ---
    {
        "id": "kr_history_1",
        "category": "korean_knowledge",
        "prompt": "조선 시대 세종대왕의 한글 창제 과정과 그 역사적 의의를 설명해 주세요.",
    },
    {
        "id": "kr_culture_1",
        "category": "korean_knowledge",
        "prompt": "한국의 김장 문화가 유네스코 무형문화유산에 등재된 이유와 김장의 사회적 의미를 서술하세요.",
    },
    {
        "id": "kr_current_1",
        "category": "korean_knowledge",
        "prompt": "한국의 저출산·고령화 문제의 원인과 이에 대한 정부 정책의 효과를 분석해 주세요.",
    },
    # --- 추론 (3) ---
    {
        "id": "reasoning_logic_1",
        "category": "reasoning",
        "prompt": (
            "A, B, C, D 네 사람이 있습니다. A는 항상 진실만 말하고, B는 항상 거짓만 말합니다. "
            "C는 A가 거짓말쟁이라고 말했고, D는 B가 진실을 말한다고 했습니다. "
            "C와 D는 각각 진실을 말하는 사람인가요, 거짓을 말하는 사람인가요? 논리적으로 풀어 주세요."
        ),
    },
    {
        "id": "reasoning_math_1",
        "category": "reasoning",
        "prompt": (
            "어떤 연못에 수련이 자라고 있습니다. 수련은 매일 면적이 2배로 늘어납니다. "
            "48일째에 연못 전체를 덮었다면, 연못의 절반을 덮은 것은 며칠째인가요? "
            "풀이 과정을 단계별로 설명하세요."
        ),
    },
    {
        "id": "reasoning_causal_1",
        "category": "reasoning",
        "prompt": (
            "도시에서 자전거 전용 도로를 확충하면 교통 체증이 줄어들까요, 늘어날까요? "
            "인과 관계를 고려하여 다양한 시나리오를 분석하세요."
        ),
    },
    # --- 창작 (3) ---
    {
        "id": "creative_poem_1",
        "category": "creative_writing",
        "prompt": "가을 단풍을 주제로 한국어 자유시를 한 편 써 주세요. 시각적 이미지와 감정을 풍부하게 담아 주세요.",
    },
    {
        "id": "creative_essay_1",
        "category": "creative_writing",
        "prompt": "'어린 시절의 골목길'이라는 주제로 서정적인 수필을 작성해 주세요. 감각적 묘사를 포함해 주세요.",
    },
    {
        "id": "creative_story_1",
        "category": "creative_writing",
        "prompt": "2050년 서울을 배경으로, AI와 인간이 공존하는 미래 사회에서 벌어지는 짧은 이야기를 써 주세요.",
    },
    # --- 코딩 (2) ---
    {
        "id": "code_function_1",
        "category": "code",
        "prompt": (
            "Python으로 주어진 문자열에서 가장 긴 팰린드롬 부분 문자열을 찾는 함수를 작성하세요. "
            "시간 복잡도를 설명하고 테스트 케이스도 포함해 주세요."
        ),
    },
    {
        "id": "code_algo_1",
        "category": "code",
        "prompt": (
            "다익스트라(Dijkstra) 알고리즘의 동작 원리를 단계별로 설명하고, "
            "우선순위 큐를 사용하는 이유와 시간 복잡도를 분석해 주세요."
        ),
    },
    # --- 설명 (3) ---
    {
        "id": "explain_science_1",
        "category": "explanation",
        "prompt": "양자 얽힘(quantum entanglement)이란 무엇인지 비전공자도 이해할 수 있도록 비유를 들어 설명해 주세요.",
    },
    {
        "id": "explain_everyday_1",
        "category": "explanation",
        "prompt": "비행기가 하늘을 나는 원리를 중학생이 이해할 수 있는 수준으로 설명해 주세요.",
    },
    {
        "id": "explain_complex_1",
        "category": "explanation",
        "prompt": "블록체인 기술의 작동 원리와 합의 메커니즘(PoW, PoS)의 차이점을 설명해 주세요.",
    },
    # --- 실용 (3) ---
    {
        "id": "practical_advice_1",
        "category": "practical",
        "prompt": "한국에서 1인 가구가 월 200만원으로 생활하기 위한 구체적인 예산 계획을 세워 주세요.",
    },
    {
        "id": "practical_howto_1",
        "category": "practical",
        "prompt": "집에서 천연 발효 식초를 만드는 방법을 재료 준비부터 완성까지 단계별로 알려 주세요.",
    },
    {
        "id": "practical_compare_1",
        "category": "practical",
        "prompt": "Python과 JavaScript를 비교하여 각각 어떤 프로젝트에 더 적합한지 장단점과 함께 설명해 주세요.",
    },
    # --- 한국어 특화 (3) ---
    {
        "id": "korean_honorific_1",
        "category": "korean_language",
        "prompt": (
            "한국어 경어법(존댓말) 체계를 설명하고, '합쇼체', '해요체', '해체'의 "
            "차이를 예문과 함께 보여 주세요. 각각 어떤 상황에서 사용하는지도 설명해 주세요."
        ),
    },
    {
        "id": "korean_grammar_1",
        "category": "korean_language",
        "prompt": (
            "한국어 조사 '-은/는'과 '-이/가'의 차이를 외국인 학습자에게 "
            "명확한 예문과 함께 설명해 주세요."
        ),
    },
    {
        "id": "korean_idiom_1",
        "category": "korean_language",
        "prompt": (
            "'소 잃고 외양간 고친다', '낮말은 새가 듣고 밤말은 쥐가 듣는다', "
            "'될성부른 나무는 떡잎부터 알아본다' — 이 세 가지 한국 속담의 "
            "의미와 유래를 설명하고, 각각의 현대적 활용 예시를 들어 주세요."
        ),
    },
]

# JSON 파일이 있으면 외부 데이터로 교체 (확장성)
PROMPTS = _load_prompts() or PROMPTS

assert len(PROMPTS) == config.TRACK7_NUM_PROMPTS, (
    f"프롬프트 수 불일치: {len(PROMPTS)} != {config.TRACK7_NUM_PROMPTS}"
)


# ── 응답 수집 ────────────────────────────────────────────────────────────────

def _collect_responses(
    models: list[str],
    checkpoint_data: dict,
) -> dict[str, dict[str, str]]:
    """
    모든 모델의 프롬프트별 응답을 수집한다.

    Returns:
        {model: {prompt_id: response_text}}
    """
    responses: dict[str, dict[str, str]] = checkpoint_data.get("responses", {})
    current_model: Optional[str] = None

    for mi, model in enumerate(models):
        if model in responses and len(responses[model]) == len(PROMPTS):
            print(f"  [{mi+1}/{len(models)}] {model} — 이미 수집 완료 (체크포인트)")
            continue

        print(f"  [{mi+1}/{len(models)}] {model} 응답 수집 중...")
        ok = runner.switch_model(model, current_model)
        if not ok:
            print(f"    SKIP — 모델 로딩 실패: {model}")
            continue
        current_model = model

        if model not in responses:
            responses[model] = {}

        for pi, p in enumerate(PROMPTS):
            pid = p["id"]
            if pid in responses[model]:
                continue

            out = runner.generate(model, p["prompt"])
            if out.get("error"):
                print(f"    [{pi+1}/{len(PROMPTS)}] {pid} — 오류: {out['error']}")
                responses[model][pid] = ""
            else:
                responses[model][pid] = out["response"]

            time.sleep(config.COOLDOWN_BETWEEN_TESTS)

        # 모델별 체크포인트 저장
        checkpoint_data["responses"] = responses
        runner.save_checkpoint(checkpoint_data, TRACK_NAME)
        print(f"    {model}: {len(responses[model])}개 응답 저장 완료")

    return responses


# ── 쌍대비교 실행 ─────────────────────────────────────────────────────────────

def _resolve_winner(winner_forward: str, winner_reverse: str) -> str:
    """
    위치 편향 제거: 정방향/역방향 비교 결과를 종합한다.

    정방향: (A=model_i, B=model_j) → winner "A" 또는 "B" 또는 "TIE"
    역방향: (A=model_j, B=model_i) → winner를 뒤집어서 해석

    두 결과가 일치하면 그것을 채택, 불일치하면 TIE.
    """
    # 역방향 결과를 정방향 관점으로 변환
    if winner_reverse == "A":
        reverse_as_forward = "B"  # 역방향에서 A(=model_j) 승 → 정방향에서 B 승
    elif winner_reverse == "B":
        reverse_as_forward = "A"  # 역방향에서 B(=model_i) 승 → 정방향에서 A 승
    else:
        reverse_as_forward = "TIE"

    if winner_forward == reverse_as_forward:
        return winner_forward
    else:
        return "TIE"


def _run_comparisons(
    models: list[str],
    responses: dict[str, dict[str, str]],
    checkpoint_data: dict,
) -> list[dict]:
    """
    모든 모델 쌍 x 프롬프트에 대해 쌍대비교를 수행한다.

    Returns:
        [{"model_a": str, "model_b": str, "prompt_id": str,
          "winner": "A"|"B"|"TIE", "forward": {...}, "reverse": {...}}, ...]
    """
    comparisons: list[dict] = checkpoint_data.get("comparisons", [])
    completed_keys: set[str] = set(checkpoint_data.get("comparison_keys", []))

    pairs = list(itertools.combinations(models, 2))
    total = len(pairs) * len(PROMPTS)
    done_count = len(completed_keys)

    print(f"\n  총 비교 수: {total} ({len(pairs)} 쌍 x {len(PROMPTS)} 프롬프트)")
    print(f"  이미 완료: {done_count}")

    for pair_idx, (model_i, model_j) in enumerate(pairs):
        for prompt_idx, p in enumerate(PROMPTS):
            pid = p["id"]
            comp_key = f"{model_i}|{model_j}|{pid}"

            if comp_key in completed_keys:
                continue

            resp_i = responses.get(model_i, {}).get(pid, "")
            resp_j = responses.get(model_j, {}).get(pid, "")

            # 빈 응답 처리
            if not resp_i and not resp_j:
                result_entry = {
                    "model_a": model_i,
                    "model_b": model_j,
                    "prompt_id": pid,
                    "winner": "TIE",
                    "forward": {"winner": "TIE", "reasoning": "양쪽 모두 응답 없음", "error": None},
                    "reverse": {"winner": "TIE", "reasoning": "양쪽 모두 응답 없음", "error": None},
                }
            elif not resp_i:
                result_entry = {
                    "model_a": model_i,
                    "model_b": model_j,
                    "prompt_id": pid,
                    "winner": "B",
                    "forward": {"winner": "B", "reasoning": "model_a 응답 없음", "error": None},
                    "reverse": {"winner": "A", "reasoning": "model_a 응답 없음", "error": None},
                }
            elif not resp_j:
                result_entry = {
                    "model_a": model_i,
                    "model_b": model_j,
                    "prompt_id": pid,
                    "winner": "A",
                    "forward": {"winner": "A", "reasoning": "model_b 응답 없음", "error": None},
                    "reverse": {"winner": "B", "reasoning": "model_b 응답 없음", "error": None},
                }
            else:
                # 정방향: A=model_i, B=model_j
                forward = judge.score_pairwise(p["prompt"], resp_i, resp_j)
                time.sleep(0.5)

                # 역방향: A=model_j, B=model_i (위치 편향 제거)
                reverse = judge.score_pairwise(p["prompt"], resp_j, resp_i)
                time.sleep(0.5)

                final_winner = _resolve_winner(forward["winner"], reverse["winner"])

                result_entry = {
                    "model_a": model_i,
                    "model_b": model_j,
                    "prompt_id": pid,
                    "winner": final_winner,
                    "forward": forward,
                    "reverse": reverse,
                }

            comparisons.append(result_entry)
            completed_keys.add(comp_key)
            done_count += 1

            # 진행률 출력
            if done_count % 10 == 0 or done_count == total:
                pct = done_count / total * 100
                print(f"  비교 {done_count}/{total} ({pct:.1f}%)")

            # 50건마다 체크포인트
            if done_count % 50 == 0:
                checkpoint_data["comparisons"] = comparisons
                checkpoint_data["comparison_keys"] = list(completed_keys)
                runner.save_checkpoint(checkpoint_data, TRACK_NAME)

    # 최종 체크포인트
    checkpoint_data["comparisons"] = comparisons
    checkpoint_data["comparison_keys"] = list(completed_keys)
    runner.save_checkpoint(checkpoint_data, TRACK_NAME)

    return comparisons


# ── 요약 생성 ────────────────────────────────────────────────────────────────

def _build_summary(
    elo_scores: dict[str, dict],
    comparisons: list[dict],
    models: list[str],
) -> dict[str, dict]:
    """모델별 Elo, 승/패, 순위 요약을 생성한다."""
    # 순위 산출
    ranked = sorted(elo_scores.items(), key=lambda x: x[1]["elo"], reverse=True)

    summary = {}
    for rank, (model, data) in enumerate(ranked, 1):
        summary[model] = {
            "elo": data["elo"],
            "ci_lower": data["ci_lower"],
            "ci_upper": data["ci_upper"],
            "wins": data["wins"],
            "losses": data["losses"],
            "rank": rank,
        }

    return summary


# ── 메인 실행 ────────────────────────────────────────────────────────────────

def run(models: Optional[list[str]] = None) -> dict:
    """
    Track 7 쌍대비교 평가를 실행한다.

    Args:
        models: 평가할 모델 목록. None이면 config.ALL_MODELS 사용.

    Returns:
        {
            "track": "pairwise",
            "results": {
                "responses": {model: {prompt_id: response}},
                "comparisons": [...],
                "elo_scores": {model: {elo, ci_lower, ci_upper, wins, losses}},
            },
            "summary": {model: {elo, ci_lower, ci_upper, wins, losses, rank}},
        }
    """
    if models is None:
        models = list(config.ALL_MODELS)

    n_models = len(models)
    n_pairs = n_models * (n_models - 1) // 2
    n_comparisons = n_pairs * len(PROMPTS)
    n_api_calls = n_comparisons * 2  # 정방향 + 역방향

    print(f"{'='*60}")
    print(f"Track 7: 쌍대비교 (Pairwise Comparison)")
    print(f"  모델 수: {n_models}")
    print(f"  프롬프트 수: {len(PROMPTS)}")
    print(f"  모델 쌍: {n_pairs} ({n_models}C2)")
    print(f"  총 비교: {n_comparisons}")
    print(f"  Claude API 호출 (위치 편향 제거): {n_api_calls}")
    print(f"{'='*60}")

    # 체크포인트 복원
    checkpoint = runner.load_checkpoint(TRACK_NAME)
    if checkpoint is None:
        checkpoint = {}

    if checkpoint:
        n_resp = sum(len(v) for v in checkpoint.get("responses", {}).values())
        n_comp = len(checkpoint.get("comparisons", []))
        if n_resp > 0 or n_comp > 0:
            print(f"  체크포인트 복원: 응답 {n_resp}건, 비교 {n_comp}건")

    # ── 1단계: 응답 수집 ─────────────────────────────────────────────────
    print(f"\n[1/3] 모델 응답 수집 ({n_models}개 모델 x {len(PROMPTS)}개 프롬프트)")

    if not runner.wait_for_ollama():
        return {
            "track": TRACK_NAME,
            "results": {"responses": {}, "comparisons": [], "elo_scores": {}},
            "summary": {},
            "error": "Ollama 서버에 연결할 수 없습니다.",
        }

    responses = _collect_responses(models, checkpoint)

    # 유효한 모델만 비교 대상에 포함 (최소 1개 응답이 있는 모델)
    valid_models = [m for m in models if m in responses and any(responses[m].values())]
    if len(valid_models) < 2:
        return {
            "track": TRACK_NAME,
            "results": {"responses": responses, "comparisons": [], "elo_scores": {}},
            "summary": {},
            "error": f"비교 가능한 모델이 부족합니다 ({len(valid_models)}개).",
        }

    # 이전 모델 언로드 (Judge 단계에서는 Ollama 불필요)
    runner.unload_all_models()

    # ── 2단계: 쌍대비교 ──────────────────────────────────────────────────
    print(f"\n[2/3] 쌍대비교 수행 (LLM-as-Judge)")
    comparisons = _run_comparisons(valid_models, responses, checkpoint)

    # ── 3단계: Bradley-Terry 피팅 ────────────────────────────────────────
    print(f"\n[3/3] Bradley-Terry 모델 피팅 → Elo 점수 산출")
    elo_scores = scoring.fit_bradley_terry(comparisons, valid_models)

    # ── 요약 및 저장 ─────────────────────────────────────────────────────
    summary = _build_summary(elo_scores, comparisons, valid_models)

    final = {
        "track": TRACK_NAME,
        "results": {
            "responses": responses,
            "comparisons": comparisons,
            "elo_scores": elo_scores,
        },
        "summary": summary,
    }

    runner.save_results_incremental(final, TRACK_NAME)

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"Track 7 완료: {len(comparisons)}건 비교")
    print(f"{'─'*60}")
    print(f"{'순위':<4} {'모델':<35} {'Elo':>7} {'95% CI':>16} {'승':>4} {'패':>4}")
    print(f"{'─'*60}")
    for model in sorted(summary, key=lambda m: summary[m]["rank"]):
        s = summary[model]
        ci = f"[{s['ci_lower']:.0f}, {s['ci_upper']:.0f}]"
        print(f"{s['rank']:<4} {model:<35} {s['elo']:>7.1f} {ci:>16} {s['wins']:>4} {s['losses']:>4}")
    print(f"{'='*60}")

    return final
