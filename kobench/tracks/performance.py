"""
Track 6: 성능 프로파일링 — prefill/decode 속도, TTFT, VRAM, 양자화 비교, 동시 요청
"""

import json
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

from kobench import config
from kobench import runner

TRACK_NAME = "performance"

# ── 한국어 필러 텍스트 생성 ──────────────────────────────────────────────────
_FILLER_UNIT = (
    "대한민국은 민주공화국이다. 대한민국의 주권은 국민에게 있고, "
    "모든 권력은 국민으로부터 나온다. 국민의 자유와 권리는 헌법에 "
    "열거되지 아니한 이유로 경시되지 아니한다. 모든 국민은 법 앞에 "
    "평등하다. 누구든지 성별·종교 또는 사회적 신분에 의하여 정치적·"
    "경제적·사회적·문화적 생활의 모든 영역에 있어서 차별을 받지 아니한다. "
)


def _make_filler_prompt(target_tokens: int) -> str:
    """Approximate a prompt of `target_tokens` length in Korean.

    Ollama tokenizers vary, but Korean text averages roughly 1 token per
    1.5-2 characters.  We over-generate and trim to a safe estimate.
    """
    chars_needed = int(target_tokens * 2.0)
    repeats = (chars_needed // len(_FILLER_UNIT)) + 1
    text = (_FILLER_UNIT * repeats)[:chars_needed]
    return text


# ── 양자화 비교용 테스트 프롬프트 ─────────────────────────────────────────────
_QUANT_TEST_PROMPTS = [
    "대한민국의 수도는 어디이며, 그 역사적 배경을 간단히 설명해 주세요.",
    "인공지능이 사회에 미치는 긍정적·부정적 영향을 세 가지씩 서술하세요.",
    "다음 파이썬 함수의 시간 복잡도를 분석하세요: def fib(n): return n if n<2 else fib(n-1)+fib(n-2)",
    "한국 전통 음식 중 발효 식품의 종류와 건강상의 이점을 설명해 주세요.",
    "기후 변화에 대응하기 위한 국제 사회의 노력과 한계를 논하세요.",
]

# ── 양자화 그룹 검출 ──────────────────────────────────────────────────────────

def _get_quant_groups(models: list[str]) -> dict[str, dict[str, str]]:
    """Group models by base name for quantization comparison.

    Returns {base_name: {quant_tag: full_model_name}} where quant_tag is
    one of 'f16', 'Q8_0', 'Q4_K_M'.
    """
    quant_tags = ("f16", "Q8_0", "Q4_K_M")
    groups: dict[str, dict[str, str]] = {}
    for m in models:
        for tag in quant_tags:
            if tag in m:
                base = m.replace(f"-{tag}", "")
                groups.setdefault(base, {})[tag] = m
                break
    # Only keep groups with at least 2 variants
    return {b: v for b, v in groups.items() if len(v) >= 2}


# ── 개별 테스트 ──────────────────────────────────────────────────────────────

def _make_result_entry(
    model: str,
    test_type: str,
    *,
    input_length: int = 0,
    output_length: int = 0,
    tokens_per_sec: float = 0.0,
    prefill_tok_s: float = 0.0,
    ttft_s: float = 0.0,
    vram_used_mb: int = 0,
    wall_time_s: float = 0.0,
    error: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    entry = {
        "model": model,
        "test_type": test_type,
        "input_length": input_length,
        "output_length": output_length,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "prefill_tok_s": round(prefill_tok_s, 2),
        "ttft_s": round(ttft_s, 4),
        "vram_used_mb": vram_used_mb,
        "wall_time_s": round(wall_time_s, 4),
        "error": error,
    }
    if extra:
        entry.update(extra)
    return entry


def _test_prefill_speed(model: str) -> list[dict]:
    """Measure prefill (prompt evaluation) speed at various input lengths."""
    results = []
    for length in config.TRACK6_INPUT_LENGTHS:
        prompt = _make_filler_prompt(length)
        options = {
            "temperature": 0.0,
            "num_predict": 1,  # minimal generation — we care about prompt eval
            "num_ctx": max(length + 128, 4096),
        }
        out = runner.generate(model, prompt, options=options)
        pe_count = out.get("prompt_eval_count", 0)
        pe_dur = out.get("prompt_eval_duration_s", 0)
        prefill_tps = pe_count / pe_dur if pe_dur > 0 else 0

        results.append(_make_result_entry(
            model, "prefill_speed",
            input_length=pe_count or length,
            output_length=out.get("eval_count", 0),
            tokens_per_sec=out.get("tokens_per_sec", 0),
            prefill_tok_s=prefill_tps,
            wall_time_s=out.get("wall_time_s", 0),
            error=out.get("error"),
        ))
        time.sleep(config.COOLDOWN_BETWEEN_TESTS)
    return results


def _test_decode_speed(model: str) -> list[dict]:
    """Measure decode (generation) speed at different output lengths."""
    results = []
    output_lengths = [50, 100, 256, 512]
    prompt = "다음 주제에 대해 자세히 설명하세요: 인공지능의 미래와 인류 사회의 변화"

    for num_predict in output_lengths:
        options = {
            "temperature": 0.7,
            "num_predict": num_predict,
            "num_ctx": 4096,
        }
        out = runner.generate(model, prompt, options=options)
        results.append(_make_result_entry(
            model, "decode_speed",
            input_length=out.get("prompt_eval_count", 0),
            output_length=out.get("eval_count", 0),
            tokens_per_sec=out.get("tokens_per_sec", 0),
            prefill_tok_s=(
                out["prompt_eval_count"] / out["prompt_eval_duration_s"]
                if out.get("prompt_eval_duration_s", 0) > 0 else 0
            ),
            wall_time_s=out.get("wall_time_s", 0),
            error=out.get("error"),
            extra={"requested_output_length": num_predict},
        ))
        time.sleep(config.COOLDOWN_BETWEEN_TESTS)
    return results


def _test_ttft(model: str) -> list[dict]:
    """Measure Time To First Token using streaming API."""
    results = []
    prompts = [
        ("short", "안녕하세요"),
        ("medium", _make_filler_prompt(500)),
        ("long", _make_filler_prompt(1000)),
    ]

    for label, prompt in prompts:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": 20, "temperature": 0.0},
        }
        ttft = 0.0
        error = None
        wall_start = time.time()

        try:
            timeout = config.MODEL_TIMEOUTS.get(model, 120)
            resp = requests.post(
                config.OLLAMA_API_GENERATE,
                json=payload,
                stream=True,
                timeout=timeout,
            )
            first_token_received = False
            for raw_line in resp.iter_lines():
                if raw_line:
                    chunk = json.loads(raw_line)
                    if chunk.get("response") and not first_token_received:
                        ttft = time.time() - wall_start
                        first_token_received = True
                    if chunk.get("done"):
                        break
            if not first_token_received:
                ttft = time.time() - wall_start
                error = "no_token_received"
        except Exception as e:
            ttft = time.time() - wall_start
            error = str(e)

        wall_time = time.time() - wall_start
        results.append(_make_result_entry(
            model, "ttft",
            input_length=len(prompt),
            ttft_s=ttft,
            wall_time_s=wall_time,
            error=error,
            extra={"prompt_label": label},
        ))
        time.sleep(config.COOLDOWN_BETWEEN_TESTS)
    return results


def _test_vram(model: str) -> list[dict]:
    """Measure VRAM usage after model is loaded."""
    vram = runner.get_vram_usage()
    return [_make_result_entry(
        model, "vram_usage",
        vram_used_mb=vram.get("vram_used_mb", 0),
        extra={
            "vram_total_mb": vram.get("vram_total_mb", 0),
            "vram_free_mb": vram.get("vram_free_mb", 0),
            "gpu_util_pct": vram.get("gpu_util_pct", 0),
        },
    )]


def _test_quant_comparison(quant_groups: dict[str, dict[str, str]]) -> list[dict]:
    """Compare generation speed across quantization variants of the same model."""
    results = []
    current_model = None

    for base_name, variants in quant_groups.items():
        print(f"  [quant_comparison] base={base_name}, variants={list(variants.keys())}")
        for tag, model_name in sorted(variants.items()):
            runner.switch_model(model_name, current_model)
            current_model = model_name
            time.sleep(config.COOLDOWN_BETWEEN_TESTS)

            vram = runner.get_vram_usage()
            speeds = []

            for i, prompt in enumerate(_QUANT_TEST_PROMPTS):
                options = {
                    "temperature": 0.0,
                    "num_predict": 128,
                    "num_ctx": 4096,
                }
                out = runner.generate(model_name, prompt, options=options)
                tps = out.get("tokens_per_sec", 0)
                speeds.append(tps)

                results.append(_make_result_entry(
                    model_name, "quant_comparison",
                    input_length=out.get("prompt_eval_count", 0),
                    output_length=out.get("eval_count", 0),
                    tokens_per_sec=tps,
                    prefill_tok_s=(
                        out["prompt_eval_count"] / out["prompt_eval_duration_s"]
                        if out.get("prompt_eval_duration_s", 0) > 0 else 0
                    ),
                    vram_used_mb=vram.get("vram_used_mb", 0),
                    wall_time_s=out.get("wall_time_s", 0),
                    error=out.get("error"),
                    extra={
                        "base_model": base_name,
                        "quant_tag": tag,
                        "prompt_index": i,
                    },
                ))
                time.sleep(config.COOLDOWN_BETWEEN_TESTS)

            avg_tps = statistics.mean(speeds) if speeds else 0
            print(f"    {tag}: avg {avg_tps:.1f} tok/s, VRAM {vram.get('vram_used_mb', 0)} MB")

    return results, current_model


def _test_max_context(model: str) -> list[dict]:
    """Test with progressively longer contexts up to 4096 tokens."""
    results = []
    context_lengths = [512, 1024, 2048, 3072, 4096]

    for ctx_len in context_lengths:
        prompt = _make_filler_prompt(ctx_len)
        options = {
            "temperature": 0.0,
            "num_predict": 32,
            "num_ctx": ctx_len + 128,
        }
        out = runner.generate(model, prompt, options=options)
        pe_count = out.get("prompt_eval_count", 0)
        pe_dur = out.get("prompt_eval_duration_s", 0)

        results.append(_make_result_entry(
            model, "max_context",
            input_length=pe_count or ctx_len,
            output_length=out.get("eval_count", 0),
            tokens_per_sec=out.get("tokens_per_sec", 0),
            prefill_tok_s=pe_count / pe_dur if pe_dur > 0 else 0,
            wall_time_s=out.get("wall_time_s", 0),
            error=out.get("error"),
            extra={"requested_context": ctx_len},
        ))
        time.sleep(config.COOLDOWN_BETWEEN_TESTS)
    return results


def _test_concurrent(model: str) -> list[dict]:
    """Test concurrent request handling with ThreadPoolExecutor."""
    results = []
    prompt = "대한민국의 경제 발전 과정을 간략히 설명하세요."

    def _single_request(idx: int) -> dict:
        options = {
            "temperature": 0.7,
            "num_predict": 64,
            "num_ctx": 4096,
        }
        start = time.time()
        out = runner.generate(model, prompt, options=options)
        wall = time.time() - start
        return {
            "request_idx": idx,
            "tokens_per_sec": out.get("tokens_per_sec", 0),
            "eval_count": out.get("eval_count", 0),
            "wall_time_s": wall,
            "error": out.get("error"),
        }

    for concurrency in config.TRACK6_CONCURRENT_LEVELS:
        print(f"    concurrency={concurrency}")
        batch_start = time.time()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_single_request, i): i
                for i in range(concurrency)
            }
            request_results = []
            for future in as_completed(futures):
                request_results.append(future.result())

        batch_wall = time.time() - batch_start
        speeds = [r["tokens_per_sec"] for r in request_results if r["tokens_per_sec"] > 0]
        avg_tps = statistics.mean(speeds) if speeds else 0
        total_tokens = sum(r["eval_count"] for r in request_results)
        aggregate_tps = total_tokens / batch_wall if batch_wall > 0 else 0

        results.append(_make_result_entry(
            model, "concurrent",
            output_length=total_tokens,
            tokens_per_sec=avg_tps,
            wall_time_s=batch_wall,
            error=None,
            extra={
                "concurrency_level": concurrency,
                "aggregate_tok_s": round(aggregate_tps, 2),
                "per_request": request_results,
            },
        ))
        time.sleep(config.COOLDOWN_BETWEEN_TESTS)

    return results


# ── 요약 생성 ────────────────────────────────────────────────────────────────

def _build_summary(results: list[dict]) -> dict:
    """Aggregate results into a per-model summary."""
    by_model: dict[str, dict] = {}

    for r in results:
        model = r["model"]
        if model not in by_model:
            by_model[model] = {
                "prefill_tok_s": [],
                "decode_tok_s": [],
                "ttft_s": [],
                "vram_used_mb": 0,
                "max_context_reached": 0,
                "concurrent_aggregate_tok_s": {},
            }
        entry = by_model[model]

        if r["test_type"] == "prefill_speed" and r["prefill_tok_s"] > 0:
            entry["prefill_tok_s"].append(r["prefill_tok_s"])
        elif r["test_type"] == "decode_speed" and r["tokens_per_sec"] > 0:
            entry["decode_tok_s"].append(r["tokens_per_sec"])
        elif r["test_type"] == "ttft" and r["ttft_s"] > 0:
            entry["ttft_s"].append(r["ttft_s"])
        elif r["test_type"] == "vram_usage":
            entry["vram_used_mb"] = r["vram_used_mb"]
        elif r["test_type"] == "max_context" and not r.get("error"):
            entry["max_context_reached"] = max(
                entry["max_context_reached"], r["input_length"]
            )
        elif r["test_type"] == "concurrent":
            level = r.get("concurrency_level") or (r.get("extra") or {}).get("concurrency_level", 0)
            agg = r.get("aggregate_tok_s") or (r.get("extra") or {}).get("aggregate_tok_s", 0)
            entry["concurrent_aggregate_tok_s"][str(level)] = agg

    summary = {}
    for model, data in by_model.items():
        summary[model] = {
            "avg_prefill_tok_s": round(statistics.mean(data["prefill_tok_s"]), 2) if data["prefill_tok_s"] else 0,
            "avg_decode_tok_s": round(statistics.mean(data["decode_tok_s"]), 2) if data["decode_tok_s"] else 0,
            "avg_ttft_s": round(statistics.mean(data["ttft_s"]), 4) if data["ttft_s"] else 0,
            "vram_used_mb": data["vram_used_mb"],
            "max_context_reached": data["max_context_reached"],
            "concurrent_aggregate_tok_s": data["concurrent_aggregate_tok_s"],
        }
    return summary


# ── 메인 실행 ────────────────────────────────────────────────────────────────

def run(models: Optional[list[str]] = None) -> dict:
    """Execute all Track 6 performance profiling tests.

    Args:
        models: List of Ollama model names to evaluate.
                 Defaults to config.ALL_MODELS.

    Returns:
        {"track": "performance", "results": [...], "summary": {...}}
    """
    if models is None:
        models = list(config.ALL_MODELS)

    print(f"{'='*60}")
    print(f"Track 6: 성능 프로파일링 — {len(models)}개 모델")
    print(f"{'='*60}")

    # Try to resume from checkpoint
    checkpoint = runner.load_checkpoint(TRACK_NAME)
    all_results: list[dict] = []
    completed_keys: set[str] = set()

    if checkpoint:
        all_results = checkpoint.get("results", [])
        completed_keys = set(checkpoint.get("completed_keys", []))
        print(f"  체크포인트 복원: {len(all_results)}건, "
              f"완료 키 {len(completed_keys)}개")

    if not runner.wait_for_ollama():
        return {
            "track": TRACK_NAME,
            "results": all_results,
            "summary": {},
            "error": "Ollama 서버에 연결할 수 없습니다.",
        }

    current_model: Optional[str] = None

    # ── Tests 1-4, 6-7: per-model tests ─────────────────────────────────
    for mi, model in enumerate(models):
        print(f"\n[{mi+1}/{len(models)}] {model}")

        # Switch model once per model
        model_key = f"model_loaded:{model}"
        if model_key not in completed_keys:
            ok = runner.switch_model(model, current_model)
            if not ok:
                print(f"  SKIP — 모델 로딩 실패: {model}")
                all_results.append(_make_result_entry(
                    model, "model_load_failed", error="warmup_failed",
                ))
                continue
            current_model = model

        # 1. Prefill speed
        key = f"prefill:{model}"
        if key not in completed_keys:
            print(f"  [1/6] Prefill speed...")
            res = _test_prefill_speed(model)
            all_results.extend(res)
            completed_keys.add(key)
            runner.save_checkpoint({
                "results": all_results,
                "completed_keys": list(completed_keys),
            }, TRACK_NAME)

        # 2. Decode speed
        key = f"decode:{model}"
        if key not in completed_keys:
            print(f"  [2/6] Decode speed...")
            res = _test_decode_speed(model)
            all_results.extend(res)
            completed_keys.add(key)
            runner.save_checkpoint({
                "results": all_results,
                "completed_keys": list(completed_keys),
            }, TRACK_NAME)

        # 3. TTFT
        key = f"ttft:{model}"
        if key not in completed_keys:
            print(f"  [3/6] TTFT (Time To First Token)...")
            res = _test_ttft(model)
            all_results.extend(res)
            completed_keys.add(key)
            runner.save_checkpoint({
                "results": all_results,
                "completed_keys": list(completed_keys),
            }, TRACK_NAME)

        # 4. VRAM
        key = f"vram:{model}"
        if key not in completed_keys:
            print(f"  [4/6] VRAM usage...")
            res = _test_vram(model)
            all_results.extend(res)
            completed_keys.add(key)
            runner.save_checkpoint({
                "results": all_results,
                "completed_keys": list(completed_keys),
            }, TRACK_NAME)

        # 6. Max context
        key = f"max_context:{model}"
        if key not in completed_keys:
            print(f"  [5/6] Max context test...")
            res = _test_max_context(model)
            all_results.extend(res)
            completed_keys.add(key)
            runner.save_checkpoint({
                "results": all_results,
                "completed_keys": list(completed_keys),
            }, TRACK_NAME)

        # 7. Concurrent requests
        key = f"concurrent:{model}"
        if key not in completed_keys:
            print(f"  [6/6] Concurrent requests...")
            res = _test_concurrent(model)
            all_results.extend(res)
            completed_keys.add(key)
            runner.save_checkpoint({
                "results": all_results,
                "completed_keys": list(completed_keys),
            }, TRACK_NAME)

        completed_keys.add(model_key)

    # ── Test 5: quantization comparison (cross-model) ────────────────────
    quant_key = "quant_comparison"
    if quant_key not in completed_keys:
        quant_groups = _get_quant_groups(models)
        if quant_groups:
            print(f"\n[양자화 비교] {len(quant_groups)}개 그룹")
            quant_results, current_model = _test_quant_comparison(quant_groups)
            all_results.extend(quant_results)
            completed_keys.add(quant_key)
            runner.save_checkpoint({
                "results": all_results,
                "completed_keys": list(completed_keys),
            }, TRACK_NAME)

    # ── Build summary and save ───────────────────────────────────────────
    summary = _build_summary(all_results)

    final = {
        "track": TRACK_NAME,
        "results": all_results,
        "summary": summary,
    }

    runner.save_results_incremental(final, TRACK_NAME)
    print(f"\n{'='*60}")
    print(f"Track 6 완료: {len(all_results)}건 결과")
    for model_name, s in summary.items():
        print(f"  {model_name}: prefill={s['avg_prefill_tok_s']} tok/s, "
              f"decode={s['avg_decode_tok_s']} tok/s, "
              f"ttft={s['avg_ttft_s']}s, vram={s['vram_used_mb']}MB")
    print(f"{'='*60}")

    return final
