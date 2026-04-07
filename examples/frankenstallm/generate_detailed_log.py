#!/home/lanco/ai-env/bin/python3
"""
Generate a detailed input/output log in Markdown format from evaluation results.

Reads all track result JSON files from results/ and produces a comprehensive
Markdown file showing every input prompt, every model's output, scores, and
timing information for each test case across all 7 tracks.

Output: reports/DETAILED_IO_LOG.md
"""

import json
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_FILE = REPORTS_DIR / "DETAILED_IO_LOG.md"

# Track name mappings for display
TRACK_NAMES = {
    "track1": "Track 1: 한국어 표준 벤치마크 (KoBEST + KMMLU)",
    "track2": "Track 2: Ko-Bench 멀티턴 평가",
    "track3": "Track 3: 한국어 심화 평가 (Korean Deep)",
    "track4": "Track 4: 코드 및 수학 문제 평가",
    "track5": "Track 5: 일관성 & 강건성",
    "track6": "Track 6: 성능 프로파일링",
    "track7": "Track 7: 쌍대비교 (Pairwise Elo)",
}


# ── Utility ────────────────────────────────────────────────────────────────────

def escape_md(text: str) -> str:
    """Minimally escape text for Markdown blockquotes (preserve readability)."""
    if not text:
        return "(empty)"
    return text


def truncate(text: str, max_len: int = 3000) -> str:
    """Truncate very long text with a notice."""
    if not text:
        return "(empty)"
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n\n... (truncated, {len(text)} chars total)"


def blockquote(text: str) -> str:
    """Format text as a Markdown blockquote."""
    if not text:
        return "> (empty)\n"
    lines = text.split("\n")
    return "\n".join(f"> {line}" for line in lines) + "\n"


def code_block(text: str, lang: str = "") -> str:
    """Format text as a fenced code block."""
    if not text:
        return "```\n(empty)\n```\n"
    return f"```{lang}\n{text}\n```\n"


def find_result_files(results_dir: Path) -> dict:
    """
    Find all track result JSON files, searching both top-level and archive dirs.

    Returns: {track_key: filepath} where track_key is like 'track1', 'track2', etc.
    """
    found = {}

    # Search patterns: top-level and archived directories
    search_dirs = [results_dir]
    for entry in results_dir.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            search_dirs.append(entry)

    for search_dir in search_dirs:
        for track_num in range(1, 8):
            if f"track{track_num}" in found:
                continue

            # Try multiple naming patterns
            patterns = [
                f"track{track_num}_*_20*.json",
                f"track{track_num}_20*.json",
            ]
            # Exclude checkpoint files
            for pattern in patterns:
                matches = sorted(
                    search_dir.glob(pattern),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for m in matches:
                    if "checkpoint" not in m.name:
                        found[f"track{track_num}"] = m
                        break
                if f"track{track_num}" in found:
                    break

            # Also check for full-name files like track1_korean_bench_20260311_024226.json
            if f"track{track_num}" not in found:
                track_name_map = {
                    1: "track1_korean_bench",
                    2: "track2_ko_bench",
                    3: "track3_korean_deep",
                    4: "track4_code_math",
                    5: "track5_consistency",
                    6: "track6_performance",
                    7: "track7_pairwise",
                }
                full_name = track_name_map.get(track_num, "")
                if full_name:
                    matches = sorted(
                        search_dir.glob(f"{full_name}_20*.json"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    for m in matches:
                        if "checkpoint" not in m.name:
                            found[f"track{track_num}"] = m
                            break

    return found


# ── Track Formatters ───────────────────────────────────────────────────────────

def format_track1(data: dict, out: list):
    """
    Track 1: 한국어 표준 벤치마크
    Structure: results = [{model, scores, details: [{id, benchmark, subject,
               expected, parsed, correct, raw_response, error}], mode}]
    """
    results = data.get("results", [])
    if not results:
        out.append("*No results found.*\n\n")
        return

    for model_result in results:
        model = model_result.get("model", "unknown")
        scores = model_result.get("scores", {})
        details = model_result.get("details", [])
        mode = model_result.get("mode", "unknown")
        error = model_result.get("error")

        out.append(f"## Model: {model}\n\n")
        out.append(f"**Mode:** {mode}\n\n")

        if error:
            out.append(f"**Error:** {error}\n\n")

        if scores:
            out.append("**Benchmark Scores:**\n\n")
            out.append("| Benchmark | Accuracy |\n")
            out.append("|-----------|----------|\n")
            for bench, acc in sorted(scores.items()):
                out.append(f"| {bench} | {acc:.4f} |\n")
            out.append("\n")

        if details:
            # Group by benchmark
            by_bench = {}
            for d in details:
                bench = d.get("benchmark", "unknown")
                by_bench.setdefault(bench, []).append(d)

            for bench, items in sorted(by_bench.items()):
                out.append(f"### Benchmark: {bench}\n\n")
                for item in items:
                    qid = item.get("id", "?")
                    subject = item.get("subject", "")
                    expected = item.get("expected", "")
                    parsed = item.get("parsed", None)
                    correct = item.get("correct", False)
                    raw_resp = item.get("raw_response", "")
                    err = item.get("error")

                    subject_str = f" ({subject})" if subject else ""
                    status = "CORRECT" if correct else "WRONG"
                    status_icon = status

                    out.append(f"#### {qid}{subject_str} — {status}\n\n")
                    out.append(f"**Expected:** `{expected}` | **Parsed:** `{parsed}`\n\n")

                    if err:
                        out.append(f"**Error:** {err}\n\n")

                    out.append("**Model Response:**\n\n")
                    out.append(code_block(truncate(raw_resp)))
                    out.append("\n---\n\n")

        out.append("\n---\n\n")


def format_track2(data: dict, out: list):
    """
    Track 2: Ko-Bench multi-turn
    Structure: results = [{model, category, question_idx, turn1_question,
               turn2_question, turn1_answer, turn2_answer, turn1_scores,
               turn2_scores, turn1_mean, turn2_mean, turn1_perf, turn2_perf, error}]
    """
    results = data.get("results", [])
    if not results:
        out.append("*No results found.*\n\n")
        return

    # Group by model, then category
    by_model = {}
    for r in results:
        model = r.get("model", "unknown")
        by_model.setdefault(model, []).append(r)

    for model, entries in sorted(by_model.items()):
        out.append(f"## Model: {model}\n\n")

        by_category = {}
        for e in entries:
            cat = e.get("category", "unknown")
            by_category.setdefault(cat, []).append(e)

        for category, cat_entries in sorted(by_category.items()):
            out.append(f"### Category: {category}\n\n")

            for entry in sorted(cat_entries, key=lambda x: x.get("question_idx", 0)):
                q_idx = entry.get("question_idx", "?")
                t1_q = entry.get("turn1_question", "")
                t2_q = entry.get("turn2_question", "")
                t1_a = entry.get("turn1_answer", "")
                t2_a = entry.get("turn2_answer", "")
                t1_scores = entry.get("turn1_scores", {})
                t2_scores = entry.get("turn2_scores", {})
                t1_mean = entry.get("turn1_mean", 0)
                t2_mean = entry.get("turn2_mean", 0)
                t1_perf = entry.get("turn1_perf", {})
                t2_perf = entry.get("turn2_perf", {})
                error = entry.get("error")

                out.append(f"#### Question {q_idx}\n\n")

                if error:
                    out.append(f"**Error:** {error}\n\n")

                # Turn 1
                out.append("**Turn 1 Input:**\n\n")
                out.append(blockquote(t1_q))
                out.append("\n**Turn 1 Output:**\n\n")
                out.append(code_block(truncate(str(t1_a))))

                # Turn 1 metrics
                t1_scores_dict = t1_scores if isinstance(t1_scores, dict) else {}
                scores_part = t1_scores_dict.get("scores", t1_scores_dict)
                reasoning = t1_scores_dict.get("reasoning", "")
                t1_perf_dict = t1_perf if isinstance(t1_perf, dict) else {}
                tps = t1_perf_dict.get("tokens_per_sec", 0)
                wall = t1_perf_dict.get("wall_time_s", 0)
                eval_count = t1_perf_dict.get("eval_count", 0)

                out.append(f"**Turn 1 Metrics:** Mean Score: {t1_mean}/10")
                if scores_part and isinstance(scores_part, dict):
                    scores_str = ", ".join(f"{k}: {v}" for k, v in scores_part.items())
                    out.append(f" | Sub-scores: {scores_str}")
                if tps:
                    out.append(f" | TPS: {tps:.1f}")
                if wall:
                    out.append(f" | Wall: {wall:.2f}s")
                if eval_count:
                    out.append(f" | Tokens: {eval_count}")
                out.append("\n\n")

                if reasoning:
                    out.append(f"**Judge Reasoning (T1):** {truncate(str(reasoning), 500)}\n\n")

                # Turn 2
                out.append("**Turn 2 Input:**\n\n")
                out.append(blockquote(t2_q))
                out.append("\n**Turn 2 Output:**\n\n")
                out.append(code_block(truncate(str(t2_a))))

                # Turn 2 metrics
                t2_scores_dict = t2_scores if isinstance(t2_scores, dict) else {}
                scores_part2 = t2_scores_dict.get("scores", t2_scores_dict)
                reasoning2 = t2_scores_dict.get("reasoning", "")
                t2_perf_dict = t2_perf if isinstance(t2_perf, dict) else {}
                tps2 = t2_perf_dict.get("tokens_per_sec", 0)
                wall2 = t2_perf_dict.get("wall_time_s", 0)
                eval_count2 = t2_perf_dict.get("eval_count", 0)

                out.append(f"**Turn 2 Metrics:** Mean Score: {t2_mean}/10")
                if scores_part2 and isinstance(scores_part2, dict):
                    scores_str2 = ", ".join(f"{k}: {v}" for k, v in scores_part2.items())
                    out.append(f" | Sub-scores: {scores_str2}")
                if tps2:
                    out.append(f" | TPS: {tps2:.1f}")
                if wall2:
                    out.append(f" | Wall: {wall2:.2f}s")
                if eval_count2:
                    out.append(f" | Tokens: {eval_count2}")
                out.append("\n\n")

                if reasoning2:
                    out.append(f"**Judge Reasoning (T2):** {truncate(str(reasoning2), 500)}\n\n")

                out.append("---\n\n")

        out.append("\n---\n\n")


def format_track3(data: dict, out: list):
    """
    Track 3: 한국어 심화 평가
    Structure: results = [{model, id, category, question, expected_answer,
               answer_type, response, score, tokens_per_sec, wall_time_s,
               error, judge_score_raw, judge_reasoning, judge_error}]
    """
    results = data.get("results", [])
    if not results:
        out.append("*No results found.*\n\n")
        return

    by_model = {}
    for r in results:
        model = r.get("model", "unknown")
        by_model.setdefault(model, []).append(r)

    for model, entries in sorted(by_model.items()):
        out.append(f"## Model: {model}\n\n")

        by_category = {}
        for e in entries:
            cat = e.get("category", "unknown")
            by_category.setdefault(cat, []).append(e)

        for category, cat_entries in sorted(by_category.items()):
            out.append(f"### Category: {category}\n\n")

            for entry in cat_entries:
                qid = entry.get("id", "?")
                question = entry.get("question", "")
                expected = entry.get("expected_answer", "")
                answer_type = entry.get("answer_type", "")
                response = entry.get("response", "")
                score = entry.get("score", 0)
                tps = entry.get("tokens_per_sec", 0)
                wall = entry.get("wall_time_s", 0)
                error = entry.get("error")
                judge_raw = entry.get("judge_score_raw")
                judge_reasoning = entry.get("judge_reasoning", "")
                judge_error = entry.get("judge_error")

                out.append(f"#### {qid} (type: {answer_type})\n\n")

                out.append("**Input:**\n\n")
                out.append(blockquote(question))

                if expected:
                    out.append(f"\n**Expected Answer:** {expected}\n\n")

                out.append("**Output:**\n\n")
                out.append(code_block(truncate(response)))

                metrics_parts = [f"Score: {score:.2f}"]
                if judge_raw is not None:
                    metrics_parts.append(f"Judge Raw: {judge_raw}/10")
                if tps:
                    metrics_parts.append(f"TPS: {tps:.1f}")
                if wall:
                    metrics_parts.append(f"Wall: {wall:.2f}s")
                out.append(f"**Metrics:** {' | '.join(metrics_parts)}\n\n")

                if error:
                    out.append(f"**Error:** {error}\n\n")
                if judge_error:
                    out.append(f"**Judge Error:** {judge_error}\n\n")
                if judge_reasoning:
                    out.append(f"**Judge Reasoning:** {truncate(str(judge_reasoning), 500)}\n\n")

                out.append("---\n\n")

        out.append("\n---\n\n")


def format_track4(data: dict, out: list):
    """
    Track 4: 코드 및 수학 문제 평가
    Structure: results = [{model, scores, python_details, sql_details,
               debug_details, math_details}]
    Each detail type has its own format.
    """
    results = data.get("results", [])
    if not results:
        out.append("*No results found.*\n\n")
        return

    for model_result in results:
        model = model_result.get("model", "unknown")
        scores = model_result.get("scores", {})
        error = model_result.get("error")

        out.append(f"## Model: {model}\n\n")

        if error:
            out.append(f"**Error:** {error}\n\n")
            continue

        if scores:
            out.append("**Overall Scores:**\n\n")
            out.append("| Metric | Score |\n")
            out.append("|--------|-------|\n")
            for metric, val in sorted(scores.items()):
                if isinstance(val, float):
                    out.append(f"| {metric} | {val:.2%} |\n")
                else:
                    out.append(f"| {metric} | {val} |\n")
            out.append("\n")

        # Python details
        py_details = model_result.get("python_details", [])
        if py_details:
            out.append("### Python Coding Problems\n\n")
            for d in py_details:
                pid = d.get("id", "?")
                passed = d.get("passed", 0)
                total = d.get("total", 0)
                pass1 = d.get("pass_at_1", 0)
                preview = d.get("response_preview", "")
                err = d.get("error")

                status = "PASS" if pass1 == 1.0 else "FAIL"
                out.append(f"#### {pid} — {status} ({passed}/{total} test cases)\n\n")

                if err:
                    out.append(f"**Error:** {err}\n\n")

                out.append("**Model Response (preview):**\n\n")
                out.append(code_block(truncate(preview), "python"))
                out.append("---\n\n")

        # SQL details
        sql_details = model_result.get("sql_details", [])
        if sql_details:
            out.append("### SQL Query Problems\n\n")
            for d in sql_details:
                sid = d.get("id", "?")
                correct = d.get("correct", False)
                preview = d.get("response_preview", "")
                err = d.get("error")

                status = "CORRECT" if correct else "WRONG"
                out.append(f"#### {sid} — {status}\n\n")

                if err:
                    out.append(f"**Error:** {err}\n\n")

                out.append("**Model Response (preview):**\n\n")
                out.append(code_block(truncate(preview), "sql"))
                out.append("---\n\n")

        # Debug details
        dbg_details = model_result.get("debug_details", [])
        if dbg_details:
            out.append("### Debugging Problems\n\n")
            for d in dbg_details:
                did = d.get("id", "?")
                correct = d.get("correct", False)
                bug_id = d.get("bug_identified", False)
                fix_works = d.get("fix_works", False)
                preview = d.get("response_preview", "")

                status = "CORRECT" if correct else "WRONG"
                out.append(f"#### {did} — {status}\n\n")
                out.append(f"**Bug Identified:** {bug_id} | **Fix Works:** {fix_works}\n\n")
                out.append("**Model Response (preview):**\n\n")
                out.append(code_block(truncate(preview), "python"))
                out.append("---\n\n")

        # Math details
        math_details = model_result.get("math_details", [])
        if math_details:
            out.append("### Math Problems\n\n")
            for d in math_details:
                mid = d.get("id", "?")
                correct = d.get("correct", False)
                expected = d.get("expected", "")
                extracted = d.get("extracted", "")
                preview = d.get("response_preview", "")

                status = "CORRECT" if correct else "WRONG"
                out.append(f"#### {mid} — {status}\n\n")
                out.append(f"**Expected:** `{expected}` | **Extracted:** `{extracted}`\n\n")
                out.append("**Model Response (preview):**\n\n")
                out.append(code_block(truncate(preview)))
                out.append("---\n\n")

        out.append("\n---\n\n")


def format_track5(data: dict, out: list):
    """
    Track 5: 일관성 & 강건성
    Structure: results = [{model, test_type, ...}]
    test_types: repetition_consistency, paraphrase_robustness,
                length_sensitivity, language_mixing,
                instruction_following, hallucination_detection
    """
    results = data.get("results", [])
    if not results:
        out.append("*No results found.*\n\n")
        return

    by_model = {}
    for r in results:
        model = r.get("model", "unknown")
        by_model.setdefault(model, []).append(r)

    for model, entries in sorted(by_model.items()):
        out.append(f"## Model: {model}\n\n")

        by_type = {}
        for e in entries:
            tt = e.get("test_type", "unknown")
            by_type.setdefault(tt, []).append(e)

        for test_type, type_entries in sorted(by_type.items()):
            out.append(f"### Test Type: {test_type}\n\n")

            for i, entry in enumerate(type_entries):
                out.append(f"#### Test Case {i + 1}\n\n")

                # Show prompt if available
                prompt = entry.get("prompt", "")
                if prompt:
                    out.append("**Input:**\n\n")
                    out.append(blockquote(truncate(str(prompt), 1000)))
                    out.append("\n")

                if test_type == "repetition_consistency":
                    num_trials = entry.get("num_trials", 0)
                    avg_edit = entry.get("avg_edit_distance_ratio", 0)
                    avg_jaccard = entry.get("avg_jaccard_similarity", 0)
                    resp_lengths = entry.get("response_lengths", [])

                    out.append(
                        f"**Metrics:** Trials: {num_trials} | "
                        f"Avg Edit Distance Ratio: {avg_edit:.4f} | "
                        f"Avg Jaccard Similarity: {avg_jaccard:.4f}\n\n"
                    )
                    if resp_lengths:
                        out.append(f"**Response Lengths:** {resp_lengths}\n\n")

                elif test_type == "paraphrase_robustness":
                    keywords = entry.get("answer_keywords", [])
                    hit_rate = entry.get("keyword_hit_rate", 0)
                    consistent = entry.get("all_consistent", False)
                    hits = entry.get("keyword_hits", [])

                    out.append(
                        f"**Metrics:** Keywords: {keywords} | "
                        f"Hit Rate: {hit_rate:.4f} | "
                        f"All Consistent: {consistent}\n\n"
                    )
                    if hits:
                        out.append(f"**Per-variant Hits:** {hits}\n\n")

                elif test_type == "length_sensitivity":
                    keywords = entry.get("answer_keywords", [])
                    length_results = entry.get("length_results", {})
                    all_correct = entry.get("all_correct", False)
                    consistency = entry.get("consistent_across_lengths", False)

                    out.append(
                        f"**Metrics:** Keywords: {keywords} | "
                        f"All Correct: {all_correct} | "
                        f"Consistent Across Lengths: {consistency}\n\n"
                    )
                    if length_results:
                        out.append("**Per-Length Results:**\n\n")
                        for label, lr in length_results.items():
                            out.append(f"- {label}: correct={lr.get('correct')}, len={lr.get('response_length')}\n")
                        out.append("\n")

                elif test_type == "language_mixing":
                    korean_ratio = entry.get("korean_ratio", 0)
                    resp_len = entry.get("response_length", 0)

                    out.append(
                        f"**Metrics:** Korean Ratio: {korean_ratio:.4f} | "
                        f"Response Length: {resp_len}\n\n"
                    )

                elif test_type == "instruction_following":
                    itype = entry.get("instruction_type", "")
                    constraint = entry.get("constraint_value", "")
                    compliant = entry.get("compliant", False)
                    detail = entry.get("detail", "")
                    resp_len = entry.get("response_length", 0)

                    status = "PASS" if compliant else "FAIL"
                    out.append(
                        f"**Metrics:** {status} | "
                        f"Type: {itype} | Constraint: {constraint} | "
                        f"Response Length: {resp_len}\n\n"
                    )
                    if detail:
                        out.append(f"**Detail:** {detail}\n\n")

                elif test_type == "hallucination_detection":
                    refused = entry.get("refused", False)
                    resp_len = entry.get("response_length", 0)
                    preview = entry.get("response_preview", "")

                    status = "REFUSED (good)" if refused else "HALLUCINATED (bad)"
                    out.append(
                        f"**Metrics:** {status} | "
                        f"Response Length: {resp_len}\n\n"
                    )
                    if preview:
                        out.append("**Response Preview:**\n\n")
                        out.append(code_block(truncate(preview, 500)))

                elif test_type == "model_load_failed":
                    err = entry.get("error", "")
                    out.append(f"**Error:** {err}\n\n")

                else:
                    # Generic fallback: dump all keys
                    for k, v in entry.items():
                        if k in ("model", "test_type"):
                            continue
                        out.append(f"**{k}:** {truncate(str(v), 300)}\n\n")

                out.append("---\n\n")

        out.append("\n---\n\n")


def format_track6(data: dict, out: list):
    """
    Track 6: 성능 프로파일링
    Structure: results = [{model, test_type, input_length, output_length,
               tokens_per_sec, prefill_tok_s, ttft_s, vram_used_mb,
               wall_time_s, error, ...extra fields}]
    """
    results = data.get("results", [])
    if not results:
        out.append("*No results found.*\n\n")
        return

    by_model = {}
    for r in results:
        model = r.get("model", "unknown")
        by_model.setdefault(model, []).append(r)

    for model, entries in sorted(by_model.items()):
        out.append(f"## Model: {model}\n\n")

        by_type = {}
        for e in entries:
            tt = e.get("test_type", "unknown")
            by_type.setdefault(tt, []).append(e)

        for test_type, type_entries in sorted(by_type.items()):
            out.append(f"### Test Type: {test_type}\n\n")

            if test_type in ("prefill_speed", "decode_speed", "max_context"):
                # Table format for these
                out.append("| Input Len | Output Len | TPS (decode) | Prefill TPS | Wall Time (s) | Error |\n")
                out.append("|-----------|------------|--------------|-------------|---------------|-------|\n")
                for e in type_entries:
                    in_len = e.get("input_length", 0)
                    out_len = e.get("output_length", 0)
                    tps = e.get("tokens_per_sec", 0)
                    prefill = e.get("prefill_tok_s", 0)
                    wall = e.get("wall_time_s", 0)
                    err = e.get("error") or ""
                    extra_info = ""
                    if e.get("requested_output_length"):
                        extra_info = f" (req: {e['requested_output_length']})"
                    if e.get("requested_context"):
                        extra_info = f" (req ctx: {e['requested_context']})"
                    out.append(
                        f"| {in_len} | {out_len}{extra_info} | "
                        f"{tps:.2f} | {prefill:.2f} | {wall:.4f} | {err} |\n"
                    )
                out.append("\n")

            elif test_type == "ttft":
                out.append("| Prompt Label | Input Len | TTFT (s) | Wall Time (s) | Error |\n")
                out.append("|-------------|-----------|----------|---------------|-------|\n")
                for e in type_entries:
                    label = e.get("prompt_label", "")
                    in_len = e.get("input_length", 0)
                    ttft = e.get("ttft_s", 0)
                    wall = e.get("wall_time_s", 0)
                    err = e.get("error") or ""
                    out.append(f"| {label} | {in_len} | {ttft:.4f} | {wall:.4f} | {err} |\n")
                out.append("\n")

            elif test_type == "vram_usage":
                for e in type_entries:
                    vram = e.get("vram_used_mb", 0)
                    vram_total = e.get("vram_total_mb", 0)
                    vram_free = e.get("vram_free_mb", 0)
                    gpu_util = e.get("gpu_util_pct", 0)
                    out.append(
                        f"**VRAM Used:** {vram} MB | "
                        f"**Total:** {vram_total} MB | "
                        f"**Free:** {vram_free} MB | "
                        f"**GPU Util:** {gpu_util}%\n\n"
                    )

            elif test_type == "quant_comparison":
                out.append("| Base Model | Quant | Prompt Idx | In Len | Out Len | TPS | Prefill TPS | VRAM (MB) | Wall (s) |\n")
                out.append("|-----------|-------|------------|--------|---------|-----|-------------|-----------|----------|\n")
                for e in type_entries:
                    base = e.get("base_model", "")
                    quant = e.get("quant_tag", "")
                    pidx = e.get("prompt_index", "")
                    in_len = e.get("input_length", 0)
                    out_len = e.get("output_length", 0)
                    tps = e.get("tokens_per_sec", 0)
                    prefill = e.get("prefill_tok_s", 0)
                    vram = e.get("vram_used_mb", 0)
                    wall = e.get("wall_time_s", 0)
                    out.append(
                        f"| {base} | {quant} | {pidx} | {in_len} | {out_len} | "
                        f"{tps:.2f} | {prefill:.2f} | {vram} | {wall:.4f} |\n"
                    )
                out.append("\n")

            elif test_type == "concurrent":
                for e in type_entries:
                    level = e.get("concurrency_level") or (e.get("extra") or {}).get("concurrency_level", "?")
                    agg_tps = e.get("aggregate_tok_s") or (e.get("extra") or {}).get("aggregate_tok_s", 0)
                    tps = e.get("tokens_per_sec", 0)
                    out_len = e.get("output_length", 0)
                    wall = e.get("wall_time_s", 0)
                    per_req = e.get("per_request") or (e.get("extra") or {}).get("per_request", [])

                    out.append(f"#### Concurrency Level: {level}\n\n")
                    out.append(
                        f"**Aggregate TPS:** {agg_tps:.2f} | "
                        f"**Avg TPS:** {tps:.2f} | "
                        f"**Total Tokens:** {out_len} | "
                        f"**Wall Time:** {wall:.4f}s\n\n"
                    )

                    if per_req:
                        out.append("| Request | TPS | Tokens | Wall (s) | Error |\n")
                        out.append("|---------|-----|--------|----------|-------|\n")
                        for pr in per_req:
                            ridx = pr.get("request_idx", "?")
                            rtps = pr.get("tokens_per_sec", 0)
                            rtok = pr.get("eval_count", 0)
                            rwall = pr.get("wall_time_s", 0)
                            rerr = pr.get("error") or ""
                            out.append(f"| {ridx} | {rtps:.2f} | {rtok} | {rwall:.4f} | {rerr} |\n")
                        out.append("\n")

            else:
                # model_load_failed or unknown
                for e in type_entries:
                    err = e.get("error", "")
                    out.append(f"**Error:** {err}\n\n")

            out.append("---\n\n")

        out.append("\n---\n\n")

    # Summary
    summary = data.get("summary", {})
    if summary:
        out.append("## Summary Table\n\n")
        out.append("| Model | Avg Prefill TPS | Avg Decode TPS | Avg TTFT (s) | VRAM (MB) | Max Context |\n")
        out.append("|-------|-----------------|----------------|--------------|-----------|-------------|\n")
        for model, s in sorted(summary.items()):
            out.append(
                f"| {model} | {s.get('avg_prefill_tok_s', 0)} | "
                f"{s.get('avg_decode_tok_s', 0)} | "
                f"{s.get('avg_ttft_s', 0)} | "
                f"{s.get('vram_used_mb', 0)} | "
                f"{s.get('max_context_reached', 0)} |\n"
            )
        out.append("\n")


def format_track7(data: dict, out: list):
    """
    Track 7: 쌍대비교 (Pairwise Elo)
    Structure: {results: {responses: {model: {prompt_id: response}},
                comparisons: [...], elo_scores: {...}},
                summary: {model: {elo, ci_lower, ci_upper, wins, losses, rank}}}
    """
    results_data = data.get("results", {})

    # If results is a list (checkpoint format), handle differently
    if isinstance(results_data, list):
        out.append("*Track 7 data appears to be in checkpoint format (list). "
                    "Attempting to display available information.*\n\n")
        return

    responses = results_data.get("responses", {})
    comparisons = results_data.get("comparisons", [])
    elo_scores = results_data.get("elo_scores", {})
    summary = data.get("summary", {})

    # Show all model responses per prompt
    if responses:
        out.append("## Model Responses\n\n")

        # Collect all prompt IDs
        all_prompts = set()
        for model_resps in responses.values():
            all_prompts.update(model_resps.keys())

        for prompt_id in sorted(all_prompts):
            out.append(f"### Prompt: {prompt_id}\n\n")

            for model in sorted(responses.keys()):
                resp = responses[model].get(prompt_id, "")
                out.append(f"#### Model: {model}\n\n")
                out.append("**Output:**\n\n")
                out.append(code_block(truncate(str(resp))))
                out.append("\n")

            out.append("---\n\n")

    # Comparisons
    if comparisons:
        out.append("## Pairwise Comparisons\n\n")
        out.append(f"Total comparisons: {len(comparisons)}\n\n")

        out.append("| Prompt | Model A | Model B | Winner | Score A | Score B | Reasoning (excerpt) |\n")
        out.append("|--------|---------|---------|--------|---------|---------|--------------------|\n")

        for comp in comparisons:
            prompt_id = comp.get("prompt_id", "?")
            model_a = comp.get("model_a", "?")
            model_b = comp.get("model_b", "?")
            winner = comp.get("winner", "?")
            score_a = comp.get("score_a", "")
            score_b = comp.get("score_b", "")
            reasoning = str(comp.get("reasoning", ""))[:80].replace("|", "/").replace("\n", " ")

            out.append(
                f"| {prompt_id} | {model_a} | {model_b} | "
                f"{winner} | {score_a} | {score_b} | {reasoning} |\n"
            )
        out.append("\n")

    # Elo Scores / Summary
    if summary:
        out.append("## Elo Rankings\n\n")
        out.append("| Rank | Model | Elo | CI Lower | CI Upper | Wins | Losses |\n")
        out.append("|------|-------|-----|----------|----------|------|--------|\n")

        sorted_models = sorted(summary.items(), key=lambda x: x[1].get("rank", 999))
        for model, s in sorted_models:
            rank = s.get("rank", "?")
            elo = s.get("elo", 0)
            ci_lo = s.get("ci_lower", 0)
            ci_hi = s.get("ci_upper", 0)
            wins = s.get("wins", 0)
            losses = s.get("losses", 0)
            out.append(
                f"| {rank} | {model} | {elo:.1f} | "
                f"{ci_lo:.1f} | {ci_hi:.1f} | {wins} | {losses} |\n"
            )
        out.append("\n")
    elif elo_scores:
        out.append("## Elo Scores\n\n")
        out.append("| Model | Elo | Wins | Losses |\n")
        out.append("|-------|-----|------|--------|\n")
        for model, s in sorted(elo_scores.items(), key=lambda x: x[1].get("elo", 0), reverse=True):
            elo = s.get("elo", 0)
            wins = s.get("wins", 0)
            losses = s.get("losses", 0)
            out.append(f"| {model} | {elo:.1f} | {wins} | {losses} |\n")
        out.append("\n")


# ── Track format dispatcher ───────────────────────────────────────────────────

FORMATTERS = {
    "track1": format_track1,
    "track2": format_track2,
    "track3": format_track3,
    "track4": format_track4,
    "track5": format_track5,
    "track6": format_track6,
    "track7": format_track7,
}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Searching for result files in: {RESULTS_DIR}")
    found_files = find_result_files(RESULTS_DIR)

    if not found_files:
        print("ERROR: No track result JSON files found in results/ directory.")
        print("Expected files like: track1_korean_bench_20260311_024226.json")
        sys.exit(1)

    print(f"Found {len(found_files)} track result files:")
    for key, path in sorted(found_files.items()):
        print(f"  {key}: {path}")

    # Build the Markdown document
    out = []
    out.append("# Detailed Input/Output Log -- FrankenStaLLM Evaluation\n\n")
    out.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    out.append(f"**Source directory:** `{RESULTS_DIR}`\n\n")

    # Table of contents
    out.append("## Table of Contents\n\n")
    for track_key in sorted(found_files.keys()):
        track_title = TRACK_NAMES.get(track_key, track_key)
        anchor = track_title.lower().replace(" ", "-").replace(":", "").replace("(", "").replace(")", "")
        out.append(f"- [{track_title}](#{anchor})\n")
    out.append("\n---\n\n")

    # Process each track
    for track_key in sorted(found_files.keys()):
        filepath = found_files[track_key]
        track_title = TRACK_NAMES.get(track_key, track_key)

        print(f"\nProcessing {track_key}: {filepath.name}...")

        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            out.append(f"# {track_title}\n\n")
            out.append(f"**Error loading file:** {e}\n\n")
            continue

        out.append(f"# {track_title}\n\n")
        out.append(f"**Source file:** `{filepath.name}`\n\n")

        # Show file-level metadata
        if "timestamp" in data:
            out.append(f"**Timestamp:** {data['timestamp']}\n\n")
        if "num_models" in data:
            out.append(f"**Number of models:** {data['num_models']}\n\n")
        if "mode" in data:
            out.append(f"**Mode:** {data['mode']}\n\n")

        # Format track-specific content
        formatter = FORMATTERS.get(track_key)
        if formatter:
            formatter(data, out)
        else:
            out.append("*No formatter available for this track.*\n\n")
            out.append(f"**Top-level keys:** {list(data.keys())}\n\n")

        out.append("\n\n")

    # Write output
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    content = "".join(out)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\nDone! Output written to: {OUTPUT_FILE}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Total lines: {content.count(chr(10))}")


if __name__ == "__main__":
    main()
