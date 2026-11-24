"""최종 보고서 HTML 생성: reports/final/data.json(final_data) + narrative.json(해설) → 한 장짜리 HTML.

python -m kobench.standard.final_report [--data reports/final/data.json] [--narrative reports/final/narrative.json]
                                          [--out reports/final/report.html] [--fragment]

기본 출력은 브라우저에서 바로 여는 단독 문서(doctype·charset·viewport 포함, GitHub Pages가 그대로 배포)다.
--fragment는 문서 골격 없이 내용만 쓴다(골격을 덧씌우는 호스팅에 올릴 때).

구성: Python이 data.json에서 차트·표 단위 '뷰'를 계산하고(값·CI·색 슬롯), 페이지 안의 d3 렌더러가 그린다.
해설 문장은 narrative.json {섹션 id: HTML}에 따로 쓴다(결과가 바뀌면 뷰는 다시 계산되고, 해설은 사람이 고친다).
색: 사용자 모델만 범주색(JANSORI 1 파랑, frankenstallm 2 주황, EVAFRILL 3 청록 — 앞 3슬롯은 모든 쌍 CVD 검증 통과),
공개 모델은 회색. 레이더는 모든 축 바깥=좋음, 축별 로그 정규화(최고=1), 순위는 기하평균 막대로 따로 낸다.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "reports" / "final"

D3 = "https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"

TASK_LABEL = {
    "kobest": "KoBEST", "kmmlu": "KMMLU", "haerae": "HAE-RAE", "click": "CLIcK", "global_mmlu_ko": "Global-MMLU-ko",
    "belebele_kor_Hang": "Belebele-ko", "kmmlu_redux": "KMMLU-Redux", "kmmlu_pro": "KMMLU-Pro", "kobalt": "KoBALT",
    "hellaswag": "HellaSwag", "arc_easy": "ARC-e", "arc_challenge": "ARC-c", "mmlu": "MMLU", "winogrande": "WinoGrande",
    "hrm8k": "HRM8K", "ifeval_ko": "IFEval-Ko", "kormedmcqa": "KorMedMCQA", "kobbq": "KoBBQ", "ko_truthful_mc1": "TruthfulQA-ko",
    "k2_knowledge": "K2 지식", "komt": "KoMT", "logickor": "LogicKor", "k2_generation": "K2 생성", "ko_arena_hard": "arena-hard-ko",
    "kosimpleqa": "KoSimpleQA", "humaneval_plus": "HumanEval+", "mbpp_plus": "MBPP+", "scenario_self": "다턴(자기 이력)",
    "scenario_oracle": "다턴(참조 이력)", "gsm8k_ko_dev": "GSM8K-ko dev", "kormedmcqa_dev": "KorMedMCQA dev",
    "scenario_dev_self": "다턴 dev",
}
SCN_CAT = {"math_update": "수학 조건 변경", "memory": "일정 기억", "instruction_persist": "지시 유지", "format_transform": "형식 변환",
           "logic_order": "순서 추론", "list_tracking": "목록 추적", "story_continue": "이어쓰기", "korean_style": "한국어 문체",
           "code_revise": "코드 수정", "unknown_entity": "모름 인정"}
JUDGE_SCALE = {"komt", "logickor", "k2_generation"}


def slot(model: str) -> int | None:
    if model.startswith("jansori"):
        return 1
    if model.startswith("frankenstallm"):
        return 2
    if model.startswith("evafrill"):
        return 3
    return None


def tl(t: str) -> str:
    return TASK_LABEL.get(t, t)


def _num(x):
    return x if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)) else None


# ── 뷰 빌더 ─────────────────────────────────────────────────────────────────

def mc_views(d: dict) -> dict:
    v = {}
    for suite in ("ko_mc_v1", "ko_mc_ext_v1", "en_mc_v1"):
        rows = d["mc"].get(suite) or []
        if not rows:
            continue
        tasks = list(dict.fromkeys(r["task"] for r in rows))
        for kind in ("all",):                 # 2026-11-22: base·instruct 한 라인 (종류는 열·호버로 표시)
            ms = list(dict.fromkeys(r["model"] for r in rows))
            kinds = {r["model"]: r["kind"] for r in rows}
            avg = []
            trows = []
            for m in ms:
                rr = {r["task"]: r for r in rows if r["model"] == m}
                vals = [rr[t]["micro"] for t in tasks if t in rr and _num(rr[t]["micro"]) is not None]
                a = sum(vals) / len(vals) if vals else None
                avg.append({"label": m, "v": a * 100 if a is not None else None, "slot": slot(m), "kind": kinds[m],
                            "tip": f"{m} ({kinds[m]}): 태스크 평균 {a * 100:.1f}% ({len(vals)}개)" if a is not None else m})
                cells = {"model": {"v": m, "slot": slot(m)}, "_kind": {"v": kinds[m]}}
                for t in tasks:
                    r = rr.get(t)
                    if r and _num(r["micro"]) is not None:
                        cells[t] = {"v": r["micro"] * 100, "fmt": "pct",
                                    "tip": f"{m} · {tl(t)}: {r['micro'] * 100:.1f}% [95% CI {r['ci_lo'] * 100:.1f}, {r['ci_hi'] * 100:.1f}], "
                                           f"n={int(r['n'])}, {int(r['nshot'])}-shot"}
                cells["avg"] = {"v": a * 100 if a is not None else None, "fmt": "pct"}
                cells["_params"] = {"v": f"{rr[next(iter(rr))]['params_b']:g}/{rr[next(iter(rr))]['active_b']:g}"}
                trows.append(cells)
            avg.sort(key=lambda x: -(x["v"] or 0))
            v[f"mc_avg_{suite}_{kind}"] = {"type": "dots", "rows": avg, "unit": "%", "xlabel": "태스크 평균 정확도(%)",
                                           "title": f"{suite}: 태스크 평균 (모든 모델, ■ base / ● instruct)"}
            cols = [{"key": "model", "label": "모델"}, {"key": "_kind", "label": "종류"}, {"key": "_params", "label": "총/활성 B"}] + \
                   [{"key": t, "label": tl(t)} for t in tasks] + [{"key": "avg", "label": "평균"}]
            trows.sort(key=lambda r: -(r["avg"]["v"] or 0))
            v[f"mc_table_{suite}_{kind}"] = {"type": "table", "columns": cols, "rows": trows, "sort": "avg",
                                             "title": f"{suite} (모든 모델)"}
    # B 효과(같은 모델 A 대비, paired CI)
    fr = []
    for suite in ("ko_mc_v1", "ko_mc_ext_v1", "en_mc_v1"):
        for r in d["mc"].get(f"{suite}__B") or []:
            if r.get("variant") in (None, "A") or _num(r.get("delta_A")) is None:
                continue
            fr.append({"label": f"{r['model']} · {tl(r['task'])}", "group": f"{r['model']} ({r['variant']})", "v": r["delta_A"] * 100,
                       "lo": r["delta_A_lo"] * 100, "hi": r["delta_A_hi"] * 100, "slot": slot(r["model"]),
                       "tip": f"{r['model']} {r['variant']} − A · {suite}/{tl(r['task'])}: {r['delta_A'] * 100:+.2f}%p "
                              f"[{r['delta_A_lo'] * 100:+.2f}, {r['delta_A_hi'] * 100:+.2f}]"})
    if fr:
        v["mc_b_effect"] = {"type": "dots", "rows": fr, "unit": "%p", "zero": True, "xlabel": "B − A (%p, paired bootstrap 95% CI)",
                            "title": "객관식 dev 선택 설정의 test 효과", "grouped": True}
    # PPL bits-per-byte
    rows = d["mc"].get("ppl_v1") or []
    for t in dict.fromkeys(r["task"] for r in rows):
        rr = [r for r in rows if r["task"] == t and _num(r["micro"]) is not None]
        rr.sort(key=lambda r: r["micro"])
        v[f"ppl_{t}"] = {"type": "dots", "rows": [{"label": r["model"], "v": r["micro"], "lo": r["ci_lo"], "hi": r["ci_hi"], "slot": slot(r["model"]), "kind": r["kind"],
                                                    "tip": f"{r['model']}: {r['micro']:.4f} bpb [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] ({r['kind']})"}
                                                   for r in rr],
                         "unit": "bpb", "xlabel": "bits per byte (낮을수록 좋음)", "lower_better": True, "title": f"PPL · {tl(t)}", "digits": 3}
    return v


def quant_views(d: dict) -> dict:
    v = {}
    for suite, unit, scale in (("ko_mc_v1", "%p", 100), ("ppl_v1", "bpb", 1)):
        rows = d["mc"].get(f"{suite}__quant") or []
        fr = []
        for r in rows:
            for ref, key in (("f16", "delta_f16"), ("BF16", "delta_bf16")):
                if _num(r.get(key)) is None or (ref == "BF16" and r["quant"] != "f16"):
                    continue          # BF16 기준은 f16 GGUF(엔진+정밀도 차이)만, 양자화 손실은 f16 기준
                lab = f"{r['family']} {r['quant']} − {ref} · {tl(r['task'])}"
                fr.append({"label": lab, "group": f"{r['family']} {r['quant']} − {ref}", "v": r[key] * scale,
                           "lo": r[f"{key}_lo"] * scale, "hi": r[f"{key}_hi"] * scale, "slot": slot(r["model"]),
                           "tip": f"{lab}: {r[key] * scale:+.4g} [{r[key + '_lo'] * scale:+.4g}, {r[key + '_hi'] * scale:+.4g}] {unit}"})
        if fr:
            v[f"quant_{suite}"] = {"type": "dots", "rows": fr, "unit": unit, "zero": True, "grouped": True,
                                   "xlabel": ("정확도 차이 (%p)" if unit == "%p" else "bits-per-byte 차이 (양수 = 나빠짐)"),
                                   "title": f"GGUF 양자화 손실 · {suite}", "digits": 2 if unit == "%p" else 4}
        trows = []
        for r in rows:
            trows.append({"model": {"v": r["family"], "slot": slot(r["model"])}, "quant": {"v": f"{r['quant']} ({r['engine']})"},
                          "task": {"v": tl(r["task"])},
                          "val": {"v": r["micro"] * scale, "fmt": "pct" if scale == 100 else "f4",
                                  "tip": f"[{r['ci_lo'] * scale:.4g}, {r['ci_hi'] * scale:.4g}], n={int(r['n'])}"},
                          "df16": _delta_cell(r, "delta_f16", scale), "dbf16": _delta_cell(r, "delta_bf16", scale)})
        if trows:
            v[f"quant_table_{suite}"] = {"type": "table", "title": f"양자화 표 · {suite}",
                                         "columns": [{"key": "model", "label": "모델"}, {"key": "quant", "label": "형식(엔진)"},
                                                     {"key": "task", "label": "태스크"},
                                                     {"key": "val", "label": "정확도 %" if scale == 100 else "bpb"},
                                                     {"key": "df16", "label": "− f16 GGUF"}, {"key": "dbf16", "label": "− BF16 vLLM"}],
                                         "rows": trows}
    return v


def _delta_cell(r, key, scale):
    if _num(r.get(key)) is None:
        return {"v": None}
    lo, hi = r[f"{key}_lo"] * scale, r[f"{key}_hi"] * scale
    sig = lo > 0 or hi < 0
    return {"v": r[key] * scale, "fmt": "d2" if scale == 100 else "d4", "sig": sig, "tip": f"95% CI [{lo:+.4g}, {hi:+.4g}]" + (" · 유의" if sig else "")}


def gen_views(d: dict) -> dict:
    v = {}
    for name in ("gen_v1__A", "gen_v1__B", "gen_v1__think", "gen_dev_v1__A", "gen_dev_v1__B"):
        rows = d["gen"].get(name) or []
        if name == "gen_v1__think":
            rows = rows + (d["gen"].get("gen_v1__thinkrd") or [])   # qwen3.5-4b README 권장 샘플링 재실행(2026-11-23)은 같은 표의 별도 행
        if not rows:
            continue
        for kind in ("instruct", "base"):
            rk = [r for r in rows if r["kind"] == kind]
            if not rk:
                continue
            tasks = list(dict.fromkeys(r["task"] for r in rk))
            trows = []
            for m, var in dict.fromkeys((r["model"], r["variant"]) for r in rk):
                rr = {r["task"]: r for r in rk if r["model"] == m and r["variant"] == var}
                a = next(iter(rr.values()))
                vlabel = "thinkrd (README 권장 샘플링)" if var == "thinkrd" else var
                cells = {"model": {"v": m, "slot": slot(m)}, "_set": {"v": vlabel + (f" / {a['unk_policy']}" if a['unk_policy'] not in ('raw', None) else "")}}
                for t in tasks:
                    r = rr.get(t)
                    if not r or _num(r["value"]) is None:
                        continue
                    judge = t in JUDGE_SCALE
                    s = 1 if judge else 100
                    tip = f"{m} · {tl(t)} ({r['metric']}): {r['value'] * s:.{2 if judge else 1}f} [{r['ci_lo'] * s:.{2 if judge else 1}f}, {r['ci_hi'] * s:.{2 if judge else 1}f}], n={int(r['n'])}"
                    if _num(r.get("delta_A")) is not None:
                        tip += f" · A 대비 {r['delta_A'] * s:+.{2 if judge else 1}f} [{r['delta_A_lo'] * s:+.{2 if judge else 1}f}, {r['delta_A_hi'] * s:+.{2 if judge else 1}f}]"
                    if _num(r.get("loop")) is not None:
                        tip += f" · 루프 {r['loop'] * 100:.1f}% · 잘림 {r['truncated'] * 100:.1f}%"
                    if r.get("judge_same_family"):
                        tip += " · judge와 같은 계열(자기 선호 가능)"
                    cells[t] = {"v": r["value"] * s, "fmt": "f2" if judge else "pct", "tip": tip, "flag": bool(r.get("judge_same_family")),
                                "warn": bool(_num(r.get("loop")) and r["loop"] > 0.05)}
                trows.append(cells)
            cols = [{"key": "model", "label": "모델"}, {"key": "_set", "label": "설정"}] + [{"key": t, "label": tl(t)} for t in tasks]
            v[f"gen_table_{name}_{kind}"] = {"type": "table", "columns": cols, "rows": trows, "title": f"{name} · {kind}"}
    rows = d["gen"].get("gen_v1__B") or []
    fr = []
    for r in rows:
        if _num(r.get("delta_A")) is None:
            continue
        s = 1 if r["task"] in JUDGE_SCALE else 100
        if r["task"] in JUDGE_SCALE:
            continue          # 척도가 다른 judge 점수는 표에서만 (한 축에 섞지 않는다)
        fr.append({"label": f"{r['model']} · {tl(r['task'])}", "group": f"{r['model']} ({r['variant']})", "v": r["delta_A"] * s,
                   "lo": r["delta_A_lo"] * s, "hi": r["delta_A_hi"] * s, "slot": slot(r["model"]),
                   "tip": f"{r['model']} {r['variant']} − A · {tl(r['task'])}: {r['delta_A'] * s:+.1f}%p [{r['delta_A_lo'] * s:+.1f}, {r['delta_A_hi'] * s:+.1f}]"})
    if fr:
        v["gen_b_effect"] = {"type": "dots", "rows": fr, "unit": "%p", "zero": True, "grouped": True, "digits": 1,
                             "xlabel": "선택 설정 − A (%p, paired bootstrap 95% CI)", "title": "생성 운용 설정의 test 효과"}
    # 운용 설정 탐색(v3) 전후
    tr = d["gen"].get("tune_v3") or []
    if tr:
        mal = lambda r, p: sum((r.get(f"{p}_{k}") or 0) for k in ("loop", "lang", "leak", "trunc")) * 100
        v["tune_mal"] = {"type": "dumbbell", "title": "오동작률 합(루프+언어 이탈+누출+잘림), A → 선택 설정 (dev)",
                         "alabel": "A (greedy)", "blabel": "선택 설정", "unit": "%", "lower_better": True,
                         "rows": sorted([{"label": r["model"], "a": mal(r, "A"), "b": mal(r, "chosen"), "slot": slot(r["model"]),
                                          "tip": f"{r['model']} → {r['chosen']}/{r['unk_policy']}: 루프 {100 * (r.get('A_loop') or 0):.1f}→{100 * (r.get('chosen_loop') or 0):.1f}%, "
                                                 f"언어 {100 * (r.get('A_lang') or 0):.1f}→{100 * (r.get('chosen_lang') or 0):.1f}%, "
                                                 f"잘림 {100 * (r.get('A_trunc') or 0):.1f}→{100 * (r.get('chosen_trunc') or 0):.1f}%"
                                                 + ("" if r["feasible"] else " · 제약 미충족")} for r in tr], key=lambda x: -x["a"])}
        v["tune_quality"] = {"type": "dumbbell", "title": "dev 품질(세 태스크 평균), A → 선택 설정", "alabel": "A (greedy)",
                             "blabel": "선택 설정", "unit": "%",
                             "rows": sorted([{"label": r["model"], "a": 100 * (r.get("A_quality") or 0), "b": 100 * (r.get("chosen_quality") or 0),
                                              "slot": slot(r["model"])} for r in tr], key=lambda x: -x["b"])}
        pct = lambda x: None if x is None else 100 * x
        v["tune_table"] = {"type": "table", "title": "모델별 운용 설정 (규칙 v3)",
                           "columns": [{"key": "model", "label": "모델"}, {"key": "set", "label": "선택"}, {"key": "ok", "label": "제약"},
                                       {"key": "qa", "label": "품질 A"}, {"key": "qc", "label": "품질 선택"},
                                       {"key": "la", "label": "루프 A"}, {"key": "lc", "label": "루프 선택"},
                                       {"key": "ga", "label": "언어 A"}, {"key": "gc", "label": "언어 선택"},
                                       {"key": "ta", "label": "잘림 A"}, {"key": "tc", "label": "잘림 선택"}],
                           "rows": [{"model": {"v": r["model"], "slot": slot(r["model"])}, "set": {"v": f"{r['chosen']} / {r['unk_policy']}"},
                                     "ok": {"v": "만족" if r["feasible"] else "미충족", "warn": not r["feasible"]},
                                     "qa": {"v": pct(r.get("A_quality")), "fmt": "pct"}, "qc": {"v": pct(r.get("chosen_quality")), "fmt": "pct"},
                                     "la": {"v": pct(r.get("A_loop")), "fmt": "pct"}, "lc": {"v": pct(r.get("chosen_loop")), "fmt": "pct"},
                                     "ga": {"v": pct(r.get("A_lang")), "fmt": "pct"}, "gc": {"v": pct(r.get("chosen_lang")), "fmt": "pct"},
                                     "ta": {"v": pct(r.get("A_trunc")), "fmt": "pct"}, "tc": {"v": pct(r.get("chosen_trunc")), "fmt": "pct"}}
                                    for r in tr]}
    return v


def scenario_views(d: dict) -> dict:
    v = {}
    rows = d.get("scenario") or []
    for task in ("scenario_self", "scenario_dev_self"):
        rr = [r for r in rows if r["task"] == task and r["variant"] == "A"]
        if not rr:
            continue
        rr.sort(key=lambda r: -(r["turn_pass"] or 0))
        cats = list(SCN_CAT)
        v[f"scn_heat_{task}"] = {"type": "heatmap", "title": f"{tl(task)} · 범주별 턴 통과율 (A)",
                                 "rows": [r["model"] for r in rr], "cols": [SCN_CAT[c] for c in cats],
                                 "slots": [slot(r["model"]) for r in rr],
                                 "values": [[100 * ((r["by_category"] or {}).get(c, {}).get("turn_pass") or 0) for c in cats] for r in rr],
                                 "total": [100 * r["turn_pass"] for r in rr], "unit": "%"}
        turns = sorted({int(k) for r in rr for k in (r["by_turn"] or {})})
        series = [{"name": r["model"], "slot": slot(r["model"]),
                   "points": [[t, 100 * r["by_turn"][str(t)]] for t in turns if str(t) in (r["by_turn"] or {})]} for r in rr]
        v[f"scn_turn_{task}"] = {"type": "lines", "title": f"{tl(task)} · 턴 위치별 통과율", "series": series,
                                 "xlabel": "턴", "ylabel": "통과율 (%)", "xticks": turns}
    so = {}
    for r in rows:
        if r["variant"] != "A" or r["task"] not in ("scenario_self", "scenario_oracle"):
            continue
        so.setdefault(r["model"], {})[r["task"]] = r
    pairs = [{"label": m, "a": 100 * x["scenario_oracle"]["turn_pass"], "b": 100 * x["scenario_self"]["turn_pass"], "slot": slot(m),
              "tip": f"{m}: 참조 이력 {100 * x['scenario_oracle']['turn_pass']:.1f}% → 자기 이력 {100 * x['scenario_self']['turn_pass']:.1f}% "
                     f"(앞 답 복사 {100 * (x['scenario_self']['copy_prev'] or 0):.1f}%, 루프 {100 * (x['scenario_self']['loop'] or 0):.1f}%)"}
             for m, x in so.items() if len(x) == 2]
    if pairs:
        v["scn_self_oracle"] = {"type": "dumbbell", "title": "참조 이력 → 자기 이력 (오류 누적 효과)", "alabel": "참조 이력(oracle)",
                                "blabel": "자기 이력(self)", "unit": "%", "rows": sorted(pairs, key=lambda x: -x["b"])}
    if d.get("transcripts"):
        v["scn_transcripts"] = {"type": "transcripts", "title": "대화 보기", "items": d["transcripts"], "cats": SCN_CAT}
    return v


def _is_moe(e: dict) -> bool:
    return not (e["params_b"] == e["active_b"] or e["active_b"] is None or e["params_b"] < 8)


def _w1(run: dict, c: int):
    return next((w for w in run.get("w1") or [] if w["concurrency"] == c), None)


def perf_views(d: dict) -> dict:
    v = {}
    rows = [r for r in d.get("perf") or [] if not r.get("error")]
    if not rows:
        return v
    # 모델·엔진별 묶기
    by = {}
    for r in rows:
        k = (r["model"], r["engine"])
        e = by.setdefault(k, {"model": r["model"], "engine": r["engine"], "params_b": r["params_b"], "active_b": r["active_b"],
                              "family": r["family"], "kind": r["kind"], "runs": {}})
        run = e["runs"].setdefault(r["run"], {"w1": [], "w2": None, **{k2: r.get(k2) for k2 in ("weights_gib", "kv_cache_gib", "kv_cache_tokens",
                                                                                                "vram_peak_mib", "startup_s", "kv_bytes_per_token",
                                                                                                "mamba_state_mib_per_seq", "mem_source")}})
        if r["workload"] == "w1":
            run["w1"].append(r)
        else:
            run["w2"] = r
    main = {}
    for (m, eng), e in by.items():
        run = e["runs"].get("prefix_cache_on") or e["runs"].get("static_batch") or next(iter(e["runs"].values()))
        e["main"] = run
        main[(m, eng)] = e
    vllm = [e for (m, eng), e in main.items() if eng in ("vllm", "evafrill")]
    for panel, pred in (("all", lambda e: True),):         # 2026-11-22: dense·MoE 한 축 (MoE는 이름 뒤 [MoE])
        es = [e for e in vllm if pred(e)]
        if not es:
            continue
        for metric, lab, ylog in (("output_tok_per_s", "출력 처리량 (tok/s)", True), ("ttft_ms_p50", "TTFT p50 (ms)", True)):
            series = [{"name": e["model"] + (" (EVAFRILL 엔진, 정적 배치)" if e["engine"] == "evafrill" else "") + (" [MoE]" if _is_moe(e) else ""),
                       "slot": slot(e["model"]), "dash": _is_moe(e), "kind": e["kind"],
                       "points": [[w["concurrency"], w[metric]] for w in sorted(e["main"]["w1"], key=lambda w: w["concurrency"]) if _num(w[metric])]}
                      for e in es]
            v[f"perf_{metric}_{panel}"] = {"type": "lines", "series": series, "xlog": True, "ylog": ylog, "xlabel": "동시 요청 수",
                                           "ylabel": lab, "xticks": [1, 4, 16, 32, 64],
                                           "title": f"{lab} · 모든 모델 (점선 = MoE A3B급)"}
    tp = []
    for e in vllm:
        a, b = _w1(e["main"], 1), _w1(e["main"], 64)
        if a and b:
            tp.append({"label": e["model"], "a": a["tpot_ms_p50"], "b": b["tpot_ms_p50"], "slot": slot(e["model"])})
    if tp:
        v["perf_tpot"] = {"type": "dumbbell", "title": "토큰 간 시간 TPOT p50: 동시 1 → 64", "alabel": "동시 1", "blabel": "동시 64",
                          "unit": "ms", "lower_better": True, "rows": sorted(tp, key=lambda x: x["b"]), "digits": 1}
    pc = []
    for (m, eng), e in by.items():
        on, off = e["runs"].get("prefix_cache_on"), e["runs"].get("prefix_cache_off")
        if on and off and on.get("w2") and off.get("w2"):
            pc.append({"label": f"{m}" + ("" if eng == "vllm" else f" ({eng})"), "a": off["w2"]["ttft_ms_p50"], "b": on["w2"]["ttft_ms_p50"],
                       "slot": slot(m)})
    if pc:
        v["perf_prefix"] = {"type": "dumbbell", "title": "공유 prefix(~2,000토큰) TTFT p50: 캐시 끔 → 켬", "alabel": "prefix caching 끔",
                            "blabel": "켬", "unit": "ms", "lower_better": True, "rows": sorted(pc, key=lambda x: x["b"]), "digits": 0}
    mem = []
    for e in vllm:
        r = e["main"]
        if _num(r.get("weights_gib")):
            kvb = r.get("kv_bytes_per_token") or ((r["kv_cache_gib"] * 2 ** 30 / r["kv_cache_tokens"]) if _num(r.get("kv_cache_tokens")) and r["kv_cache_tokens"] else None)
            src = f" · {r['mem_source']}" if r.get("mem_source") else ""
            mst = f", Mamba 상태 시퀀스당 {r['mamba_state_mib_per_seq']:.0f} MiB 고정" if _num(r.get("mamba_state_mib_per_seq")) else ""
            mem.append({"label": e["model"], "v": r["weights_gib"], "slot": slot(e["model"]), "kv": kvb, "src": src, "mst": mst,
                        "tip": f"{e['model']}: 가중치 {r['weights_gib']:.2f} GiB" + (f", KV {kvb / 1024:.0f} KiB/토큰" if kvb else "") + mst + src
                               + (f", nvidia-smi 최고 {r['vram_peak_mib'] / 1024:.0f} GiB(선할당 포함)" if _num(r.get('vram_peak_mib')) else "")})
    if mem:
        v["perf_weights"] = {"type": "bars", "title": "가중치 메모리 (GiB)", "unit": "GiB", "rows": sorted(mem, key=lambda x: x["v"]), "digits": 1}
        kv = [{"label": x["label"], "v": x["kv"] / 1024, "slot": x["slot"],
               "tip": f"{x['label']}: KV {x['kv'] / 1024:.0f} KiB/토큰{x['mst']}{x['src']}"} for x in mem if x["kv"]]
        if kv:
            v["perf_kv"] = {"type": "bars", "title": "KV 캐시 크기 (KiB/토큰)", "unit": "KiB", "rows": sorted(kv, key=lambda x: x["v"]), "digits": 0}
    # GGUF / 엔진 비교 (같은 모델의 vLLM BF16 vs llama-server 형식별)
    eng_rows = []
    for (m, eng), e in main.items():
        if eng != "llamacpp" and not m.endswith("-bf16"):
            continue
        fam = "JANSORI" if m.startswith("jansori") else ("frankenstallm" if m.startswith("frankenstallm") else None)
        if not fam:
            continue
        a, b = _w1(e["main"], 1), _w1(e["main"], 64)
        lab = f"{fam} {'BF16 vLLM' if eng == 'vllm' else m.split('-gguf-')[-1].upper() + ' llama.cpp'}"
        if a and b:
            eng_rows.append({"label": lab, "a": a["output_tok_per_s"], "b": b["output_tok_per_s"], "slot": slot(m)})
    if eng_rows:
        v["perf_engine"] = {"type": "dumbbell", "title": "엔진·형식별 출력 처리량: 동시 1 → 64 (tok/s)", "alabel": "동시 1",
                            "blabel": "동시 64", "unit": "tok/s", "xlog": True, "rows": sorted(eng_rows, key=lambda x: x["label"]), "digits": 0}
    v.update(radar_views(vllm))
    return v


RADAR_AXES = [("thru", "최대 처리량", "tok/s"), ("speed", "단일 스트림 속도", "tok/s"), ("ttft", "첫 토큰 빠르기", "ms"),
              ("load", "부하 내성", "×"), ("mem", "메모리 효율", "GiB"), ("prefix", "prefix 캐시 이득", "×")]


def radar_metrics(e: dict) -> dict:
    r = e["main"]
    w = sorted(r["w1"], key=lambda x: x["concurrency"])
    a, b = _w1(r, 1), _w1(r, 64)
    out = {}
    if w:
        out["thru"] = (max(x["output_tok_per_s"] for x in w), f"{max(x['output_tok_per_s'] for x in w):,.0f} tok/s")
    if a:
        out["speed"] = (1000 / a["tpot_ms_p50"], f"TPOT {a['tpot_ms_p50']:.2f} ms")
        out["ttft"] = (1 / a["ttft_ms_p50"], f"TTFT {a['ttft_ms_p50']:.1f} ms")
    if a and b:
        out["load"] = (a["tpot_ms_p50"] / b["tpot_ms_p50"], f"TPOT {a['tpot_ms_p50']:.2f}→{b['tpot_ms_p50']:.2f} ms")
    if _num(r.get("weights_gib")):
        out["mem"] = (1 / r["weights_gib"], f"{r['weights_gib']:.2f} GiB")
    on, off = e["runs"].get("prefix_cache_on"), e["runs"].get("prefix_cache_off")
    if on and off and on.get("w2") and off.get("w2"):
        out["prefix"] = (off["w2"]["ttft_ms_p50"] / on["w2"]["ttft_ms_p50"],
                         f"TTFT {off['w2']['ttft_ms_p50']:.0f}→{on['w2']['ttft_ms_p50']:.0f} ms")
    return out


def radar_views(es: list[dict]) -> dict:
    if len(es) < 2:
        return {}
    mets = {e["model"]: radar_metrics(e) for e in es}
    lo = {k: min(m[k][0] for m in mets.values() if k in m) for k, *_ in RADAR_AXES if any(k in m for m in mets.values())}
    hi = {k: max(m[k][0] for m in mets.values() if k in m) for k in lo}
    axes = [(k, lab, u) for k, lab, u in RADAR_AXES if k in lo]

    def norm(k, x):           # 로그 정규화: 최저 0.15, 최고 1
        if hi[k] <= lo[k]:
            return 1.0
        return 0.15 + 0.85 * (math.log(x) - math.log(lo[k])) / (math.log(hi[k]) - math.log(lo[k]))

    rows = []
    for m, mm in mets.items():
        ratios = [mm[k][0] / hi[k] for k, *_ in axes if k in mm]
        g = math.exp(sum(math.log(x) for x in ratios) / len(ratios)) if ratios else None
        rows.append({"label": m, "v": g, "slot": slot(m), "n_axes": len(ratios),
                     "tip": f"{m}: 축별 (값/최고) 기하평균 {g:.3f} · 축 {len(ratios)}/{len(axes)}개"
                            + ("" if len(ratios) == len(axes) else " (없는 축 제외)")})
    rows.sort(key=lambda x: -(x["v"] or 0))
    pub = [m for m in mets if slot(m) is None]
    med = [median([norm(k, mets[m][k][0]) for m in pub if k in mets[m]]) if any(k in mets[m] for m in pub) else None for k, *_ in axes]
    users = [m for m in mets if slot(m) is not None]
    radar = {"type": "radar", "title": "사용자 모델 성능 모양 (공개 모델 중앙값 회색)",
             "axes": [{"key": k, "label": lab} for k, lab, _ in axes],
             "models": [{"name": m, "slot": slot(m), "vals": [norm(k, mets[m][k][0]) if k in mets[m] else None for k, *_ in axes],
                         "raw": [mets[m][k][1] if k in mets[m] else "해당 없음" for k, *_ in axes]} for m in users],
             "median": med}
    return {"perf_radar": radar,
            "perf_geomean": {"type": "bars", "title": "속도·메모리 종합 (축별 값/최고의 기하평균, 1 = 모든 축 최고)", "rows": rows, "unit": "",
                             "digits": 3}}


def model_view(d: dict) -> dict:
    rows = []
    for m, s in d["models"].items():
        if s.get("backend") == "llamacpp":
            continue
        rows.append({"model": {"v": m, "slot": slot(m)}, "kind": {"v": s["kind"]}, "family": {"v": s["family"]},
                     "params": {"v": s["params_b"], "fmt": "f1"}, "active": {"v": s["active_b"], "fmt": "f1"},
                     "owner": {"v": "사용자" if s["owner"] == "self" else "공개"},
                     "rev": {"v": (s.get("revision") or "")[:10]}})
    return {"models": {"type": "table", "title": "평가 대상 모델", "rows": rows, "sort": None,
                       "columns": [{"key": "model", "label": "모델"}, {"key": "kind", "label": "종류"}, {"key": "family", "label": "계열"},
                                   {"key": "params", "label": "총 B"}, {"key": "active", "label": "활성 B"}, {"key": "owner", "label": "출처"},
                                   {"key": "rev", "label": "revision"}]}}


# ── 사용자 모델 일관 표시 ─────────────────────────────────────────────────────
# 사용자 모델 4개 항목(EVAFRILL은 base 트랙 pretrain, instruct 트랙 slerp)이 모든 표·그림에 같은 이름·색·위치로 나오게 한다.
# 측정이 없는 곳은 조용히 빠지지 않도록 표에는 사유 행을, 그림에는 사유 주석을 붙인다.
USER_ENTRIES = [("jansori-bf16", "JANSORI", "base"), ("evafrill-mo-3b-pretrain", "EVAFRILL-Mo pretrain", "base"),
                ("frankenstallm-bf16", "frankenstallm", "instruct"), ("evafrill-mo-3b", "EVAFRILL-Mo slerp", "instruct")]
USER_LABEL = {m: lab for m, lab, _ in USER_ENTRIES}
FAMILY_NAME = {1: "JANSORI", 2: "frankenstallm", 3: "EVAFRILL-Mo"}
_LABEL_RE = None


def display(text: str) -> str:
    """문자열 안의 사용자 모델 id를 표시 이름으로 (긴 id부터, 하이픈으로 이어지는 더 긴 id는 건드리지 않음)."""
    import re

    global _LABEL_RE
    if _LABEL_RE is None:
        ids = sorted(USER_LABEL, key=len, reverse=True)
        _LABEL_RE = re.compile(r"(?<![\w-])(" + "|".join(map(re.escape, ids)) + r")(?![\w.-])")
    return _LABEL_RE.sub(lambda m_: USER_LABEL[m_.group(1)], text) if isinstance(text, str) else text


INSTRUCT_VIEWS = ("mc_avg_ko_mc_v1_instruct", "mc_table_ko_mc_v1_instruct", "mc_avg_ko_mc_ext_v1_instruct", "mc_table_ko_mc_ext_v1_instruct",
                  "mc_avg_en_mc_v1_instruct", "mc_table_en_mc_v1_instruct")
BASE_ONLY = "base 모델(사전학습만, chat 템플릿 없음)이라 지시 수행·대화 평가 대상이 아님 — 코드는 base 이어쓰기 표, 지식·언어는 객관식·PPL 표"


def miss_reason(vid: str, s: int) -> str | None:
    """보기 vid에 사용자 모델 계열 s(1 JANSORI, 2 frankenstallm, 3 EVAFRILL)가 없을 때의 사유. None이면 주석을 달지 않는다."""
    if vid in ("models", "scn_transcripts", "cover", "user_card") or vid.startswith("ppl_"):
        return None
    if vid.startswith("uni_"):
        return "통일 조건 측정 없음"
    if vid.endswith("_moe"):
        return "dense 3B급 — dense 패널에 표시"
    if vid.startswith(("quant_", "perf_engine")):
        return "Mamba-2 하이브리드 구조라 GGUF·llama.cpp 변환 불가" if s == 3 else None
    if vid == "perf_prefix":
        return "자체 증분 엔진(정적 배치)에 prefix 캐시 기능이 없음" if s == 3 else None
    if "think" in vid:
        return "thinking 모드가 없는 모델"
    if vid == "mc_b_effect":
        return "dev 선택 결과 A 유지(B = A) — 별도 B 실행 없음"
    if vid in INSTRUCT_VIEWS:
        return "base 모델 — 같은 스위트의 base 표에 있음" if s == 1 else None
    if vid.startswith("mc_") and vid.endswith("_base"):
        return "instruct 모델 — 같은 스위트의 instruct 표에 있음" if s == 2 else None
    if vid.startswith("gen_table") and vid.endswith("_base"):
        if s == 2:
            return "instruct 모델 — 코드는 instruct 표의 HumanEval+·MBPP+(chat 형식)"
        return "base 체크포인트는 운용 설정 탐색 대상이 아님(B = A, A 표 참고)" if s == 3 and "__B" in vid else None
    if vid.startswith(("gen_", "tune_", "scn_")):
        return BASE_ONLY if s == 1 else "측정 없음"
    return "측정 없음"


def _view_slots(x: dict) -> set:
    t = x["type"]
    if t == "table":
        return {(r.get("model") or {}).get("slot") for r in x["rows"]}
    if t in ("dots", "bars", "dumbbell"):
        return {r.get("slot") for r in x["rows"] if r.get("v", r.get("b")) is not None}
    if t == "lines":
        return {s_["slot"] for s_ in x["series"] if s_["points"]}
    if t == "heatmap":
        return set(x["slots"])
    if t == "radar":
        return {m["slot"] for m in x["models"]}
    if t == "scatter":
        return {p_["slot"] for p_ in x["points"]}
    return set()


def finalize(views: dict) -> dict:
    for vid, x in views.items():
        t = x["type"]
        # 1) 표시 이름
        if t == "table":
            for r in x["rows"]:
                if r.get("model"):
                    r["model"]["v"] = display(r["model"]["v"])
        elif t in ("dots", "bars", "dumbbell"):
            for r in x["rows"]:
                r["label"] = display(r["label"])
                if "group" in r:
                    r["group"] = display(r["group"])
                if r.get("tip"):
                    r["tip"] = display(r["tip"])
        elif t == "lines":
            for s_ in x["series"]:
                s_["name"] = display(s_["name"])
        elif t == "heatmap":
            x["rows"] = [display(r) for r in x["rows"]]
        elif t == "radar":
            for m in x["models"]:
                m["name"] = display(m["name"])
        elif t == "scatter":
            for p_ in x["points"]:
                p_["label"] = display(p_["label"])
                p_["tip"] = display(p_.get("tip", ""))
        # 2) 빠진 사용자 모델 사유
        have = _view_slots(x)
        miss = [(FAMILY_NAME[s_], r) for s_ in (1, 2, 3) if s_ not in have and (r := miss_reason(vid, s_))]
        if t == "table" and not x.get("nopin"):
            # 사용자 모델 행을 위로(공개 모델 순서는 유지), 없는 사용자 모델은 사유 행
            x["rows"] = [r for r in x["rows"] if (r.get("model") or {}).get("slot")] + \
                        [{"model": {"v": n, "slot": [k for k, v_ in FAMILY_NAME.items() if v_ == n][0]}, "_note": r} for n, r in miss] + \
                        [r for r in x["rows"] if not (r.get("model") or {}).get("slot")]
        elif miss:
            x["missing"] = [list(m) for m in miss]
    return views


def _rank(vals: dict, m: str, lower: bool) -> tuple[int, int] | None:
    if m not in vals or vals[m] is None:
        return None
    xs = sorted((v for v in vals.values() if v is not None), reverse=not lower)
    return xs.index(vals[m]) + 1, len(xs)


def _track_metrics(d: dict) -> list[dict]:
    """사용자 카드·커버리지 표의 지표. 각 지표는 비교군(같은 종류) 안의 {model: 값}."""
    out = []
    kinds = {m: s["kind"] for m, s in d["models"].items()}
    for suite, lab in (("ko_mc_v1", "한국어 객관식 평균"), ("ko_mc_ext_v1", "한국어 확장 객관식 평균"), ("en_mc_v1", "영어 객관식 평균")):
        rows = d["mc"].get(suite) or []
        acc = {}
        for r in rows:
            if _num(r["micro"]) is not None:
                acc.setdefault(r["model"], []).append(r["micro"])
        for kind in ("base", "instruct"):
            vals = {m: 100 * sum(v) / len(v) for m, v in acc.items() if kinds.get(m) == kind}
            out.append({"area": "객관식", "label": lab, "kind": kind, "vals": vals, "fmt": "pct", "unit": "%", "lower": False, "cover": "mc"})
    rows = [r for r in d["mc"].get("ppl_v1") or [] if r["task"] == "wiki40b_ko" and _num(r["micro"]) is not None]
    for kind in ("base", "instruct"):
        out.append({"area": "언어 모델링", "label": "한국어 PPL (bits/byte)", "kind": kind, "fmt": "f3", "unit": "bpb", "lower": True, "cover": "ppl",
                    "vals": {r["model"]: r["micro"] for r in rows if kinds.get(r["model"]) == kind}})
    ga = d["gen"].get("gen_v1__A") or []
    for task, area in (("hrm8k", "생성(규칙)"), ("ifeval_ko", "생성(규칙)"), ("kormedmcqa", "생성(규칙)"), ("kobbq", "생성(규칙)"),
                       ("komt", "judge"), ("logickor", "judge"), ("ko_arena_hard", "judge"), ("humaneval_plus", "코드"), ("mbpp_plus", "코드")):
        for kind in (("instruct", "base") if area == "코드" else ("instruct",)):
            vals = {r["model"]: (r["value"] if task in JUDGE_SCALE else 100 * r["value"]) for r in ga
                    if r["task"] == task and r["kind"] == kind and _num(r["value"]) is not None}
            if vals:
                out.append({"area": area, "label": tl(task) + (" (base 이어쓰기)" if kind == "base" else ""), "kind": kind, "vals": vals,
                            "fmt": "f2" if task in JUDGE_SCALE else "pct", "unit": "점" if task in JUDGE_SCALE else "%", "lower": False,
                            "cover": "judge" if area == "judge" else ("code" if area == "코드" else "gen")})
    sc = {r["model"]: 100 * r["turn_pass"] for r in d.get("scenario") or [] if r["task"] == "scenario_self" and r["variant"] == "A"
          and _num(r.get("turn_pass")) is not None}
    if sc:
        out.append({"area": "다턴", "label": "다턴 턴 통과율 (자기 이력)", "kind": "instruct", "vals": sc, "fmt": "pct", "unit": "%", "lower": False,
                    "cover": "scenario"})
    # 속도·메모리: dense 3B급 BF16(vLLM, EVAFRILL은 자체 엔진) 비교군
    pr = [r for r in d.get("perf") or [] if not r.get("error") and r.get("engine") in ("vllm", "evafrill")
          and r.get("run") in ("prefix_cache_on", "static_batch")]
    dense = lambda r: r["params_b"] == r["active_b"] or r["active_b"] is None or r["params_b"] < 8
    for c, lab in ((1, "출력 속도 · 동시 1"), (64, "출력 처리량 · 동시 64")):
        vals = {r["model"]: r["output_tok_per_s"] for r in pr if r["workload"] == "w1" and r["concurrency"] == c and dense(r)}
        out.append({"area": "속도", "label": lab, "kind": "dense", "vals": vals, "fmt": "f0", "unit": "tok/s", "lower": False, "cover": "perf"})
    vals = {r["model"]: r["weights_gib"] for r in pr if r["workload"] == "w1" and r["concurrency"] == 1 and dense(r) and _num(r.get("weights_gib"))}
    out.append({"area": "메모리", "label": "가중치 메모리", "kind": "dense", "vals": vals, "fmt": "f2", "unit": "GiB", "lower": True, "cover": "perf"})
    return out


def _fmtv(x, f):
    return {"pct": f"{x:.1f}", "f2": f"{x:.2f}", "f3": f"{x:.3f}", "f0": f"{x:,.0f}"}.get(f, f"{x:.3g}")


def user_views(d: dict) -> dict:
    mets = _track_metrics(d)
    kinds = {m: s["kind"] for m, s in d["models"].items()}
    # 사용자 카드: 지표(행) × 사용자 모델 4항목(열), 값 · 같은 비교군 순위
    cols = [{"key": "area", "label": "영역"}, {"key": "metric", "label": "지표"}] + \
           [{"key": m, "label": f"{lab} ({kind})"} for m, lab, kind in USER_ENTRIES]
    rows = []
    seen = set()
    for mt in mets:
        key = (mt["area"], mt["label"])
        if key in seen:
            continue
        seen.add(key)
        same = [x for x in mets if (x["area"], x["label"]) == key]
        row = {"area": {"v": mt["area"]}, "metric": {"v": f"{mt['label']}" + (" ↓" if mt["lower"] else "")}}
        for m, lab, kind in USER_ENTRIES:
            grp = next((x for x in same if x["kind"] == kind or x["kind"] == "dense"), None)
            if grp is None:
                row[m] = {"v": "해당 없음", "muted": True,
                          "tip": f"{lab}: {BASE_ONLY if kind == 'base' else 'instruct 모델 — 같은 지표의 instruct 행 참고'}"}
                continue
            if m not in grp["vals"] and grp["kind"] == "dense" and kind == "base" and m.startswith("evafrill"):
                row[m] = {"v": "slerp 열 참고", "muted": True, "tip": "속도·메모리는 같은 구조라 최종 체크포인트(slerp)로만 측정"}
                continue
            rk = _rank(grp["vals"], m, grp["lower"])
            if rk is None:
                row[m] = {"v": "미측정", "warn": True, "tip": f"{lab}: 이 지표의 결과 파일이 없음"}
                continue
            xs = [v for v in grp["vals"].values() if v is not None]
            best = (min if grp["lower"] else max)(grp["vals"], key=lambda k: grp["vals"][k])
            grp_name = {"base": "base", "instruct": "instruct", "dense": "dense 3B급"}[grp["kind"]]
            row[m] = {"v": f"{_fmtv(grp['vals'][m], grp['fmt'])} · {rk[0]}/{rk[1]}위", "slot": slot(m),
                      "tip": f"{lab} · {mt['label']}: {_fmtv(grp['vals'][m], grp['fmt'])} {grp['unit']} — {grp_name} {rk[1]}종 중 {rk[0]}위, "
                             f"중앙값 {_fmtv(median(xs), grp['fmt'])}, 최고 {_fmtv(grp['vals'][best], grp['fmt'])} ({display(best)})"}
        rows.append(row)
    card = {"type": "table", "title": "사용자 모델 4항목 한눈에: 값 · 같은 비교군 안 순위", "columns": cols, "rows": rows, "sort": None,
            "nopin": True}
    # 커버리지: 모델(행) × 평가 영역(열)
    areas = [("mc", "객관식"), ("ppl", "PPL"), ("gen", "생성(규칙)"), ("judge", "judge"), ("code", "코드"), ("scenario", "다턴"),
             ("tune", "운용 설정"), ("think", "thinking"), ("quant", "GGUF 양자화"), ("perf", "속도·메모리")]
    have = {a: set() for a, _ in areas}
    for mt in mets:
        have[mt["cover"]].update(mt["vals"])
    have["perf"].update(r["model"] for r in d.get("perf") or [] if not r.get("error"))
    have["tune"].update(r["model"] for r in d["gen"].get("tune_v3") or [])
    have["think"].update(r["model"] for r in d["gen"].get("gen_v1__think") or [])
    qfam = {r["family"] for r in d["mc"].get("ko_mc_v1__quant") or []}
    thinking = set((d.get("gen_config") or {}).get("thinking", {}).get("models", []) or [])
    order = [m for m, _, _ in USER_ENTRIES] + sorted([m for m, s in d["models"].items() if s["owner"] != "self" and s.get("backend") != "llamacpp"],
                                                    key=lambda m: (kinds[m] != "base", m))
    crow = []
    for m in order:
        if m not in d["models"]:
            continue
        k = kinds[m]
        r = {"model": {"v": display(m), "slot": slot(m)}, "kind": {"v": k}}
        for a, lab in areas:
            if m in have[a]:
                r[a] = {"v": "✓"}
            elif a == "quant":
                fam = d["models"][m]["family"]
                r[a] = ({"v": "✓"} if fam in qfam else {"v": "구조상 불가", "muted": True, "tip": "Mamba-2 하이브리드 — GGUF 미지원"}) if slot(m) \
                    else {"v": "대상 아님", "muted": True, "tip": "양자화 비교는 사용자 모델 배포 GGUF만"}
            elif a == "perf" and m == "evafrill-mo-3b-pretrain":
                r[a] = {"v": "slerp와 같음", "muted": True, "tip": "같은 구조라 최종 체크포인트로만 측정"}
            elif a == "think":
                r[a] = {"v": "모드 없음" if m not in thinking else "미측정", "muted": True}
            elif a in ("gen", "judge", "scenario", "tune") and k == "base":
                r[a] = {"v": "base", "muted": True, "tip": "base 모델은 지시 수행·대화 평가 대상 아님"}
            elif a == "code" and k == "base":
                r[a] = {"v": "미측정", "warn": True}
            else:
                r[a] = {"v": "미측정", "warn": True}
        crow.append(r)
    cover = {"type": "table", "title": "평가 범위: 모델 × 평가 영역 (✓ 측정, 회색 = 해당 없음과 사유)", "nopin": True, "sort": None,
             "columns": [{"key": "model", "label": "모델"}, {"key": "kind", "label": "종류"}] + [{"key": a, "label": lab} for a, lab in areas],
             "rows": crow}
    return {"user_card": card, "cover": cover}


# ── 통일 비교: 모든 모델 한 라인 (configs/standard/unified.yaml 정규화·영역) ─────────────────────
JUDGE10 = {"komt", "logickor"}


def _se(lo, hi):
    return (hi - lo) / (2 * 1.96) if _num(lo) is not None and _num(hi) is not None else 0.0


def uni_scores(d: dict) -> dict:
    """모델별 정규화 태스크 점수(0~1)·영역 점수·종합 점수. 종합은 '모든 모델에 값이 있는 영역'만 같은 가중치로 평균한다."""
    cfg = d.get("unified") or {}
    chance = cfg.get("chance", {})
    areas = cfg.get("areas", {})
    kinds = {m: s["kind"] for m, s in d["models"].items()}
    tasks: dict = {}

    def put(m, t, v, lo, hi):
        tasks.setdefault(m, {})[t] = (v, lo, hi)

    for suite in ("ko_mc_v1", "ko_mc_ext_v1", "en_mc_v1", "uni_mc_v1"):
        acc = {}
        for r in d["mc"].get(suite) or []:
            if _num(r.get("micro")) is None or r["task"] not in chance:
                continue
            c = chance[r["task"]]
            f = lambda x: max(0.0, (x - c) / (1 - c))                    # noqa: E731  무작위 보정
            if suite == "uni_mc_v1":
                put(r["model"], r["task"], f(r["micro"]), f(r["ci_lo"]), f(r["ci_hi"]))
            else:
                acc.setdefault(r["model"], []).append((f(r["micro"]), _se(f(r["ci_lo"]), f(r["ci_hi"]))))
        for m, xs in acc.items():                                           # 스위트 = 태스크 평균 1개
            v = sum(x for x, _ in xs) / len(xs)
            se = math.sqrt(sum(e * e for _, e in xs)) / len(xs)
            put(m, suite, v, v - 1.96 * se, v + 1.96 * se)
    ppl = [r for r in d["mc"].get("ppl_v1") or [] if r["task"] == "wiki40b_ko" and _num(r.get("micro")) is not None]
    if ppl:
        best = min(r["micro"] for r in ppl)
        for r in ppl:
            put(r["model"], "ppl_wiki40b_ko", best / r["micro"], best / r["ci_hi"], best / r["ci_lo"])
    for r in d["gen"].get("uni_v1__U") or []:
        if _num(r.get("value")) is None:
            continue
        t = r["task"]
        f = (lambda x: x / 10) if t in JUDGE10 else (lambda x: (x - 1) / 9) if t == "k2_generation" else (lambda x: x)
        put(r["model"], t, f(r["value"]), f(r["ci_lo"]), f(r["ci_hi"]))
    for r in d.get("scenario") or []:
        if r.get("variant") == "U" and r["task"] == "scenario_self" and _num(r.get("turn_pass")) is not None:
            put(r["model"], "scenario_self", r["turn_pass"], r["turn_pass"], r["turn_pass"])
    models = [m for m in tasks if not (m.startswith("evafrill-mo-3b-") and m != "evafrill-mo-3b-pretrain")]
    out = {}
    for m in models:
        ar = {}
        for a, ts in areas.items():
            xs = [tasks[m][t] for t in ts if t in tasks[m]]
            if xs and len(xs) == len([t for t in ts if any(t in tasks[mm] for mm in models)]):   # 영역 안 측정된 태스크를 모두 가진 경우만
                v = sum(x[0] for x in xs) / len(xs)
                ar[a] = (v, math.sqrt(sum(_se(x[1], x[2]) ** 2 for x in xs)) / len(xs))
        out[m] = {"kind": kinds.get(m), "tasks": tasks[m], "areas": ar}
    common = [a for a in areas if out and all(a in o["areas"] for o in out.values())]
    for o in out.values():
        xs = [o["areas"][a] for a in common]
        if xs:
            v = sum(x[0] for x in xs) / len(xs)
            se = math.sqrt(sum(x[1] ** 2 for x in xs)) / len(xs)
            o["comp"] = (v, v - 1.96 * se, v + 1.96 * se)
    return {"models": out, "common": common, "areas": list(areas)}


def uni_views(d: dict) -> dict:
    u = uni_scores(d)
    M, common = u["models"], u["common"]
    if not M or not common:
        return {}
    v = {}
    rk = sorted((m for m in M if "comp" in M[m]), key=lambda m: -M[m]["comp"][0])
    area_txt = " · ".join(common)
    v["uni_rank"] = {"type": "dots", "title": f"통일 종합 점수 (모든 모델, 영역 {len(common)}개 같은 가중치: {area_txt})", "unit": "", "digits": 3,
                     "xlabel": "종합 점수 (0~1, 영역별 정규화 점수의 평균, 95% CI 근사)",
                     "rows": [{"label": m, "v": M[m]["comp"][0], "lo": M[m]["comp"][1], "hi": M[m]["comp"][2], "slot": slot(m), "kind": M[m]["kind"],
                               "tip": f"{m} ({M[m]['kind']}): {M[m]['comp'][0]:.3f} — " + ", ".join(f"{a} {M[m]['areas'][a][0]:.2f}" for a in common)}
                              for m in rk]}
    allareas = [a for a in u["areas"] if any(a in M[m]["areas"] for m in M)]
    v["uni_area_heat"] = {"type": "heatmap", "title": "영역별 정규화 점수 (0~100, 모든 모델 · 종합 순)", "rows": rk, "cols": allareas,
                          "slots": [slot(m) for m in rk], "kinds": [M[m]["kind"] for m in rk], "unit": "",
                          "values": [[(100 * M[m]["areas"][a][0]) if a in M[m]["areas"] else None for a in allareas] for m in rk],
                          "total": [100 * M[m]["comp"][0] for m in rk], "total_label": "종합"}
    ts = list(dict.fromkeys(t for m in rk for t in M[m]["tasks"]))
    v["uni_task_heat"] = {"type": "heatmap", "title": "태스크별 정규화 점수 (0~100; 객관식은 무작위 보정, PPL은 최고 bpb/bpb)", "rows": rk,
                          "cols": [tl(t) if t in TASK_LABEL else t for t in ts], "slots": [slot(m) for m in rk], "kinds": [M[m]["kind"] for m in rk],
                          "values": [[(100 * M[m]["tasks"][t][0]) if t in M[m]["tasks"] else None for t in ts] for m in rk],
                          "total": [100 * M[m]["comp"][0] for m in rk], "total_label": "종합", "unit": ""}
    pr = {}
    for r in d.get("perf") or []:
        if not r.get("error") and r.get("engine") in ("vllm", "evafrill") and r.get("run") in ("prefix_cache_on", "static_batch") \
                and r.get("workload") == "w1" and r.get("concurrency") == 64:
            pr[r["model"]] = r
    pts_t, pts_a, pts_s = [], [], []
    for m in rk:
        spec = d["models"][m]
        c = M[m]["comp"]
        base = {"label": m, "y": c[0], "lo": c[1], "hi": c[2], "slot": slot(m), "kind": M[m]["kind"]}
        pts_t.append({**base, "x": spec["params_b"], "tip": f"{m}: 종합 {c[0]:.3f}, 총 {spec['params_b']:g}B"})
        pts_a.append({**base, "x": spec["active_b"] or spec["params_b"], "tip": f"{m}: 종합 {c[0]:.3f}, 활성 {spec['active_b'] or spec['params_b']:g}B"})
        pm = pr.get(m) or (pr.get("evafrill-mo-3b") if m == "evafrill-mo-3b-pretrain" else None)
        if pm:
            pts_s.append({**base, "x": pm["output_tok_per_s"], "tip": f"{m}: 종합 {c[0]:.3f}, 동시 64 {pm['output_tok_per_s']:,.0f} tok/s"
                                                                       + (" (slerp 측정값)" if m == "evafrill-mo-3b-pretrain" else "")})
    common_note = f"종합 = 영역 {len(common)}개"
    v["uni_scatter_total"] = {"type": "scatter", "points": pts_t, "xlog": True, "frontier": "lower_x", "xlabel": "총 파라미터 (B, 로그)",
                              "ylabel": "통일 종합 점수", "title": f"종합 점수 × 총 파라미터 (메모리) — 선 = 같은 크기 이하에서 최고 ({common_note})"}
    v["uni_scatter_active"] = {"type": "scatter", "points": pts_a, "xlog": True, "frontier": "lower_x", "xlabel": "활성 파라미터 (B, 로그)",
                               "ylabel": "통일 종합 점수", "title": f"종합 점수 × 활성 파라미터 (토큰당 연산) ({common_note})"}
    if pts_s:
        v["uni_scatter_speed"] = {"type": "scatter", "points": pts_s, "xlog": True, "frontier": "higher_x", "xlabel": "동시 64 출력 처리량 (tok/s, 로그)",
                                  "ylabel": "통일 종합 점수", "title": f"종합 점수 × 처리량 — 선 = 같은 속도 이상에서 최고 ({common_note}, 속도 미측정 모델 제외)"}
    return v


def build_views(d: dict) -> dict:
    v = {}
    for f in (model_view, user_views, uni_views, mc_views, quant_views, gen_views, scenario_views, perf_views):
        v.update(f(d))
    return finalize(v)


# ── 페이지 ───────────────────────────────────────────────────────────────────

SECTIONS = [
    ("summary", "요약", []),
    ("users", "사용자 모델 3종 한눈에", ["user_card", "cover"]),
    ("uni", "통일 비교: 모든 모델을 한 라인에", ["uni_rank", "uni_area_heat", "uni_scatter_total", "uni_scatter_active", "uni_scatter_speed",
                                         "uni_task_heat"]),
    ("scope", "평가 대상과 방법", ["models"]),
    ("mc", "객관식과 PPL (base·instruct 공통 로그우도, 한 라인)", ["mc_avg_ko_mc_v1_all", "mc_avg_ko_mc_ext_v1_all", "mc_avg_en_mc_v1_all",
                                                              "mc_table_ko_mc_v1_all", "mc_table_ko_mc_ext_v1_all", "mc_table_en_mc_v1_all", "ppl_",
                                                              "mc_b_effect"]),
    ("gen", "생성 벤치 (instruct 생성·judge, base 코드 이어쓰기)", ["gen_table_gen_v1__A_instruct", "gen_table_gen_v1__B_instruct", "gen_b_effect",
                                    "gen_table_gen_v1__think_instruct", "gen_table_gen_v1__A_base"]),
    ("tune", "모델별 운용 설정과 오동작", ["tune_mal", "tune_quality", "tune_table"]),
    ("scenario", "다턴 시나리오", ["scn_heat_scenario_self", "scn_turn_scenario_self", "scn_self_oracle", "scn_transcripts",
                                 "scn_heat_scenario_dev_self"]),
    ("quant", "GGUF 양자화 손실", ["quant_ko_mc_v1", "quant_ppl_v1", "quant_table_ko_mc_v1_", "quant_table_ppl_v1"]),
    ("perf", "속도와 메모리 (모든 모델 한 축)", ["perf_output_tok_per_s_all", "perf_ttft_ms_p50_all",
                            "perf_tpot", "perf_prefix", "perf_weights", "perf_kv", "perf_engine", "perf_radar", "perf_geomean"]),
    ("strengths", "강점과 약점", []),
    ("caveats", "오염 가능성, 실패·미확인 항목", []),
    ("repro", "재현 명령과 남은 과제", []),
]


def page(views: dict, narrative: dict, generated_at: str) -> str:
    used = set()
    toc, body = [], []
    for sid, title, vs in SECTIONS:
        toc.append(f'<a href="#{sid}">{html.escape(title)}</a>')
        parts = [f'<section id="{sid}"><h2>{html.escape(title)}</h2>']
        nar = narrative.get(sid)
        if isinstance(nar, dict):
            parts.append(nar.get("intro", ""))
        elif nar:
            parts.append(nar)
        for vid in vs:
            key = next((k for k in views if k == vid or (vid.endswith("_") and k.startswith(vid))), None)
            keys = [k for k in views if k == vid or k.startswith(vid.rstrip("_"))] if vid.endswith("_") else ([key] if key else [])
            for k in keys:
                used.add(k)
                cap = narrative.get(f"view:{k}", "")
                parts.append(f'<figure class="view" id="v-{k}"><figcaption>{html.escape(views[k].get("title", k))}</figcaption>'
                             f'<div class="viz" data-view="{k}"></div>{f"<p class=cap>{cap}</p>" if cap else ""}</figure>')
        if isinstance(nar, dict) and nar.get("outro"):
            parts.append(nar["outro"])
        parts.append("</section>")
        body.append("".join(parts))
    rest = [k for k in views if k not in used]
    if rest:
        body.append('<section id="appendix"><h2>부록: 나머지 표·그림</h2>' + "".join(
            f'<figure class="view" id="v-{k}"><figcaption>{html.escape(views[k].get("title", k))}</figcaption><div class="viz" data-view="{k}"></div></figure>'
            for k in rest) + "</section>")
        toc.append('<a href="#appendix">부록</a>')
    data_js = json.dumps(views, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    title = narrative.get("_title", "한국어 소형 LLM 심층 평가")
    sub = narrative.get("_subtitle", "")
    return TEMPLATE.replace("%%TITLE%%", html.escape(title)).replace("%%SUB%%", sub).replace("%%TOC%%", "".join(toc)) \
        .replace("%%BODY%%", "\n".join(body)).replace("%%DATA%%", data_js).replace("%%GEN%%", html.escape(generated_at)).replace("%%D3%%", D3)


TEMPLATE = r"""<title>%%TITLE%%</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@500;600;700&family=Noto+Sans+KR:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* 레이아웃: 한 열 본문(최대 960px) + 상단 고정 목차. 차트는 본문 폭, 넓은 표만 가로 스크롤. */
:root{
  --bg:#f7f8fa; --surface:#fdfdfc; --ink:#14171c; --ink2:#4d535c; --muted:#868b93; --rule:#e2e4e8; --axis:#c3c6cc;
  --accent:#2f4a7a; --chip:#eef1f6; --warn:#b4541c; --good:#1f7a3a;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --pub:#9aa0a8; --publine:#b9bdc4; --median:#6f747c;
  --f-display:"IBM Plex Sans KR","Noto Sans KR",system-ui,sans-serif; --f-body:"Noto Sans KR",system-ui,-apple-system,"Segoe UI",sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;
  --bg:#111317; --surface:#181b20; --ink:#eceef1; --ink2:#b7bcc4; --muted:#8a9099; --rule:#2a2e35; --axis:#3a3f47;
  --accent:#9db5e3; --chip:#222731; --warn:#f08a50; --good:#4cc46e; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --pub:#6c727b; --publine:#4c525a; --median:#a6abb3;}}
:root[data-theme="dark"]{color-scheme:dark;
  --bg:#111317; --surface:#181b20; --ink:#eceef1; --ink2:#b7bcc4; --muted:#8a9099; --rule:#2a2e35; --axis:#3a3f47;
  --accent:#9db5e3; --chip:#222731; --warn:#f08a50; --good:#4cc46e; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --pub:#6c727b; --publine:#4c525a; --median:#a6abb3;}
body{background:var(--bg);color:var(--ink);font-family:var(--f-body);font-size:15px;line-height:1.65;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));}
.rep{max-width:960px;margin:0 auto;padding-block:28px 64px;}
header.top h1{font-family:var(--f-display);font-size:30px;line-height:1.25;margin:0 0 6px;text-wrap:balance;letter-spacing:-0.01em}
header.top .sub{color:var(--ink2);margin:0 0 4px;max-width:70ch}
header.top .meta{color:var(--muted);font-size:12.5px;font-family:var(--f-mono)}
nav.toc{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:var(--bg);display:flex;flex-wrap:wrap;gap:6px;padding-block:10px;
  border-bottom:1px solid var(--rule);margin:18px 0 8px}
body::before{content:"";position:fixed;inset:0 0 auto;height:env(safe-area-inset-top,0px);z-index:10;background:var(--bg)}
nav.toc a{font-size:12.5px;color:var(--ink2);text-decoration:none;background:var(--chip);padding:3px 9px;border-radius:999px}
nav.toc a:hover,nav.toc a:focus-visible{color:var(--ink);outline:2px solid var(--accent);outline-offset:1px}
section{padding-top:18px}
h2{font-family:var(--f-display);font-size:21px;margin:22px 0 10px;text-wrap:balance}
h3{font-family:var(--f-display);font-size:16.5px;margin:18px 0 6px}
p,li{max-width:72ch} p{margin:8px 0}
code,.mono{font-family:var(--f-mono);font-size:.88em}
pre{font-family:var(--f-mono);font-size:12.5px;background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:10px 12px;overflow-x:auto}
figure.view{margin:18px 0 22px;background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:12px 14px}
figcaption{font-weight:600;font-size:14px;margin-bottom:6px;color:var(--ink)}
p.cap{font-size:13px;color:var(--ink2);margin:6px 0 0}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12.5px;color:var(--ink2);margin:2px 0 6px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:5px}
.legend i.dot{width:9px;height:9px;border-radius:50%}
.tblwrap{overflow-x:auto}
table.t{border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums;min-width:100%}
table.t th,table.t td{padding:4px 8px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
table.t th{position:sticky;top:0;background:var(--surface);color:var(--ink2);font-weight:600;cursor:pointer;user-select:none}
table.t th:first-child,table.t td:first-child,table.t th:nth-child(2),table.t td:nth-child(2){text-align:left}
table.t td.flag::after{content:" †";color:var(--muted)}
table.t td.warn{color:var(--warn);font-weight:600}
table.t td.sig{font-weight:700}
table.t tr:hover td{background:var(--chip)}
.sw{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle;background:var(--pub)}
svg text{fill:var(--ink2);font-family:var(--f-body);font-size:11.5px}
svg .axis line,svg .axis path{stroke:var(--axis)} svg .grid line{stroke:var(--rule)}
svg .lab{fill:var(--ink);font-size:12px}
#tip{position:fixed;pointer-events:none;z-index:50;background:var(--surface);color:var(--ink);border:1px solid var(--rule);border-radius:6px;
  padding:6px 9px;font-size:12.5px;max-width:340px;box-shadow:0 4px 14px rgba(0,0,0,.12);display:none}
.tr-ctl{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px}
.tr-ctl select{font:inherit;font-size:13px;padding:3px 6px;background:var(--surface);color:var(--ink);border:1px solid var(--rule);border-radius:5px}
.turn{border-top:1px solid var(--rule);padding:8px 0}
.turn .u{color:var(--ink2);font-size:13px;white-space:pre-wrap}
.turn .a{white-space:pre-wrap;font-size:13px;background:var(--bg);border-radius:6px;padding:6px 9px;margin-top:4px;max-height:260px;overflow:auto}
.pill{display:inline-block;font-size:11px;padding:0 7px;border-radius:999px;margin-left:6px}
.pill.ok{background:color-mix(in srgb,var(--good) 18%,transparent);color:var(--good)}
.pill.ng{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}
.note{font-size:13px;color:var(--ink2)}
.note.miss{border-left:3px solid var(--axis);padding-left:8px}
td.na{color:var(--muted);font-size:12.5px}
@media (prefers-reduced-motion: reduce){*{transition:none!important}}
</style>
<div class="rep">
<header class="top"><h1>%%TITLE%%</h1><p class="sub">%%SUB%%</p><p class="meta">데이터 생성 %%GEN%%</p></header>
<nav class="toc" aria-label="목차">%%TOC%%</nav>
%%BODY%%
</div>
<div id="tip" role="tooltip"></div>
<script src="%%D3%%"></script>
<script>
const V=%%DATA%%;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const SL=s=>s?css('--s'+s):css('--pub');
const tip=document.getElementById('tip');
function showTip(ev,t){if(!t)return;tip.textContent=t;tip.style.display='block';const w=tip.offsetWidth,h=tip.offsetHeight;
  let x=ev.clientX+12,y=ev.clientY+12;if(x+w>innerWidth-8)x=ev.clientX-w-12;if(y+h>innerHeight-8)y=ev.clientY-h-12;tip.style.left=x+'px';tip.style.top=y+'px';}
function hideTip(){tip.style.display='none';}
const F={pct:v=>v.toFixed(1),f1:v=>v.toFixed(1),f2:v=>v.toFixed(2),f4:v=>v.toFixed(4),d2:v=>(v>0?'+':'')+v.toFixed(2),d4:v=>(v>0?'+':'')+v.toFixed(4)};
function fmt(c){if(c==null||c.v==null)return'—';if(typeof c.v!=='number')return String(c.v);return(F[c.fmt]||(v=>String(+v.toFixed(3))))(c.v);}
function legendUser(el,items){const L=document.createElement('div');L.className='legend';
  items.forEach(([name,color,dot])=>{const s=document.createElement('span');const i=document.createElement('i');if(dot)i.className='dot';i.style.background=color;s.append(i,name);L.append(s);});el.append(L);}
function userLegend(el,labels){const seen=new Map();labels.forEach(([m,s])=>{if(s&&!seen.has(s))seen.set(s,m)});
  const it=[...seen].sort().map(([s,m])=>[m.replace(/-bf16$/,''),SL(s),true]);it.push(['공개 모델',css('--pub'),true]);legendUser(el,it);}

const R={};
R.table=(el,v)=>{const w=document.createElement('div');w.className='tblwrap';const t=document.createElement('table');t.className='t';
  let rows=v.rows.slice();let sortKey=v.sort,dir=-1;
  function draw(){t.innerHTML='';const th=t.createTHead().insertRow();v.columns.forEach(c=>{const h=document.createElement('th');h.textContent=c.label+(sortKey===c.key?(dir<0?' ▾':' ▴'):'');h.tabIndex=0;
      const go=()=>{if(sortKey===c.key)dir=-dir;else{sortKey=c.key;dir=-1}rows.sort((a,b)=>{const x=a[c.key]?.v,y=b[c.key]?.v;if(x==null)return 1;if(y==null)return-1;return(typeof x==='number'?(x-y):String(x).localeCompare(String(y)))*dir});draw();};
      h.onclick=go;h.onkeydown=e=>{if(e.key==='Enter')go()};th.append(h);});
    const tb=t.createTBody();rows.forEach(r=>{const tr=tb.insertRow();
      if(r._note){const td=tr.insertCell();const s=document.createElement('span');s.className='sw';s.style.background=SL(r.model.slot);td.append(s,r.model.v);
        const tn=tr.insertCell();tn.colSpan=v.columns.length-1;tn.className='na';tn.textContent='해당 없음 — '+r._note;return;}
      v.columns.forEach((c,i)=>{const td=tr.insertCell();const cell=r[c.key];
      if(i===0&&cell){const s=document.createElement('span');s.className='sw';s.style.background=SL(cell.slot);td.append(s);}
      td.append(fmt(cell));if(cell){if(cell.muted)td.classList.add('na');if(cell.flag)td.classList.add('flag');if(cell.warn)td.classList.add('warn');if(cell.sig)td.classList.add('sig');
        if(cell.tip){td.onmousemove=e=>showTip(e,cell.tip);td.onmouseleave=hideTip;}}});});}
  draw();w.append(t);el.append(w);};

function frame(el,h,ml){const W=Math.max(320,el.clientWidth);const m={l:ml||150,r:24,t:8,b:36};const svg=d3.select(el).append('svg').attr('width',W).attr('height',h).attr('viewBox',`0 0 ${W} ${h}`);return{svg,W,m};}
function trunc(s,n){return s.length>n?s.slice(0,n-1)+'…':s;}

R.dots=(el,v)=>{const rows=v.rows.filter(r=>r.v!=null);if(!rows.length)return;userLegend(el,rows.map(r=>[r.label.split(' · ')[0],r.slot]));
  const narrow=el.clientWidth<520;const rh=v.grouped?16:19;let groups=[];
  if(v.grouped){const g=d3.group(rows,r=>r.group);groups=[...g];}
  const lines=v.grouped?groups.reduce((n,[k,a])=>n+a.length+1,0):rows.length;
  const ml=narrow?120:Math.min(260,8+7*d3.max(rows,r=>(v.grouped?r.label.split(' · ').slice(1).join(' · '):r.label).length));
  const {svg,W,m}=frame(el,lines*rh+m0(),ml);function m0(){return 52}
  const ext=d3.extent(rows.flatMap(r=>[r.lo??r.v,r.hi??r.v]).concat(v.zero?[0]:[]));const pad=(ext[1]-ext[0])*0.06||1;
  const x=d3.scaleLinear().domain([ext[0]-pad,ext[1]+pad]).nice().range([m.l,W-m.r]);
  const g=svg.append('g');g.append('g').attr('class','grid').selectAll('line').data(x.ticks(6)).join('line').attr('x1',x).attr('x2',x).attr('y1',m.t).attr('y2',lines*rh+m.t);
  g.append('g').attr('class','axis').attr('transform',`translate(0,${lines*rh+m.t+4})`).call(d3.axisBottom(x).ticks(narrow?4:6).tickSizeOuter(0));
  svg.append('text').attr('x',(m.l+W-m.r)/2).attr('y',lines*rh+m.t+36).attr('text-anchor','middle').text(v.xlabel||'');
  if(v.zero)g.append('line').attr('x1',x(0)).attr('x2',x(0)).attr('y1',m.t).attr('y2',lines*rh+m.t).attr('stroke',css('--muted')).attr('stroke-dasharray','3 3');
  let yi=0;const place=[];
  if(v.grouped){groups.forEach(([k,a])=>{place.push({head:k,y:m.t+yi*rh+rh*0.7});yi++;a.forEach(r=>{place.push({r,y:m.t+yi*rh+rh/2,lab:r.label.split(' · ').slice(1).join(' · ')});yi++;});});}
  else rows.forEach(r=>{place.push({r,y:m.t+yi*rh+rh/2,lab:r.label});yi++;});
  place.forEach(p=>{if(p.head){svg.append('text').attr('class','lab').attr('x',4).attr('y',p.y).style('font-weight',600).text(trunc(p.head,60));return;}
    const r=p.r,c=SL(r.slot);const row=svg.append('g');
    row.append('text').attr('x',m.l-8).attr('y',p.y+4).attr('text-anchor','end').attr('class',r.slot?'lab':'').text(trunc(p.lab,narrow?16:40));
    if(r.lo!=null)row.append('line').attr('x1',x(r.lo)).attr('x2',x(r.hi)).attr('y1',p.y).attr('y2',p.y).attr('stroke',c).attr('stroke-width',2).attr('stroke-linecap','round');
    const sig=v.zero&&r.lo!=null&&(r.lo>0||r.hi<0);
    const fillc=sig||!v.zero?c:css('--surface');
    if(r.kind==='base')row.append('rect').attr('x',x(r.v)-4.5).attr('y',p.y-4.5).attr('width',9).attr('height',9).attr('rx',1.5).attr('fill',fillc).attr('stroke',c).attr('stroke-width',2);
    else row.append('circle').attr('cx',x(r.v)).attr('cy',p.y).attr('r',4.5).attr('fill',fillc).attr('stroke',c).attr('stroke-width',2);
    row.append('rect').attr('x',0).attr('width',W).attr('y',p.y-rh/2).attr('height',rh).attr('fill','transparent')
      .on('mousemove',e=>showTip(e,r.tip||`${r.label}: ${r.v.toFixed(v.digits??1)}${v.unit||''}`)).on('mouseleave',hideTip);});
  if(v.zero){const n=document.createElement('p');n.className='note';n.textContent='채운 점 = 95% CI가 0을 포함하지 않음(유의), 빈 점 = 포함.';el.append(n);}
  if(rows.some(r=>r.kind==='base')&&rows.some(r=>r.kind&&r.kind!=='base')){const n=document.createElement('p');n.className='note';n.textContent='■ 네모 = base(사전학습) 모델, ● 원 = instruct 모델.';el.append(n);}};

R.dumbbell=(el,v)=>{const rows=v.rows;if(!rows.length)return;userLegend(el,rows.map(r=>[r.label,r.slot]));
  legendUser(el,[[v.alabel,css('--muted'),true],[v.blabel+' (모델 색)',css('--ink2'),true]]);
  const narrow=el.clientWidth<520;const rh=20;const ml=narrow?120:Math.min(240,8+7*d3.max(rows,r=>r.label.length));
  const {svg,W,m}=frame(el,rows.length*rh+52,ml);const vals=rows.flatMap(r=>[r.a,r.b]);
  const x=(v.xlog?d3.scaleLog().domain([d3.min(vals)*0.8,d3.max(vals)*1.2]):d3.scaleLinear().domain([Math.min(0,d3.min(vals)),d3.max(vals)*1.05])).nice().range([m.l,W-m.r]);
  const H=rows.length*rh+m.t;svg.append('g').attr('class','grid').selectAll('line').data(x.ticks(v.xlog?4:6)).join('line').attr('x1',x).attr('x2',x).attr('y1',m.t).attr('y2',H);
  svg.append('g').attr('class','axis').attr('transform',`translate(0,${H+4})`).call(d3.axisBottom(x).ticks(v.xlog?4:6,v.xlog?'~s':undefined).tickSizeOuter(0));
  svg.append('text').attr('x',(m.l+W-m.r)/2).attr('y',H+36).attr('text-anchor','middle').text((v.unit||'')+(v.lower_better?' · 낮을수록 좋음':''));
  rows.forEach((r,i)=>{const y=m.t+i*rh+rh/2,c=SL(r.slot);const g=svg.append('g');
    g.append('text').attr('x',m.l-8).attr('y',y+4).attr('text-anchor','end').attr('class',r.slot?'lab':'').text(trunc(r.label,narrow?16:34));
    g.append('line').attr('x1',x(r.a)).attr('x2',x(r.b)).attr('y1',y).attr('y2',y).attr('stroke',c).attr('stroke-width',2).attr('opacity',.55);
    g.append('circle').attr('cx',x(r.a)).attr('cy',y).attr('r',4).attr('fill',css('--surface')).attr('stroke',css('--muted')).attr('stroke-width',2);
    g.append('circle').attr('cx',x(r.b)).attr('cy',y).attr('r',4.5).attr('fill',c);
    const d=v.digits??1;g.append('rect').attr('x',0).attr('width',W).attr('y',y-rh/2).attr('height',rh).attr('fill','transparent')
      .on('mousemove',e=>showTip(e,r.tip||`${r.label}: ${v.alabel} ${(+r.a).toFixed(d)} → ${v.blabel} ${(+r.b).toFixed(d)} ${v.unit||''}`)).on('mouseleave',hideTip);});};

R.bars=(el,v)=>{const rows=v.rows.filter(r=>r.v!=null);if(!rows.length)return;userLegend(el,rows.map(r=>[r.label,r.slot]));
  const narrow=el.clientWidth<520;const rh=19;const ml=narrow?120:Math.min(240,8+7*d3.max(rows,r=>r.label.length));
  const {svg,W,m}=frame(el,rows.length*rh+44,ml);const x=d3.scaleLinear().domain([0,d3.max(rows,r=>r.v)*1.08]).nice().range([m.l,W-m.r-40]);
  const H=rows.length*rh+m.t;svg.append('g').attr('class','axis').attr('transform',`translate(0,${H+4})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
  rows.forEach((r,i)=>{const y=m.t+i*rh+2,c=SL(r.slot);const g=svg.append('g');
    g.append('text').attr('x',m.l-8).attr('y',y+rh/2+2).attr('text-anchor','end').attr('class',r.slot?'lab':'').text(trunc(r.label,narrow?16:34));
    g.append('path').attr('d',`M${x(0)},${y}h${Math.max(0,x(r.v)-x(0)-3)}a3,3 0 0 1 3,3v${rh-10}a3,3 0 0 1 -3,3h${-Math.max(0,x(r.v)-x(0)-3)}z`).attr('fill',c);
    g.append('text').attr('x',x(r.v)+5).attr('y',y+rh/2+2).text((+r.v).toFixed(v.digits??1));
    g.append('rect').attr('x',0).attr('width',W).attr('y',y-2).attr('height',rh).attr('fill','transparent')
      .on('mousemove',e=>showTip(e,r.tip||`${r.label}: ${(+r.v).toFixed(v.digits??2)} ${v.unit||''}`)).on('mouseleave',hideTip);});};

R.lines=(el,v)=>{const S=v.series.filter(s=>s.points.length);if(!S.length)return;
  const users=S.filter(s=>s.slot);legendUser(el,users.map(s=>[s.name.replace(/-bf16/,''),SL(s.slot)]).concat([['공개 모델',css('--publine')]]));
  const W=Math.max(320,el.clientWidth),H=300,m={l:58,r:16,t:10,b:42};const svg=d3.select(el).append('svg').attr('width',W).attr('height',H).attr('viewBox',`0 0 ${W} ${H}`);
  const xs=S.flatMap(s=>s.points.map(p=>p[0])),ys=S.flatMap(s=>s.points.map(p=>p[1]));
  const x=(v.xlog?d3.scaleLog():d3.scaleLinear()).domain(d3.extent(xs)).range([m.l,W-m.r]);
  const y=(v.ylog?d3.scaleLog().domain([d3.min(ys)*0.8,d3.max(ys)*1.2]):d3.scaleLinear().domain([Math.min(0,d3.min(ys)),d3.max(ys)*1.05])).nice().range([H-m.b,m.t]);
  svg.append('g').attr('class','grid').selectAll('line').data(y.ticks(5)).join('line').attr('x1',m.l).attr('x2',W-m.r).attr('y1',y).attr('y2',y);
  svg.append('g').attr('class','axis').attr('transform',`translate(0,${H-m.b})`).call(d3.axisBottom(x).tickValues(v.xticks||x.ticks(5)).tickFormat(d3.format('~s')).tickSizeOuter(0));
  svg.append('g').attr('class','axis').attr('transform',`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5,v.ylog?'~s':undefined).tickSizeOuter(0));
  svg.append('text').attr('x',(m.l+W-m.r)/2).attr('y',H-6).attr('text-anchor','middle').text(v.xlabel||'');
  svg.append('text').attr('transform',`translate(14,${(H-m.b+m.t)/2}) rotate(-90)`).attr('text-anchor','middle').text(v.ylabel||'');
  const line=d3.line().x(p=>x(p[0])).y(p=>y(p[1]));
  const ord=S.slice().sort((a,b)=>(a.slot?1:0)-(b.slot?1:0));
  ord.forEach(s=>{const c=s.slot?SL(s.slot):css('--publine');svg.append('path').datum(s.points).attr('d',line).attr('fill','none').attr('stroke',c).attr('stroke-width',s.slot?2.5:1.5).attr('stroke-dasharray',s.dash?'5 3':null);
    svg.selectAll(null).data(s.points).join('circle').attr('cx',p=>x(p[0])).attr('cy',p=>y(p[1])).attr('r',s.slot?4:3).attr('fill',c).attr('stroke',css('--surface')).attr('stroke-width',1.5);
    if(s.slot){const p=s.points[s.points.length-1];svg.append('text').attr('class','lab').attr('x',Math.min(x(p[0])+6,W-m.r-60)).attr('y',y(p[1])-8).text(trunc(s.name.replace(/-bf16/,''),18));}});
  const bis=d3.bisector(d=>d).center;const xv=[...new Set(xs)].sort((a,b)=>a-b);
  const cross=svg.append('line').attr('y1',m.t).attr('y2',H-m.b).attr('stroke',css('--muted')).attr('stroke-dasharray','2 3').style('display','none');
  svg.append('rect').attr('x',m.l).attr('y',m.t).attr('width',W-m.l-m.r).attr('height',H-m.t-m.b).attr('fill','transparent')
    .on('mousemove',e=>{const [mx]=d3.pointer(e);const xx=xv[bis(xv,x.invert(mx))];cross.attr('x1',x(xx)).attr('x2',x(xx)).style('display',null);
      const vals=S.map(s=>[s.name,(s.points.find(p=>p[0]===xx)||[])[1]]).filter(a=>a[1]!=null).sort((a,b)=>b[1]-a[1]);
      showTip(e,`${v.xlabel} ${xx}\n`+vals.map(([n,val])=>`${n}: ${d3.format(',.4~r')(val)}`).join('\n'));tip.style.whiteSpace='pre';})
    .on('mouseleave',()=>{cross.style('display','none');hideTip();tip.style.whiteSpace='';});};

R.heatmap=(el,v)=>{const narrow=el.clientWidth<560;const W=Math.max(320,el.clientWidth),rh=22,ml=narrow?120:190,mt=78,cw=(W-ml-56)/v.cols.length;
  const H=mt+v.rows.length*rh+8;const svg=d3.select(el).append('svg').attr('width',W).attr('height',H).attr('viewBox',`0 0 ${W} ${H}`);
  const col=d3.scaleSequential(d3.interpolateRgb(css('--surface'),css('--s1'))).domain([0,100]);
  v.cols.forEach((c,j)=>svg.append('text').attr('transform',`translate(${ml+j*cw+cw/2},${mt-6}) rotate(-40)`).text(c));
  svg.append('text').attr('x',ml+v.cols.length*cw+8).attr('y',mt-6).text(v.total_label||'전체');
  v.rows.forEach((r,i)=>{const y=mt+i*rh;svg.append('text').attr('x',ml-8).attr('y',y+rh/2+4).attr('text-anchor','end').attr('class',v.slots[i]?'lab':'').text(trunc(r,narrow?16:28));
    if(v.slots[i])svg.append('circle').attr('cx',8).attr('cy',y+rh/2).attr('r',4).attr('fill',SL(v.slots[i]));
    if(v.kinds&&v.kinds[i]==='base')svg.append('text').attr('x',18).attr('y',y+rh/2+4).style('font-size','10px').style('fill',css('--muted')).text('B');
    v.values[i].forEach((val,j)=>{const na=val==null;svg.append('rect').attr('x',ml+j*cw+1).attr('y',y+1).attr('width',cw-2).attr('height',rh-2).attr('rx',3).attr('fill',na?css('--chip'):col(val))
      .on('mousemove',e=>showTip(e,na?`${r} · ${v.cols[j]}: 측정 없음`:`${r} · ${v.cols[j]}: ${val.toFixed(1)}${v.unit??'%'}`)).on('mouseleave',hideTip);
      if(cw>30)svg.append('text').attr('x',ml+j*cw+cw/2).attr('y',y+rh/2+4).attr('text-anchor','middle').style('fill',!na&&val>60?'#fff':css(na?'--muted':'--ink')).style('pointer-events','none').text(na?'—':val.toFixed(0));});
    svg.append('text').attr('x',ml+v.cols.length*cw+8).attr('y',y+rh/2+4).attr('class','lab').text(v.total[i].toFixed(1));});};

R.radar=(el,v)=>{const W=Math.max(320,Math.min(el.clientWidth,620)),H=W*0.82,cx=W/2,cy=H/2+4,rad=Math.min(W,H)/2-70;
  legendUser(el,v.models.map(m=>[m.name.replace(/-bf16/,''),SL(m.slot)]).concat([['공개 모델 중앙값',css('--median')]]));
  const svg=d3.select(el).append('svg').attr('width',W).attr('height',H).attr('viewBox',`0 0 ${W} ${H}`);const n=v.axes.length;
  const ang=i=>-Math.PI/2+i*2*Math.PI/n,pt=(i,r)=>[cx+Math.cos(ang(i))*r*rad,cy+Math.sin(ang(i))*r*rad];
  [0.25,0.5,0.75,1].forEach(k=>svg.append('polygon').attr('points',d3.range(n).map(i=>pt(i,k)).join(' ')).attr('fill','none').attr('stroke',css('--rule')));
  v.axes.forEach((a,i)=>{const [x,y]=pt(i,1);svg.append('line').attr('x1',cx).attr('y1',cy).attr('x2',x).attr('y2',y).attr('stroke',css('--rule'));
    const [lx,ly]=pt(i,1.16);svg.append('text').attr('x',lx).attr('y',ly+4).attr('text-anchor',Math.abs(lx-cx)<8?'middle':(lx>cx?'start':'end')).attr('class','lab').text(a.label);});
  const poly=vals=>vals.map((x,i)=>pt(i,x??0)).join(' ');
  if(v.median)svg.append('polygon').attr('points',poly(v.median)).attr('fill','none').attr('stroke',css('--median')).attr('stroke-width',1.5).attr('stroke-dasharray','4 3');
  v.models.forEach(m=>{const c=SL(m.slot);svg.append('polygon').attr('points',poly(m.vals)).attr('fill',c).attr('fill-opacity',.08).attr('stroke',c).attr('stroke-width',2.2);
    m.vals.forEach((x,i)=>{if(x==null)return;const [px,py]=pt(i,x);svg.append('circle').attr('cx',px).attr('cy',py).attr('r',4.5).attr('fill',c).attr('stroke',css('--surface')).attr('stroke-width',1.5)
      .on('mousemove',e=>showTip(e,`${m.name} · ${v.axes[i].label}: ${m.raw[i]}`)).on('mouseleave',hideTip);});});
  const n2=document.createElement('p');n2.className='note';n2.textContent='모든 축은 바깥이 좋음. 축마다 측정 모델 중 최고=바깥 테두리, 로그 척도. 축 순서는 고정이며 다각형 면적은 순위가 아니다(순위는 아래 기하평균 막대). 점에 올리면 원래 값.';el.append(n2);};

R.scatter=(el,v)=>{const P=v.points.filter(p=>p.x!=null&&p.y!=null);if(!P.length)return;
  legendUser(el,[...new Map(P.filter(p=>p.slot).map(p=>[p.slot,p.label]))].map(([s,l])=>[l,SL(s),true]).concat([['공개 모델',css('--pub'),true]]));
  const W=Math.max(320,el.clientWidth),H=340,m={l:52,r:16,t:12,b:44};const svg=d3.select(el).append('svg').attr('width',W).attr('height',H).attr('viewBox',`0 0 ${W} ${H}`);
  const xs=P.map(p=>p.x),ys=P.flatMap(p=>[p.lo??p.y,p.hi??p.y]);
  const x=(v.xlog?d3.scaleLog().domain([d3.min(xs)/1.3,d3.max(xs)*1.3]):d3.scaleLinear().domain(d3.extent(xs))).range([m.l,W-m.r]);
  const y=d3.scaleLinear().domain([Math.min(0,d3.min(ys)),d3.max(ys)*1.08]).nice().range([H-m.b,m.t]);
  svg.append('g').attr('class','grid').selectAll('line').data(y.ticks(5)).join('line').attr('x1',m.l).attr('x2',W-m.r).attr('y1',y).attr('y2',y);
  svg.append('g').attr('class','axis').attr('transform',`translate(0,${H-m.b})`).call(d3.axisBottom(x).ticks(5,'~s').tickSizeOuter(0));
  svg.append('g').attr('class','axis').attr('transform',`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSizeOuter(0));
  svg.append('text').attr('x',(m.l+W-m.r)/2).attr('y',H-6).attr('text-anchor','middle').text(v.xlabel||'');
  svg.append('text').attr('transform',`translate(13,${(H-m.b+m.t)/2}) rotate(-90)`).attr('text-anchor','middle').text(v.ylabel||'');
  if(v.frontier){const srt=P.slice().sort((a,b)=>v.frontier==='lower_x'?a.x-b.x:b.x-a.x);let best=-Infinity;const f=[];
    srt.forEach(p=>{if(p.y>best){best=p.y;f.push(p);}});const fl=f.slice().sort((a,b)=>a.x-b.x);
    svg.append('path').datum(fl).attr('d',d3.line().x(p=>x(p.x)).y(p=>y(p.y)).curve(d3.curveStepAfter)).attr('fill','none').attr('stroke',css('--muted')).attr('stroke-dasharray','4 3');}
  const ord=P.slice().sort((a,b)=>(a.slot?1:0)-(b.slot?1:0));
  ord.forEach(p=>{const c=SL(p.slot),g=svg.append('g'),cx=x(p.x),cy=y(p.y),r=p.slot?6:4.5;
    if(p.lo!=null&&p.slot)g.append('line').attr('x1',cx).attr('x2',cx).attr('y1',y(p.lo)).attr('y2',y(p.hi)).attr('stroke',c).attr('stroke-width',1.5).attr('opacity',.6);
    if(p.kind==='base')g.append('rect').attr('x',cx-r).attr('y',cy-r).attr('width',2*r).attr('height',2*r).attr('rx',1.5).attr('fill',c).attr('stroke',css('--surface')).attr('stroke-width',1.5);
    else g.append('circle').attr('cx',cx).attr('cy',cy).attr('r',r).attr('fill',c).attr('stroke',css('--surface')).attr('stroke-width',1.5);
    if(p.slot)g.append('text').attr('class','lab').attr('x',Math.min(cx+9,W-m.r-90)).attr('y',cy-8).text(trunc(p.label,22));
    g.append('circle').attr('cx',cx).attr('cy',cy).attr('r',11).attr('fill','transparent').on('mousemove',e=>showTip(e,p.tip||p.label)).on('mouseleave',hideTip);});
  const n=document.createElement('p');n.className='note';n.textContent='■ base · ● instruct. 점선 계단 = 효율 경계(그보다 왼쪽·위에 다른 모델이 없는 점). 사용자 모델 세로 막대 = 95% CI(근사).';el.append(n);};

R.transcripts=(el,v)=>{const ctl=document.createElement('div');ctl.className='tr-ctl';const mk=(id,opts,lab)=>{const s=document.createElement('select');s.id=id;s.setAttribute('aria-label',lab);
    opts.forEach(([k,t])=>{const o=document.createElement('option');o.value=k;o.textContent=t;s.append(o)});ctl.append(s);return s;};
  const models=[...new Set(v.items.map(x=>x.model+'|'+x.variant))];const cats=[...new Set(v.items.map(x=>x.category))];
  const sm=mk('tr-model',models.map(k=>[k,k.replace('|',' · 설정 ')]),'모델과 설정'),sc=mk('tr-cat',cats.map(c=>[c,v.cats[c]||c]),'범주');
  const box=document.createElement('div');el.append(ctl,box);
  function draw(){box.innerHTML='';const [m,va]=sm.value.split('|');const it=v.items.find(x=>x.model===m&&x.variant===va&&x.category===sc.value);
    if(!it){box.textContent='이 조합의 출력이 없습니다.';return;}
    it.turns.forEach((t,i)=>{const d=document.createElement('div');d.className='turn';const h=document.createElement('div');h.className='u';
      h.textContent=`턴 ${i+1} · 사용자: ${t.user}`;const p=document.createElement('span');p.className='pill '+(t.pass?'ok':'ng');p.textContent=t.pass==null?'채점 없음':(t.pass?'통과':'실패');
      h.append(p);const a=document.createElement('div');a.className='a';a.textContent=t.answer||'(빈 답)';d.append(h,a);box.append(d);});}
  sm.onchange=draw;sc.onchange=draw;draw();};

function renderAll(){document.querySelectorAll('.viz').forEach(el=>{const v=V[el.dataset.view];el.innerHTML='';if(!v){el.textContent='데이터 없음';return;}
  try{(R[v.type]||(()=>{el.textContent='알 수 없는 형식 '+v.type}))(el,v);}catch(e){el.textContent='그리기 오류: '+e.message;console.error(e);}
  if(v.missing&&v.missing.length){const n=document.createElement('p');n.className='note miss';
    n.textContent='이 그림에 없는 사용자 모델 — '+v.missing.map(([a,b])=>a+': '+b).join(' / ');el.append(n);}});}
if(window.d3){renderAll();let t;addEventListener('resize',()=>{clearTimeout(t);t=setTimeout(renderAll,200)});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',renderAll);
  new MutationObserver(renderAll).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});}
else document.querySelectorAll('.viz').forEach(el=>el.textContent='차트 라이브러리를 불러오지 못했습니다.');
</script>
"""


def standalone(fragment: str) -> str:
    """page() 조각(<title>·<link>·<style> 머리 + 본문·스크립트)을 단독 HTML 문서로 감싼다."""
    cut = fragment.index("</style>") + len("</style>")
    head, body = fragment[:cut], fragment[cut:]
    return ('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            "<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n"
            f"{head}\n</head>\n<body>{body}</body>\n</html>\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(FINAL / "data.json"))
    ap.add_argument("--narrative", default=str(FINAL / "narrative.json"))
    ap.add_argument("--out", default=str(FINAL / "report.html"))
    ap.add_argument("--fragment", action="store_true", help="문서 골격 없이 내용만 쓴다")
    a = ap.parse_args(argv)
    d = json.loads(Path(a.data).read_text(encoding="utf-8"))
    nar = json.loads(Path(a.narrative).read_text(encoding="utf-8")) if Path(a.narrative).exists() else {}
    views = build_views(d)
    out = page(views, nar, d.get("generated_at", ""))
    Path(a.out).write_text(out if a.fragment else standalone(out), encoding="utf-8")
    print(f"{a.out}: 뷰 {len(views)}개, {Path(a.out).stat().st_size / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
