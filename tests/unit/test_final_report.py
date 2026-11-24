"""최종 보고서 뷰 계산(final_report)."""

import math

from kobench.standard import final_report as fr


def _perf(model, params, active, tp1, tp64, thru, ttft, w, on=None, off=None):
    rows = []
    base = {"model": model, "engine": "vllm", "params_b": params, "active_b": active, "kind": "instruct", "family": "x",
            "weights_gib": w, "kv_cache_gib": 10.0, "kv_cache_tokens": 1000, "vram_peak_mib": 1000, "startup_s": 1}
    for c, tp, th in ((1, tp1, thru / 10), (64, tp64, thru)):
        rows.append({**base, "run": "prefix_cache_on", "workload": "w1", "concurrency": c, "output_tok_per_s": th, "ttft_ms_p50": ttft,
                     "tpot_ms_p50": tp})
    if on:
        rows.append({**base, "run": "prefix_cache_on", "workload": "w2", "concurrency": 16, "ttft_ms_p50": on})
        rows.append({**base, "run": "prefix_cache_off", "workload": "w2", "concurrency": 16, "ttft_ms_p50": off})
    return rows


def test_radar_outward_is_better_and_geomean():
    perf = _perf("frankenstallm-bf16", 3, 3, 2.0, 4.0, 10000, 10, 6, on=100, off=300) + \
        _perf("qwen3-4b-2507", 4, 4, 4.0, 4.0, 5000, 20, 8, on=100, off=200) + \
        _perf("qwen3-30b-a3b-2507", 30, 3, 3.0, 6.0, 8000, 15, 60, on=100, off=150)
    v = fr.perf_views({"perf": perf})
    r = v["perf_radar"]
    assert [m["name"] for m in r["models"]] == ["frankenstallm-bf16"]          # 사용자 모델만 색 다각형
    keys = [a["key"] for a in r["axes"]]
    fr_vals = dict(zip(keys, r["models"][0]["vals"]))
    assert fr_vals["thru"] == 1.0 and fr_vals["speed"] == 1.0 and fr_vals["ttft"] == 1.0 and fr_vals["mem"] == 1.0 and fr_vals["prefix"] == 1.0
    assert math.isclose(fr_vals["load"], 0.15 + 0.85 * (math.log(0.5) - math.log(0.5)) / (math.log(1.0) - math.log(0.5)))  # 최저 = 0.15
    g = {x["label"]: x["v"] for x in v["perf_geomean"]["rows"]}
    assert g["frankenstallm-bf16"] < 1 and max(g, key=g.get) == "frankenstallm-bf16"
    assert {x["label"] for x in v["perf_prefix"]["rows"]} == {"frankenstallm-bf16", "qwen3-4b-2507", "qwen3-30b-a3b-2507"}
    s = {x["name"]: x for x in v["perf_output_tok_per_s_all"]["series"]}          # dense·MoE 한 축, MoE는 점선
    assert s["qwen3-30b-a3b-2507 [MoE]"]["dash"] and not s["qwen3-4b-2507"]["dash"]


def test_quant_rows_use_f16_reference_and_flag_significance():
    q = [{"suite": "ko_mc_v1", "family": "JANSORI", "model": "jansori-gguf-q4_k_m", "quant": "Q4_K_M", "engine": "llama.cpp", "task": "kobest",
          "micro": 0.57, "ci_lo": 0.56, "ci_hi": 0.58, "n": 100, "delta_bf16": -0.01, "delta_bf16_lo": -0.02, "delta_bf16_hi": -0.001,
          "delta_f16": -0.008, "delta_f16_lo": -0.015, "delta_f16_hi": -0.001}]
    v = fr.quant_views({"mc": {"ko_mc_v1__quant": q}})
    labs = [r["label"] for r in v["quant_ko_mc_v1"]["rows"]]
    assert labs == ["JANSORI Q4_K_M − f16 · KoBEST"]                  # BF16 기준 차이는 f16 GGUF 행에만
    cell = v["quant_table_ko_mc_v1"]["rows"][0]["df16"]
    assert cell["sig"] and round(cell["v"], 2) == -0.8


def test_page_places_views_and_appendix():
    views = {"models": {"type": "table", "columns": [], "rows": []}, "extra": {"type": "bars", "rows": []}}
    html = fr.page(views, {"summary": "<p>요약</p>", "view:models": "설명"}, "t")
    assert 'data-view="models"' in html and 'id="appendix"' in html and "<p>요약</p>" in html and "<title>" in html


def test_display_names_user_entries_without_touching_longer_ids():
    assert fr.display("evafrill-mo-3b") == "EVAFRILL-Mo slerp"
    assert fr.display("evafrill-mo-3b-pretrain") == "EVAFRILL-Mo pretrain"
    assert fr.display("evafrill-mo-3b-sft-v2") == "evafrill-mo-3b-sft-v2"          # 중간 체크포인트는 id 그대로
    assert fr.display("jansori-bf16 · KoBEST") == "JANSORI · KoBEST"
    assert fr.display("jansori-gguf-q8_0") == "jansori-gguf-q8_0"


def test_finalize_pins_user_rows_and_explains_missing_models():
    views = {
        "gen_table_gen_v1__A_instruct": {"type": "table", "columns": [{"key": "model", "label": "모델"}], "rows": [
            {"model": {"v": "qwen3-4b-2507", "slot": None}}, {"model": {"v": "frankenstallm-bf16", "slot": 2}},
            {"model": {"v": "evafrill-mo-3b", "slot": 3}}]},
        "perf_weights": {"type": "bars", "rows": [{"label": "jansori-bf16", "v": 6.0, "slot": 1},
                                                  {"label": "frankenstallm-bf16", "v": 6.0, "slot": 2}]},
        "perf_prefix": {"type": "dumbbell", "rows": [{"label": "jansori-bf16", "a": 1, "b": 0.5, "slot": 1},
                                                     {"label": "frankenstallm-bf16", "a": 1, "b": 0.5, "slot": 2}]},
    }
    v = fr.finalize(views)
    rows = v["gen_table_gen_v1__A_instruct"]["rows"]
    assert [r["model"]["v"] for r in rows] == ["frankenstallm", "EVAFRILL-Mo slerp", "JANSORI", "qwen3-4b-2507"]
    assert "base 모델" in rows[2]["_note"]                       # 빠진 사용자 모델은 사유 행
    assert v["perf_weights"]["missing"] == [["EVAFRILL-Mo", "측정 없음"]]
    assert "prefix 캐시" in v["perf_prefix"]["missing"][0][1]
    assert v["perf_weights"]["rows"][0]["label"] == "JANSORI"


def test_user_card_ranks_within_same_kind_and_marks_not_applicable():
    models = {"jansori-bf16": {"kind": "base", "owner": "self", "family": "JANSORI", "backend": "vllm"},
              "qwen3-4b-base": {"kind": "base", "owner": "public", "family": "Qwen3", "backend": "vllm"},
              "frankenstallm-bf16": {"kind": "instruct", "owner": "self", "family": "frankenstallm", "backend": "vllm"},
              "qwen3-4b-2507": {"kind": "instruct", "owner": "public", "family": "Qwen3", "backend": "vllm"}}
    mc = [{"model": m, "kind": models[m]["kind"], "task": "kobest", "micro": x} for m, x in
          (("jansori-bf16", 0.5), ("qwen3-4b-base", 0.7), ("frankenstallm-bf16", 0.4), ("qwen3-4b-2507", 0.3))]
    gen = [{"model": "frankenstallm-bf16", "kind": "instruct", "task": "hrm8k", "value": 0.1},
           {"model": "qwen3-4b-2507", "kind": "instruct", "task": "hrm8k", "value": 0.6}]
    d = {"models": models, "mc": {"ko_mc_v1": mc}, "gen": {"gen_v1__A": gen}, "perf": [], "scenario": []}
    v = fr.user_views(d)
    card = {r["metric"]["v"]: r for r in v["user_card"]["rows"]}
    assert card["한국어 객관식 평균"]["jansori-bf16"]["v"] == "50.0 · 2/2위"
    assert card["한국어 객관식 평균"]["frankenstallm-bf16"]["v"] == "40.0 · 1/2위"       # instruct끼리 순위
    assert card["HRM8K"]["jansori-bf16"]["v"] == "해당 없음"                            # base는 지시 생성 대상 아님
    cover = {r["model"]["v"]: r for r in v["cover"]["rows"]}
    assert cover["JANSORI"]["gen"]["v"] == "base" and cover["frankenstallm"]["gen"]["v"] == "✓"


def test_uni_scores_chance_correct_and_composite_uses_common_areas_only():
    models = {"a": {"kind": "base", "params_b": 3, "active_b": 3, "owner": "public"},
              "b": {"kind": "instruct", "params_b": 30, "active_b": 3, "owner": "public"}}
    mc = [{"model": "a", "task": "kmmlu", "micro": 0.25, "ci_lo": 0.24, "ci_hi": 0.26},
          {"model": "b", "task": "kmmlu", "micro": 0.625, "ci_lo": 0.6, "ci_hi": 0.65}]
    ppl = [{"model": "a", "task": "wiki40b_ko", "micro": 1.0, "ci_lo": 0.9, "ci_hi": 1.1},
           {"model": "b", "task": "wiki40b_ko", "micro": 0.5, "ci_lo": 0.45, "ci_hi": 0.55}]
    gen = [{"model": "b", "task": "komt", "value": 8.0, "ci_lo": 7.0, "ci_hi": 9.0}]       # a에는 없음 → 종합에서 빠지는 영역
    d = {"models": models, "mc": {"ko_mc_v1": mc, "ppl_v1": ppl}, "gen": {"uni_v1__U": gen}, "scenario": [],
         "unified": {"chance": {"kmmlu": 0.25}, "areas": {"지식": ["ko_mc_v1"], "대화": ["komt"], "언어 모델링": ["ppl_wiki40b_ko"]}}}
    u = fr.uni_scores(d)
    assert u["common"] == ["지식", "언어 모델링"]
    assert u["models"]["a"]["tasks"]["ko_mc_v1"][0] == 0.0 and math.isclose(u["models"]["b"]["tasks"]["ko_mc_v1"][0], 0.5)
    assert math.isclose(u["models"]["b"]["tasks"]["komt"][0], 0.8)
    assert math.isclose(u["models"]["a"]["comp"][0], (0.0 + 0.5) / 2) and math.isclose(u["models"]["b"]["comp"][0], (0.5 + 1.0) / 2)
    v = fr.uni_views(d)
    assert [r["label"] for r in v["uni_rank"]["rows"]] == ["b", "a"] and v["uni_rank"]["rows"][1]["kind"] == "base"
