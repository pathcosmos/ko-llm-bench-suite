"""최종 보고서용 데이터 수집: 표준 MC/PPL(A·B), 양자화, 생성(A·B·think), 성능, 설정 선택 근거를 하나의 JSON으로 모은다.

python -m kobench.standard.final_data --out ../reports/final/data.json
(먼저 report / gen_report table을 다시 만든다: --regen)
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from kobench.standard.registry import CONFIG_DIR, load_models

ROOT = Path(__file__).resolve().parents[2]


def _csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if v in ("", None):
                r[k] = None
                continue
            try:
                r[k] = float(v) if any(c in v for c in ".eE") or v.lstrip("-").isdigit() else v
            except ValueError:
                pass
            if v in ("True", "False"):
                r[k] = v == "True"
    return rows


def regen() -> None:
    py = sys.executable
    for s in ("ko_mc_v1", "ko_mc_ext_v1", "en_mc_v1", "ppl_v1", "uni_mc_v1"):
        subprocess.run([py, "-m", "kobench.standard.report", "--suite", s, "--ref", "jansori-bf16"], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
        if s not in ("ppl_v1", "uni_mc_v1"):
            subprocess.run([py, "-m", "kobench.standard.report", "--suite", s, "--ref", "jansori-bf16", "--table", "B"], cwd=ROOT,
                           check=True, stdout=subprocess.DEVNULL)
    for s in ("ko_mc_v1", "ppl_v1"):
        subprocess.run([py, "-m", "kobench.standard.report", "--suite", s, "--table", "Q"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    for t in ("A", "B", "think", "thinkrd", "tune"):
        subprocess.run([py, "-m", "kobench.standard.gen_report", "table", "--table", t], cwd=ROOT, check=True)
    subprocess.run([py, "-m", "kobench.standard.gen_report", "table", "--table", "A", "--suite", "gen_dev_v1"], cwd=ROOT, check=True)
    subprocess.run([py, "-m", "kobench.standard.gen_report", "table", "--table", "U", "--suite", "uni_v1"], cwd=ROOT, check=True)
    subprocess.run([py, "-m", "kobench.standard.gen_report", "table", "--table", "B", "--suite", "gen_dev_v1"], cwd=ROOT, check=True)
    subprocess.run([py, "-m", "kobench.standard.gen_report", "summarize"], cwd=ROOT, check=True)


def evafrill_memory(model_id: str) -> dict:
    """EVAFRILL 자체 엔진은 vLLM처럼 가중치·KV 크기를 보고하지 않는다 → 체크포인트 파일 크기와 config로 계산한 값(표시: 계산값).
    KV는 attention 층(hybrid_pattern의 'A')만 토큰당 2·n_kv_heads·head_dim·2바이트, Mamba 층 상태는 시퀀스당 고정 크기."""
    from huggingface_hub import try_to_load_from_cache

    spec = load_models().get(model_id)
    if spec is None:
        return {}
    sub = f"{spec.subfolder}/" if spec.subfolder else ""
    st = try_to_load_from_cache(spec.path, f"{sub}model.safetensors", revision=spec.revision)
    cf = try_to_load_from_cache(spec.path, f"{sub}config.json", revision=spec.revision)
    if not isinstance(st, str) or not isinstance(cf, str):
        return {}
    c = json.loads(Path(cf).read_text(encoding="utf-8"))
    pat = c.get("hybrid_pattern", "").split()
    n_attn = pat.count("A") if pat else c["n_layers"]
    head_dim = c["d_model"] // c["n_heads"]
    d_inner = c["mamba_expand"] * c["d_model"]
    mamba_state = (len(pat) - n_attn) * (d_inner // c["mamba_head_dim"]) * c["mamba_head_dim"] * c["mamba_d_state"] * 2
    return {"weights_gib": Path(st).stat().st_size / 2 ** 30, "kv_bytes_per_token": n_attn * 2 * c["n_kv_heads"] * head_dim * 2,
            "mamba_state_mib_per_seq": mamba_state / 2 ** 20, "mem_source": "계산값(체크포인트 크기·config)"}


def perf_rows() -> list[dict]:
    out = []
    for p in sorted((ROOT / "results" / "perf").glob("*/*.json")):
        if p.name.endswith(".error.json"):
            out.append({"model": p.parent.name, "engine": p.name.split(".")[0], "error": json.loads(p.read_text())["error"]})
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for tag, r in d["runs"].items():
            base = {"model": d["model"], "engine": d["engine"], "params_b": d["params_b"], "active_b": d["active_b"], "kind": d["kind"],
                    "family": d["family"], "run": tag, "startup_s": r.get("startup_s"), "vram_peak_mib": r.get("vram_peak_mib"),
                    "weights_gib": r.get("weights_gib"), "kv_cache_gib": r.get("kv_cache_gib"), "kv_cache_tokens": r.get("kv_cache_tokens"),
                    "torch_max_allocated_gib": r.get("torch_max_allocated_gib")}
            if d["engine"] == "evafrill" and base["weights_gib"] is None:
                base.update(evafrill_memory(d["model"]))
            for w in r.get("w1", []) or []:
                out.append({**base, "workload": "w1", **{k: w.get(k) for k in ("concurrency", "output_tok_per_s", "req_per_s", "ttft_ms_p50",
                                                                             "ttft_ms_p90", "tpot_ms_p50", "tpot_ms_p90", "ok", "requests")}})
            if r.get("w2"):
                w = r["w2"]
                out.append({**base, "workload": "w2", **{k: w.get(k) for k in ("concurrency", "output_tok_per_s", "ttft_ms_p50", "ttft_ms_p90",
                                                                             "tpot_ms_p50", "ok", "requests")}})
    return out


def _policy(model_id: str, models) -> str:
    from kobench.standard.gen_report import chosen
    try:
        return chosen(model_id)[1]
    except Exception:  # noqa: BLE001
        return "raw"


def scenario_rows(models) -> list[dict]:
    """다턴 시나리오 점수(모델·변형·태스크별): 턴/시나리오 통과, 범주별, 턴 위치별, 반복 지표."""
    out = []
    for p in sorted((ROOT / "results" / "gen").glob("*/*/scenario_*.score.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        mid = d["model"]
        if mid not in models:
            continue
        pol = _policy(mid, models)
        m = d["metrics"].get(pol) or next(iter(d["metrics"].values()))
        out.append({"model": mid, "variant": d["variant"], "task": d["task"], "policy": pol if pol in d["metrics"] else next(iter(d["metrics"])),
                    **{k: m.get(k) for k in ("turn_pass", "scenario_pass", "n", "loop", "copy_prev", "overlap_prev", "rep3")},
                    "by_category": m.get("by_category"), "by_turn": m.get("by_turn")})
    return out


TRANSCRIPT_MODELS = ("frankenstallm-bf16", "evafrill-mo-3b", "qwen3-4b-2507", "kanana-2-3b")


def transcripts(models, per_cat: int = 1, task: str = "scenario_self") -> list[dict]:
    """대화 보기용 표본: 범주마다 시드 순 첫 항목, 지정 모델의 A와 선택 설정 출력(자체 생성 문항이라 원문 포함 가능).
    표시는 읽기 쉽게 <unk>를 개행으로 되돌린다(채점 정책과 무관)."""
    from kobench.standard import scenario
    from kobench.standard.gen_report import chosen
    from kobench.standard.gen_score import apply_policy

    split = "dev" if "dev" in task else "test"
    items = {it["id"]: it for it in scenario.build(split, "self")}
    pick, seen = [], {}
    for iid, it in items.items():
        c = it["meta"]["category"]
        if seen.get(c, 0) < per_cat:
            seen[c] = seen.get(c, 0) + 1
            pick.append(iid)
    out = []
    for mid in TRANSCRIPT_MODELS:
        if mid not in models:
            continue
        v_ch, pol = chosen(mid)
        for var in dict.fromkeys(["A", v_ch]):
            f = ROOT / "results" / "gen" / mid / var / f"{task}.jsonl"
            sc = ROOT / "results" / "gen" / mid / var / f"{task}.score.json"
            if not f.exists():
                continue
            rows = {json.loads(l)["id"]: json.loads(l) for l in f.open(encoding="utf-8")}
            score = {}
            if sc.exists():
                d = json.loads(sc.read_text(encoding="utf-8"))
                its = d["items"].get(pol) or next(iter(d["items"].values()))
                score = {x["id"]: x for x in its}
            for iid in pick:
                if iid not in rows:
                    continue
                it, r, s_ = items[iid], rows[iid], score.get(iid, {})
                out.append({"model": mid, "variant": var, "id": iid, "category": it["meta"]["category"],
                            "turns": [{"user": u, "answer": apply_policy(o, "unk_nl" if pol != "raw" else "raw"), "pass": (t or {}).get("pass")}
                                      for u, o, t in zip(it["turns"], r["outputs"], s_.get("turns") or [None] * len(r["outputs"]))]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "reports" / "final" / "data.json"))
    ap.add_argument("--regen", action="store_true")
    a = ap.parse_args(argv)
    if a.regen:
        regen()
    models = load_models()
    rs, rg = ROOT / "reports" / "standard", ROOT / "reports" / "gen"
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": {m: {"family": s.family, "kind": s.kind, "owner": s.owner, "backend": s.backend, "params_b": s.params_b,
                       "active_b": s.active_b, "revision": s.revision, "path": s.path if s.owner == "self" else None,
                       "quant": s.extra.get("quant")} for m, s in models.items()},
        "mc": {f.stem: _csv(f) for f in sorted(rs.glob("*.csv"))},
        "gen": {f.stem: _csv(f) for f in sorted(rg.glob("*.csv"))},
        "perf": perf_rows(),
        "scenario": scenario_rows(models),
        "transcripts": transcripts(models),
        "best_settings": yaml.safe_load((CONFIG_DIR / "best_settings.yaml").read_text(encoding="utf-8")),
        "dev_selection": yaml.safe_load((CONFIG_DIR / "dev_selection.yaml").read_text(encoding="utf-8")),
        "gen_config": yaml.safe_load((CONFIG_DIR / "gen.yaml").read_text(encoding="utf-8")),
        "suites": yaml.safe_load((CONFIG_DIR / "suites.yaml").read_text(encoding="utf-8")),
        "unified": yaml.safe_load((CONFIG_DIR / "unified.yaml").read_text(encoding="utf-8")),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"{out}: mc {len(data['mc'])} 표, gen {len(data['gen'])} 표, perf {len(data['perf'])}행")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
