"""생성 벤치 dev 선택과 표 생성.

python -m kobench.standard.gen_report select          # 생성 dev 선택 → configs/standard/best_settings.yaml 의 gen 항목
python -m kobench.standard.gen_report table --table A  # reports/gen/gen_v1__A.{md,csv}
python -m kobench.standard.gen_report table --table B  # dev 선택 설정 (같은 모델 A 대비 paired 차이)
python -m kobench.standard.gen_report table --table think

대표 지표(문항 단위): hrm8k exact_match_strict / ifeval_ko prompt_level_strict_acc / 객관식 acc(형식 실패 = 오답) /
코드 pass_plus / komt·logickor·k2_generation judge 점수(1~10, logickor 0~10) / ko_arena_hard 승률(기준 모델 대비, 강한 판정 3배) /
kosimpleqa correct. CI = 문항 bootstrap 95%(1000회), 기준 모델 대비 차이는 같은 문항으로 paired bootstrap.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from kobench.standard.gen import out_dir, read_jsonl
from kobench.standard.gen_data import load_gen_config
from kobench.standard.registry import CONFIG_DIR, load_models

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "gen"
MAIN = {"hrm8k": "exact_match_strict", "gsm8k_ko_dev": "exact_match_strict", "ifeval_ko": "prompt_level_strict_acc",
        "kormedmcqa": "acc", "kormedmcqa_dev": "acc", "kobbq": "acc", "ko_truthful_mc1": "acc", "k2_knowledge": "acc",
        "humaneval_plus": "pass_plus", "mbpp_plus": "pass_plus", "komt": "score", "logickor": "score", "k2_generation": "score",
        "ko_arena_hard": "score", "kosimpleqa": "correct", "scenario_self": "turn_pass", "scenario_oracle": "turn_pass",
        "scenario_dev_self": "turn_pass", "scenario_dev_oracle": "turn_pass"}
SCALE = {"komt": 1.0, "logickor": 1.0, "k2_generation": 1.0}        # judge 점수는 원 척도, 나머지는 %로 표시
B = 1000
RNG_SEED = 1234


def load_scores(model_id: str, variant: str, task: str, policy: str | None = None) -> dict[str, float] | None:
    p = out_dir(model_id, variant) / f"{task}.score.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text(encoding="utf-8"))
    items = s["items"]
    pol = policy if policy in items else next(iter(items))
    key = MAIN[task]
    out = {}
    for it in items[pol]:
        v = it.get(key)
        if v is None:
            v = 0.0 if task in ("komt", "logickor", "k2_generation", "ko_arena_hard") else v   # judge 파싱 실패는 아래에서 별도 집계
            if v is None:
                continue
        out[it["id"]] = float(v)
    return out


def boot_ci(x: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a: dict, b: dict) -> tuple[float, float, float] | None:
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    d = np.array([a[i] - b[i] for i in ids])
    lo, hi = boot_ci(d)
    return float(d.mean()), lo, hi


# ── dev 선택 ────────────────────────────────────────────────────────────────

def dev_score(model_id: str, variant: str, policy: str | None) -> dict | None:
    """dev 점수(두 태스크 대표 지표 평균)와 dev 루프율(두 태스크 전체 응답 중 루프 비율, 규칙 v2)."""
    tasks = load_gen_config()["suites"]["gen_dev_v1"]
    vals, loops = {}, []
    for t in tasks:
        p = out_dir(model_id, variant) / f"{t}.score.json"
        s = load_scores(model_id, variant, t, policy)
        if not s or not p.exists():
            return None
        vals[t] = float(np.mean(list(s.values())))
        sc = json.loads(p.read_text(encoding="utf-8"))
        pol = policy if policy in sc["items"] else next(iter(sc["items"]))
        loops += [it.get("loop", 0) for it in sc["items"][pol]]
    return {"tasks": vals, "mean": float(np.mean(list(vals.values()))), "loop": float(np.mean(loops)) if loops else None}


def policies(model_id: str) -> list[str]:
    from kobench.standard.gen_score import policies_for

    return policies_for(model_id)


LOOP_GUARD = 0.05
TOL = 0.001


def choose(cands: dict[str, dict], base: str) -> tuple[str, float, list[str]]:
    """규칙 v2: 루프율 > 0.05 후보 제외(모두 넘으면 루프율 최저) → 점수 최고, base(A)가 남아 있고 차이 ≤ 0.001이면 base.
    반환: (선택, base 대비 gain, 제외된 후보)."""
    excluded = [k for k, d in cands.items() if (d.get("loop") or 0) > LOOP_GUARD]
    ok = {k: d for k, d in cands.items() if k not in excluded}
    if not ok:
        m = min((d.get("loop") or 0) for d in cands.values())
        ok = {k: d for k, d in cands.items() if (d.get("loop") or 0) == m}
    top = max(ok, key=lambda k: ok[k]["mean"])
    if base in ok and ok[top]["mean"] - ok[base]["mean"] <= TOL:
        top = base
    gain = cands[top]["mean"] - cands[base]["mean"] if base in cands else 0.0
    return top, gain, excluded


def cmd_select(a) -> int:
    rule = yaml.safe_load((CONFIG_DIR / "dev_selection.yaml").read_text(encoding="utf-8"))["generation"]
    models = load_models()
    bp = CONFIG_DIR / "best_settings.yaml"
    best = yaml.safe_load(bp.read_text(encoding="utf-8")) or {}
    gen = best.setdefault("gen", {})
    now = datetime.now(timezone.utc).isoformat()
    DEC = ["A", "card", "rep"]                    # v2: rep 후보 추가

    for mid, spec in models.items():              # 공개 instruct
        if spec.kind != "instruct" or spec.owner == "self" or spec.backend != "vllm":
            continue
        cands = {v: d for v in DEC if (d := dev_score(mid, v, None))}
        if "A" not in cands:
            continue
        ch, gain, ex = choose(cands, "A")
        gen[mid] = {"chosen": ch, "unk_policy": "raw", "dev": cands, "gain_over_A_dev": round(gain, 5),
                    "excluded_by_loop_guard": ex, "rule": "v2", "selected_at": now}
        print(f"{mid}: {ch} (gain {gain:+.4f}, 제외 {ex})")
    mid = "frankenstallm-bf16"                    # frankenstallm: decoding × unk_policy
    ss = rule["search_space"][mid]
    cands = {f"{v}|{p}": d for v in DEC for p in ss["unk_policy"] if (d := dev_score(mid, v, p))}
    if cands:
        ch, gain, ex = choose(cands, "A|unk_nl")
        v, p = ch.split("|")
        gen[mid] = {"chosen": v, "unk_policy": p, "dev": cands, "gain_over_A_dev": round(gain, 5),
                    "excluded_by_loop_guard": ex, "rule": "v2", "selected_at": now}
        print(f"{mid}: {ch} (gain {gain:+.4f}, 제외 {ex})")
    ev = rule["search_space"]["evafrill"]         # EVAFRILL: 1차 체크포인트(A, v1 규칙) → 2차 decoding(v2)
    c1 = {f"{ck}|{p}": d for ck in ev["checkpoints"] for p in ev["unk_policy"] if (d := dev_score(ck, "A", p))}
    if c1:
        top = max(c1, key=lambda k: c1[k]["mean"])
        base = "evafrill-mo-3b|unk_nl"
        gain = c1[top]["mean"] - c1[base]["mean"] if base in c1 else 0.0
        ck_sel, pol = (top if gain > TOL else base).split("|")
        c2 = {f"{v}|{p}": d for v in DEC for p in ev["unk_policy"] if (d := dev_score(ck_sel, v, p))}
        entry = {"checkpoint": ck_sel, "checkpoint_dev": c1, "checkpoint_gain_over_slerp": round(gain, 5), "selected_at": now}
        if len({k.split("|")[0] for k in c2}) == len(DEC):
            ch, g2, ex = choose(c2, f"A|{pol}")
            v, p = ch.split("|")
            entry.update({"chosen": v, "unk_policy": p, "dev": c2, "gain_over_A_dev": round(g2, 5),
                          "excluded_by_loop_guard": ex, "rule": "v2"})
        else:
            entry.update({"chosen": "A", "unk_policy": pol, "pending": "2차(decoding) dev 미완료"})
        gen["evafrill"] = entry
        for m2 in ev["checkpoints"]:
            gen.setdefault(m2, {})["unk_policy"] = entry["unk_policy"]
        print(f"evafrill: ckpt {ck_sel}, {entry.get('chosen')}|{entry['unk_policy']}")
    best["gen_rule_file"] = "dev_selection.yaml#generation, #generation_v2"
    bp.write_text(yaml.safe_dump(best, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return 0


# ── 규칙 v3: 모델별 운용 최적 설정 (오동작 제약 + 품질 최대) ─────────────────────
TUNE_TASKS = ("gsm8k_ko_dev", "kormedmcqa_dev", "scenario_dev_self")
LIMITS = {"loop": 0.02, "lang": 0.02, "leak": 0.005, "trunc": 0.05}


def tune_dev(model_id: str, variant: str, policy: str | None) -> dict | None:
    """품질 = 세 dev 태스크 대표 지표 평균. 오동작률 = 세 태스크 모든 턴 출력에서의 비율(코드 범주 턴은 언어 판정 제외)."""
    from kobench.standard.gen_score import apply_policy, malfunction

    vals, mal = {}, []
    for t in TUNE_TASKS:
        sc = load_scores(model_id, variant, t, policy)
        rows = read_jsonl(out_dir(model_id, variant) / f"{t}.jsonl")
        if not sc or not rows:
            return None
        vals[t] = float(np.mean(list(sc.values())))
        pol = policy or "raw"
        for r in rows:
            code = r["id"].startswith("code_revise/")
            for o, f in zip(r["outputs"], r["finish"]):
                mal.append(malfunction(apply_policy(o, pol), f, check_lang=not code))
    rates = {k: float(np.mean([m[k] for m in mal])) for k in ("loop", "lang", "leak", "empty", "trunc")}
    return {"tasks": vals, "mean": float(np.mean(list(vals.values()))), "malfunction": rates, "n_outputs": len(mal)}


def violation(d: dict) -> float:
    m = d["malfunction"]
    return sum(max(0.0, m[k] - lim) / max(lim, 1e-3) for k, lim in LIMITS.items())


def choose_v3(cands: dict[str, dict], base: str) -> tuple[str, list[str]]:
    """제약(LIMITS)을 만족하는 후보 중 품질 최고. 1위와 base의 품질 차이가 0.001 이하이고 base가 만족하면 base.
    아무것도 만족하지 않으면 위반 정도(제약 대비 초과 비율 합)가 가장 작은 후보(같으면 품질)."""
    ok = [k for k, d in cands.items() if violation(d) == 0]
    if not ok:
        return min(cands, key=lambda k: (violation(cands[k]), -cands[k]["mean"])), []
    top = max(ok, key=lambda k: cands[k]["mean"])
    if base in ok and cands[top]["mean"] - cands[base]["mean"] <= TOL:
        top = base
    return top, ok


def cmd_tune(a) -> int:
    """--stage 1: 디코딩 후보 중 선택, --stage 2: 1단계 선택 + 수정자(+sys/+orpo) 중 최종 선택. 결과 best_settings.yaml gen[모델]."""
    bp = CONFIG_DIR / "best_settings.yaml"
    best = yaml.safe_load(bp.read_text(encoding="utf-8")) or {}
    gen = best.setdefault("gen", {})
    now = datetime.now(timezone.utc).isoformat()
    for mid in a.models:
        pols = policies(mid)
        key = "evafrill" if mid.startswith("evafrill-mo-3b") and a.evafrill else mid
        ent = gen.setdefault(key, {})
        if a.stage == 1:
            vs = ["A", "card", "rep", "rp11", "rp12", "fp03"]
        else:
            d1 = ent["stage1"]["chosen"]
            vs = [d1, f"{d1}+sys"] + ([f"{d1}+orpo"] if mid == "frankenstallm-bf16" else [])
        cands = {f"{v}|{p}": d for v in vs for p in pols if (d := tune_dev(mid, v, None if p == "raw" else p))}
        if not cands:
            print(f"{mid}: 후보 결과 없음")
            continue
        base = f"{vs[0]}|{pols[0]}" if a.stage == 2 else f"A|{pols[0]}"
        ch, ok = choose_v3(cands, base if base in cands else next(iter(cands)))
        v, p = ch.split("|")
        rec = {"chosen": v, "unk_policy": p, "feasible": ok, "dev": cands, "selected_at": now, "rule": "v3"}
        if a.stage == 1:
            ent["stage1"] = rec
        else:
            ent["stage2"] = rec
            ent.update({"chosen": v, "unk_policy": p, "rule": "v3", "selected_at": now,
                        "gain_over_A_dev": round(cands[ch]["mean"] - cands.get(f"A|{pols[0]}", cands[ch])["mean"], 5)
                        if f"A|{pols[0]}" in cands else None})
            if key == "evafrill":
                ent["checkpoint"] = mid
                for m2 in ("evafrill-mo-3b", "evafrill-mo-3b-sft-v2", "evafrill-mo-3b-dpo-r1", "evafrill-mo-3b-dpo-r2",
                           "evafrill-mo-3b-dpo-r3", "evafrill-mo-3b-orpo"):
                    gen.setdefault(m2, {})["unk_policy"] = p
        m = cands[ch]["malfunction"]
        print(f"{mid} stage{a.stage}: {ch} 품질 {cands[ch]['mean']:.4f} 루프 {m['loop']:.3f} 언어 {m['lang']:.3f} 누출 {m['leak']:.3f} "
              f"잘림 {m['trunc']:.3f} (제약 만족 {len(ok)}/{len(cands)})")
    best["gen_rule_file"] = "dev_selection.yaml#generation_v3"
    bp.write_text(yaml.safe_dump(best, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return 0


def cmd_evafrill_ckpt(a) -> int:
    """EVAFRILL 체크포인트 선택(v1과 같음: 6 체크포인트 A, gen_dev_v1 평균). 결과는 gen.evafrill.checkpoint."""
    rule = yaml.safe_load((CONFIG_DIR / "dev_selection.yaml").read_text(encoding="utf-8"))["generation"]["search_space"]["evafrill"]
    bp = CONFIG_DIR / "best_settings.yaml"
    best = yaml.safe_load(bp.read_text(encoding="utf-8")) or {}
    c1 = {f"{ck}|{p}": d for ck in rule["checkpoints"] for p in rule["unk_policy"] if (d := dev_score(ck, "A", p))}
    top = max(c1, key=lambda k: c1[k]["mean"])
    # 기준점 slerp(최종)은 1위와 같은 unk 정책 값과 비교한다(체크포인트 효과만 보기 위해). 2026-11-20 수정: 이전에는
    # slerp|unk_nl과 비교해 정책 차이가 체크포인트 이득으로 잡혔다(sft-v2|unk_drop = slerp|unk_drop 동점인데 sft-v2 선택).
    base = f"evafrill-mo-3b|{top.split('|')[1]}"
    gain = c1[top]["mean"] - c1[base]["mean"] if base in c1 else 0.0
    ck = (top if gain > TOL else base).split("|")[0]
    e = best.setdefault("gen", {}).setdefault("evafrill", {})
    e.update({"checkpoint": ck, "checkpoint_dev": c1, "checkpoint_gain_over_slerp": round(gain, 5)})
    bp.write_text(yaml.safe_dump(best, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(ck)
    return 0


def repetition_stats(model_id: str, variant: str, task: str, policy: str) -> dict | None:
    """생성 jsonl에서 반복 지표(모든 턴): 루프율, 중복 3-gram 평균, 최대 길이 잘림 비율."""
    from kobench.standard.gen_score import apply_policy, is_loop, rep3

    rows = read_jsonl(out_dir(model_id, variant) / f"{task}.jsonl")
    outs = [(apply_policy(o, policy), f) for r in rows for o, f in zip(r["outputs"], r["finish"])]
    if not outs:
        return None
    r3 = [x for x in (rep3(o) for o, _ in outs) if x is not None]
    return {"loop": float(np.mean([is_loop(o) for o, _ in outs])), "rep3": float(np.mean(r3)) if r3 else None,
            "truncated": float(np.mean([f == "length" for _, f in outs]))}


# ── 표 ─────────────────────────────────────────────────────────────────────

def chosen(model_id: str) -> tuple[str, str]:
    b = (yaml.safe_load((CONFIG_DIR / "best_settings.yaml").read_text(encoding="utf-8")) or {}).get("gen", {})
    e = b.get(model_id, {})
    spec = load_models()[model_id]
    if spec.backend == "evafrill" and "evafrill" in b:
        ev = b["evafrill"]
        return ev.get("chosen", "A"), ev.get("unk_policy", "unk_nl")
    return e.get("chosen", "A"), e.get("unk_policy", "raw" if spec.owner != "self" else "unk_nl")


def fmt(task: str, v: float) -> str:
    return f"{v:.2f}" if task in SCALE else f"{100 * v:.1f}"


def cmd_table(a) -> int:
    cfg = load_gen_config()
    models = load_models()
    tasks = cfg["suites"][a.suite]
    rows = []
    ref = a.ref
    for mid, spec in models.items():
        if spec.backend == "llamacpp":
            continue
        for t in tasks:
            if spec.kind not in cfg["tasks"][t]["kinds"] and a.table != "U":
                continue                  # 통일 표 U는 모든 모델에 같은 태스크 (configs/standard/unified.yaml)
            v_ch, pol = chosen(mid)
            variant = {"A": "A", "B": v_ch, "think": "think", "thinkrd": "thinkrd", "U": "U"}[a.table]
            if a.table == "U" and spec.backend == "evafrill" and mid not in ("evafrill-mo-3b", "evafrill-mo-3b-pretrain"):
                continue
            if a.table == "A" and spec.backend == "evafrill" and spec.kind != "base" and mid != "evafrill-mo-3b" and a.suite != "gen_dev_v1":
                continue          # A 표의 EVAFRILL instruct는 최종(slerp)만 (dev 선택 체크포인트는 B 표). base(pretrain)는 base 코드 표에 남긴다
            s = load_scores(mid, variant, t, pol)
            if not s:
                continue
            x = np.array(list(s.values()))
            lo, hi = boot_ci(x)
            row = {"model": mid, "kind": spec.kind, "family": spec.family, "params_b": spec.params_b, "active_b": spec.active_b,
                   "variant": variant, "unk_policy": pol, "task": t, "metric": MAIN[t], "n": len(x), "value": float(x.mean()),
                   "ci_lo": lo, "ci_hi": hi}
            rs = load_scores(ref, "A" if a.table != "B" else chosen(ref)[0], t, chosen(ref)[1]) if ref else None
            if rs and mid != ref:
                pdiff = paired(s, rs)
                if pdiff:
                    row.update({"diff_vs_ref": pdiff[0], "diff_lo": pdiff[1], "diff_hi": pdiff[2]})
            if a.table == "B":
                sa = load_scores(mid, "A", t, pol)
                if sa and variant != "A":
                    pdiff = paired(s, sa)
                    if pdiff:
                        row.update({"delta_A": pdiff[0], "delta_A_lo": pdiff[1], "delta_A_hi": pdiff[2]})
            rs_ = repetition_stats(mid, variant, t, pol)
            if rs_:
                row.update({"loop": rs_["loop"], "rep3": rs_["rep3"], "truncated": rs_["truncated"]})
            judge_fam = spec.family in cfg["judge"]["same_family_as_judge"] and cfg["tasks"][t]["scorer"] == "judge"
            row["judge_same_family"] = bool(judge_fam)
            rows.append(row)
    REPORTS.mkdir(parents=True, exist_ok=True)
    stem = REPORTS / f"{a.suite}__{a.table}"
    keys = ["model", "kind", "family", "params_b", "active_b", "variant", "unk_policy", "task", "metric", "n", "value", "ci_lo",
            "ci_hi", "diff_vs_ref", "diff_lo", "diff_hi", "delta_A", "delta_A_lo", "delta_A_hi", "judge_same_family",
            "loop", "rep3", "truncated"]
    with open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    # markdown: 모델 × 태스크
    lines = [f"# 생성 벤치 {a.suite} — {a.table} 표", "",
             f"값: % (judge 점수는 원 척도), [95% CI]. 기준 모델 {ref} 대비 paired 차이는 CSV. † = judge와 같은 계열(자기 선호 가능).", ""]
    for kind in ("instruct", "base"):
        ks = [t for t in tasks if any(r["task"] == t and r["kind"] == kind for r in rows)]
        ms = list(dict.fromkeys(r["model"] for r in rows if r["kind"] == kind))
        if not ks:
            continue
        lines += [f"## {kind}", "", "| 모델 | 총/활성 B | 설정 | " + " | ".join(ks) + " |", "|---|---|---|" + "---|" * len(ks)]
        for m in ms:
            rr = {r["task"]: r for r in rows if r["model"] == m}
            any_r = next(iter(rr.values()))
            cells = []
            for t in ks:
                r = rr.get(t)
                if not r:
                    cells.append("—")
                    continue
                c = f"{fmt(t, r['value'])} [{fmt(t, r['ci_lo'])}, {fmt(t, r['ci_hi'])}]"
                if "delta_A" in r:
                    c += f" (ΔA {fmt(t, r['delta_A']) if t in SCALE else f'{100 * r['delta_A']:+.1f}'})"
                if r["judge_same_family"]:
                    c += " †"
                cells.append(c)
            lines.append(f"| {m} | {any_r['params_b']}/{any_r['active_b']} | {any_r['variant']}/{any_r['unk_policy']} | " + " | ".join(cells) + " |")
        lines.append("")
    # 반복 지표 표 (모델 × 태스크 루프율 %)
    lines += ["## 반복 지표 (루프율 % / 중복 3-gram % / 최대 길이 잘림 %)", "",
              "| 모델 | 설정 | " + " | ".join(tasks) + " |", "|---|---|" + "---|" * len(tasks)]
    for m in dict.fromkeys(r["model"] for r in rows):
        rr = {r["task"]: r for r in rows if r["model"] == m}
        anyr = next(iter(rr.values()))
        cells = [(f"{100 * rr[t]['loop']:.0f} / {100 * (rr[t]['rep3'] or 0):.0f} / {100 * rr[t]['truncated']:.0f}" if t in rr and rr[t].get("loop") is not None else "—")
                 for t in tasks]
        lines.append(f"| {m} | {anyr['variant']} | " + " | ".join(cells) + " |")
    lines.append("")
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{stem}.md ({len(rows)}행)")
    return 0


def tune_rows(best: dict) -> list[dict]:
    """규칙 v3 결과 요약: 모델별 A(같은 unk 정책)와 최종 선택의 dev 품질·오동작률, 후보 수, 제약 만족 여부."""
    out = []
    for key, e in (best.get("gen") or {}).items():
        if not isinstance(e, dict) or "stage2" not in e:
            continue
        s1, s2 = e["stage1"], e["stage2"]
        ch = f"{e['chosen']}|{e['unk_policy']}"
        d_ch = s2["dev"].get(ch) or s1["dev"].get(ch)
        a_key = next((k for k in s1["dev"] if k.split("|")[0] == "A" and k.split("|")[1] == e["unk_policy"]),
                     next((k for k in s1["dev"] if k.startswith("A|")), None))
        d_a = s1["dev"].get(a_key) if a_key else None
        row = {"model": e.get("checkpoint", key) if key == "evafrill" else key, "chosen": e["chosen"], "unk_policy": e["unk_policy"],
               "feasible": bool(s2["feasible"]), "n_candidates": len(s1["dev"]) + len(s2["dev"]) - 1}
        for tag, d in (("A", d_a), ("chosen", d_ch)):
            if not d:
                continue
            row[f"{tag}_quality"] = d["mean"]
            for t, v in d["tasks"].items():
                row[f"{tag}_{t}"] = v
            for k, v in d["malfunction"].items():
                row[f"{tag}_{k}"] = v
        out.append(row)
    return out


def cmd_tune_table(a) -> int:
    best = yaml.safe_load((CONFIG_DIR / "best_settings.yaml").read_text(encoding="utf-8")) or {}
    rows = tune_rows(best)
    REPORTS.mkdir(parents=True, exist_ok=True)
    stem = REPORTS / "tune_v3"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    pct = lambda v: "—" if v is None else f"{100 * v:.1f}"
    lines = ["# 생성 운용 설정 탐색 (규칙 v3, dev = tune_dev_v1)", "",
             f"제약: 루프 ≤{100 * LIMITS['loop']:.0f}%, 언어 이탈 ≤{100 * LIMITS['lang']:.0f}%, 템플릿 누출 ≤{100 * LIMITS['leak']:.1f}%, "
             f"잘림 ≤{100 * LIMITS['trunc']:.0f}%. 품질 = gsm8k_ko_dev·kormedmcqa_dev·scenario_dev_self 평균(%). "
             "'만족 ✗'는 어떤 후보도 제약을 다 지키지 못해 위반 최소 후보를 고른 경우.", "",
             "| 모델 | 선택 | 만족 | 품질 A → 선택 | 루프 A → 선택 | 언어 이탈 | 누출 | 잘림 |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda k: f"{pct(r.get('A_' + k))} → {pct(r.get('chosen_' + k))}"
        lines.append(f"| {r['model']} | {r['chosen']}/{r['unk_policy']} | {'✓' if r['feasible'] else '✗'} | {f('quality')} | {f('loop')} | "
                     f"{f('lang')} | {f('leak')} | {f('trunc')} |")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{stem}.md ({len(rows)}행)")
    return 0


def cmd_summarize(a) -> int:
    """공개 저장소용 요약: results/gen의 각 <task>.score.json에서 문항별 결과(items)를 뺀 <task>.summary.json.
    모델 출력(jsonl)·judge 기록·문항별 점수는 로컬(NFS)에만 둔다(.gitignore)."""
    root = Path(out_dir("x", "A")).parents[1]
    n = 0
    for f in sorted(root.glob("*/*/*.score.json")):
        if f.parts[-3].startswith("_") or "__limit" in f.name:
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        d.pop("items", None)
        out = f.with_name(f.name.replace(".score.json", ".summary.json"))
        txt = json.dumps(d, ensure_ascii=False, indent=1) + "\n"
        if not out.exists() or out.read_text(encoding="utf-8") != txt:
            out.write_text(txt, encoding="utf-8")
        n += 1
    print(f"summary {n}개 ({root})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    tu = sub.add_parser("tune")
    tu.add_argument("--models", nargs="+", required=True)
    tu.add_argument("--stage", type=int, choices=[1, 2], required=True)
    tu.add_argument("--evafrill", action="store_true", help="EVAFRILL 선택 체크포인트 (결과 키 gen.evafrill)")
    sub.add_parser("evafrill_ckpt")
    sub.add_parser("summarize")
    t = sub.add_parser("table")
    t.add_argument("--table", default="A", choices=["A", "B", "think", "thinkrd", "tune", "U"])
    t.add_argument("--suite", default="gen_v1")
    t.add_argument("--ref", default="frankenstallm-bf16")
    a = ap.parse_args(argv)
    if a.cmd == "table" and a.table == "tune":
        return cmd_tune_table(a)
    return {"select": cmd_select, "table": cmd_table, "tune": cmd_tune, "evafrill_ckpt": cmd_evafrill_ckpt, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
